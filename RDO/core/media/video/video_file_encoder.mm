#include "video_file_encoder.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace rdo
{
namespace core
{
namespace media
{
namespace video
{

VideoFileEncoder::VideoFileEncoder(id<MTLDevice> device)
    : device_(device)
    , texture_cache_(nullptr)
    , writer_(nil)
    , writer_input_(nil)
    , pixel_buffer_adaptor_(nil)
    , append_queue_(dispatch_queue_create("rdo.video.encoder.append", DISPATCH_QUEUE_SERIAL))
    , command_queue_(nil)
    , is_recording_(false)
    , has_error_(false)
    , error_message_()
    , width_(0)
    , height_(0)
    , fps_(30.0f)
    , frame_duration_(CMTimeMake(1, 30))
    , frame_index_(0)
{
    if (device_)
    {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device_, nullptr, &texture_cache_);
        command_queue_ = [device_ newCommandQueue];
    }
}

VideoFileEncoder::~VideoFileEncoder()
{
    Stop();
    if (texture_cache_)
    {
        CVMetalTextureCacheFlush(texture_cache_, 0);
        CFRelease(texture_cache_);
        texture_cache_ = nullptr;
    }
}

void VideoFileEncoder::SetError(const std::string& message)
{
    has_error_ = true;
    error_message_ = message;
    std::cerr << "VideoFileEncoder: " << message << std::endl;
}

bool VideoFileEncoder::InitializeWriter(const std::string& file_path)
{
    if (file_path.empty())
    {
        SetError("Empty output file path");
        return false;
    }

    NSString* path = [NSString stringWithUTF8String:file_path.c_str()];
    if (!path || [path length] == 0)
    {
        SetError("Invalid output file path");
        return false;
    }

    NSURL* output_url = [NSURL fileURLWithPath:path];
    if (!output_url)
    {
        SetError("Failed to create output URL");
        return false;
    }

    [[NSFileManager defaultManager] removeItemAtURL:output_url error:nil];

    NSString* ext = [[path pathExtension] lowercaseString];
    AVFileType file_type = [ext isEqualToString:@"mp4"] ? AVFileTypeMPEG4 : AVFileTypeQuickTimeMovie;

    NSError* error = nil;
    writer_ = [[AVAssetWriter alloc] initWithURL:output_url fileType:file_type error:&error];
    if (!writer_ || error)
    {
        SetError(error ? [[error localizedDescription] UTF8String] : "Failed to create AVAssetWriter");
        writer_ = nil;
        return false;
    }

    NSString* codec = AVVideoCodecTypeH264;
    if ([ext isEqualToString:@"mov"])
    {
        if (@available(macOS 10.13, *))
        {
            codec = AVVideoCodecTypeHEVC;
        }
    }

    const float target_bpp =
        [codec isEqualToString:AVVideoCodecTypeHEVC] ? 0.045f : 0.070f;
    const double raw_bitrate = static_cast<double>(width_) *
                               static_cast<double>(height_) *
                               static_cast<double>(fps_) *
                               static_cast<double>(target_bpp);
    const int bitrate = std::clamp(static_cast<int>(std::round(raw_bitrate)), 400000, 20000000);
    const int keyframe_interval = std::max(1, static_cast<int>(std::round(fps_ * 2.0f)));
    const int source_fps = std::max(1, static_cast<int>(std::round(fps_)));

    NSMutableDictionary* compression = [@{
        AVVideoAverageBitRateKey: @(bitrate),
        AVVideoExpectedSourceFrameRateKey: @(source_fps),
        AVVideoMaxKeyFrameIntervalKey: @(keyframe_interval)
    } mutableCopy];

    if ([codec isEqualToString:AVVideoCodecTypeH264])
    {
        compression[AVVideoProfileLevelKey] = AVVideoProfileLevelH264MainAutoLevel;
    }

    NSDictionary* video_settings = @{
        AVVideoCodecKey: codec,
        AVVideoWidthKey: @(width_),
        AVVideoHeightKey: @(height_),
        AVVideoCompressionPropertiesKey: compression
    };

    writer_input_ = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                         outputSettings:video_settings];
    // 오프라인 인코딩: 실시간 입력 플래그 비활성화 (압축 효율/품질 안정화)
    writer_input_.expectsMediaDataInRealTime = NO;

    if (![writer_ canAddInput:writer_input_])
    {
        SetError("Writer cannot add video input");
        CleanupWriter();
        return false;
    }
    [writer_ addInput:writer_input_];

    NSDictionary* source_attributes = @{
        (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString*)kCVPixelBufferWidthKey: @(width_),
        (NSString*)kCVPixelBufferHeightKey: @(height_),
        (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES,
        (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };

    pixel_buffer_adaptor_ = [AVAssetWriterInputPixelBufferAdaptor
        assetWriterInputPixelBufferAdaptorWithAssetWriterInput:writer_input_
                                    sourcePixelBufferAttributes:source_attributes];

    if (![writer_ startWriting])
    {
        NSError* start_error = [writer_ error];
        SetError(start_error ? [[start_error localizedDescription] UTF8String] : "Failed to start writing");
        CleanupWriter();
        return false;
    }

    [writer_ startSessionAtSourceTime:kCMTimeZero];
    return true;
}

void VideoFileEncoder::CleanupWriter()
{
    writer_input_ = nil;
    pixel_buffer_adaptor_ = nil;
    writer_ = nil;
}

bool VideoFileEncoder::Start(const std::string& file_path, int width, int height, float fps)
{
    Stop();

    has_error_ = false;
    error_message_.clear();

    if (!device_)
    {
        SetError("Metal device is null");
        return false;
    }
    if (!texture_cache_)
    {
        SetError("Metal texture cache is unavailable");
        return false;
    }
    if (width <= 0 || height <= 0)
    {
        SetError("Invalid output size");
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = std::max(1.0f, fps);
    frame_duration_ = CMTimeMakeWithSeconds(1.0 / fps_, 60000);
    frame_index_ = 0;

    if (!InitializeWriter(file_path))
    {
        return false;
    }

    is_recording_ = true;
    return true;
}

void VideoFileEncoder::Stop()
{
    if (!writer_)
    {
        is_recording_ = false;
        return;
    }

    is_recording_ = false;

    if (writer_input_)
    {
        [writer_input_ markAsFinished];
    }

    // pending append 작업이 모두 끝나도록 보장
    dispatch_sync(append_queue_, ^{});

    AVAssetWriter* writer = writer_;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [writer finishWritingWithCompletionHandler:^{
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    if ([writer status] == AVAssetWriterStatusFailed)
    {
        NSError* error = [writer error];
        if (error)
        {
            SetError([[error localizedDescription] UTF8String]);
        }
    }

    CleanupWriter();
    frame_index_ = 0;
}

void VideoFileEncoder::EncodeFrame(id<MTLTexture> source_texture, id<MTLCommandBuffer> cmd_buffer)
{
    if (!is_recording_ || !source_texture || !cmd_buffer || !writer_input_ || !pixel_buffer_adaptor_)
    {
        return;
    }

    if ((int)[source_texture width] != width_ || (int)[source_texture height] != height_)
    {
        // 해상도 변경 프레임은 스킵 (세션 고정 해상도 유지)
        return;
    }

    if (![writer_input_ isReadyForMoreMediaData])
    {
        return;
    }

    CVPixelBufferPoolRef pool = [pixel_buffer_adaptor_ pixelBufferPool];
    if (!pool)
    {
        return;
    }

    CVPixelBufferRef pixel_buffer = nullptr;
    CVReturn pb_result = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixel_buffer);
    if (pb_result != kCVReturnSuccess || !pixel_buffer)
    {
        return;
    }

    CVMetalTextureRef metal_texture = nullptr;
    CVReturn tex_result = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        texture_cache_,
        pixel_buffer,
        nullptr,
        MTLPixelFormatBGRA8Unorm,
        width_,
        height_,
        0,
        &metal_texture
    );

    if (tex_result != kCVReturnSuccess || !metal_texture)
    {
        CVPixelBufferRelease(pixel_buffer);
        return;
    }

    id<MTLTexture> destination_texture = CVMetalTextureGetTexture(metal_texture);
    if (!destination_texture)
    {
        CFRelease(metal_texture);
        CVPixelBufferRelease(pixel_buffer);
        return;
    }

    id<MTLBlitCommandEncoder> blit = [cmd_buffer blitCommandEncoder];
    [blit copyFromTexture:source_texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width_, height_, 1)
                toTexture:destination_texture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];

    const int64_t frame_idx = frame_index_.fetch_add(1);
    const CMTime pts = CMTimeMake(frame_duration_.value * frame_idx, frame_duration_.timescale);

    AVAssetWriterInput* writer_input = writer_input_;
    AVAssetWriterInputPixelBufferAdaptor* adaptor = pixel_buffer_adaptor_;
    dispatch_queue_t append_queue = append_queue_;

    [cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
        dispatch_async(append_queue, ^{
            if (completed_buffer.status == MTLCommandBufferStatusCompleted &&
                [writer_input isReadyForMoreMediaData])
            {
                [adaptor appendPixelBuffer:pixel_buffer withPresentationTime:pts];
            }
            CVPixelBufferRelease(pixel_buffer);
            CFRelease(metal_texture);
        });
    }];
}

void VideoFileEncoder::EncodeFrame(id<MTLTexture> source_texture)
{
    if (!is_recording_ || !source_texture || !command_queue_)
    {
        return;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    EncodeFrame(source_texture, command_buffer);
    [command_buffer commit];

    // 오프라인 인코딩 경로: 프레임 순서를 보장하기 위해 동기 완료 대기
    [command_buffer waitUntilCompleted];
    dispatch_sync(append_queue_, ^{});
}

} // namespace video
} // namespace media
} // namespace core
} // namespace rdo
