#include "video_engine.h"
#include <iostream>

namespace rdo
{
namespace core
{
namespace media
{
namespace video
{

VideoEngine::VideoEngine(id<MTLDevice> device)
    : device_(device)
    , texture_cache_(nullptr)
    , display_texture_(nil)
    , ready_texture_(nil)
    , ready_pixel_buffer_(nullptr)
    , new_frame_ready_(false)
    , asset_(nil)
    , asset_reader_(nil)
    , video_output_(nil)
    , video_track_(nil)
    , duration_(0.0f)
    , current_time_(0.0f)
    , width_(0)
    , height_(0)
    , video_fps_(30.0f)
    , playback_rate_(1.0f)
    , mode_(EngineMode::Preview)
    , is_loading_(false)
    , is_loaded_(false)
    , is_playing_(false)
    , should_stop_(false)
    , frame_request_pending_(false)
    , has_error_(false)
    , playback_start_video_time_(0.0)
    , frame_count_(0)
{
    InitializeTextureCache();
}

VideoEngine::~VideoEngine()
{
    std::cerr << "VideoEngine: Destructor called" << std::endl;
    
    // 1. 모든 재생 활동 중지
    is_playing_ = false;
    should_stop_ = true;
    
    // 2. 디코딩 스레드 종료 대기 (최우선)
    if (decoding_thread_.joinable())
    {
        std::cerr << "VideoEngine: Waiting for decoding thread..." << std::endl;
        decoding_thread_.join();
        std::cerr << "VideoEngine: Decoding thread joined" << std::endl;
    }
    
    // 3. 로딩 스레드 종료 대기
    if (loading_thread_.joinable())
    {
        std::cerr << "VideoEngine: Waiting for loading thread..." << std::endl;
        loading_thread_.join();
        std::cerr << "VideoEngine: Loading thread joined" << std::endl;
    }
    
    // 4. AssetReader 정리
    CleanupAssetReader();
    
    // 5. 텍스처 정리 (mutex 보호)
    {
        std::lock_guard<std::mutex> lock(texture_mutex_);
        display_texture_ = nil;
        ready_texture_ = nil;
    }
    
    // 6. 픽셀 버퍼 해제
    if (ready_pixel_buffer_)
    {
        CVPixelBufferRelease(ready_pixel_buffer_);
        ready_pixel_buffer_ = nullptr;
    }
    
    // 7. 텍스처 캐시 정리 (마지막에)
    CleanupTextureCache();
    
    // 8. AVFoundation 객체 해제
    asset_ = nil;
    video_track_ = nil;
    
    std::cerr << "VideoEngine: Destructor complete" << std::endl;
}

bool VideoEngine::InitializeTextureCache()
{
    if (!device_)
    {
        std::cerr << "VideoEngine: Metal device is null" << std::endl;
        return false;
    }
    
    if (texture_cache_)
    {
        return true;
    }
    
    CVReturn result = CVMetalTextureCacheCreate(
        kCFAllocatorDefault,
        nullptr,
        device_,
        nullptr,
        &texture_cache_
    );
    
    if (result != kCVReturnSuccess)
    {
        std::cerr << "VideoEngine: Failed to create texture cache: " << result << std::endl;
        return false;
    }
    
    return true;
}

void VideoEngine::CleanupTextureCache()
{
    if (texture_cache_)
    {
        std::cerr << "VideoEngine: Flushing texture cache" << std::endl;
        // 완전히 플러시 (모든 텍스처 해제)
        CVMetalTextureCacheFlush(texture_cache_, 0);
        
        // 약간의 대기 시간
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        CFRelease(texture_cache_);
        texture_cache_ = nullptr;
        std::cerr << "VideoEngine: Texture cache released" << std::endl;
    }
}

#pragma mark - 비동기 로딩

void VideoEngine::LoadVideoAsync(const std::string& file_path)
{
    // 이미 로딩 중이면 무시
    if (is_loading_)
    {
        std::cerr << "VideoEngine: Already loading, ignoring new request" << std::endl;
        return;
    }
    
    std::cerr << "VideoEngine: LoadVideoAsync called for: " << file_path << std::endl;
    
    // 이전 상태 정리
    Stop();
    
    // 이전 로딩 스레드 정리 (Stop에서 decoding_thread는 이미 정리됨)
    if (loading_thread_.joinable())
    {
        std::cerr << "VideoEngine: Waiting for previous loading thread..." << std::endl;
        loading_thread_.join();
        std::cerr << "VideoEngine: Previous loading thread joined" << std::endl;
    }
    
    // 상태 초기화
    is_loading_ = true;
    is_loaded_ = false;
    has_error_ = false;
    error_message_.clear();
    file_path_ = file_path;
    
    // 백그라운드에서 로딩 시작
    loading_thread_ = std::thread(&VideoEngine::LoadingThread, this, file_path);
}

void VideoEngine::LoadingThread(const std::string& file_path)
{
    @autoreleasepool {
        if (InitializeVideo(file_path))
        {
            is_loaded_ = true;
            std::cerr << "VideoEngine: Loaded " << width_.load() << "x" << height_.load()
                      << ", " << duration_.load() << "s, " << video_fps_ << "fps" << std::endl;
        }
        else
        {
            has_error_ = true;
            if (error_message_.empty())
            {
                error_message_ = "Failed to load video";
            }
            std::cerr << "VideoEngine: " << error_message_ << std::endl;
        }
        
        is_loading_ = false;
    }
}

bool VideoEngine::InitializeVideo(const std::string& file_path)
{
    // 이전 리소스 정리
    CleanupAssetReader();
    asset_ = nil;
    video_track_ = nil;
    
    if (file_path.empty())
    {
        error_message_ = "Empty file path";
        return false;
    }
    
    // AVAsset 생성
    NSString* path = [NSString stringWithUTF8String:file_path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path];
    
    if (!url)
    {
        error_message_ = "Invalid file path";
        return false;
    }
    
    asset_ = [AVAsset assetWithURL:url];
    if (!asset_)
    {
        error_message_ = "Failed to create AVAsset";
        return false;
    }
    
    // 비디오 트랙 가져오기
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    NSArray<AVAssetTrack*>* tracks = [asset_ tracksWithMediaType:AVMediaTypeVideo];
    #pragma clang diagnostic pop
    
    if (tracks.count == 0)
    {
        error_message_ = "No video track found";
        return false;
    }
    
    video_track_ = tracks[0];
    
    // 영상 정보
    CGSize size = [video_track_ naturalSize];
    width_ = static_cast<int>(size.width);
    height_ = static_cast<int>(size.height);
    duration_ = CMTimeGetSeconds([asset_ duration]);
    
    video_fps_ = [video_track_ nominalFrameRate];
    if (video_fps_ <= 0 || video_fps_ > 120)
    {
        video_fps_ = 30.0f;
    }
    
    // AssetReader 생성
    if (!CreateAssetReader())
    {
        return false;
    }
    
    // 디코딩 스레드 시작
    should_stop_ = false;
    frame_request_pending_ = true; // 로드 직후 정지 상태에서도 첫 프레임은 준비
    decoding_thread_ = std::thread(&VideoEngine::DecodingThread, this);
    
    return true;
}

#pragma mark - AssetReader 관리

bool VideoEngine::CreateAssetReader()
{
    if (!asset_ || !video_track_)
    {
        error_message_ = "No asset or track";
        return false;
    }
    
    // 기존 reader 정리
    if (asset_reader_)
    {
        [asset_reader_ cancelReading];
        asset_reader_ = nil;
    }
    video_output_ = nil;
    
    NSError* error = nil;
    asset_reader_ = [[AVAssetReader alloc] initWithAsset:asset_ error:&error];
    
    if (error || !asset_reader_)
    {
        error_message_ = "Failed to create AVAssetReader";
        return false;
    }
    
    // 출력 설정 - BGRA 우선 (컬러 정상 표시)
    // 옵션 1: BGRA (가장 안정적, 컬러 출력)
    NSDictionary* settings = @{
        (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
        (NSString*)kCVPixelBufferMetalCompatibilityKey: @(YES)
    };
    
    video_output_ = [[AVAssetReaderTrackOutput alloc] initWithTrack:video_track_ 
                                                      outputSettings:settings];
    
    if (!video_output_ || ![asset_reader_ canAddOutput:video_output_])
    {
        // 폴백: BGRA (IOSurface 없이)
        settings = @{
            (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (NSString*)kCVPixelBufferMetalCompatibilityKey: @(YES)
        };
        
        video_output_ = [[AVAssetReaderTrackOutput alloc] initWithTrack:video_track_ 
                                                          outputSettings:settings];
        
        if (!video_output_ || ![asset_reader_ canAddOutput:video_output_])
        {
            error_message_ = "Failed to create video output";
            return false;
        }
        
        std::cerr << "VideoEngine: Using BGRA without IOSurface" << std::endl;
    }
    else
    {
        std::cerr << "VideoEngine: Using BGRA with IOSurface (zero-copy)" << std::endl;
    }
    
    video_output_.alwaysCopiesSampleData = NO; // Zero-copy!
    [asset_reader_ addOutput:video_output_];
    
    return true;
}

bool VideoEngine::StartAssetReader()
{
    if (!asset_reader_)
    {
        return false;
    }
    
    AVAssetReaderStatus status = [asset_reader_ status];
    
    if (status == AVAssetReaderStatusReading)
    {
        return true;
    }
    
    if (status != AVAssetReaderStatusUnknown)
    {
        // 재생성 필요
        if (!CreateAssetReader())
        {
            return false;
        }
    }
    
    if (![asset_reader_ startReading])
    {
        NSError* error = [asset_reader_ error];
        std::cerr << "VideoEngine: startReading failed: " 
                  << (error ? [[error localizedDescription] UTF8String] : "unknown") << std::endl;
        return false;
    }
    
    return true;
}

void VideoEngine::CleanupAssetReader()
{
    @autoreleasepool {
        if (asset_reader_)
        {
            AVAssetReaderStatus status = [asset_reader_ status];
            if (status == AVAssetReaderStatusReading)
            {
                std::cerr << "VideoEngine: Cancelling asset reader" << std::endl;
                [asset_reader_ cancelReading];
                
                // cancelReading이 완료될 때까지 잠시 대기
                int wait_count = 0;
                while ([asset_reader_ status] == AVAssetReaderStatusReading && wait_count < 50)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    wait_count++;
                }
            }
            asset_reader_ = nil;
        }
        video_output_ = nil;
    }
}

#pragma mark - 디코딩 스레드

void VideoEngine::DecodingThread()
{
    std::cerr << "VideoEngine: Decoding thread started" << std::endl;
    
    bool reader_started = false;
    auto frame_duration = std::chrono::microseconds(static_cast<int>(1000000.0 / video_fps_));
    
    while (!should_stop_)
    {
        @autoreleasepool {  // 각 루프마다 autorelease pool
        
        bool should_continue = false;
        
        // AssetReader 확인 및 시작
        if (!asset_reader_ || !video_output_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            should_continue = true;
        }
        
        if (!should_continue && !reader_started)
        {
            if (!StartAssetReader())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                should_continue = true;
            }
            else
            {
                reader_started = true;
                std::cerr << "VideoEngine: AssetReader started" << std::endl;
            }
        }
        
        // 상태 확인
        if (!should_continue)
        {
            AVAssetReaderStatus status = [asset_reader_ status];
            
            if (status == AVAssetReaderStatusFailed)
            {
                std::cerr << "VideoEngine: Reader failed, recreating..." << std::endl;
                CleanupAssetReader();
                if (CreateAssetReader())
                {
                    reader_started = false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                should_continue = true;
            }
            else if (status == AVAssetReaderStatusCompleted)
            {
                // 영상 끝
                if (is_playing_)
                {
                    // 루프 재생
                    CleanupAssetReader();
                    if (CreateAssetReader())
                    {
                        reader_started = false;
                        current_time_ = 0.0f;
                        playback_start_time_ = std::chrono::steady_clock::now();
                        playback_start_video_time_ = 0.0;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                should_continue = true;
            }
            else if (status != AVAssetReaderStatusReading)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                should_continue = true;
            }
        }
        
        // 재생 중이 아니면 프레임 읽기 중지
        // 단, frame_request_pending_이 true면 단일 프레임 디코드는 허용
        if (!should_continue && !is_playing_)
        {
            if (!frame_request_pending_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                should_continue = true;
            }
        }
        
        // 이미 ready 프레임이 있으면 대기 (double buffering) - 재생 중일 때만
        if (!should_continue && new_frame_ready_ && is_playing_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            should_continue = true;
        }
        
        if (!should_continue)
        {
            // 다음 프레임 읽기
            CMSampleBufferRef sampleBuffer = [video_output_ copyNextSampleBuffer];
            
            if (!sampleBuffer)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            else
            {
                // 프레임 처리
                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
                CMTime pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
                
                if (imageBuffer)
                {
                    // 텍스처 생성
                    id<MTLTexture> texture = CreateTextureFromPixelBuffer(imageBuffer);
                    
                    if (texture)
                    {
                        std::lock_guard<std::mutex> lock(texture_mutex_);
                        
                        // 이전 ready 버퍼 해제
                        if (ready_pixel_buffer_)
                        {
                            CVPixelBufferRelease(ready_pixel_buffer_);
                            ready_pixel_buffer_ = nullptr;
                        }
                        
                        // 새 프레임 설정
                        ready_pixel_buffer_ = CVPixelBufferRetain(imageBuffer);
                        ready_texture_ = texture;
                        current_time_ = CMTimeGetSeconds(pts);
                        new_frame_ready_ = true;
                        if (!is_playing_)
                        {
                            frame_request_pending_ = false;
                        }
                    }
                }
                
                // 중요: sampleBuffer를 즉시 해제하여 메모리 누수 방지
                CFRelease(sampleBuffer);
                
                // 재생 중이면 프레임 속도에 맞춰 pacing
                if (is_playing_)
                {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                        now - last_frame_time_);
                    
                    if (elapsed < frame_duration)
                    {
                        std::this_thread::sleep_for(frame_duration - elapsed);
                    }
                    
                    last_frame_time_ = std::chrono::steady_clock::now();
                }
            }
        }
        
        } // autoreleasepool - 각 루프마다
    }
    
    std::cerr << "VideoEngine: Decoding thread exiting" << std::endl;
}

#pragma mark - 텍스처 생성

id<MTLTexture> VideoEngine::CreateTextureFromPixelBuffer(CVPixelBufferRef pixelBuffer)
{
    if (!pixelBuffer || !texture_cache_)
    {
        return nil;
    }
    
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    OSType format = CVPixelBufferGetPixelFormatType(pixelBuffer);
    
    MTLPixelFormat metalFormat;
    size_t planeIndex = 0;
    
    if (format == kCVPixelFormatType_32BGRA)
    {
        // BGRA - 컬러 정상 출력
        metalFormat = MTLPixelFormatBGRA8Unorm;
        planeIndex = 0;
    }
    else if (format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
             format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    {
        // YUV - Y plane만 사용 (흑백)
        // 참고: 컬러가 필요하면 Y/UV 각각 텍스처 생성 후 셰이더에서 변환 필요
        metalFormat = MTLPixelFormatR8Unorm;
        planeIndex = 0;
    }
    else
    {
        std::cerr << "CreateTextureFromPixelBuffer: Unsupported format: " << format << std::endl;
        return nil;
    }
    
    CVMetalTextureRef metalTexture = nullptr;
    CVReturn result = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        texture_cache_,
        pixelBuffer,
        nullptr,
        metalFormat,
        static_cast<int>(width),
        static_cast<int>(height),
        planeIndex,
        &metalTexture
    );
    
    if (result != kCVReturnSuccess || !metalTexture)
    {
        return nil;
    }
    
    id<MTLTexture> texture = CVMetalTextureGetTexture(metalTexture);
    CFRelease(metalTexture);
    
    return texture;
}

#pragma mark - 재생 제어

void VideoEngine::Play()
{
    if (!is_loaded_)
    {
        return;
    }
    
    if (is_playing_)
    {
        return; // 이미 재생 중
    }
    
    // Stop 후 재시작할 수 있도록 스레드 재시작
    if (!decoding_thread_.joinable())
    {
        should_stop_ = false;
        
        // AssetReader가 없으면 재생성
        if (!asset_reader_)
        {
            if (!CreateAssetReader())
            {
                std::cerr << "VideoEngine: Failed to create AssetReader" << std::endl;
                return;
            }
        }
        
        decoding_thread_ = std::thread(&VideoEngine::DecodingThread, this);
        std::cerr << "VideoEngine: Decoding thread restarted" << std::endl;
    }
    
    is_playing_ = true;
    frame_request_pending_ = false;
    playback_start_time_ = std::chrono::steady_clock::now();
    playback_start_video_time_ = current_time_;
    last_frame_time_ = std::chrono::steady_clock::now();
}

void VideoEngine::Pause()
{
    is_playing_ = false;
}

void VideoEngine::Stop()
{
    std::cerr << "VideoEngine: Stop called" << std::endl;
    
    // 재생 중지
    is_playing_ = false;
    should_stop_ = true;
    frame_request_pending_ = false;
    
    // 디코딩 스레드 종료 대기
    if (decoding_thread_.joinable())
    {
        std::cerr << "VideoEngine: Waiting for decoding thread to stop..." << std::endl;
        decoding_thread_.join();
        std::cerr << "VideoEngine: Decoding thread stopped" << std::endl;
    }
    
    // AssetReader 정리
    CleanupAssetReader();
    
    // 텍스처 정리
    {
        std::lock_guard<std::mutex> lock(texture_mutex_);
        display_texture_ = nil;
        ready_texture_ = nil;
        if (ready_pixel_buffer_)
        {
            CVPixelBufferRelease(ready_pixel_buffer_);
            ready_pixel_buffer_ = nullptr;
        }
        new_frame_ready_ = false;
    }
    
    // 텍스처 캐시 플러시
    if (texture_cache_)
    {
        CVMetalTextureCacheFlush(texture_cache_, 0);
    }
    
    // 시간 리셋
    current_time_ = 0.0f;
    
    std::cerr << "VideoEngine: Stopped" << std::endl;
}

void VideoEngine::SeekToTime(float time)
{
    if (!is_loaded_)
    {
        return;
    }
    
    time = std::max(0.0f, std::min(time, duration_.load()));
    
    bool wasPlaying = is_playing_;
    
    // 재생 일시 정지
    is_playing_ = false;
    should_stop_ = true;
    
    // 디코딩 스레드 종료 대기
    if (decoding_thread_.joinable())
    {
        decoding_thread_.join();
    }
    
    // 텍스처 클리어
    {
        std::lock_guard<std::mutex> lock(texture_mutex_);
        display_texture_ = nil;
        ready_texture_ = nil;
        if (ready_pixel_buffer_)
        {
            CVPixelBufferRelease(ready_pixel_buffer_);
            ready_pixel_buffer_ = nullptr;
        }
        new_frame_ready_ = false;
    }
    
    // AssetReader 재생성 (시간 범위 설정)
    CleanupAssetReader();
    
    if (CreateAssetReader())
    {
        CMTime seekTime = CMTimeMakeWithSeconds(time, 600);
        [asset_reader_ setTimeRange:CMTimeRangeMake(seekTime, kCMTimePositiveInfinity)];
    }
    
    current_time_ = time;
    frame_request_pending_ = true; // 정지 상태에서도 seek 결과 프레임 1장을 준비
    
    // 스레드 재시작 준비
    should_stop_ = false;
    
    // 재생 중이었다면 다시 시작
    if (wasPlaying)
    {
        Play();
    }
    else
    {
        // 일시정지 상태였다면 디코딩 스레드만 재시작 (첫 프레임 표시용)
        if (!decoding_thread_.joinable())
        {
            decoding_thread_ = std::thread(&VideoEngine::DecodingThread, this);
        }
    }
}

#pragma mark - Update (메인 스레드)

void VideoEngine::Update()
{
    // 새 프레임이 준비되었으면 교체
    if (new_frame_ready_)
    {
        std::lock_guard<std::mutex> lock(texture_mutex_);
        
        if (ready_texture_)
        {
            display_texture_ = ready_texture_;
            ready_texture_ = nil;
            new_frame_ready_ = false;
            
            // 주기적 캐시 플러시 (메모리 누수 방지)
            frame_count_++;
            if (frame_count_ >= FLUSH_INTERVAL)
            {
                if (texture_cache_)
                {
                    CVMetalTextureCacheFlush(texture_cache_, 0);
                }
                frame_count_ = 0;
            }
        }
    }
}

id<MTLTexture> VideoEngine::GetCurrentFrameTexture()
{
    std::lock_guard<std::mutex> lock(texture_mutex_);
    return display_texture_;
}

} // namespace video
} // namespace media
} // namespace core
} // namespace rdo
