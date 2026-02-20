#pragma once

#import <Metal/Metal.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include <atomic>
#include <string>

namespace rdo
{
namespace core
{
namespace media
{
namespace video
{

// 독립 출력용 비디오 인코더 (타임라인/프리뷰 경로와 분리)
class VideoFileEncoder
{
public:
    explicit VideoFileEncoder(id<MTLDevice> device);
    ~VideoFileEncoder();

    bool Start(const std::string& file_path, int width, int height, float fps);
    void Stop();

    bool IsRecording() const { return is_recording_; }
    bool HasError() const { return has_error_; }
    std::string GetErrorMessage() const { return error_message_; }

    // 오프라인 인코딩용: 내부 CommandBuffer 생성 후 동기 인코딩
    void EncodeFrame(id<MTLTexture> source_texture);

    // 외부 커맨드 버퍼를 사용하는 실시간/파이프라인 연동용
    void EncodeFrame(id<MTLTexture> source_texture, id<MTLCommandBuffer> cmd_buffer);

private:
    bool InitializeWriter(const std::string& file_path);
    void CleanupWriter();
    void SetError(const std::string& message);

    id<MTLDevice> device_;
    CVMetalTextureCacheRef texture_cache_;

    AVAssetWriter* writer_;
    AVAssetWriterInput* writer_input_;
    AVAssetWriterInputPixelBufferAdaptor* pixel_buffer_adaptor_;
    dispatch_queue_t append_queue_;
    id<MTLCommandQueue> command_queue_;

    std::atomic<bool> is_recording_;
    std::atomic<bool> has_error_;
    std::string error_message_;

    int width_;
    int height_;
    float fps_;
    CMTime frame_duration_;
    std::atomic<int64_t> frame_index_;
};

} // namespace video
} // namespace media
} // namespace core
} // namespace rdo
