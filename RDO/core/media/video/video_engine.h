#pragma once

#import <Metal/Metal.h>
#import <AVFoundation/AVFoundation.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreVideo/CoreVideo.h>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

namespace rdo
{
namespace core
{
namespace media
{
namespace video
{

// Metal 기반 zero-copy 영상 재생 엔진
// 설계 원칙:
// 1. 로딩은 백그라운드에서 비동기로
// 2. 디코딩 스레드가 프레임 속도 제어
// 3. 메인 스레드는 텍스처 교체만 (최소 작업)
// 4. Double buffering으로 안정적 프레임 전달

class VideoEngine
{
public:
    VideoEngine(id<MTLDevice> device);
    ~VideoEngine();
    
    // 영상 파일 로드 (비동기 - 즉시 반환)
    void LoadVideoAsync(const std::string& file_path);
    
    // 로딩 상태
    bool IsLoading() const { return is_loading_; }
    bool IsLoaded() const { return is_loaded_; }
    bool HasError() const { return has_error_; }
    std::string GetErrorMessage() const { return error_message_; }
    
    // 재생 제어
    void Play();
    void Pause();
    void Stop();
    void SeekToTime(float time);
    
    // 현재 프레임 텍스처 (zero-copy)
    id<MTLTexture> GetCurrentFrameTexture();
    
    // 프레임 업데이트 (메인 스레드에서 호출, 가볍게 동작)
    void Update();
    
    // 영상 정보
    float GetDuration() const { return duration_; }
    float GetCurrentTime() const { return current_time_; }
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    bool IsPlaying() const { return is_playing_; }
    float GetFPS() const { return video_fps_; }
    
    // 재생 속도
    void SetPlaybackRate(float rate) { playback_rate_ = rate; }
    float GetPlaybackRate() const { return playback_rate_; }
    
    // 모드 설정 (Preview/Export 분리)
    enum class EngineMode {
        Preview,  // 프리뷰 모드 (Timeline 제어 가능)
        Export    // Export 모드 (Timeline 제어 불가, 독립적)
    };
    
    void SetMode(EngineMode mode) { mode_ = mode; }
    EngineMode GetMode() const { return mode_; }
    bool IsPreviewMode() const { return mode_ == EngineMode::Preview; }
    
private:
    // 로딩 스레드
    void LoadingThread(const std::string& file_path);
    bool InitializeVideo(const std::string& file_path);
    
    // 디코딩 스레드
    void DecodingThread();
    
    // AssetReader 관리
    bool CreateAssetReader();
    bool StartAssetReader();
    void CleanupAssetReader();
    
    // Metal 텍스처 캐시
    bool InitializeTextureCache();
    void CleanupTextureCache();
    
    // 텍스처 생성 (zero-copy)
    id<MTLTexture> CreateTextureFromPixelBuffer(CVPixelBufferRef pixelBuffer);
    
    // Metal
    id<MTLDevice> device_;
    CVMetalTextureCacheRef texture_cache_;
    
    // 현재 표시 중인 텍스처 (double buffering)
    id<MTLTexture> display_texture_;    // 현재 화면에 표시 중
    id<MTLTexture> ready_texture_;      // 다음에 표시할 텍스처
    CVPixelBufferRef ready_pixel_buffer_; // ready_texture의 소스
    std::atomic<bool> new_frame_ready_;
    std::mutex texture_mutex_;
    
    // AVFoundation
    AVAsset* asset_;
    AVAssetReader* asset_reader_;
    AVAssetReaderTrackOutput* video_output_;
    AVAssetTrack* video_track_;
    
    // 영상 정보
    std::string file_path_;
    std::atomic<float> duration_;
    std::atomic<float> current_time_;
    std::atomic<int> width_;
    std::atomic<int> height_;
    float video_fps_;
    float playback_rate_;
    
    // 엔진 모드 (Preview/Export 분리)
    EngineMode mode_;
    
    // 상태 플래그
    std::atomic<bool> is_loading_;
    std::atomic<bool> is_loaded_;
    std::atomic<bool> is_playing_;
    std::atomic<bool> should_stop_;
    std::atomic<bool> frame_request_pending_;
    std::atomic<bool> has_error_;
    std::string error_message_;
    
    // 스레드
    std::thread loading_thread_;
    std::thread decoding_thread_;
    
    // 프레임 타이밍
    std::chrono::steady_clock::time_point last_frame_time_;
    std::chrono::steady_clock::time_point playback_start_time_;
    double playback_start_video_time_;
    
    // 캐시 플러시 (메모리 누수 방지를 위해 더 자주 플러시)
    int frame_count_;
    static constexpr int FLUSH_INTERVAL = 30;  // 60 -> 30으로 변경하여 더 자주 플러시
};

} // namespace video
} // namespace media
} // namespace core
} // namespace rdo
