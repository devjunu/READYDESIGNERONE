#pragma once

#include <memory>
#include <vector>

namespace rdo {
namespace core {
namespace media {
namespace video {
class VideoEngine;
} // namespace video
} // namespace media
} // namespace core
} // namespace rdo

namespace example {

class Timeline {
public:
    Timeline();
    ~Timeline() = default;

    // UI 렌더링
    void Render(float delta_time);

    // 현재 시간 반환
    float GetCurrentTime() const { return time_; }

    // 재생 상태
    bool IsPlaying() const { return is_playing_; }
    void SetPlaying(bool playing);
    void TogglePlayPause();

    // 시간 제어
    void SetTime(float time);
    void GoToStart();
    void GoToEnd();

    // 범위 설정
    void SetRange(float start, float end);
    float GetStartTime() const { return start_time_; }
    float GetEndTime() const { return end_time_; }

    // FPS 설정
    void SetFPS(float fps) { fps_ = fps; }
    float GetFPS() const { return fps_; }

    // VideoEngine 등록/해제 (Preview 모드만 허용)
    // 노드 시스템과 완전 분리: NodeEditor에서만 호출
    bool RegisterPreviewVideoEngine(rdo::core::media::video::VideoEngine* engine);
    void UnregisterPreviewVideoEngine(rdo::core::media::video::VideoEngine* engine);

private:
    // Timeline 상태
    bool is_playing_;
    float time_;
    float fps_;
    float start_time_;
    float end_time_;

    // 등록된 VideoEngine 목록
    std::vector<rdo::core::media::video::VideoEngine*> video_engines_;

    // UI 업데이트 (내부 로직)
    void UpdateTime(float delta_time);
    
    // VideoEngine 제어 (내부)
    void UpdateVideoEngines();
};

} // namespace example
