#include "timeline.h"
#include "core/media/video/video_engine.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cmath>

namespace example {

Timeline::Timeline()
    : is_playing_(false)
    , time_(0.0f)
    , fps_(30.0f)
    , start_time_(0.0f)
    , end_time_(10.0f)
    , video_engines_()
{
}

void Timeline::Render(float delta_time)
{
    ImGui::Begin("Timeline", nullptr, ImGuiWindowFlags_NoScrollbar);

    // 키보드 단축키: 스페이스바로 재생/일시정지
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImGui::IsKeyPressed(ImGuiKey_Space))
    {
        TogglePlayPause();
    }

    // 시간 업데이트 (재생 중일 때만)
    UpdateTime(delta_time);
    
    // VideoEngine 동기화
    UpdateVideoEngines();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float width = ImGui::GetContentRegionAvail().x;

    // === 상단 컨트롤 바 ===
    ImGui::BeginGroup();
    {
        // 재생 컨트롤 (TouchDesigner 스타일)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.45f, 1.0f));

        // 처음으로
        if (ImGui::Button("|<<", ImVec2(40, 0)))
        {
            GoToStart();
        }
        ImGui::SameLine();

        // 재생/일시정지 (색상 강조)
        if (is_playing_)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.3f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.4f, 0.3f, 1.0f));
        }
        else
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
        }

        if (ImGui::Button(is_playing_ ? " || " : " > ", ImVec2(40, 0)))
        {
            TogglePlayPause();
        }

        ImGui::PopStyleColor(2);
        ImGui::SameLine();

        // 끝으로
        if (ImGui::Button(">>|", ImVec2(40, 0)))
        {
            GoToEnd();
        }

        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();

        // 시간 표시 (TouchDesigner 스타일)
        int current_frame = static_cast<int>(time_ * fps_);
        int total_frames = static_cast<int>(end_time_ * fps_);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.9f, 1.0f, 1.0f));
        ImGui::Text("Frame: %04d", current_frame);
        ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::TextDisabled("/");
        ImGui::SameLine();

        ImGui::Text("%04d", total_frames);

        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.5f, 1.0f));
        ImGui::Text("%.3f sec", time_);
        ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();

        // FPS 표시
        ImGui::PushItemWidth(80);
        ImGui::DragFloat("##fps", &fps_, 1.0f, 1.0f, 120.0f, "%.0f fps");
        ImGui::PopItemWidth();
    }
    ImGui::EndGroup();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // === 타임라인 바 (수평 스크러버) ===
    const float timeline_height = 60.0f;
    const ImVec2 timeline_pos = ImGui::GetCursorScreenPos();
    const ImVec2 timeline_size(width, timeline_height);

    // 타임라인 배경
    const ImU32 bg_color = ImGui::GetColorU32(ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
    draw_list->AddRectFilled(timeline_pos,
                             ImVec2(timeline_pos.x + timeline_size.x, timeline_pos.y + timeline_size.y),
                             bg_color);

    // 범위 영역 표시 (start ~ end)
    const float time_range = end_time_ - start_time_;
    const float pixels_per_second = timeline_size.x / time_range;

    // 눈금 그리기
    const ImU32 tick_color = ImGui::GetColorU32(ImVec4(0.4f, 0.4f, 0.45f, 1.0f));
    const ImU32 major_tick_color = ImGui::GetColorU32(ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
    const ImU32 text_color = ImGui::GetColorU32(ImVec4(0.7f, 0.7f, 0.75f, 1.0f));

    // 눈금 간격 계산 (적응형)
    float tick_interval = 1.0f; // 기본 1초
    if (time_range < 5.0f)
        tick_interval = 0.5f;
    else if (time_range > 60.0f)
        tick_interval = 10.0f;
    else if (time_range > 30.0f)
        tick_interval = 5.0f;

    // 눈금 그리기
    for (float t = std::ceil(start_time_ / tick_interval) * tick_interval;
         t <= end_time_;
         t += tick_interval)
    {
        float x = timeline_pos.x + (t - start_time_) * pixels_per_second;

        bool is_major = (std::fmod(t, tick_interval * 5) < 0.01f);
        float tick_height = is_major ? 15.0f : 8.0f;
        ImU32 color = is_major ? major_tick_color : tick_color;

        draw_list->AddLine(ImVec2(x, timeline_pos.y),
                          ImVec2(x, timeline_pos.y + tick_height),
                          color, 1.0f);

        // 주요 눈금에 시간 표시
        if (is_major)
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f", t);
            draw_list->AddText(ImVec2(x + 2, timeline_pos.y + 2), text_color, buf);
        }
    }

    // 프레임 눈금 (하단)
    const ImU32 frame_tick_color = ImGui::GetColorU32(ImVec4(0.3f, 0.5f, 0.6f, 0.5f));
    const float safe_fps = std::max(fps_, 1.0f);
    const float frame_interval = 1.0f / safe_fps; // 1프레임
    const float pixels_per_frame = pixels_per_second * frame_interval;
    const float min_frame_tick_spacing = 6.0f; // 너무 촘촘한 tick 방지

    int frame_step = 1;
    if (pixels_per_frame < min_frame_tick_spacing)
    {
        frame_step = std::max(
            1,
            static_cast<int>(std::ceil(min_frame_tick_spacing / std::max(pixels_per_frame, 0.001f)))
        );
    }

    const float frame_tick_interval = frame_interval * frame_step;
    for (float t = start_time_; t <= end_time_; t += frame_tick_interval)
    {
        float x = timeline_pos.x + (t - start_time_) * pixels_per_second;
        draw_list->AddLine(ImVec2(x, timeline_pos.y + timeline_size.y - 5),
                          ImVec2(x, timeline_pos.y + timeline_size.y),
                          frame_tick_color, 0.5f);
    }

    // 재생 헤드 (빨간 선)
    float playhead_x = timeline_pos.x + (time_ - start_time_) * pixels_per_second;
    const ImU32 playhead_color = ImGui::GetColorU32(ImVec4(1.0f, 0.3f, 0.2f, 1.0f));
    const ImU32 playhead_bg_color = ImGui::GetColorU32(ImVec4(1.0f, 0.3f, 0.2f, 0.2f));

    // 재생 헤드 배경 (반투명)
    draw_list->AddRectFilled(ImVec2(playhead_x - 1, timeline_pos.y),
                             ImVec2(playhead_x + 1, timeline_pos.y + timeline_size.y),
                             playhead_bg_color);

    // 재생 헤드 라인
    draw_list->AddLine(ImVec2(playhead_x, timeline_pos.y),
                      ImVec2(playhead_x, timeline_pos.y + timeline_size.y),
                      playhead_color, 2.0f);

    // 재생 헤드 삼각형 (상단)
    const float triangle_size = 8.0f;
    draw_list->AddTriangleFilled(
        ImVec2(playhead_x, timeline_pos.y),
        ImVec2(playhead_x - triangle_size * 0.5f, timeline_pos.y + triangle_size),
        ImVec2(playhead_x + triangle_size * 0.5f, timeline_pos.y + triangle_size),
        playhead_color
    );

    // 타임라인 인터랙션 (클릭 & 드래그)
    ImGui::SetCursorScreenPos(timeline_pos);
    ImGui::InvisibleButton("##timeline_bar", timeline_size);

    if (ImGui::IsItemActive() || ImGui::IsItemClicked())
    {
        float mouse_x = ImGui::GetMousePos().x;
        float relative_x = mouse_x - timeline_pos.x;
        float new_time = start_time_ + (relative_x / timeline_size.x) * time_range;
        SetTime(new_time);

        if (ImGui::IsItemActive())
        {
            is_playing_ = false; // 드래그 중에는 일시정지
        }
    }

    // 범위 조절 (하단 바)
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::BeginGroup();
    {
        ImGui::Text("Range:");
        ImGui::SameLine();

        ImGui::PushItemWidth(100);
        bool range_changed = false;
        range_changed |= ImGui::DragFloat("##start", &start_time_, 0.1f, 0.0f, end_time_ - 0.1f, "%.2f");
        ImGui::SameLine();
        ImGui::Text("-");
        ImGui::SameLine();
        range_changed |= ImGui::DragFloat("##end", &end_time_, 0.1f, start_time_ + 0.1f, 3600.0f, "%.2f");
        ImGui::PopItemWidth();

        if (range_changed)
        {
            time_ = std::clamp(time_, start_time_, end_time_);
        }
    }
    ImGui::EndGroup();

    ImGui::End();
}

void Timeline::UpdateTime(float delta_time)
{
    // VideoEngine이 있으면 UpdateVideoEngines()에서 시간을 동기화하므로
    // 여기서는 VideoEngine이 없을 때만 시간을 증가시킴
    if (is_playing_ && delta_time > 0.0f && video_engines_.empty())
    {
        time_ += delta_time;
        if (time_ > end_time_)
        {
            time_ = start_time_;  // 루프
        }
    }
}

void Timeline::SetTime(float time)
{
    const float safe_fps = std::max(fps_, 1.0f);
    const float frame_duration = 1.0f / safe_fps;

    float quantized_time = std::clamp(time, start_time_, end_time_);
    quantized_time = std::round(quantized_time / frame_duration) * frame_duration;
    quantized_time = std::clamp(quantized_time, start_time_, end_time_);

    // 동일 프레임 내에서는 seek 생략 (불필요한 디코더 재시작 방지)
    if (std::abs(quantized_time - time_) < frame_duration * 0.5f)
    {
        time_ = quantized_time;
        return;
    }

    time_ = quantized_time;

    // 모든 VideoEngine에 시간 동기화
    for (auto* engine : video_engines_)
    {
        if (engine && engine->IsLoaded())
        {
            float engine_time = engine->GetCurrentTime();
            if (std::abs(engine_time - time_) >= frame_duration * 0.5f)
            {
                engine->SeekToTime(time_);
            }
        }
    }
}

void Timeline::GoToStart()
{
    time_ = start_time_;
    is_playing_ = false;
    
    // 모든 VideoEngine에 시간 동기화
    for (auto* engine : video_engines_)
    {
        if (engine && engine->IsLoaded())
        {
            engine->SeekToTime(time_);
            engine->Pause();
        }
    }
}

void Timeline::GoToEnd()
{
    time_ = end_time_;
    is_playing_ = false;
    
    // 모든 VideoEngine에 시간 동기화
    for (auto* engine : video_engines_)
    {
        if (engine && engine->IsLoaded())
        {
            engine->SeekToTime(time_);
            engine->Pause();
        }
    }
}

void Timeline::SetRange(float start, float end)
{
    if (end <= start)
    {
        end = start + (1.0f / std::max(fps_, 1.0f));
    }
    start_time_ = start;
    end_time_ = end;
    // 현재 시간이 범위를 벗어나지 않도록 조정
    time_ = std::clamp(time_, start_time_, end_time_);
}

void Timeline::SetPlaying(bool playing)
{
    if (is_playing_ == playing)
        return;
    
    is_playing_ = playing;
    
    // 모든 VideoEngine에 재생 상태 동기화
    for (auto* engine : video_engines_)
    {
        if (engine && engine->IsLoaded())
        {
            if (playing)
            {
                engine->Play();
            }
            else
            {
                engine->Pause();
            }
        }
    }
}

void Timeline::TogglePlayPause()
{
    SetPlaying(!is_playing_);
}

bool Timeline::RegisterPreviewVideoEngine(rdo::core::media::video::VideoEngine* engine)
{
    if (!engine)
        return false;
    
    // Preview 모드만 허용 (엄격한 제한)
    if (!engine->IsPreviewMode())
    {
        return false;  // Export 모드는 등록 불가
    }
    
    // 중복 등록 방지
    bool newly_registered = false;
    auto it = std::find(video_engines_.begin(), video_engines_.end(), engine);
    if (it == video_engines_.end())
    {
        video_engines_.push_back(engine);
        newly_registered = true;
    }

    // Timeline의 현재 시간/범위를 VideoEngine에 동기화
    // 단일 VideoEngine 구성에서는 재로딩 시에도 범위를 갱신
    if (engine->IsLoaded())
    {
        float duration = engine->GetDuration();
        if (duration > 0.0f && (newly_registered || video_engines_.size() == 1))
        {
            SetRange(0.0f, duration);
            SetFPS(engine->GetFPS());
        }

        if (newly_registered)
        {
            engine->SeekToTime(time_);
        }
    }

    return newly_registered;
}

void Timeline::UnregisterPreviewVideoEngine(rdo::core::media::video::VideoEngine* engine)
{
    if (!engine)
        return;
    
    auto it = std::find(video_engines_.begin(), video_engines_.end(), engine);
    if (it != video_engines_.end())
    {
        video_engines_.erase(it);
    }
}

void Timeline::UpdateVideoEngines()
{
    // Timeline과 VideoEngine의 재생 상태 동기화 확인
    for (auto* engine : video_engines_)
    {
        if (!engine || !engine->IsLoaded())
            continue;
        
        // Timeline의 재생 상태와 VideoEngine의 재생 상태가 다르면 동기화
        bool engine_playing = engine->IsPlaying();
        if (is_playing_ != engine_playing)
        {
            if (is_playing_)
            {
                engine->Play();
            }
            else
            {
                engine->Pause();
            }
        }
        
        // VideoEngine의 현재 시간을 Timeline에 동기화 (둘 다 재생 중일 때만)
        if (is_playing_ && engine_playing)
        {
            float engine_time = engine->GetCurrentTime();
            float duration = engine->GetDuration();
            
            // VideoEngine의 시간을 Timeline에 반영
            if (duration > 0.0f)
            {
                time_ = engine_time;
                
                // 범위를 벗어나면 루프
                if (time_ > end_time_)
                {
                    time_ = start_time_;
                    engine->SeekToTime(time_);
                }
            }
        }
    }
}

} // namespace example
