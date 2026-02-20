#include "performance_monitor.h"
#include <imgui.h>
#include <algorithm>
#include <numeric>

namespace example
{

PerformanceMonitor g_performance_monitor;

PerformanceMonitor::PerformanceMonitor()
    : timers_()
    , frame_start_()
    , frame_time_ms_(0.0)
    , show_window_(true)
{
}

void PerformanceMonitor::begin(const std::string& name)
{
    timers_[name].start = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::end(const std::string& name)
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto& timer = timers_[name];
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - timer.start);
    double ms = duration.count() / 1000.0;
    
    timer.samples.push_back(ms);
    if (timer.samples.size() > timer.max_samples)
    {
        timer.samples.erase(timer.samples.begin());
    }
}

PerformanceMonitor::Stats PerformanceMonitor::get_stats(const std::string& name) const
{
    Stats stats = {0.0, 0.0, 0.0, 0};
    
    auto it = timers_.find(name);
    if (it == timers_.end() || it->second.samples.empty())
    {
        return stats;
    }
    
    const auto& samples = it->second.samples;
    stats.sample_count = samples.size();
    
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.avg_ms = sum / samples.size();
    stats.min_ms = *std::min_element(samples.begin(), samples.end());
    stats.max_ms = *std::max_element(samples.begin(), samples.end());
    
    return stats;
}

void PerformanceMonitor::render_ui()
{
    if (!show_window_)
    {
        return;
    }
    
    ImGui::Begin("Performance Monitor", &show_window_);
    
    // 전체 프레임 시간
    ImGui::Text("Frame Time: %.2f ms (%.1f FPS)", frame_time_ms_, 1000.0 / frame_time_ms_);
    ImGui::Separator();
    
    // 각 타이머 통계
    if (ImGui::BeginTable("Timers", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Avg (ms)");
        ImGui::TableSetupColumn("Min (ms)");
        ImGui::TableSetupColumn("Max (ms)");
        ImGui::TableSetupColumn("Samples");
        ImGui::TableHeadersRow();
        
        for (const auto& pair : timers_)
        {
            Stats stats = get_stats(pair.first);
            
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", pair.first.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.avg_ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.min_ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.max_ms);
            ImGui::TableNextColumn();
            ImGui::Text("%zu", stats.sample_count);
        }
        
        ImGui::EndTable();
    }
    
    if (ImGui::Button("Reset"))
    {
        reset();
    }
    
    ImGui::End();
}

void PerformanceMonitor::end_frame()
{
    if (frame_start_.time_since_epoch().count() != 0)
    {
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start_);
        frame_time_ms_ = duration.count() / 1000.0;
    }
    
    frame_start_ = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::reset()
{
    for (auto& pair : timers_)
    {
        pair.second.samples.clear();
    }
    frame_time_ms_ = 0.0;
}

} // namespace example
