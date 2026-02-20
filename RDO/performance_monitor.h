#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace example
{

// 성능 모니터링 시스템
class PerformanceMonitor
{
public:
    struct Stats
    {
        double avg_ms;
        double min_ms;
        double max_ms;
        size_t sample_count;
    };
    
    PerformanceMonitor();
    
    // 타이머 시작
    void begin(const std::string& name);
    
    // 타이머 종료 및 기록
    void end(const std::string& name);
    
    // 통계 가져오기
    Stats get_stats(const std::string& name) const;
    
    // 모든 통계 출력 (ImGui 윈도우)
    void render_ui();
    
    // 프레임 종료 (평균 계산)
    void end_frame();
    
    // 리셋
    void reset();
    
    // 전체 프레임 시간
    double frame_time_ms() const { return frame_time_ms_; }

private:
    struct TimerData
    {
        std::chrono::high_resolution_clock::time_point start;
        std::vector<double> samples;
        size_t max_samples = 120;  // 2초 분량 (60fps 기준)
    };
    
    std::unordered_map<std::string, TimerData> timers_;
    std::chrono::high_resolution_clock::time_point frame_start_;
    double frame_time_ms_;
    bool show_window_;
};

// 전역 성능 모니터
extern PerformanceMonitor g_performance_monitor;

// RAII 스타일 타이머
class ScopedTimer
{
public:
    ScopedTimer(const std::string& name) : name_(name)
    {
        g_performance_monitor.begin(name_);
    }
    
    ~ScopedTimer()
    {
        g_performance_monitor.end(name_);
    }

private:
    std::string name_;
};

// 편의 매크로
#define PERF_SCOPE(name) ScopedTimer _timer_##__LINE__(name)

} // namespace example
