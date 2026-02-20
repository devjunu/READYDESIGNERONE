#pragma once

namespace example
{

// 프리뷰 스케일 설정 (전역 싱글톤)
class PreviewSettings
{
public:
    enum class Scale
    {
        Full = 0,      // 1/1 - 원본 해상도
        Half = 1,      // 1/2 - 50%
        Quarter = 2,   // 1/4 - 25%
    };
    
    static PreviewSettings& Instance()
    {
        static PreviewSettings instance;
        return instance;
    }
    
    // 스케일 설정
    void SetScale(Scale scale) { scale_ = scale; }
    Scale GetScale() const { return scale_; }
    
    // 해상도 계산 (원본 해상도 → 스케일된 해상도)
    void ApplyScale(int original_width, int original_height, int& scaled_width, int& scaled_height) const
    {
        switch (scale_)
        {
            case Scale::Full:
                scaled_width = original_width;
                scaled_height = original_height;
                break;
                
            case Scale::Half:
                scaled_width = original_width / 2;
                scaled_height = original_height / 2;
                // 최소 해상도 보장
                if (scaled_width < 64) scaled_width = 64;
                if (scaled_height < 64) scaled_height = 64;
                break;
                
            case Scale::Quarter:
                scaled_width = original_width / 4;
                scaled_height = original_height / 4;
                // 최소 해상도 보장
                if (scaled_width < 64) scaled_width = 64;
                if (scaled_height < 64) scaled_height = 64;
                break;
        }
    }
    
    // 스케일 비율 반환
    float GetScaleFactor() const
    {
        switch (scale_)
        {
            case Scale::Full:    return 1.0f;
            case Scale::Half:    return 0.5f;
            case Scale::Quarter: return 0.25f;
            default:             return 1.0f;
        }
    }
    
    // UI용 문자열
    const char* GetScaleName() const
    {
        switch (scale_)
        {
            case Scale::Full:    return "1/1 (Full)";
            case Scale::Half:    return "1/2 (Half)";
            case Scale::Quarter: return "1/4 (Quarter)";
            default:             return "Unknown";
        }
    }
    
    // 메모리 절약량 추정 (%)
    int GetMemorySavings() const
    {
        switch (scale_)
        {
            case Scale::Full:    return 0;
            case Scale::Half:    return 75;   // 1/4 메모리 사용
            case Scale::Quarter: return 94;   // 1/16 메모리 사용
            default:             return 0;
        }
    }
    
private:
    PreviewSettings() : scale_(Scale::Half) {}  // 기본값: 1/2 (성능 최우선)
    PreviewSettings(const PreviewSettings&) = delete;
    PreviewSettings& operator=(const PreviewSettings&) = delete;
    
    Scale scale_;
};

} // namespace example
