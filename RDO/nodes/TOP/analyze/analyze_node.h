#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

@protocol MTLDevice;
@protocol MTLTexture;
@protocol MTLComputePipelineState;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;
@protocol MTLBuffer;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// 분석 모드
enum class AnalyzeMode {
    Average = 0,
    MinPixel = 1,
    MaxPixel = 2,
    CountPixels = 3,
    Sum = 4
};

// 분석 채널
enum class AnalyzeChannel {
    Luminance = 0,
    Red = 1,
    Green = 2,
    Blue = 3,
    Alpha = 4,
    RGBAverage = 5
};

// 분석 결과 구조체
struct AnalyzeResult {
    float r, g, b, a;
    float value;        // 단일 값 (평균, 카운트 등)
    int pixel_x, pixel_y;  // 픽셀 위치 (MinPixel, MaxPixel)
};

// Analyze 노드 - 이미지 통계 분석
class AnalyzeNode : public TOPNodeBase
{
public:
    AnalyzeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~AnalyzeNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Analyze"; }
    std::string GetCategory() const override { return "TOP/Analysis"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Analyze 파라미터
    AnalyzeMode GetMode() const { return mode_; }
    void SetMode(AnalyzeMode mode) { mode_ = mode; }

    AnalyzeChannel GetChannel() const { return channel_; }
    void SetChannel(AnalyzeChannel channel) { channel_ = channel; }

    // 분석 결과 접근
    const AnalyzeResult& GetResult() const { return result_; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> analyze_pipeline_;
    id<MTLCommandQueue> command_queue_;
    id<MTLBuffer> result_buffer_;

    // 파라미터
    AnalyzeMode mode_;
    AnalyzeChannel channel_;

    // 분석 결과
    AnalyzeResult result_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateAnalyzeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
