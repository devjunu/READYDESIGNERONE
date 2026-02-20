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

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Background Subtract 노드
class BackgroundSubtractNode : public TOPNodeBase
{
public:
    BackgroundSubtractNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~BackgroundSubtractNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "BackgroundSubtract"; }
    std::string GetCategory() const override { return "TOP/Filter"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Background Subtract 파라미터
    float GetThreshold() const { return threshold_; }
    void SetThreshold(float threshold) { threshold_ = threshold; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> background_subtract_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float threshold_;    // 0.0 ~ 1.0

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateBackgroundSubtractNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
