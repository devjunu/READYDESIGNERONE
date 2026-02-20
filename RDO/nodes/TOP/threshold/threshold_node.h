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

// Threshold 노드
class ThresholdNode : public TOPNodeBase
{
public:
    ThresholdNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ThresholdNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Threshold"; }
    std::string GetCategory() const override { return "TOP/Filter"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Threshold 파라미터
    float GetThreshold() const { return threshold_; }
    void SetThreshold(float threshold) { threshold_ = threshold; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> threshold_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float threshold_;

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input_texture_;
    float last_threshold_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateThresholdNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
