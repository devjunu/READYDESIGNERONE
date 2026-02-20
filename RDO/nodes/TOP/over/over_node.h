#pragma once

#import <Metal/Metal.h>
#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Over 노드 - 알파 오버 합성 (전문가용)
class OverNode : public TOPNodeBase
{
public:
    OverNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~OverNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Over"; }
    std::string GetCategory() const override { return "TOP/Composite"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Over 파라미터
    float GetOpacity() const { return opacity_; }
    void SetOpacity(float opacity) { opacity_ = opacity; }
    
    bool GetPreMultiplied() const { return pre_multiplied_; }
    void SetPreMultiplied(bool pre) { pre_multiplied_ = pre; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    id<MTLComputePipelineState> premult_pipeline_;
    
    // 파라미터
    float opacity_;
    bool pre_multiplied_;  // 입력이 이미 프리멀티플라이드인지
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateOverNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
