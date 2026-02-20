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

// Glow Effect 노드
class GlowNode : public TOPNodeBase
{
public:
    GlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~GlowNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Glow"; }
    std::string GetCategory() const override { return "TOP/Style"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Glow 파라미터
    float GetIntensity() const { return intensity_; }
    void SetIntensity(float intensity) { intensity_ = intensity; }
    float GetBlurSize() const { return blur_size_; }
    void SetBlurSize(float size) { blur_size_ = size; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> horizontal_blur_pipeline_;
    id<MTLComputePipelineState> vertical_blur_pipeline_;
    id<MTLComputePipelineState> composite_pipeline_;
    
    // 파라미터
    float intensity_;
    float blur_size_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateGlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
