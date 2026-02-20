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

// HSV Adjust 노드 - Hue, Saturation, Value 조정
class HSVAdjustNode : public TOPNodeBase
{
public:
    HSVAdjustNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~HSVAdjustNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "HSV Adjust"; }
    std::string GetCategory() const override { return "TOP/Color"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // HSV 파라미터
    float GetHue() const { return hue_; }
    float GetSaturation() const { return saturation_; }
    float GetValue() const { return value_; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 파라미터
    float hue_;         // -180 to 180 degrees
    float saturation_;  // 0 to 2 (1 = original)
    float value_;       // 0 to 2 (1 = original)
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateHSVAdjustNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
