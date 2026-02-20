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

// Level 노드 - 레벨 조정 (Shadows, Midtones, Highlights)
class LevelNode : public TOPNodeBase
{
public:
    LevelNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~LevelNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Level"; }
    std::string GetCategory() const override { return "TOP/Color"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Level 파라미터
    float GetInputMin() const { return input_min_; }
    float GetInputMax() const { return input_max_; }
    float GetGamma() const { return gamma_; }
    float GetOutputMin() const { return output_min_; }
    float GetOutputMax() const { return output_max_; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 파라미터
    float input_min_;   // Input Black Point (0-1)
    float input_max_;   // Input White Point (0-1)
    float gamma_;       // Midtones adjustment
    float output_min_;  // Output Black Point (0-1)
    float output_max_;  // Output White Point (0-1)
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateLevelNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
