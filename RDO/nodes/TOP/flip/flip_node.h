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

// Flip 노드 - 상하/좌우 반전
class FlipNode : public TOPNodeBase
{
public:
    FlipNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~FlipNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Flip"; }
    std::string GetCategory() const override { return "TOP/Transform"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Flip 파라미터
    bool GetFlipHorizontal() const { return flip_horizontal_; }
    bool GetFlipVertical() const { return flip_vertical_; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 파라미터
    bool flip_horizontal_;
    bool flip_vertical_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateFlipNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
