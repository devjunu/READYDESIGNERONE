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

// Null 노드 - 패스스루 (네트워크 정리용)
class NullNode : public TOPNodeBase
{
public:
    NullNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~NullNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Null"; }
    std::string GetCategory() const override { return "TOP/Utility"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // RenderContext 오버로드 (Preview/Final 모드 + Zero-copy)
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool,
        const RenderContext& context
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateNullNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
