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

// Lookup (LUT) 노드
class LookupNode : public TOPNodeBase
{
public:
    LookupNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~LookupNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Lookup"; }
    std::string GetCategory() const override { return "TOP/Color"; }
    
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    void InvalidateCache() override;
    
private:
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    float mix_amount_;
    
    bool InitializeMetal();
};

std::unique_ptr<NodeBase> CreateLookupNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
