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

enum class MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min
};

// Math 노드 - 수학 연산
class MathNode : public TOPNodeBase
{
public:
    MathNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~MathNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Math"; }
    std::string GetCategory() const override { return "TOP/Math"; }
    
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    void InvalidateCache() override;
    
private:
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    MathOp operation_;
    
    bool InitializeMetal();
};

std::unique_ptr<NodeBase> CreateMathNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
