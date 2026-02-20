#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"

struct ImVec2;

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

class SineNode : public CHOPNodeBase
{
public:
    SineNode(Graph<Node>& graph, const ImVec2& pos);
    virtual ~SineNode() = default;
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Sine"; }
    std::string GetCategory() const override { return "CHOP/Math"; }
    
    float Evaluate(const std::vector<float>& inputs, float time) override;
    
private:
    float result_;
};

std::unique_ptr<NodeBase> CreateSineNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
