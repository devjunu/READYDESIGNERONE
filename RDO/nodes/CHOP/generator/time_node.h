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

class TimeNode : public CHOPNodeBase
{
public:
    TimeNode(Graph<Node>& graph, const ImVec2& pos);
    virtual ~TimeNode() = default;
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Time"; }
    std::string GetCategory() const override { return "CHOP/Generator"; }
    
    float Evaluate(const std::vector<float>& inputs, float time) override;
    
    void UpdateTime(float delta_time);
    
private:
    float current_time_;
};

std::unique_ptr<NodeBase> CreateTimeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
