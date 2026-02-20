#include "time_node.h"
#include "../../../core/node_system/node_registry.h"
#include <imgui.h>
#include <imnodes.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

TimeNode::TimeNode(Graph<Node>& graph, const ImVec2& pos)
    : current_time_(0.0f)
{
    node_id_ = graph.insert_node(Node(NodeType::time));
    
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "float", "time"));
    
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void TimeNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Time");
    ImNodes::EndNodeTitleBar();
    
    ImGui::Text("%.2f s", current_time_);
    
    ImGui::Spacing();
    
    // Output
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("time").x;
        ImGui::Indent(100.0f - label_width);
        ImGui::TextUnformatted("time");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void TimeNode::RenderInspector()
{
    ImGui::Text("Time");
    ImGui::Separator();
    ImGui::Text("Current Time: %.3f s", current_time_);
}

float TimeNode::Evaluate(const std::vector<float>& inputs, float time)
{
    current_time_ = time;
    return current_time_;
}

void TimeNode::UpdateTime(float delta_time)
{
    current_time_ += delta_time;
}

std::unique_ptr<NodeBase> CreateTimeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<TimeNode>(graph, pos);
}

REGISTER_NODE(Time, "Time", "CHOP/Generator", NodeFamily::CHOP, CreateTimeNode, "Current time in seconds");

} // namespace nodes
} // namespace example
