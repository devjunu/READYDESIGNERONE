#include "multiply_node.h"
#include "../../../core/node_system/node_registry.h"
#include <imgui.h>
#include <imnodes.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

MultiplyNode::MultiplyNode(Graph<Node>& graph, const ImVec2& pos)
    : result_(0.0f)
{
    node_id_ = graph.insert_node(Node(NodeType::multiply));
    
    int lhs_id = graph.insert_node(Node(NodeType::value, 1.0f));
    int rhs_id = graph.insert_node(Node(NodeType::value, 1.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(lhs_id, NodeFamily::CHOP, PortDirection::Input, "float", "lhs"));
    AddInputPort(Port(rhs_id, NodeFamily::CHOP, PortDirection::Input, "float", "rhs"));
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "float", "result"));
    
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void MultiplyNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Multiply");
    ImNodes::EndNodeTitleBar();
    
    // LHS
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        
        if (graph.num_edges_from_node(port.id) == 0) {
            ImGui::PushItemWidth(100.0f);
            ImGui::DragFloat("##lhs", &graph.node(port.id).value, 0.01f);
            ImGui::PopItemWidth();
        } else {
            ImGui::TextUnformatted("lhs");
        }
        
        ImNodes::EndInputAttribute();
    }
    
    // RHS
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        
        if (graph.num_edges_from_node(port.id) == 0) {
            ImGui::PushItemWidth(100.0f);
            ImGui::DragFloat("##rhs", &graph.node(port.id).value, 0.01f);
            ImGui::PopItemWidth();
        } else {
            ImGui::TextUnformatted("rhs");
        }
        
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    ImGui::Text("= %.2f", result_);
    ImGui::Spacing();
    
    // Output
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("result").x;
        ImGui::Indent(120.0f - label_width);
        ImGui::TextUnformatted("result");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void MultiplyNode::RenderInspector()
{
    ImGui::Text("Multiply");
    ImGui::Separator();
    ImGui::Text("Result: %.3f", result_);
}

float MultiplyNode::Evaluate(const std::vector<float>& inputs, float time)
{
    float lhs = inputs.size() > 0 ? inputs[0] : 1.0f;
    float rhs = inputs.size() > 1 ? inputs[1] : 1.0f;
    result_ = lhs * rhs;
    return result_;
}

std::unique_ptr<NodeBase> CreateMultiplyNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<MultiplyNode>(graph, pos);
}

REGISTER_NODE(Multiply, "Multiply", "CHOP/Math", NodeFamily::CHOP, CreateMultiplyNode, "Multiply two numbers");

} // namespace nodes
} // namespace example
