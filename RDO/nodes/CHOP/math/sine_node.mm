#include "sine_node.h"
#include "../../../core/node_system/node_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <cmath>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

SineNode::SineNode(Graph<Node>& graph, const ImVec2& pos)
    : result_(0.0f)
{
    node_id_ = graph.insert_node(Node(NodeType::sine));
    
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(input_id, NodeFamily::CHOP, PortDirection::Input, "float", "input"));
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "float", "output"));
    
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void SineNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Sine");
    ImNodes::EndNodeTitleBar();
    
    // Input
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        
        if (graph.num_edges_from_node(port.id) == 0) {
            ImGui::PushItemWidth(100.0f);
            ImGui::DragFloat("##input", &graph.node(port.id).value, 0.01f);
            ImGui::PopItemWidth();
        } else {
            ImGui::TextUnformatted("input");
        }
        
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    ImGui::Text("sin() = %.2f", result_);
    ImGui::Spacing();
    
    // Output
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("output").x;
        ImGui::Indent(120.0f - label_width);
        ImGui::TextUnformatted("output");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void SineNode::RenderInspector()
{
    ImGui::Text("Sine");
    ImGui::Separator();
    ImGui::Text("Output: %.3f", result_);
}

float SineNode::Evaluate(const std::vector<float>& inputs, float time)
{
    float input = inputs.size() > 0 ? inputs[0] : 0.0f;
    result_ = std::sin(input);
    return result_;
}

std::unique_ptr<NodeBase> CreateSineNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<SineNode>(graph, pos);
}

REGISTER_NODE(Sine, "Sine", "CHOP/Generator", NodeFamily::CHOP, CreateSineNode, "Sine wave function");

} // namespace nodes
} // namespace example
