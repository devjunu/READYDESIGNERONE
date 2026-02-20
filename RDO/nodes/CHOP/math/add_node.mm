#include "add_node.h"
#include "../../../core/node_system/node_registry.h"
#include <imgui.h>
#include <imnodes.h>

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

AddNode::AddNode(Graph<Node>& graph, const ImVec2& pos)
    : result_(0.0f)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::add));
    
    // 포트 생성
    int lhs_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int rhs_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    // 포트 추가
    AddInputPort(Port(lhs_id, NodeFamily::CHOP, PortDirection::Input, "float", "lhs"));
    AddInputPort(Port(rhs_id, NodeFamily::CHOP, PortDirection::Input, "float", "rhs"));
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "float", "result"));
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void AddNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Add");
    ImNodes::EndNodeTitleBar();
    
    // LHS 입력
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
    
    // RHS 입력
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
    
    // 결과 표시
    ImGui::Text("= %.2f", result_);
    
    ImGui::Spacing();
    
    // 출력 포트
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

void AddNode::RenderInspector()
{
    ImGui::Text("Add");
    ImGui::Separator();
    ImGui::Text("Result: %.3f", result_);
}

float AddNode::Evaluate(const std::vector<float>& inputs, float time)
{
    float lhs = inputs.size() > 0 ? inputs[0] : 0.0f;
    float rhs = inputs.size() > 1 ? inputs[1] : 0.0f;
    result_ = lhs + rhs;
    return result_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateAddNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<AddNode>(graph, pos);
}

// 자동 등록
REGISTER_NODE(Add, "Add", "CHOP/Math", NodeFamily::CHOP, CreateAddNode, "Add two numbers");

} // namespace nodes
} // namespace example
