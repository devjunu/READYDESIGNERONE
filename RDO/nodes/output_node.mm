#include "output_node.h"
#include <imgui.h>
#include <imnodes.h>
#include <algorithm>

namespace example
{
namespace nodes
{
void RenderOutputNode(const OutputNode& node, Graph<Node>& graph)
{
    const float node_width = 100.0f;
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(11, 109, 191, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(45, 126, 194, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(81, 148, 204, 255));
    ImNodes::BeginNode(node.id);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("output");
    ImNodes::EndNodeTitleBar();

    ImGui::Dummy(ImVec2(node_width, 0.f));
    {
        ImNodes::BeginInputAttribute(node.r);
        const float label_width = ImGui::CalcTextSize("r").x;
        ImGui::TextUnformatted("r");
        if (graph.num_edges_from_node(node.r) == 0ull)
        {
            ImGui::SameLine();
            ImGui::PushItemWidth(node_width - label_width);
            ImGui::DragFloat("##hidelabel", &graph.node(node.r).value, 0.01f, 0.f, 1.0f);
            ImGui::PopItemWidth();
        }
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    {
        ImNodes::BeginInputAttribute(node.g);
        const float label_width = ImGui::CalcTextSize("g").x;
        ImGui::TextUnformatted("g");
        if (graph.num_edges_from_node(node.g) == 0ull)
        {
            ImGui::SameLine();
            ImGui::PushItemWidth(node_width - label_width);
            ImGui::DragFloat("##hidelabel", &graph.node(node.g).value, 0.01f, 0.f, 1.f);
            ImGui::PopItemWidth();
        }
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    {
        ImNodes::BeginInputAttribute(node.b);
        const float label_width = ImGui::CalcTextSize("b").x;
        ImGui::TextUnformatted("b");
        if (graph.num_edges_from_node(node.b) == 0ull)
        {
            ImGui::SameLine();
            ImGui::PushItemWidth(node_width - label_width);
            ImGui::DragFloat("##hidelabel", &graph.node(node.b).value, 0.01f, 0.f, 1.0f);
            ImGui::PopItemWidth();
        }
        ImNodes::EndInputAttribute();
    }
    ImNodes::EndNode();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
}

void CreateOutputNode(Graph<Node>& graph, std::vector<OutputNode>& nodes, const ImVec2& pos, int& root_node_id)
{
    const Node value(NodeType::value, 0.f);
    const Node out(NodeType::output);

    OutputNode ui_node;
    ui_node.r = graph.insert_node(value);
    ui_node.g = graph.insert_node(value);
    ui_node.b = graph.insert_node(value);
    ui_node.id = graph.insert_node(out);

    graph.insert_edge(ui_node.id, ui_node.r);
    graph.insert_edge(ui_node.id, ui_node.g);
    graph.insert_edge(ui_node.id, ui_node.b);

    nodes.push_back(ui_node);
    ImNodes::SetNodeScreenSpacePos(ui_node.id, pos);
    root_node_id = ui_node.id;
}

void DeleteOutputNode(Graph<Node>& graph, std::vector<OutputNode>& nodes, int node_id, int& root_node_id)
{
    auto iter = std::find_if(nodes.begin(), nodes.end(), [node_id](const OutputNode& node) -> bool {
        return node.id == node_id;
    });
    if (iter != nodes.end())
    {
        graph.erase_node(iter->r);
        graph.erase_node(iter->g);
        graph.erase_node(iter->b);
        graph.erase_node(iter->id);
        root_node_id = -1;
        nodes.erase(iter);
    }
}
} // namespace nodes
} // namespace example
