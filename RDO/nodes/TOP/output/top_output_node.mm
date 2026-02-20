#include "top_output_node.h"
#include "../../../core/node_system/node_registry.h"
#include <imgui.h>
#include <imnodes.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

TopOutputNode::TopOutputNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , last_width_(0)
    , last_height_(0)
{
    node_id_ = graph.insert_node(Node(NodeType::value));
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));

    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));

    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

TopOutputNode::~TopOutputNode()
{
    output_texture_ = nil;
    last_texture_pool_ = nullptr;
}

void TopOutputNode::Render(Graph<Node>& graph)
{
    (void)graph;
    (void)device_;

    const float node_width = 210.0f;
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(166, 93, 32, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(190, 111, 40, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(210, 125, 48, 255));

    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("TOP Output");
    ImNodes::EndNodeTitleBar();

    ImGui::Dummy(ImVec2(node_width, 0.0f));
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted("input");
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();
    if (output_texture_ != nil)
    {
        ImGui::Text("Connected");
        ImGui::Text("%d x %d", last_width_, last_height_);
    }
    else
    {
        ImGui::TextDisabled("No input");
    }

    ImNodes::EndNode();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
}

void TopOutputNode::RenderInspector()
{
    ImGui::TextUnformatted("TOP Output");
    ImGui::Separator();
    ImGui::TextWrapped("Independent output sink. Encoding/export is managed from Output Preview panel.");

    ImGui::Spacing();
    if (output_texture_ != nil)
    {
        ImGui::Text("Resolution: %d x %d", last_width_, last_height_);
    }
    else
    {
        ImGui::TextDisabled("No input texture connected.");
    }
}

void TopOutputNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    (void)cmd_buffer;
    (void)texture_pool;

    // 이 노드는 입력 텍스처를 참조만 하며, TexturePool 소유권을 갖지 않는다.
    SetLastTexturePool(nullptr);

    if (inputs.empty() || inputs[0] == nil)
    {
        output_texture_ = nil;
        last_width_ = 0;
        last_height_ = 0;
        return;
    }

    output_texture_ = inputs[0];
    last_width_ = static_cast<int>([output_texture_ width]);
    last_height_ = static_cast<int>([output_texture_ height]);
}

void TopOutputNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool,
    const RenderContext& context
)
{
    (void)context;
    ProcessGPU(inputs, cmd_buffer, texture_pool);
}

void TopOutputNode::InvalidateCache()
{
    output_texture_ = nil;
    last_texture_pool_ = nullptr;
}

std::unique_ptr<NodeBase> CreateTopOutputNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<TopOutputNode>(graph, pos, device);
}

REGISTER_NODE(TOPOutput, "Output", "TOP/Output", NodeFamily::TOP, CreateTopOutputNode, "Dedicated independent output sink for TOP streams");

} // namespace nodes
} // namespace example

