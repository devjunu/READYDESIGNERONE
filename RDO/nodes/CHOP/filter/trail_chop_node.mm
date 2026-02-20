// Trail CHOP: GPU 최적화 버전
// Ring Buffer 히스토리 + Zero-Copy GPU 접근
#include "trail_chop_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../core/node_system/node_manager.h"
#include <imgui.h>
#include <imnodes.h>
#include <algorithm>
#include <cstring>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

namespace
{
inline int ClampInt(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
}

TrailCHOPNode::TrailCHOPNode(Graph<Node>& graph, const ImVec2& pos)
    : graph_ref_(&graph)
{
    node_id_ = graph.insert_node(Node(NodeType::value));

    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(input_id, NodeFamily::CHOP, PortDirection::Input, "channels", "input"));

    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "channels", "output"));

    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void TrailCHOPNode::SetWindowLength(int length)
{
    int clamped = ClampInt(length, 1, 1000);  // 최대 1000 프레임으로 제한
    window_length_ = clamped;
    
    // 히스토리 크기 조정
    std::lock_guard<std::mutex> lock(history_mutex_);
    while (static_cast<int>(history_.size()) > window_length_) {
        history_.pop_front();
    }
}

void TrailCHOPNode::SetSmoothAlpha(float alpha)
{
    smooth_alpha_ = std::max(0.0f, std::min(1.0f, alpha));
}

void TrailCHOPNode::SetSmoothSegments(int segments)
{
    smooth_segments_ = ClampInt(segments, 2, 16);  // 최대 16으로 제한 (성능)
}

void TrailCHOPNode::RebuildChannels(size_t count)
{
    last_ema_.assign(count, 0.0f);
    channel_names_.clear();
    channel_names_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        channel_names_.push_back("ch" + std::to_string(i));
    }
    output_channels_.assign(count, 0.0f);
    
    // 주의: 히스토리는 초기화하지 않음!
    // 채널 수가 변경되어도 이전 히스토리 유지
    // CHOPToTOPNode에서 채널 수 차이를 처리함
}

void TrailCHOPNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Trail");
    ImNodes::EndNodeTitleBar();

    ImNodes::BeginInputAttribute(input_ports_[0].id);
    ImGui::TextUnformatted(input_ports_[0].name.c_str());
    ImNodes::EndInputAttribute();

    ImGui::Spacing();
    ImGui::Text("Blobs: %d", blob_count_);
    ImGui::Text("History: %d/%d", GetHistoryFrameCount(), window_length_);

    ImNodes::BeginOutputAttribute(output_ports_[0].id);
    const float label_width = ImGui::CalcTextSize(output_ports_[0].name.c_str()).x;
    ImGui::Indent(200.0f - label_width);
    ImGui::TextUnformatted(output_ports_[0].name.c_str());
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();
}

void TrailCHOPNode::RenderInspector()
{
    ImGui::Text("Trail CHOP (GPU Optimized)");
    ImGui::Separator();
    ImGui::Text("Blobs: %d", blob_count_);
    ImGui::Text("Channels: %d", GetChannelCount());
    ImGui::Text("History Frames: %d / %d", GetHistoryFrameCount(), window_length_);
    
    int win = window_length_;
    if (ImGui::SliderInt("Window Length", &win, 1, 1000)) {
        SetWindowLength(win);
    }

    if (ImGui::Checkbox("Smooth", &smooth_enabled_)) {
        // no-op
    }
    if (smooth_enabled_) {
        if (ImGui::SliderFloat("Smooth Alpha", &smooth_alpha_, 0.0f, 1.0f, "%.2f")) {
            SetSmoothAlpha(smooth_alpha_);
        }
        int seg = smooth_segments_;
        if (ImGui::SliderInt("Smooth Segments", &seg, 2, 16)) {
            SetSmoothSegments(seg);
        }
        ImGui::TextDisabled("EMA: new = a*cur + (1-a)*prev");
    }
    
    ImGui::Separator();
    ImGui::TextDisabled("GPU Zero-Copy: History stored in deque");
    ImGui::TextDisabled("CHOPToTOP directly accesses history");
}

void TrailCHOPNode::UpdateConnectedCHOPNode(NodeManager* node_manager)
{
    if (!node_manager || !graph_ref_ || input_ports_.empty()) {
        connected_chop_input_ = nullptr;
        return;
    }
    int input_port_id = input_ports_[0].id;
    connected_chop_input_ = nullptr;

    for (const auto& edge : graph_ref_->edges()) {
        if (edge.to != input_port_id) continue;
        NodeBase* node = node_manager->GetNodeByPortId(edge.from);
        if (auto* chop = dynamic_cast<AudioCHOPNodeBase*>(node)) {
            connected_chop_input_ = chop;
            return;
        }
    }
}

void TrailCHOPNode::Evaluate(const std::vector<std::vector<float>>& inputs, float /*time*/)
{
    // 입력 채널 가져오기
    const std::vector<float>* input_ptr = nullptr;
    
    if (connected_chop_input_) {
        input_ptr = &connected_chop_input_->GetOutputChannels();
    } else if (!inputs.empty()) {
        input_ptr = &inputs[0];
    }

    if (!input_ptr || input_ptr->empty()) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_channels_.clear();
        channel_names_.clear();
        last_ema_.clear();
        blob_count_ = 0;
        
        std::lock_guard<std::mutex> hlock(history_mutex_);
        history_.clear();
        return;
    }

    const std::vector<float>& input_channels = *input_ptr;
    const size_t channel_count = input_channels.size();

    // 채널 수 변경 시에만 재구축
    if (output_channels_.size() != channel_count) {
        RebuildChannels(channel_count);
    }
    
    // 블롭 수 계산 (채널 수 / 8, BlobTrackInfoNode 형식)
    // 채널: num_blobs, u0, v0, w0, h0, area0, id0, vx0, vy0, u1, v1, ...
    if (channel_count > 0) {
        blob_count_ = (static_cast<int>(channel_count) - 1) / 8;
    } else {
        blob_count_ = 0;
    }

    // 출력 채널 업데이트 (EMA는 출력에만 적용, 히스토리에는 원본 저장)
    if (smooth_enabled_) {
        const float alpha = smooth_alpha_;
        const float one_minus_alpha = 1.0f - alpha;
        for (size_t ch = 0; ch < channel_count; ++ch) {
            float v = input_channels[ch];
            last_ema_[ch] = alpha * v + one_minus_alpha * last_ema_[ch];
            output_channels_[ch] = last_ema_[ch];
        }
    } else {
        std::memcpy(output_channels_.data(), input_channels.data(), 
                    channel_count * sizeof(float));
    }
    
    // ========== 히스토리 저장 (Ring Buffer) ==========
    // 중요: 원본 입력 데이터를 저장 (EMA 적용 전)
    // 스플라인 보간은 CHOPToTOPNode에서 수행
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        // 윈도우 크기 초과 시 가장 오래된 프레임 제거
        while (static_cast<int>(history_.size()) >= window_length_) {
            history_.pop_front();
        }
        
        // 원본 입력 데이터 저장 (스무딩 적용 안 함)
        history_.push_back(input_channels);
    }
}

const std::vector<std::string>& TrailCHOPNode::GetChannelNames() const
{
    std::lock_guard<std::mutex> lock(output_mutex_);
    return channel_names_;
}

std::unique_ptr<NodeBase> CreateTrailCHOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<TrailCHOPNode>(graph, pos);
}

REGISTER_NODE(Trail, "Trail", "CHOP/Filter", NodeFamily::CHOP, CreateTrailCHOPNode, "Record channel history with GPU-optimized access");

} // namespace nodes
} // namespace example
