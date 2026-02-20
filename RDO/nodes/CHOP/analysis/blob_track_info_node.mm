#include "blob_track_info_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../core/node_system/node_manager.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

BlobTrackInfoNode::BlobTrackInfoNode(Graph<Node>& graph, const ImVec2& pos)
    : connected_blob_track_(nullptr)
    , graph_ref_(&graph)
    , last_blob_count_(0)
{
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // TOP 입력 포트 추가 (Blob Track TOP 연결용)
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "blobtrack"));
    
    // CHOP 출력 포트 추가 (생성자에서 미리 추가 - port_to_node_ 등록을 위해)
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::CHOP, PortDirection::Output, "channels", "output"));
    
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

void BlobTrackInfoNode::Render(Graph<Node>& graph)
{
    // 연결된 Blob Track 노드 업데이트 (터치디자이너처럼 자동으로)
    // NodeManager는 Render에서 접근할 수 없으므로, Evaluate에서 처리
    
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Blob Track Info");
    ImNodes::EndNodeTitleBar();
    
    // Input
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 연결 상태 표시 (디버깅용)
    if (connected_blob_track_) {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Connected");
        
        // 연결된 Blob Track 노드에서 직접 blob 데이터 가져오기
        int blob_count = connected_blob_track_->GetBlobCount();
        ImGui::Text("Blob Track Blobs: %d", blob_count);
        
        // 채널 정보 업데이트 (Render에서도 업데이트)
        if (blob_count != last_blob_count_) {
            UpdateChannelNames(blob_count);
            last_blob_count_ = blob_count;
            
            // Evaluate 호출하여 채널 데이터 생성
            Evaluate({}, 0.0f);
        }
    } else {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Not Connected");
        if (!input_ports_.empty()) {
            int input_port_id = input_ports_[0].id;
            ImGui::TextDisabled("Input Port ID: %d", input_port_id);
            
            // 연결 확인을 위한 디버깅 정보
            int edge_count = 0;
            for (const auto& edge : graph.edges()) {
                if (edge.to == input_port_id) {
                    edge_count++;
                    ImGui::TextDisabled("  Edge found: from=%d to=%d", edge.from, edge.to);
                }
            }
            if (edge_count == 0) {
                ImGui::TextDisabled("  No edges connected");
            }
        }
    }
    
    ImGui::Spacing();
    
    // 채널 정보 표시
    ImGui::Text("Channels: %d", GetChannelCount());
    if (!channel_names_.empty()) {
        ImGui::Text("Blobs: %d", last_blob_count_);
    } else {
        ImGui::TextDisabled("No channels (no blobs detected)");
    }
    
    ImGui::Spacing();
    
    // Outputs (동적 채널)
    // 터치디자이너에서는 각 채널이 개별 출력 포트로 표시되지 않고
    // 출력 포트 표시 (생성자에서 이미 추가됨)
    {
        if (!output_ports_.empty()) {
            const Port& port = output_ports_[0];
            ImNodes::BeginOutputAttribute(port.id);
            const float label_width = ImGui::CalcTextSize(port.name.c_str()).x;
            ImGui::Indent(node_width - label_width);
            ImGui::TextUnformatted(port.name.c_str());
            ImNodes::EndOutputAttribute();
        }
    }
    
    ImNodes::EndNode();
}

void BlobTrackInfoNode::RenderInspector()
{
    ImGui::Text("Blob Track Info");
    ImGui::Separator();
    
    ImGui::Text("Channels: %d", GetChannelCount());
    ImGui::Text("Blobs: %d", last_blob_count_);
    
    if (!channel_names_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Channel Names:");
        ImGui::BeginChild("ChannelList", ImVec2(0, 200), true);
        for (size_t i = 0; i < channel_names_.size() && i < 20; i++) {
            ImGui::Text("  %s", channel_names_[i].c_str());
        }
        if (channel_names_.size() > 20) {
            ImGui::TextDisabled("  ... and %zu more", channel_names_.size() - 20);
        }
        ImGui::EndChild();
    }
}


void BlobTrackInfoNode::UpdateChannelNames(int blob_count)
{
    channel_names_.clear();
    
    // num_blobs 채널 추가
    channel_names_.push_back("num_blobs");
    
    // 각 blob마다 채널 추가 (터치디자이너 형식)
    // u, v, width, height, area, id, vx, vy (velocity)
    for (int i = 1; i <= blob_count; i++) {
        channel_names_.push_back("u" + std::to_string(i));
        channel_names_.push_back("v" + std::to_string(i));
        channel_names_.push_back("width" + std::to_string(i));
        channel_names_.push_back("height" + std::to_string(i));
        channel_names_.push_back("area" + std::to_string(i));
        channel_names_.push_back("id" + std::to_string(i));
        channel_names_.push_back("vx" + std::to_string(i));  // velocity x (터치디자이너 호환)
        channel_names_.push_back("vy" + std::to_string(i));  // velocity y (터치디자이너 호환)
    }
}


void BlobTrackInfoNode::UpdateConnectedBlobTrackNode(NodeManager* node_manager)
{
    if (!node_manager || !graph_ref_ || input_ports_.empty()) {
        connected_blob_track_ = nullptr;
        return;
    }
    
    int input_port_id = input_ports_[0].id;
    connected_blob_track_ = nullptr;  // 초기화
    
    // Graph에서 입력 포트에 연결된 엣지 찾기
    bool found_edge = false;
    for (const auto& edge : graph_ref_->edges()) {
        if (edge.to == input_port_id) {
            found_edge = true;
            // 출력 포트 ID (edge.from)를 가진 노드 찾기
            int from_port_id = edge.from;
            
            // NodeManager를 통해 포트를 가진 노드 찾기
            NodeBase* connected_node = node_manager->GetNodeByPortId(from_port_id);
            if (!connected_node) {
                // 포트를 가진 노드를 찾을 수 없음
                continue;
            }
            
            // BlobTrackNode인지 확인
            BlobTrackNode* blob_track = dynamic_cast<BlobTrackNode*>(connected_node);
            if (blob_track) {
                connected_blob_track_ = blob_track;
                return;  // 성공적으로 찾음
            }
            // BlobTrackNode가 아니면 계속 찾기
        }
    }
    
    // 연결이 없거나 BlobTrackNode가 아니면 nullptr 유지
}

void BlobTrackInfoNode::Evaluate(const std::vector<std::vector<float>>& inputs, float time)
{
    // 연결된 Blob Track 노드는 RenderAllNodes에서 자동으로 업데이트됨
    // (터치디자이너처럼 자동으로 연결 감지)
    // 하지만 Evaluate가 호출되기 전에 업데이트되어야 함
    // Render에서 이미 업데이트되므로 여기서는 그냥 사용
    output_channels_.clear();

    if (!connected_blob_track_) {
        UpdateChannelNames(0);
        last_blob_count_ = 0;
        return;
    }
    
    // Blob 데이터 가져오기
    const auto& blobs = connected_blob_track_->GetBlobs();
    const int max_blobs = connected_blob_track_->GetMaxBlobs();
    int blob_count = static_cast<int>(std::min<size_t>(blobs.size(), static_cast<size_t>(max_blobs)));
    
    // Lost blob 포함 여부 확인
    bool include_lost = connected_blob_track_->GetIncludeLostBlobsInTable();
    bool include_expired = connected_blob_track_->GetIncludeExpiredBlobsInTable();
    
    // 전체 blob 개수 계산 (lost blob 포함 시)
    int total_blob_count = blob_count;
    if (include_lost) {
        total_blob_count += static_cast<int>(std::min<size_t>(
            connected_blob_track_->GetLostBlobs().size(),
            static_cast<size_t>(max_blobs - total_blob_count)));
    }
    total_blob_count = std::min(total_blob_count, max_blobs);
    
    // 채널 이름 업데이트 (blob 개수가 변경된 경우)
    if (total_blob_count != last_blob_count_) {
        UpdateChannelNames(total_blob_count);
        last_blob_count_ = total_blob_count;
    }
    
    // 텍스처 크기 가져오기 (정규화 좌표 계산용)
    id<MTLTexture> output_texture = connected_blob_track_->GetOutputTexture();
    float texture_width = 1.0f;
    float texture_height = 1.0f;
    if (output_texture != nil) {
        texture_width = static_cast<float>([output_texture width]);
        texture_height = static_cast<float>([output_texture height]);
    }
    
    // 활성 blob들
    std::vector<BlobInfo> all_blobs = blobs;
    
    // Lost blob 포함 (옵션에 따라)
    if (include_lost) {
        const auto& lost_blobs = connected_blob_track_->GetLostBlobs();
        for (const auto& lost : lost_blobs) {
            all_blobs.push_back(lost);
        }
    }
    
    // 채널 데이터 생성 (터치디자이너 형식) - 벡터 재사용으로 할당 최소화
    const size_t channel_count = channel_names_.size();
    if (output_channels_.size() != channel_count) {
        output_channels_.resize(channel_count);
    }
    // memset이 assign보다 빠름
    if (!output_channels_.empty()) {
        memset(output_channels_.data(), 0, channel_count * sizeof(float));
    }
    
    // num_blobs (전체 blob 개수, lost 포함)
    if (!output_channels_.empty()) {
        output_channels_[0] = static_cast<float>(total_blob_count);
    }
    
    // 각 blob의 데이터 (u, v, width, height, area, id, vx, vy)
    size_t write_idx = 1; // 0은 num_blobs
    for (size_t i = 0; i < static_cast<size_t>(total_blob_count) && write_idx + 7 < channel_count; ++i) {
        const auto& blob = all_blobs[i];
        float u = (texture_width > 0) ? blob.x / texture_width : 0.0f;
        float v = (texture_height > 0) ? blob.y / texture_height : 0.0f;
        
        output_channels_[write_idx + 0] = u;                       // u
        output_channels_[write_idx + 1] = v;                       // v
        output_channels_[write_idx + 2] = blob.width;              // width
        output_channels_[write_idx + 3] = blob.height;             // height
        output_channels_[write_idx + 4] = blob.area;               // area
        output_channels_[write_idx + 5] = static_cast<float>(blob.id); // id
        output_channels_[write_idx + 6] = blob.vx;                 // vx
        output_channels_[write_idx + 7] = blob.vy;                 // vy
        write_idx += 8;
    }
}

std::unique_ptr<NodeBase> CreateBlobTrackInfoNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<BlobTrackInfoNode>(graph, pos);
}

REGISTER_NODE(BlobTrackInfo, "Blob Track Info", "CHOP/Analysis", NodeFamily::CHOP, CreateBlobTrackInfoNode, "Extract blob tracking data as CHOP channels (TouchDesigner compatible)");

} // namespace nodes
} // namespace example
