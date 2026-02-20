#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include "../../TOP/blob_track/blob_track_node.h"
#include <vector>
#include <string>
#include <unordered_map>

struct ImVec2;

namespace example { class NodeManager; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Blob Track Info CHOP 노드 - 터치디자이너 호환
// Blob Track TOP에 연결하여 blob 데이터를 CHOP 채널로 출력
class BlobTrackInfoNode : public AudioCHOPNodeBase
{
public:
    BlobTrackInfoNode(Graph<Node>& graph, const ImVec2& pos);
    virtual ~BlobTrackInfoNode() = default;
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "BlobTrackInfo"; }
    std::string GetCategory() const override { return "CHOP/Analysis"; }
    
    // 멀티 채널 평가 인터페이스
    void Evaluate(const std::vector<std::vector<float>>& inputs, float time) override;
    
    // 채널 이름 접근
    const std::vector<std::string>& GetChannelNames() const { return channel_names_; }
    int GetChannelCount() const { return static_cast<int>(channel_names_.size()); }
    
    // 연결된 Blob Track 노드 업데이트 (Render나 Evaluate에서 호출)
    void UpdateConnectedBlobTrackNode(NodeManager* node_manager);
    
private:
    // 연결된 Blob Track TOP 노드
    BlobTrackNode* connected_blob_track_;
    
    // Graph 참조 (연결된 노드 찾기용)
    Graph<Node>* graph_ref_;
    
    // 채널 이름 생성 (터치디자이너 형식)
    void UpdateChannelNames(int blob_count);
    
    std::vector<std::string> channel_names_;  // 채널 이름들 (num_blobs, u1, v1, width1, height1, area1, id1, ...)
    int last_blob_count_;  // 마지막 blob 개수 (채널 이름 업데이트용)
};

std::unique_ptr<NodeBase> CreateBlobTrackInfoNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
