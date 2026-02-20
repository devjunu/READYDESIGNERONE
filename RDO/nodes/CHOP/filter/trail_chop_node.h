#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <mutex>
#include <string>
#include <vector>
#include <deque>

struct ImVec2;

namespace example { class NodeManager; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Trail CHOP: 입력 CHOP의 채널 히스토리를 Ring Buffer로 기록
// GPU 렌더링을 위한 히스토리 데이터 직접 노출
// 4K 60fps에서 1만개 이상 블롭 처리 최적화
class TrailCHOPNode : public AudioCHOPNodeBase
{
public:
    TrailCHOPNode(Graph<Node>& graph, const ImVec2& pos);
    ~TrailCHOPNode() override = default;

    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Trail"; }
    std::string GetCategory() const override { return "CHOP/Filter"; }

    // 멀티 채널 평가
    void Evaluate(const std::vector<std::vector<float>>& inputs, float time) override;

    // 파라미터
    int  GetWindowLength() const { return window_length_; }
    void SetWindowLength(int length);

    bool IsSmoothEnabled() const { return smooth_enabled_; }
    void SetSmoothEnabled(bool enabled) { smooth_enabled_ = enabled; }
    float GetSmoothAlpha() const { return smooth_alpha_; }
    void SetSmoothAlpha(float alpha);
    int   GetSmoothSegments() const { return smooth_segments_; }
    void  SetSmoothSegments(int segments);

    // 메타
    const std::vector<std::string>& GetChannelNames() const;
    int  GetChannelCount() const { return static_cast<int>(channel_names_.size()); }

    // 그래프 연결
    void UpdateConnectedCHOPNode(NodeManager* node_manager);
    AudioCHOPNodeBase* GetConnectedCHOPInput() const { return connected_chop_input_; }
    
    // ========== GPU 최적화: 히스토리 데이터 직접 접근 ==========
    // CHOPToTOPNode가 직접 접근하여 GPU 렌더링에 사용
    
    // 히스토리 프레임 수 (실제 저장된 프레임)
    int GetHistoryFrameCount() const { 
        std::lock_guard<std::mutex> lock(history_mutex_);
        return static_cast<int>(history_.size()); 
    }
    
    // 히스토리 데이터 접근 (읽기 전용)
    // 반환: [frame][channel] 형태의 2D 배열
    const std::deque<std::vector<float>>& GetHistory() const {
        return history_;
    }
    
    // 뮤텍스 접근 (외부 잠금용)
    std::mutex& GetHistoryMutex() const { return history_mutex_; }
    
    // 블롭 수 (채널 수 / 8, BlobTrackInfoNode 형식)
    int GetBlobCount() const { return blob_count_; }

private:
    void RebuildChannels(size_t count);

    AudioCHOPNodeBase* connected_chop_input_ = nullptr;
    Graph<Node>*       graph_ref_ = nullptr;

    std::vector<std::string> channel_names_;

    // params
    int   window_length_ = 50;
    bool  smooth_enabled_ = false;
    float smooth_alpha_ = 0.5f;
    int   smooth_segments_ = 8;
    std::vector<float> last_ema_;

    mutable std::mutex output_mutex_;
    
    // ========== 히스토리 저장 (Ring Buffer) ==========
    mutable std::mutex history_mutex_;
    std::deque<std::vector<float>> history_;  // [frame][channel]
    int blob_count_ = 0;  // 채널 수 / 8
};

std::unique_ptr<NodeBase> CreateTrailCHOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
