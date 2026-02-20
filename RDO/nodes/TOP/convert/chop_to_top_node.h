#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include "../../../texture_pool.h"
#include <vector>
#include <string>
#include <mutex>

struct ImVec2;

namespace example { class NodeManager; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// CHOP to TOP 노드 - GPU 최적화 트레일 렌더링
// 4K 60fps에서 1만개 이상 블롭 처리 가능
// TrailCHOPNode의 히스토리 데이터를 직접 접근하여 렌더링
class CHOPToTOPNode : public TOPNodeBase
{
public:
    CHOPToTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~CHOPToTOPNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "CHOPToTOP"; }
    std::string GetCategory() const override { return "TOP/Convert"; }
    
    // GPU 처리 인터페이스
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // RenderContext 오버로드 (Preview/Final 모드 지원)
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool,
        const RenderContext& context
    ) override;

    // 파라미터 접근
    int GetWidth() const { return width_; }
    void SetWidth(int w) { width_ = std::max(1, std::min(8192, w)); }
    int GetHeight() const { return height_; }
    void SetHeight(int h) { height_ = std::max(1, std::min(8192, h)); }
    
    // 해상도 자동 감지 모드
    bool GetAutoResolution() const { return auto_resolution_; }
    void SetAutoResolution(bool auto_res) { auto_resolution_ = auto_res; }
    
    // CHOP 입력 데이터 설정
    void SetCHOPInput(const std::vector<float>& channels, const std::vector<std::string>& channel_names);
    
    // 연결된 CHOP 노드 업데이트
    void UpdateConnectedCHOPNode(NodeManager* node_manager);
    
    // 포인트 데이터 접근 (Line TOP에서 사용)
    void GetPointData(std::vector<float>& points_data, int& point_count) const;
    
    // 연결된 CHOP 입력 노드 접근
    AudioCHOPNodeBase* GetConnectedCHOPInput() const { return connected_chop_input_; }
    
private:
    // 연결된 CHOP 입력 노드
    AudioCHOPNodeBase* connected_chop_input_;
    
    // Graph 참조
    Graph<Node>* graph_ref_;
    bool InitializeMetal();
    
    id<MTLDevice> device_;
    id<MTLComputePipelineState> clear_pipeline_;
    id<MTLComputePipelineState> fade_pipeline_;  // Accumulation Buffer용
    id<MTLComputePipelineState> draw_lines_pipeline_;
    
    // Render Pipeline: 대규모 포인트용 (500개 이상)
    id<MTLRenderPipelineState> line_render_pipeline_;
    static constexpr int kRenderPipelineThreshold = 100;  // 임계값 낮춤
    
    // 파라미터
    int width_;
    int height_;
    bool auto_resolution_;
    
    // Trail 시각화 개선 파라미터
    float line_thickness_;              // 기본 선 두께
    float thickness_variation_;         // 두께 변화량 (0.0 = 고정, 1.0 = 최대 변화)
    bool use_color_gradient_;          // 색상 그라데이션 사용 여부
    float gradient_intensity_;          // 그라데이션 강도 (0.0 = 없음, 1.0 = 최대)
    bool use_velocity_colors_;          // 속도 기반 색상 사용 여부
    float velocity_threshold_;         // 속도 임계값 (이 이상이면 빨간색)
    float max_trail_length_;           // 최대 trail 길이 (프레임 수, 0 = 무제한)
    float trail_start_color_[4];       // Trail 시작 색상 (최신 위치)
    float trail_end_color_[4];         // Trail 끝 색상 (오래된 위치)
    float velocity_color_[4];           // 빠른 움직임 색상
    
    static constexpr float DEFAULT_LINE_THICKNESS = 2.0f;
    static constexpr float DEFAULT_FADE_OUT = 0.5f;
    static constexpr float DEFAULT_LINE_COLOR[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    static constexpr float DEFAULT_START_COLOR[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // 흰색
    static constexpr float DEFAULT_END_COLOR[4] = {0.5f, 0.5f, 0.5f, 0.0f};    // 회색 투명
    static constexpr float DEFAULT_VELOCITY_COLOR[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // 빨간색
    
    // CHOP 입력 데이터
    std::vector<float> input_channels_;
    std::vector<std::string> input_channel_names_;
    std::mutex channels_mutex_;
    
    // 포인트 데이터
    mutable std::vector<float> points_data_;
    mutable int point_count_;
    mutable std::mutex points_mutex_;
    
    // 버퍼 풀링
    id<MTLBuffer> points_buffer_;
    size_t points_buffer_capacity_;
    static constexpr size_t kInitialPointCapacity = 8192;
    
    // 스플라인 사전 계산
    std::vector<float> interpolated_points_;
};

std::unique_ptr<NodeBase> CreateCHOPToTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
