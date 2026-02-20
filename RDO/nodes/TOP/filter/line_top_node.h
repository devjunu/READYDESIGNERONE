#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include "../../../texture_pool.h"
#include <vector>
#include <mutex>


struct ImVec2;

namespace example { class NodeManager; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;
using ::example::AudioCHOPNodeBase;

// Line TOP 노드 - 터치디자이너 호환
// 입력 텍스처의 선에 대한 스타일 설정 (두께, 색상, fade out)
class LineTOPNode : public TOPNodeBase
{
public:
    LineTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~LineTOPNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Line"; }
    std::string GetCategory() const override { return "TOP/Composite"; }
    
    // GPU 처리 인터페이스
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화
    void InvalidateCache() override;
    
    // 연결된 CHOP 노드 업데이트
    void UpdateConnectedCHOPNode(NodeManager* node_manager);
    
    // 파라미터 접근
    void GetLineColor(float color[4]) const;
    void SetLineColor(const float color[4]);
    float GetLineThickness() const { return line_thickness_; }
    void SetLineThickness(float t) { line_thickness_ = std::max(0.5f, std::min(10.0f, t)); }
    float GetFadeOut() const { return fade_out_; }
    void SetFadeOut(float f) { fade_out_ = std::max(0.0f, std::min(1.0f, f)); }
    
    // Smooth 옵션 접근
    bool IsSmoothEnabled() const { return smooth_enabled_; }
    void SetSmoothEnabled(bool enabled) { smooth_enabled_ = enabled; }
    int GetSmoothSegments() const { return smooth_segments_; }
    void SetSmoothSegments(int segments) { smooth_segments_ = std::max(2, std::min(50, segments)); }
    
    // 해상도 파라미터 (터치디자이너 Common 파라미터 호환)
    int GetWidth() const { return width_; }
    void SetWidth(int w) { width_ = std::max(1, std::min(8192, w)); }
    int GetHeight() const { return height_; }
    void SetHeight(int h) { height_ = std::max(1, std::min(8192, h)); }
    
private:
    bool InitializeMetal();
    
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 연결된 CHOP 입력 노드
    AudioCHOPNodeBase* connected_chop_input_;
    int last_chop_input_port_id_;  // 연결 변경 감지용
    
    // Graph 참조 (연결된 노드 찾기용)
    Graph<Node>* graph_ref_;
    
    // CHOP 입력 데이터
    std::vector<float> input_channels_;
    std::vector<std::string> input_channel_names_;
    std::mutex channels_mutex_;
    
    // 파싱 결과 캐싱 (Trail 형식)
    bool is_trail_format_cached_;
    bool format_cache_valid_;
    
    // 파라미터
    float line_color_[4];
    float line_thickness_;
    float fade_out_;  // 0.0 = no fade, 1.0 = full fade
    
    // Smooth 옵션
    bool smooth_enabled_;  // 부드러운 곡선 활성화 여부
    int smooth_segments_;  // Spline 보간 세그먼트 수 (2 ~ 50)
    
    // 해상도 파라미터 (터치디자이너 Common 파라미터 호환)
    int width_;
    int height_;
};

std::unique_ptr<NodeBase> CreateLineTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example

