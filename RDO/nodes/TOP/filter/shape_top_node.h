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

// Shape TOP 노드 - 터치디자이너 호환
// CHOP 데이터에서 shape 정보를 읽어서 박스, 원 등을 그리기
class ShapeTOPNode : public TOPNodeBase
{
public:
    enum class ShapeType
    {
        Box = 0,
        Circle = 1
    };
    
    ShapeTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ShapeTOPNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Shape"; }
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
    ShapeType GetShapeType() const { return shape_type_; }
    void SetShapeType(ShapeType type) { shape_type_ = type; }
    
    void GetFillColor(float color[4]) const;
    void SetFillColor(const float color[4]);
    void GetStrokeColor(float color[4]) const;
    void SetStrokeColor(const float color[4]);
    float GetStrokeWidth() const { return stroke_width_; }
    void SetStrokeWidth(float w) { stroke_width_ = std::max(0.0f, std::min(10.0f, w)); }
    bool GetFillEnabled() const { return fill_enabled_; }
    void SetFillEnabled(bool enabled) { fill_enabled_ = enabled; }
    
    // 해상도 파라미터 (터치디자이너 Common 파라미터 호환)
    int GetWidth() const { return width_; }
    void SetWidth(int w) { width_ = std::max(1, std::min(8192, w)); }
    int GetHeight() const { return height_; }
    void SetHeight(int h) { height_ = std::max(1, std::min(8192, h)); }
    
private:
    bool InitializeMetal();
    
    id<MTLDevice> device_;
    id<MTLComputePipelineState> box_pipeline_;
    id<MTLComputePipelineState> circle_pipeline_;
    id<MTLComputePipelineState> clear_pipeline_;
    id<MTLRenderPipelineState> render_pipeline_;  // instanced quad render
    
    // 연결된 CHOP 입력 노드 (void*로 저장하여 순환 참조 방지)
    void* connected_chop_input_ptr_;
    
    // Graph 참조
    Graph<Node>* graph_ref_;
    
    // 파라미터
    ShapeType shape_type_;
    float fill_color_[4];
    float stroke_color_[4];
    float stroke_width_;
    bool fill_enabled_;
    
    // 해상도 파라미터 (터치디자이너 Common 파라미터 호환)
    int width_;
    int height_;
    
    // CHOP 입력 데이터
    std::vector<float> input_channels_;
    std::vector<std::string> input_channel_names_;
    std::mutex channels_mutex_;
    
    // 버퍼 풀링 (매 프레임 할당 제거)
    id<MTLBuffer> shapes_buffer_;        // 미리 할당된 shape 버퍼
    size_t shapes_buffer_capacity_;      // 현재 버퍼 용량 (shape 개수)
    static constexpr size_t kInitialShapeCapacity = 2048;  // 초기 용량
    
    // GPU Indirect Draw 지원
    id<MTLRenderPipelineState> indirect_render_pipeline_;  // GPU 버퍼 직접 렌더링
    void* connected_blob_track_ptr_;  // BlobTrackNode 직접 연결 (GPU 버퍼 접근용)
    bool use_gpu_indirect_;           // GPU Indirect Draw 사용 여부
};

std::unique_ptr<NodeBase> CreateShapeTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
