#pragma once

#import <Metal/Metal.h>
#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Ramp 타입
enum class RampType {
    Linear,
    Radial,
    Angle,
    Box
};

// Ramp 노드 - 그라디언트 생성
class RampNode : public TOPNodeBase
{
public:
    RampNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~RampNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Ramp"; }
    std::string GetCategory() const override { return "TOP/Generator"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Ramp 파라미터
    RampType GetRampType() const { return ramp_type_; }
    void SetRampType(RampType type) { ramp_type_ = type; }
    
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    void SetResolution(int width, int height) { width_ = width; height_ = height; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> linear_pipeline_;
    id<MTLComputePipelineState> radial_pipeline_;
    id<MTLComputePipelineState> angle_pipeline_;
    id<MTLComputePipelineState> box_pipeline_;
    
    // 파라미터
    RampType ramp_type_;
    float color_start_[4];  // RGBA
    float color_end_[4];    // RGBA
    float center_x_;
    float center_y_;
    float angle_;
    int width_;
    int height_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateRampNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
