#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

@protocol MTLDevice;
@protocol MTLTexture;
@protocol MTLComputePipelineState;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Color Correction 노드
class ColorCorrectionNode : public TOPNodeBase
{
public:
    ColorCorrectionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ColorCorrectionNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "ColorCorrection"; }
    std::string GetCategory() const override { return "TOP/Color"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Color Correction 파라미터
    float GetBrightness() const { return brightness_; }
    float GetContrast() const { return contrast_; }
    float GetSaturation() const { return saturation_; }
    float GetHue() const { return hue_; }

    void SetBrightness(float brightness) { brightness_ = brightness; }
    void SetContrast(float contrast) { contrast_ = contrast; }
    void SetSaturation(float saturation) { saturation_ = saturation; }
    void SetHue(float hue) { hue_ = hue; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> color_correction_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float brightness_;   // -1.0 ~ 1.0
    float contrast_;     // 0.0 ~ 2.0
    float saturation_;   // 0.0 ~ 2.0
    float hue_;          // -180.0 ~ 180.0 degrees

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateColorCorrectionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
