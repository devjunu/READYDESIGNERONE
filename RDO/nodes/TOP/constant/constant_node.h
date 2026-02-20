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

// Constant 노드 - 단색 텍스처 생성
class ConstantNode : public TOPNodeBase
{
public:
    ConstantNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ConstantNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Constant"; }
    std::string GetCategory() const override { return "TOP/Generator"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Constant 파라미터
    float GetRed() const { return color_[0]; }
    float GetGreen() const { return color_[1]; }
    float GetBlue() const { return color_[2]; }
    float GetAlpha() const { return color_[3]; }
    
    void SetColor(float r, float g, float b, float a) {
        color_[0] = r; color_[1] = g; color_[2] = b; color_[3] = a;
    }
    
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    void SetResolution(int width, int height) { width_ = width; height_ = height; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 파라미터
    float color_[4];  // RGBA
    int width_;
    int height_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateConstantNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
