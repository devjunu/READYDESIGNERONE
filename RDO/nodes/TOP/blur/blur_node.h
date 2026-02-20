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

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Gaussian Blur 노드
class BlurNode : public TOPNodeBase
{
public:
    BlurNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~BlurNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Blur"; }
    std::string GetCategory() const override { return "TOP/Filter"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Blur 파라미터
    float GetBlurRadius() const { return blur_radius_; }
    void SetBlurRadius(float radius) { blur_radius_ = radius; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> horizontal_blur_pipeline_;
    id<MTLComputePipelineState> vertical_blur_pipeline_;
    id<MTLCommandQueue> command_queue_;
    id<MTLTexture> intermediate_texture_;
    
    // 파라미터
    float blur_radius_;
    
    // 성능 최적화: 캐싱
    id<MTLTexture> last_input_texture_;
    float last_blur_radius_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateBlurNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
