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

// Resolution 노드
class ResolutionNode : public TOPNodeBase
{
public:
    ResolutionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ResolutionNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Resolution"; }
    std::string GetCategory() const override { return "TOP/Transform"; }

    // TOPNodeBase 인터페이스 구현
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

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Resolution 파라미터
    int GetTargetWidth() const { return target_width_; }
    int GetTargetHeight() const { return target_height_; }
    void SetTargetWidth(int width) { target_width_ = width; }
    void SetTargetHeight(int height) { target_height_ = height; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> resize_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    int target_width_;
    int target_height_;

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input_texture_;
    int last_target_width_;
    int last_target_height_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateResolutionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
