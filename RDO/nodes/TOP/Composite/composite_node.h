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

enum class CompositeMode
{
    Normal = 0,
    Add = 1,
    Multiply = 2,
    Screen = 3,
    Overlay = 4
};

// Composite 노드
class CompositeNode : public TOPNodeBase
{
public:
    CompositeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~CompositeNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Composite"; }
    std::string GetCategory() const override { return "TOP/Composite"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Composite 파라미터
    float GetOpacity() const { return opacity_; }
    void SetOpacity(float opacity) { opacity_ = opacity; }
    CompositeMode GetMode() const { return mode_; }
    void SetMode(CompositeMode mode) { mode_ = mode; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> composite_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float opacity_;
    CompositeMode mode_;

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input1_texture_;
    id<MTLTexture> last_input2_texture_;
    float last_opacity_;
    CompositeMode last_mode_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateCompositeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
