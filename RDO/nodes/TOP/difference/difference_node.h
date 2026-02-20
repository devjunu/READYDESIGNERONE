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

// Difference 노드 - 두 이미지의 차이 계산 (모션 감지용)
class DifferenceNode : public TOPNodeBase
{
public:
    DifferenceNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~DifferenceNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Difference"; }
    std::string GetCategory() const override { return "TOP/Composite"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Difference 파라미터
    float GetAmplify() const { return amplify_; }
    void SetAmplify(float amplify) { amplify_ = amplify; }

    bool GetAbsolute() const { return absolute_; }
    void SetAbsolute(bool absolute) { absolute_ = absolute; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> difference_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float amplify_;        // 차이값 증폭 (기본 1.0)
    bool absolute_;        // 절대값 사용 여부

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input1_texture_;
    id<MTLTexture> last_input2_texture_;
    float last_amplify_;
    bool last_absolute_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateDifferenceNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
