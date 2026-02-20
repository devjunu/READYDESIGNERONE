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

// Transform 노드
class TransformNode : public TOPNodeBase
{
public:
    TransformNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~TransformNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Transform"; }
    std::string GetCategory() const override { return "TOP/Transform"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Transform 파라미터
    float GetRotation() const { return rotation_; }
    float GetScaleX() const { return scale_x_; }
    float GetScaleY() const { return scale_y_; }
    float GetTranslateX() const { return translate_x_; }
    float GetTranslateY() const { return translate_y_; }

    void SetRotation(float rotation) { rotation_ = rotation; }
    void SetScaleX(float scale_x) { scale_x_ = scale_x; }
    void SetScaleY(float scale_y) { scale_y_ = scale_y; }
    void SetTranslateX(float translate_x) { translate_x_ = translate_x; }
    void SetTranslateY(float translate_y) { translate_y_ = translate_y; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> transform_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    float rotation_;      // 0.0 ~ 360.0 degrees
    float scale_x_;       // 0.1 ~ 5.0
    float scale_y_;       // 0.1 ~ 5.0
    float translate_x_;   // -1.0 ~ 1.0
    float translate_y_;   // -1.0 ~ 1.0

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateTransformNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
