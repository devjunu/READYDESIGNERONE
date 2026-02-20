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

using ::example::nodes::Node;

// OpticalFlow 노드 - 모션 흐름 감지 (간단한 버전)
class OpticalFlowNode : public TOPNodeBase
{
public:
    OpticalFlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~OpticalFlowNode();

    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "OpticalFlow"; }
    std::string GetCategory() const override { return "TOP/Analysis"; }

    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    void InvalidateCache() override;

    float GetAmplify() const { return amplify_; }
    void SetAmplify(float val) { amplify_ = val; }

private:
    id<MTLDevice> device_;
    id<MTLComputePipelineState> flow_pipeline_;
    id<MTLCommandQueue> command_queue_;

    float amplify_;
    id<MTLTexture> previous_frame_;

    bool InitializeMetal();
};

std::unique_ptr<NodeBase> CreateOpticalFlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
