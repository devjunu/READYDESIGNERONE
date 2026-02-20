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

// Morphology 연산 타입
enum class MorphOp {
    Dilate = 0,
    Erode = 1,
    Opening = 2,    // Erode → Dilate
    Closing = 3,    // Dilate → Erode
    Gradient = 4,   // Dilate - Erode
    TopHat = 5,     // Input - Opening
    BlackHat = 6    // Closing - Input
};

// Morphology 노드 - 통합 형태학적 연산
class MorphologyNode : public TOPNodeBase
{
public:
    MorphologyNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~MorphologyNode();

    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Morphology"; }
    std::string GetCategory() const override { return "TOP/Filter"; }

    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    void InvalidateCache() override;

    MorphOp GetOperation() const { return operation_; }
    void SetOperation(MorphOp op) { operation_ = op; }

    int GetKernelSize() const { return kernel_size_; }
    void SetKernelSize(int size) { kernel_size_ = size; }

private:
    id<MTLDevice> device_;
    id<MTLComputePipelineState> dilate_pipeline_;
    id<MTLComputePipelineState> erode_pipeline_;
    id<MTLComputePipelineState> subtract_pipeline_;
    id<MTLCommandQueue> command_queue_;

    MorphOp operation_;
    int kernel_size_;

    bool InitializeMetal();
};

std::unique_ptr<NodeBase> CreateMorphologyNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
