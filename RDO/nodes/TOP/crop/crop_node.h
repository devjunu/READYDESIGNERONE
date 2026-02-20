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

// Crop 노드 - 자르기
class CropNode : public TOPNodeBase
{
public:
    CropNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~CropNode();
    
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Crop"; }
    std::string GetCategory() const override { return "TOP/Transform"; }
    
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    void InvalidateCache() override;
    
private:
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    float left_;
    float right_;
    float top_;
    float bottom_;
    
    bool InitializeMetal();
};

std::unique_ptr<NodeBase> CreateCropNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
