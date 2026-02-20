#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <memory>
#include <string>
#include <vector>

struct ImVec2;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// TOP 전용 독립 출력 노드 (타임라인/프리뷰 경로와 분리)
class TopOutputNode : public TOPNodeBase
{
public:
    TopOutputNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    ~TopOutputNode() override;

    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Output"; }
    std::string GetCategory() const override { return "TOP/Output"; }

    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool,
        const RenderContext& context
    ) override;

    void InvalidateCache() override;

private:
    id<MTLDevice> device_;
    int last_width_;
    int last_height_;
};

std::unique_ptr<NodeBase> CreateTopOutputNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example

