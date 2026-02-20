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

// Edge Detection 타입
enum class EdgeType {
    Sobel,
    Prewitt,
    Canny
};

// Edge 노드 - 엣지 감지
class EdgeNode : public TOPNodeBase
{
public:
    EdgeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~EdgeNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Edge"; }
    std::string GetCategory() const override { return "TOP/Filter"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Edge 파라미터
    EdgeType GetEdgeType() const { return edge_type_; }
    void SetEdgeType(EdgeType type) { edge_type_ = type; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> sobel_pipeline_;
    id<MTLComputePipelineState> prewitt_pipeline_;
    id<MTLComputePipelineState> canny_pipeline_;
    
    // 파라미터
    EdgeType edge_type_;
    float threshold_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateEdgeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
