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

// Kernel 형태 (dilate_node.h와 동일)
enum class ErosionKernelShape {
    Rect = 0,
    Ellipse = 1,
    Cross = 2
};

// Erode 노드 - 픽셀 축소 (형태학적 연산)
class ErodeNode : public TOPNodeBase
{
public:
    ErodeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~ErodeNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Erode"; }
    std::string GetCategory() const override { return "TOP/Filter"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Erode 파라미터
    int GetKernelSize() const { return kernel_size_; }
    void SetKernelSize(int size) { kernel_size_ = size; }

    int GetIterations() const { return iterations_; }
    void SetIterations(int iterations) { iterations_ = iterations; }

    ErosionKernelShape GetKernelShape() const { return kernel_shape_; }
    void SetKernelShape(ErosionKernelShape shape) { kernel_shape_ = shape; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> erode_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    int kernel_size_;      // 커널 크기 (3, 5, 7, ...)
    int iterations_;       // 반복 횟수
    ErosionKernelShape kernel_shape_;  // 커널 형태

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input_texture_;
    int last_kernel_size_;
    int last_iterations_;
    ErosionKernelShape last_kernel_shape_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateErodeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
