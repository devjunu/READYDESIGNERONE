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

// Kernel 형태
enum class KernelShape {
    Rect = 0,
    Ellipse = 1,
    Cross = 2
};

// Dilate 노드 - 픽셀 확장 (형태학적 연산)
class DilateNode : public TOPNodeBase
{
public:
    DilateNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~DilateNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Dilate"; }
    std::string GetCategory() const override { return "TOP/Filter"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // Dilate 파라미터
    int GetKernelSize() const { return kernel_size_; }
    void SetKernelSize(int size) { kernel_size_ = size; }

    int GetIterations() const { return iterations_; }
    void SetIterations(int iterations) { iterations_ = iterations; }

    KernelShape GetKernelShape() const { return kernel_shape_; }
    void SetKernelShape(KernelShape shape) { kernel_shape_ = shape; }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> dilate_pipeline_;
    id<MTLCommandQueue> command_queue_;

    // 파라미터
    int kernel_size_;      // 커널 크기 (3, 5, 7, ...)
    int iterations_;       // 반복 횟수
    KernelShape kernel_shape_;  // 커널 형태

    // 성능 최적화: 캐싱
    id<MTLTexture> last_input_texture_;
    int last_kernel_size_;
    int last_iterations_;
    KernelShape last_kernel_shape_;

    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateDilateNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
