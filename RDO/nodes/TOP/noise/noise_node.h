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

// Noise 타입
enum class NoiseType {
    White,
    Perlin,
    Simplex,
    Cellular
};

// Noise 노드 - 노이즈 생성
class NoiseNode : public TOPNodeBase
{
public:
    NoiseNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~NoiseNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Noise"; }
    std::string GetCategory() const override { return "TOP/Generator"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Noise 파라미터
    NoiseType GetNoiseType() const { return noise_type_; }
    void SetNoiseType(NoiseType type) { noise_type_ = type; }
    
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    void SetResolution(int width, int height) { width_ = width; height_ = height; }
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> white_pipeline_;
    id<MTLComputePipelineState> perlin_pipeline_;
    id<MTLComputePipelineState> simplex_pipeline_;
    id<MTLComputePipelineState> cellular_pipeline_;
    
    // 파라미터
    NoiseType noise_type_;
    float scale_;
    float octaves_;
    float persistence_;
    float seed_;
    int width_;
    int height_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateNoiseNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
