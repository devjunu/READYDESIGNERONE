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

// Switch 노드 - 여러 입력 중 하나 선택
class SwitchNode : public TOPNodeBase
{
public:
    SwitchNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~SwitchNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Switch"; }
    std::string GetCategory() const override { return "TOP/Utility"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;
    
    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Switch 파라미터
    int GetSelectedIndex() const { return selected_index_; }
    void SetSelectedIndex(int index) { selected_index_ = index; }
    
    int GetInputCount() const { return input_count_; }
    void SetInputCount(int count);
    
private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> pipeline_;
    
    // 파라미터
    int selected_index_;
    int input_count_;
    
    // Metal 리소스 초기화
    bool InitializeMetal();
    
    // 포트 재구성
    void RebuildPorts(Graph<Node>& graph);
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateSwitchNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
