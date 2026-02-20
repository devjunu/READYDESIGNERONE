#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Add 노드 (SIG)
class AddNode : public CHOPNodeBase
{
public:
    AddNode(Graph<Node>& graph, const ImVec2& pos);
    virtual ~AddNode() = default;
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Add"; }
    std::string GetCategory() const override { return "CHOP/Math"; }
    
    // CHOPNodeBase 인터페이스 구현
    float Evaluate(const std::vector<float>& inputs, float time) override;
    
private:
    float result_;
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateAddNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
