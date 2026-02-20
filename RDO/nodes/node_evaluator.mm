#include "node_evaluator.h"
#include <imgui.h>

// Redefine ImU32 if not already defined
#ifndef IM_COL32
typedef unsigned int ImU32;
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stack>
#include <unordered_map>

namespace example
{
namespace nodes
{
static float current_time_seconds = 0.f;

// 캐싱 시스템: 그래프 평가 결과 저장
struct GraphEvaluationCache
{
    std::stack<int> postorder;
    ImU32 last_result;
    bool valid;
    
    GraphEvaluationCache() : valid(false), last_result(0) {}
};

static std::unordered_map<int, GraphEvaluationCache> evaluation_cache;

void UpdateTime(float time_seconds)
{
    current_time_seconds = time_seconds;
}

template<class T>
T clamp(T x, T a, T b)
{
    return std::min(b, std::max(x, a));
}

ImU32 EvaluateGraph(Graph<Node>& graph, const int root_node)
{
    // 성능 최적화: 그래프가 변경되지 않았고 캐시가 유효하면 캐시된 결과 반환
    auto& cache = evaluation_cache[root_node];
    
    if (!graph.is_dirty() && cache.valid)
    {
        return cache.last_result;
    }
    
    // DFS 순회 (그래프가 변경된 경우에만 수행)
    std::stack<int> postorder;
    dfs_traverse(
        graph, root_node, [&postorder](const int node_id) -> void { postorder.push(node_id); });

    std::stack<float> value_stack;
    while (!postorder.empty())
    {
        const int id = postorder.top();
        postorder.pop();
        const Node node = graph.node(id);

        switch (node.type)
        {
        case NodeType::add:
        {
            const float rhs = value_stack.top();
            value_stack.pop();
            const float lhs = value_stack.top();
            value_stack.pop();
            value_stack.push(lhs + rhs);
        }
        break;
        case NodeType::multiply:
        {
            const float rhs = value_stack.top();
            value_stack.pop();
            const float lhs = value_stack.top();
            value_stack.pop();
            value_stack.push(rhs * lhs);
        }
        break;
        case NodeType::sine:
        {
            const float x = value_stack.top();
            value_stack.pop();
            const float res = std::abs(std::sin(x));
            value_stack.push(res);
        }
        break;
        case NodeType::time:
        {
            value_stack.push(current_time_seconds);
        }
        break;
        case NodeType::value:
        {
            // If the edge does not have an edge connecting to another node, then just use the value
            // at this node. It means the node's input pin has not been connected to anything and
            // the value comes from the node's UI.
            if (graph.num_edges_from_node(id) == 0ull)
            {
                value_stack.push(node.value);
            }
        }
        break;
        default:
            break;
        }
    }

    // The final output node isn't evaluated in the loop -- instead we just pop
    // the three values which should be in the stack.
    assert(value_stack.size() == 3ull);
    const int b = static_cast<int>(255.f * clamp(value_stack.top(), 0.f, 1.f) + 0.5f);
    value_stack.pop();
    const int g = static_cast<int>(255.f * clamp(value_stack.top(), 0.f, 1.f) + 0.5f);
    value_stack.pop();
    const int r = static_cast<int>(255.f * clamp(value_stack.top(), 0.f, 1.f) + 0.5f);
    value_stack.pop();

    ImU32 result = IM_COL32(r, g, b, 255);
    
    // 결과 캐싱
    cache.last_result = result;
    cache.valid = true;
    graph.mark_clean();

    return result;
}
} // namespace nodes
} // namespace example
