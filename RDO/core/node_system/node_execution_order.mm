#include "node_execution_order.h"
#include <algorithm>
#include <sstream>

namespace example
{

std::vector<int> NodeExecutionOrder::ComputeExecutionOrder(
    const Graph<nodes::Node>& graph,
    const std::vector<int>& output_nodes,
    const std::unordered_map<int, int>& port_to_node
)
{
    std::unordered_set<int> visited;
    std::unordered_set<int> recursion_stack;
    std::vector<int> result;
    bool has_cycle = false;

    last_error_.clear();

    // 각 출력 노드에서 DFS 시작
    for (int output_node_id : output_nodes) {
        if (visited.find(output_node_id) == visited.end()) {
            DFS(output_node_id, graph, port_to_node, visited, recursion_stack, result, has_cycle);

            if (has_cycle) {
                std::ostringstream oss;
                oss << "Cycle detected in node graph! Output node: " << output_node_id;
                last_error_ = oss.str();
                return {};  // 빈 벡터 반환
            }
        }
    }

    // 현재 DFS 구현은 "의존성 먼저 방문 후 현재 노드 push"이므로
    // result 자체가 이미 topological order(입력 -> 출력)이다.
    // reverse를 하면 출력 -> 입력 순으로 뒤집혀 의존성이 깨진다.

    return result;
}

bool NodeExecutionOrder::HasCycles(const Graph<nodes::Node>& graph)
{
    // 사이클 검사는 ComputeExecutionOrder에서 자동으로 수행됨
    // 이 메서드는 단순히 last_error_가 설정되었는지 확인
    return !last_error_.empty();
}

void NodeExecutionOrder::DFS(
    int node_id,
    const Graph<nodes::Node>& graph,
    const std::unordered_map<int, int>& port_to_node,
    std::unordered_set<int>& visited,
    std::unordered_set<int>& recursion_stack,
    std::vector<int>& result,
    bool& has_cycle
)
{
    // 이미 사이클 발견되었으면 조기 종료
    if (has_cycle) {
        return;
    }

    // 이미 방문한 노드면 스킵
    if (visited.find(node_id) != visited.end()) {
        return;
    }

    // 재귀 스택에 이미 있으면 사이클!
    if (recursion_stack.find(node_id) != recursion_stack.end()) {
        has_cycle = true;
        return;
    }

    // 재귀 스택에 추가
    recursion_stack.insert(node_id);

    // 입력 노드들 (의존성) 먼저 방문
    auto edges = graph.edges();
    for (const auto& edge : edges) {
        // edge.to(포트 ID)를 노드 ID로 변환
        auto to_node_it = port_to_node.find(edge.to);
        if (to_node_it == port_to_node.end()) continue;

        int to_node = to_node_it->second;

        // to_node가 현재 노드면, edge.from의 노드가 의존성
        if (to_node == node_id) {
            auto from_node_it = port_to_node.find(edge.from);
            if (from_node_it == port_to_node.end()) continue;

            int from_node = from_node_it->second;
            DFS(from_node, graph, port_to_node, visited, recursion_stack, result, has_cycle);

            if (has_cycle) {
                return;
            }
        }
    }

    // 재귀 스택에서 제거
    recursion_stack.erase(node_id);

    // 방문 완료 표시
    visited.insert(node_id);

    // Post-order로 결과에 추가
    result.push_back(node_id);
}

} // namespace example
