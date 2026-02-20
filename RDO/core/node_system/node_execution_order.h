#pragma once

#include <vector>
#include <unordered_set>
#include "../../graph.h"
#include "../../nodes/node_types.h"

namespace example
{

// Topological Sort를 통한 노드 실행 순서 계산
// DFS 기반으로 의존성을 추적하여 올바른 실행 순서를 보장
class NodeExecutionOrder
{
public:
    NodeExecutionOrder() = default;
    ~NodeExecutionOrder() = default;

    // Topological Sort 수행 (DFS 기반)
    // output_nodes: 최종 출력 노드들 (여기서부터 역방향으로 추적)
    // port_to_node: 포트 ID -> 노드 ID 매핑
    // 반환값: 실행 순서대로 정렬된 노드 ID 목록
    std::vector<int> ComputeExecutionOrder(
        const Graph<nodes::Node>& graph,
        const std::vector<int>& output_nodes,
        const std::unordered_map<int, int>& port_to_node
    );

    // 순환 의존성 검사
    // 반환값: true면 순환 의존성 존재 (에러 상태)
    bool HasCycles(const Graph<nodes::Node>& graph);

    // 마지막 에러 메시지 (순환 의존성 발견 시)
    const std::string& GetLastError() const { return last_error_; }

private:
    // DFS 헬퍼 함수
    void DFS(
        int node_id,
        const Graph<nodes::Node>& graph,
        const std::unordered_map<int, int>& port_to_node,
        std::unordered_set<int>& visited,
        std::unordered_set<int>& recursion_stack,
        std::vector<int>& result,
        bool& has_cycle
    );

    // 에러 메시지 저장
    std::string last_error_;
};

} // namespace example
