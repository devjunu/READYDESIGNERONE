#pragma once

#import <Metal/Metal.h>
#include "node_base.h"
#include "node_registry.h"
#include "../../graph.h"
#include "../../nodes/node_types.h"
#include <unordered_map>
#include <memory>
#include <vector>

namespace example
{

class GPUBatchProcessor;
class TexturePool;

// nodes 네임스페이스의 Node를 사용
using nodes::Node;

    // 노드 매니저: 모든 노드의 생명주기와 렌더링 관리
class NodeManager
{
public:
    NodeManager(Graph<Node>& graph, id<MTLDevice> device);
    ~NodeManager();
    
    // ============ 노드 생성/삭제 ============
    
    // 노드 생성 (레지스트리에서 타입으로 생성)
    NodeBase* CreateNode(const std::string& full_name, const ImVec2& pos);
    
    // 노드 삭제
    void DeleteNode(int node_id);
    
    // 선택된 노드들 삭제
    void DeleteSelectedNodes(const std::vector<int>& selected_node_ids);
    
    // ============ 렌더링 ============
    
    // 모든 노드 렌더링 (단일 루프!)
    void RenderAllNodes();
    
    // 선택된 노드의 인스펙터 렌더링 (O(1) 조회!)
    void RenderInspectorForNode(int node_id);
    
    // 모든 노드 업데이트 (VideoEngine 등)
    void UpdateAllNodes();
    
    // ============ GPU 처리 ============

    // PIX 노드들의 GPU 배치 처리 등록 (기존 - 역호환성)
    void EnqueueGPUProcessing(GPUBatchProcessor& processor, TexturePool& pool,
                             std::unordered_map<int, id<MTLTexture>>& texture_registry);

    // PIX 노드들의 GPU 배치 처리 등록 (RenderContext 포함)
    void EnqueueGPUProcessing(GPUBatchProcessor& processor, TexturePool& pool,
                             std::unordered_map<int, id<MTLTexture>>& texture_registry,
                             const RenderContext& context);

    // Final 렌더링 (최종 출력용)
    void RenderFinal(int output_node_id, int width, int height);

    // 텍스처 레지스트리 업데이트 (zero-copy 연결용)
    void UpdateTextureRegistry(std::unordered_map<int, id<MTLTexture>>& texture_registry);

    // Output Preview용 최종 TOP 출력 텍스처 조회
    id<MTLTexture> GetPrimaryOutputTexture(
        const std::unordered_map<int, id<MTLTexture>>& texture_registry) const;

    // TOP/Output 노드 전용 독립 출력 텍스처 조회
    id<MTLTexture> GetDedicatedOutputTexture() const;
    
    // ============ 노드 조회 ============
    
    // 노드 ID로 조회 (O(1))
    NodeBase* GetNode(int node_id) const;
    
    // 선택된 노드 설정/조회
    void SetSelectedNode(int node_id);
    int GetSelectedNodeId() const { return selected_node_id_; }
    NodeBase* GetSelectedNode() const;
    
    // 모든 노드 개수
    size_t GetNodeCount() const { return nodes_.size(); }
    
    // 패밀리별 노드 개수
    size_t GetNodeCountByFamily(NodeFamily family) const;
    
    // 모든 노드 순회 (콜백 함수)
    template<typename Func>
    void ForEachNode(Func func) const
    {
        for (const auto& pair : nodes_)
        {
            func(pair.first, pair.second.get());
        }
    }
    
    // 캐시 무효화 (프리뷰 스케일 변경 시)
    void InvalidateAllCaches();
    
    // ============ 타입 안전 연결 검증 ============
    
    // 두 포트 간 연결 가능 여부 체크
    bool CanConnect(int from_port_id, int to_port_id) const;
    
    // 포트 ID로 노드 찾기 (연결된 노드 찾기용)
    NodeBase* GetNodeByPortId(int port_id) const;
    
private:
    // 노드 ID -> NodeBase 맵 (O(1) 조회)
    std::unordered_map<int, std::unique_ptr<NodeBase>> nodes_;
    
    // 그래프 참조
    Graph<Node>& graph_;
    
    // Metal 디바이스
    id<MTLDevice> device_;
    
    // 선택된 노드 ID
    int selected_node_id_;
    
    // 패밀리별 노드 목록 (필터링 최적화)
    std::unordered_map<NodeFamily, std::vector<int>> nodes_by_family_;
    
    // 포트 ID -> 노드 ID 역참조 맵 (빠른 노드 찾기)
    std::unordered_map<int, int> port_to_node_;
};

} // namespace example
