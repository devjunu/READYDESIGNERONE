#include "node_manager.h"
#include "node_execution_order.h"
#include "../../gpu_batch_processor.h"
#include "../../texture_pool.h"
#include "../../nodes/CHOP/analysis/blob_track_info_node.h"
#include "../../nodes/CHOP/filter/trail_chop_node.h"
#include "../../nodes/TOP/convert/chop_to_top_node.h"
#include "../../nodes/TOP/filter/line_top_node.h"
#include "../../nodes/TOP/filter/shape_top_node.h"
#include "../../nodes/TOP/movie_file_in/movie_file_in_node.h"
#include <imgui.h>
#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace example
{

// nodes 네임스페이스의 Node를 사용
using nodes::Node;

NodeManager::NodeManager(Graph<Node>& graph, id<MTLDevice> device)
    : graph_(graph)
    , device_(device)
    , selected_node_id_(-1)
{
}

NodeManager::~NodeManager()
{
    std::cerr << "NodeManager: Destructor called, " << nodes_.size() << " nodes" << std::endl;
    
    // 모든 노드 삭제 시 그래프에서도 제거
    for (auto& pair : nodes_)
    {
        NodeBase* node = pair.second.get();
        
        // 노드 ID 삭제
        if (graph_.node_exists(node->GetNodeId()))
        {
            graph_.erase_node(node->GetNodeId());
        }
        
        // 모든 포트 ID 삭제
        for (const auto& port : node->GetInputPorts())
        {
            if (graph_.node_exists(port.id))
            {
                graph_.erase_node(port.id);
            }
        }
        for (const auto& port : node->GetOutputPorts())
        {
            if (graph_.node_exists(port.id))
            {
                graph_.erase_node(port.id);
            }
        }
    }
    
    // 노드들을 명시적으로 clear (소멸자 순서 보장)
    std::cerr << "NodeManager: Clearing nodes" << std::endl;
    nodes_.clear();
    std::cerr << "NodeManager: Destructor complete" << std::endl;
}

NodeBase* NodeManager::CreateNode(const std::string& full_name, const ImVec2& pos)
{
    // 레지스트리에서 노드 생성
    auto node = NodeRegistry::Instance().CreateNode(full_name, graph_, pos, device_);
    if (!node) return nullptr;
    
    int node_id = node->GetNodeId();
    
    // 포트 ID -> 노드 ID 매핑
    for (const auto& port : node->GetInputPorts())
    {
        port_to_node_[port.id] = node_id;
    }
    for (const auto& port : node->GetOutputPorts())
    {
        port_to_node_[port.id] = node_id;
    }
    
    // 패밀리별 인덱스 추가
    NodeFamily family = node->GetFamily();
    nodes_by_family_[family].push_back(node_id);
    
    // 노드 저장
    NodeBase* node_ptr = node.get();
    nodes_[node_id] = std::move(node);
    
    // Timeline과 노드 시스템 완전 분리: 등록 로직 제거
    // NodeEditor에서 직접 처리 (Preview 모드만)
    
    return node_ptr;
}

void NodeManager::DeleteNode(int node_id)
{
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return;
    
    NodeBase* node = it->second.get();
    
    // Timeline과 노드 시스템 완전 분리: 해제 로직 제거
    // NodeEditor에서 직접 처리 (SyncPreviewVideoEnginesWithTimeline에서 자동 해제)
    
    // 선택 해제
    if (selected_node_id_ == node_id)
    {
        selected_node_id_ = -1;
    }
    
    // 패밀리별 인덱스에서 제거
    NodeFamily family = node->GetFamily();
    auto& family_list = nodes_by_family_[family];
    family_list.erase(std::remove(family_list.begin(), family_list.end(), node_id), family_list.end());
    
    // 포트 매핑 제거
    for (const auto& port : node->GetInputPorts())
    {
        port_to_node_.erase(port.id);
        if (graph_.node_exists(port.id))
        {
            graph_.erase_node(port.id);
        }
    }
    for (const auto& port : node->GetOutputPorts())
    {
        port_to_node_.erase(port.id);
        if (graph_.node_exists(port.id))
        {
            graph_.erase_node(port.id);
        }
    }
    
    // 그래프에서 노드 제거
    if (graph_.node_exists(node_id))
    {
        graph_.erase_node(node_id);
    }
    
    // 노드 제거
    nodes_.erase(it);
}

void NodeManager::DeleteSelectedNodes(const std::vector<int>& selected_node_ids)
{
    for (int node_id : selected_node_ids)
    {
        DeleteNode(node_id);
    }
}

void NodeManager::RenderAllNodes()
{
    // BlobTrackInfoNode의 연결 업데이트 (터치디자이너처럼 자동으로)
    for (auto& pair : nodes_)
    {
        NodeBase* node = pair.second.get();
        if (auto* info_node = dynamic_cast<nodes::BlobTrackInfoNode*>(node))
        {
            info_node->UpdateConnectedBlobTrackNode(this);
        }
        // TrailCHOPNode의 연결 업데이트
        if (auto* trail_node = dynamic_cast<nodes::TrailCHOPNode*>(node))
        {
            trail_node->UpdateConnectedCHOPNode(this);
        }
        // CHOPToTOPNode의 연결 업데이트
        if (auto* chop_to_top = dynamic_cast<nodes::CHOPToTOPNode*>(node))
        {
            chop_to_top->UpdateConnectedCHOPNode(this);
        }
        // LineTOPNode의 연결 업데이트
        if (auto* line_top = dynamic_cast<nodes::LineTOPNode*>(node))
        {
            line_top->UpdateConnectedCHOPNode(this);
        }
        // ShapeTOPNode의 연결 업데이트
        if (auto* shape_top = dynamic_cast<nodes::ShapeTOPNode*>(node))
        {
            shape_top->UpdateConnectedCHOPNode(this);
        }
    }
    
    // CHOP 노드 평가 (연결 순서대로)
    // 일관된 시간 사용 (60fps 가정, 매 프레임마다 증가)
    static float global_eval_time = 0.0f;
    global_eval_time += 0.016f;  // 대략 60fps 가정
    
    // 먼저 모든 CHOP 노드를 평가하여 출력 채널 준비
    // Trail CHOP는 연결된 입력을 먼저 평가해야 하므로 별도 처리
    for (auto& pair : nodes_)
    {
        NodeBase* node = pair.second.get();
        if (auto* trail_node = dynamic_cast<nodes::TrailCHOPNode*>(node))
        {
            // 연결된 입력 노드가 있으면 먼저 평가
            auto* input_node = trail_node->GetConnectedCHOPInput();
            if (input_node)
            {
                if (auto* input_chop = dynamic_cast<AudioCHOPNodeBase*>(input_node))
                {
                    input_chop->Evaluate({}, global_eval_time);
                }
            }
            // Trail CHOP 평가 (일관된 시간 사용)
            trail_node->Evaluate({}, global_eval_time);
        }
    }
    
    // 단일 루프로 모든 노드 렌더링!
    for (auto& pair : nodes_)
    {
        NodeBase* node = pair.second.get();
        node->Render(graph_);
    }
}

void NodeManager::RenderInspectorForNode(int node_id)
{
    // O(1) 조회!
    NodeBase* node = GetNode(node_id);
    if (!node)
    {
        ImGui::TextDisabled("No node selected");
        return;
    }
    
    // 노드 정보 표시
    ImGui::TextUnformatted(node->GetTypeName().c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", NodeFamilyToString(node->GetFamily()));
    ImGui::Separator();
    
    // 노드별 인스펙터 렌더링
    node->RenderInspector();
}

void NodeManager::EnqueueGPUProcessing(GPUBatchProcessor& processor, TexturePool& pool,
                                      std::unordered_map<int, id<MTLTexture>>& texture_registry)
{
    RenderContext context;
    context.mode = RenderMode::Preview;
    EnqueueGPUProcessing(processor, pool, texture_registry, context);
}

void NodeManager::UpdateTextureRegistry(std::unordered_map<int, id<MTLTexture>>& texture_registry)
{
    // PIX 노드의 출력 텍스처를 레지스트리에 등록
    auto it = nodes_by_family_.find(NodeFamily::TOP);
    if (it == nodes_by_family_.end()) return;
    
    for (int node_id : it->second)
    {
        NodeBase* base_node = GetNode(node_id);
        if (!base_node) continue;
        
        TOPNodeBase* pix_node = dynamic_cast<TOPNodeBase*>(base_node);
        if (!pix_node) continue;
        
        // MovieFileInNode의 경우 VideoEngine에서 직접 최신 텍스처 가져오기
        // (ProcessGPU 실행 전에도 최신 프레임 텍스처를 보장)
        if (auto* movie_node = dynamic_cast<nodes::MovieFileInNode*>(pix_node))
        {
            auto* engine = movie_node->GetVideoEngine();
            if (engine && engine->IsLoaded())
            {
                id<MTLTexture> current_texture = engine->GetCurrentFrameTexture();
                if (current_texture != nil)
                {
                    // 출력 포트의 텍스처를 레지스트리에 등록
                    for (const auto& port : pix_node->GetOutputPorts())
                    {
                        if (port.family == NodeFamily::TOP)
                        {
                            texture_registry[port.id] = current_texture;
                        }
                    }
                    continue;  // 다음 노드로
                }
            }
        }
        
        // 일반 노드: GetOutputTexture() 사용 (ProcessGPU 실행 후 텍스처)
        for (const auto& port : pix_node->GetOutputPorts())
        {
            id<MTLTexture> texture = pix_node->GetOutputTexture();
            if (texture != nil)
            {
                texture_registry[port.id] = texture;
            }
        }
    }
}

NodeBase* NodeManager::GetNode(int node_id) const
{
    auto it = nodes_.find(node_id);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

void NodeManager::SetSelectedNode(int node_id)
{
    selected_node_id_ = node_id;
}

NodeBase* NodeManager::GetSelectedNode() const
{
    return GetNode(selected_node_id_);
}

size_t NodeManager::GetNodeCountByFamily(NodeFamily family) const
{
    auto it = nodes_by_family_.find(family);
    return (it != nodes_by_family_.end()) ? it->second.size() : 0;
}

NodeBase* NodeManager::GetNodeByPortId(int port_id) const
{
    auto it = port_to_node_.find(port_id);
    if (it == port_to_node_.end()) return nullptr;
    return GetNode(it->second);
}

id<MTLTexture> NodeManager::GetPrimaryOutputTexture(
    const std::unordered_map<int, id<MTLTexture>>& texture_registry) const
{
    auto it = nodes_by_family_.find(NodeFamily::TOP);
    if (it == nodes_by_family_.end() || it->second.empty())
    {
        return nil;
    }

    std::unordered_set<int> top_node_set(it->second.begin(), it->second.end());

    auto is_terminal_top_node = [this, &top_node_set](int node_id) -> bool {
        NodeBase* base_node = GetNode(node_id);
        TOPNodeBase* top_node = dynamic_cast<TOPNodeBase*>(base_node);
        if (!top_node)
        {
            return false;
        }

        for (const auto& output_port : top_node->GetOutputPorts())
        {
            if (output_port.family != NodeFamily::TOP) continue;

            for (const auto& edge : graph_.edges())
            {
                if (edge.from != output_port.id) continue;

                auto target_it = port_to_node_.find(edge.to);
                if (target_it != port_to_node_.end() &&
                    top_node_set.find(target_it->second) != top_node_set.end())
                {
                    return false;
                }
            }
        }
        return true;
    };

    std::vector<int> candidate_nodes;
    candidate_nodes.reserve(it->second.size());

    // 선택된 TOP 노드가 terminal이면 최우선
    if (selected_node_id_ != -1 &&
        top_node_set.find(selected_node_id_) != top_node_set.end() &&
        is_terminal_top_node(selected_node_id_))
    {
        candidate_nodes.push_back(selected_node_id_);
    }

    // 나머지 terminal 노드는 최신 생성순(큰 ID 우선)으로 검색
    for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
    {
        int node_id = *rit;
        if (node_id == selected_node_id_) continue;
        if (is_terminal_top_node(node_id))
        {
            candidate_nodes.push_back(node_id);
        }
    }

    // terminal이 없으면 TOP 전체를 fallback 대상으로 사용
    if (candidate_nodes.empty())
    {
        for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
        {
            candidate_nodes.push_back(*rit);
        }
    }

    for (int node_id : candidate_nodes)
    {
        NodeBase* base_node = GetNode(node_id);
        TOPNodeBase* top_node = dynamic_cast<TOPNodeBase*>(base_node);
        if (!top_node) continue;

        // 1) 레지스트리 최신 텍스처 우선
        for (const auto& output_port : top_node->GetOutputPorts())
        {
            if (output_port.family != NodeFamily::TOP) continue;

            auto tex_it = texture_registry.find(output_port.id);
            if (tex_it != texture_registry.end() && tex_it->second != nil)
            {
                return tex_it->second;
            }
        }

        // 2) 노드 내부 output_texture fallback
        id<MTLTexture> fallback_texture = top_node->GetOutputTexture();
        if (fallback_texture != nil)
        {
            return fallback_texture;
        }
    }

    return nil;
}

id<MTLTexture> NodeManager::GetDedicatedOutputTexture() const
{
    auto it = nodes_by_family_.find(NodeFamily::TOP);
    if (it == nodes_by_family_.end() || it->second.empty())
    {
        return nil;
    }

    auto is_dedicated_output_node = [](NodeBase* node) -> bool {
        if (!node) return false;
        const std::string category = node->GetCategory();
        const std::string type = node->GetTypeName();
        return (category == "TOP/Output") || (type == "Output");
    };

    // 선택된 노드가 Output 노드면 최우선
    if (selected_node_id_ != -1)
    {
        NodeBase* selected = GetNode(selected_node_id_);
        if (is_dedicated_output_node(selected))
        {
            if (TOPNodeBase* output_node = dynamic_cast<TOPNodeBase*>(selected))
            {
                id<MTLTexture> texture = output_node->GetOutputTexture();
                if (texture != nil)
                {
                    return texture;
                }
            }
        }
    }

    // 최신 생성된 Output 노드부터 검색
    for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
    {
        NodeBase* node = GetNode(*rit);
        if (!is_dedicated_output_node(node))
        {
            continue;
        }

        TOPNodeBase* output_node = dynamic_cast<TOPNodeBase*>(node);
        if (!output_node)
        {
            continue;
        }

        id<MTLTexture> texture = output_node->GetOutputTexture();
        if (texture != nil)
        {
            return texture;
        }
    }

    return nil;
}

bool NodeManager::CanConnect(int from_port_id, int to_port_id) const
{
    // 포트를 소유한 노드 찾기
    auto from_it = port_to_node_.find(from_port_id);
    auto to_it = port_to_node_.find(to_port_id);
    
    // 둘 다 V2 노드가 아니면 "모르겠다" (기존 시스템에 위임)
    if (from_it == port_to_node_.end() && to_it == port_to_node_.end())
    {
        return true;  // 기존 시스템에서 검증하도록 허용
    }
    
    // 하나만 V2 노드면 일단 허용 (추후 VideoFileLoader도 V2로 변환 예정)
    if (from_it == port_to_node_.end() || to_it == port_to_node_.end())
    {
        return true;  // 임시로 허용 (호환성)
    }
    
    NodeBase* from_node = GetNode(from_it->second);
    NodeBase* to_node = GetNode(to_it->second);
    
    if (!from_node || !to_node) return false;
    
    // 포트 찾기
    const Port* from_port = from_node->FindPortById(from_port_id);
    const Port* to_port = to_node->FindPortById(to_port_id);
    
    if (!from_port || !to_port) return false;
    
    // 타입 안전 연결 검증
    return example::CanConnect(*from_port, *to_port);
}

void NodeManager::UpdateAllNodes()
{
    // VideoFileLoaderNode 등 업데이트가 필요한 노드들을 업데이트
    // Timeline과 완전 분리: Update만 호출 (Timeline 등록은 NodeEditor에서 처리)
    for (auto& pair : nodes_)
    {
        NodeBase* node = pair.second.get();
        
        // VideoFileLoaderNode인 경우 Update 호출
        // Category가 "TOP/IO"로 시작하는지 확인 (서브카테고리 포함)
        std::string category = node->GetCategory();
        if (category.find("IO") != std::string::npos || category == "IO")
        {
            // NodeBase에 Update 메서드가 있는지 확인 (MovieFileInNode 등)
            // forward declaration으로 타입 확인 불가하므로, 
            // NodeBase에 가상 Update 메서드 추가하거나, 
            // 여기서는 category로만 판단하여 Update 호출
            // (MovieFileInNode는 NodeBase를 상속하므로 직접 호출 불가)
            // 대신 NodeEditor에서 직접 처리하도록 변경
        }
    }
}

// RenderContext를 받는 새로운 EnqueueGPUProcessing
void NodeManager::EnqueueGPUProcessing(GPUBatchProcessor& processor, TexturePool& pool,
                                      std::unordered_map<int, id<MTLTexture>>& texture_registry,
                                      const RenderContext& context)
{
    // PIX 노드만 필터링
    auto it = nodes_by_family_.find(NodeFamily::TOP);
    if (it == nodes_by_family_.end())
    {
        return;
    }

    // Step 1: 출력 노드 찾기
    std::vector<int> output_nodes;
    std::unordered_set<int> top_node_set(it->second.begin(), it->second.end());

    // 강제 출력 노드가 지정된 경우 해당 체인만 실행
    if (context.forced_output_node_id >= 0 &&
        top_node_set.find(context.forced_output_node_id) != top_node_set.end())
    {
        output_nodes.push_back(context.forced_output_node_id);
    }

    if (output_nodes.empty())
    {
        // 자동 모드: outgoing edge가 없는 TOP 노드들을 출력 노드로 사용
        for (int node_id : it->second)
        {
            bool is_output = true;

            // 이 노드에서 나가는 엣지가 다른 TOP 노드로 가는지 확인
            for (const auto& port : GetNode(node_id)->GetOutputPorts())
            {
                if (port.family != NodeFamily::TOP) continue;

                for (const auto& edge : graph_.edges())
                {
                    if (edge.from == port.id)
                    {
                        // 도착지가 TOP 노드인지 확인
                        auto target_node_it = port_to_node_.find(edge.to);
                        if (target_node_it != port_to_node_.end() &&
                            top_node_set.find(target_node_it->second) != top_node_set.end())
                        {
                            is_output = false;
                            break;
                        }
                    }
                }
                if (!is_output) break;
            }

            if (is_output)
            {
                output_nodes.push_back(node_id);
            }
        }
    }

    // 출력 노드가 없으면 모든 노드를 처리 (순환 그래프 또는 모든 노드가 연결됨)
    if (output_nodes.empty())
    {
        output_nodes = it->second;
    }

    // Step 2: Topological Sort로 실행 순서 계산
    NodeExecutionOrder sorter;
    std::vector<int> execution_order = sorter.ComputeExecutionOrder(graph_, output_nodes, port_to_node_);

    // execution_order가 비어있으면 fallback (순환 의존성 또는 에러)
    if (execution_order.empty())
    {
        if (!sorter.GetLastError().empty())
        {
            NSLog(@"⚠️ [EnqueueGPU] Cycle detected: %s", sorter.GetLastError().c_str());
        }
        else
        {
            NSLog(@"⚠️ [EnqueueGPU] Topological sort returned empty, using default order");
        }
        // Fallback: 원래 순서대로 처리
        execution_order = it->second;
    }

    // Step 3: 정렬된 순서대로 GPU 작업 등록
    // 중요: 입력 텍스처 lookup과 출력 텍스처 레지스트리 갱신을
    // enqueue 시점이 아닌 flush 시점에 수행해 프레임 내 의존성을 보장한다.
    auto* registry_ptr = &texture_registry;

    for (int node_id : execution_order)
    {
        NodeBase* base_node = GetNode(node_id);
        if (!base_node) continue;

        TOPNodeBase* pix_node = dynamic_cast<TOPNodeBase*>(base_node);
        if (!pix_node) continue;

        if (context.skip_movie_file_in_nodes &&
            dynamic_cast<nodes::MovieFileInNode*>(pix_node) != nullptr)
        {
            continue;
        }

        // 입력 포트별 source 포트 ID를 미리 계산
        std::vector<int> source_port_ids;
        const auto& input_ports = pix_node->GetInputPorts();
        for (const auto& port : input_ports)
        {
            if (port.family != NodeFamily::TOP) continue;

            int source_port_id = -1;
            for (const auto& edge : graph_.edges())
            {
                if (edge.to == port.id)
                {
                    source_port_id = edge.from;
                    break;
                }
            }
            source_port_ids.push_back(source_port_id);
        }

        // 출력 포트 ID 캐싱
        std::vector<int> output_port_ids;
        for (const auto& port : pix_node->GetOutputPorts())
        {
            if (port.family == NodeFamily::TOP)
            {
                output_port_ids.push_back(port.id);
            }
        }

        // GPU 작업 등록 (RenderContext 전달)
        processor.enqueue([pix_node, source_port_ids, output_port_ids, &pool, context, registry_ptr](id<MTLCommandBuffer> cmd_buffer) {
            std::vector<id<MTLTexture>> input_textures;
            input_textures.reserve(source_port_ids.size());

            for (int source_port_id : source_port_ids)
            {
                id<MTLTexture> input_texture = nil;
                if (source_port_id != -1)
                {
                    auto tex_it = registry_ptr->find(source_port_id);
                    if (tex_it != registry_ptr->end())
                    {
                        input_texture = tex_it->second;
                    }
                }
                input_textures.push_back(input_texture);
            }

            pix_node->ProcessGPU(input_textures, cmd_buffer, &pool, context);

            id<MTLTexture> output_texture = pix_node->GetOutputTexture();
            if (output_texture != nil)
            {
                for (int output_port_id : output_port_ids)
                {
                    (*registry_ptr)[output_port_id] = output_texture;
                }
            }
            else
            {
                for (int output_port_id : output_port_ids)
                {
                    registry_ptr->erase(output_port_id);
                }
            }
        });
    }
}

// Final 렌더링 (최종 출력용)
void NodeManager::RenderFinal(int output_node_id, int width, int height)
{
    // TODO: Final 렌더링 구현
    // 특정 출력 노드만 렌더링하는 기능
    // 현재는 placeholder
    NSLog(@"RenderFinal: output_node_id=%d, size=%dx%d", output_node_id, width, height);
}

void NodeManager::InvalidateAllCaches()
{
    // 먼저 TexturePool 참조를 얻어야 함
    // 하지만 NodeManager는 TexturePool을 직접 소유하지 않음
    // 따라서 노드들이 다음 ProcessGPU 호출 시 자동으로 정리하도록 함

    // 모든 PIX 노드의 캐시를 무효화
    for (auto& pair : nodes_)
    {
        if (TOPNodeBase* pix_node = dynamic_cast<TOPNodeBase*>(pair.second.get()))
        {
            pix_node->InvalidateCache();
        }
    }

    // 참고: 실제 TexturePool 반환은 각 노드의 ProcessGPU에서
    // 크기가 변경되었을 때 자동으로 처리됨
}

} // namespace example
