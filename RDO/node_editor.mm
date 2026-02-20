#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes.h>
#include "node_editor.h"
#include "graph.h"
#include "nodes/node_types.h"
#include "timeline.h"
#include <iostream>

// 노드 시스템 코어
#include "core/node_system/node_manager.h"
#include "core/node_system/node_registry.h"
#include "core/node_system/port.h"

// 새 아키텍처 노드들
#include "nodes/TOP/blur/blur_node.h"
#include "nodes/TOP/glow/glow_node.h"
#include "nodes/CHOP/math/add_node.h"
#include "nodes/CHOP/math/multiply_node.h"
#include "nodes/CHOP/math/sine_node.h"
#include "nodes/CHOP/generator/time_node.h"
#include "nodes/TOP/movie_file_in/movie_file_in_node.h"

// 레거시 노드들 (마이그레이션 진행 중)
#include "nodes/node_evaluator.h"
#include "nodes/TOP/threshold/threshold_node.h"
#include "nodes/TOP/composite/composite_node.h"
#include "nodes/TOP/transform/transform_node.h"
#include "nodes/TOP/resolution/resolution_node.h"
#include "nodes/TOP/color_correction/color_correction_node.h"
#include "nodes/TOP/background_subtract/background_subtract_node.h"

#include "gpu_batch_processor.h"
#include "texture_pool.h"
#include "performance_monitor.h"
#include "preview_settings.h"
#include "core/media/video/video_file_encoder.h"

#import <QuartzCore/QuartzCore.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace example
{
namespace
{
using namespace nodes;

static bool emulate_three_button_mouse = false;

class ColorNodeEditor
{
public:
    ColorNodeEditor()
        : graph_()
        , node_manager_(nullptr)
        , root_node_id_(-1), minimap_location_(ImNodesMiniMapLocation_BottomRight)
        , device_(nil), last_time_(0.0), texture_registry_()
        , gpu_batch_processor_(nullptr), texture_pool_(nullptr)
        , first_time_(true), selected_node_id_(-1), selected_node_type_(NodeType::value)
        , timeline_()
        , export_path_()
        , export_fps_(30.0f)
        , export_start_time_(0.0f)
        , export_end_time_(5.0f)
        , export_auto_end_(true)
        , export_in_progress_(false)
        , export_status_("Idle")
    {
    }
    
    ~ColorNodeEditor()
    {
        // 소멸 순서가 중요: 노드들이 texture_pool_을 참조할 수 있으므로
        // node_manager_를 먼저 해제해야 함
        std::cerr << "ColorNodeEditor: Destructor called" << std::endl;
        
        if (node_manager_)
        {
            std::cerr << "ColorNodeEditor: Destroying node_manager" << std::endl;
            node_manager_.reset();
            std::cerr << "ColorNodeEditor: node_manager destroyed" << std::endl;
        }
        
        if (gpu_batch_processor_)
        {
            std::cerr << "ColorNodeEditor: Destroying gpu_batch_processor" << std::endl;
            gpu_batch_processor_.reset();
            std::cerr << "ColorNodeEditor: gpu_batch_processor destroyed" << std::endl;
        }
        
        if (texture_pool_)
        {
            std::cerr << "ColorNodeEditor: Destroying texture_pool" << std::endl;
            texture_pool_.reset();
            std::cerr << "ColorNodeEditor: texture_pool destroyed" << std::endl;
        }
        
        std::cerr << "ColorNodeEditor: Destructor complete" << std::endl;
    }
    
    void SetDevice(id<MTLDevice> device)
    {
        device_ = device;
        
        // NodeManager 초기화 (Timeline과 완전 분리)
        if (device_ && !node_manager_)
        {
            node_manager_ = std::make_unique<NodeManager>(graph_, device_);
        }
        
        // GPU 배치 처리 시스템 초기화
        if (device_ && !gpu_batch_processor_)
        {
            gpu_batch_processor_ = std::make_unique<GPUBatchProcessor>(device_);
        }
        
        // 텍스처 풀 초기화
        if (device_ && !texture_pool_)
        {
            texture_pool_ = std::make_unique<TexturePool>(device_);
        }
    }

    void show()
    {
        PERF_SCOPE("Total Frame");

        // Update timer context
        double current_time = CACurrentMediaTime();

        // Delta time 계산
        float delta_time = 0.0f;
        if (last_time_ > 0.0)
        {
            delta_time = static_cast<float>(current_time - last_time_);
        }
        last_time_ = current_time;

        if (export_in_progress_)
        {
            ProcessOfflineEncodeStep();
        }

        // Create main DockSpace
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpace Window", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // Create DockSpace
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        // Setup default docking layout on first run
        if (first_time_)
        {
            first_time_ = false;

            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

            // Split the dockspace
            ImGuiID dock_main_id = dockspace_id;
            ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.20f, nullptr, &dock_main_id);
            ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.25f, nullptr, &dock_main_id);
            ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.25f, nullptr, &dock_main_id);

            // Dock windows to their default positions
            ImGui::DockBuilderDockWindow("Node Editor", dock_main_id);
            ImGui::DockBuilderDockWindow("Inspector", dock_id_left);
            ImGui::DockBuilderDockWindow("Properties", dock_id_left);
            ImGui::DockBuilderDockWindow("Output Preview", dock_id_right);
            ImGui::DockBuilderDockWindow("Preview Settings", dock_id_right);  // Output Preview와 같은 위치에 도킹
            ImGui::DockBuilderDockWindow("Timeline", dock_id_bottom);  // Timeline을 기본으로 설정
            ImGui::DockBuilderDockWindow("Controls", dock_id_bottom);
            ImGui::DockBuilderDockWindow("Performance Monitor", dock_id_bottom);

            ImGui::DockBuilderFinish(dockspace_id);
        }

        // Menu bar
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("View"))
            {
                if (ImGui::BeginMenu("Mini-map"))
                {
                    const char* names[] = {
                        "Top Left",
                        "Top Right",
                        "Bottom Left",
                        "Bottom Right",
                    };
                    int locations[] = {
                        ImNodesMiniMapLocation_TopLeft,
                        ImNodesMiniMapLocation_TopRight,
                        ImNodesMiniMapLocation_BottomLeft,
                        ImNodesMiniMapLocation_BottomRight,
                    };

                    for (int i = 0; i < 4; i++)
                    {
                        bool selected = minimap_location_ == locations[i];
                        if (ImGui::MenuItem(names[i], NULL, &selected))
                            minimap_location_ = locations[i];
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Style"))
            {
                if (ImGui::MenuItem("Classic"))
                {
                    ImGui::StyleColorsClassic();
                    ImNodes::StyleColorsClassic();
                }
                if (ImGui::MenuItem("Dark"))
                {
                    ImGui::StyleColorsDark();
                    ImNodes::StyleColorsDark();
                }
                if (ImGui::MenuItem("Light"))
                {
                    ImGui::StyleColorsLight();
                    ImNodes::StyleColorsLight();
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenuBar();
        }

        ImGui::End();

        // Timeline UI는 인코딩 중에도 항상 표시/동작
        timeline_.Render(delta_time);

        if (!export_in_progress_)
        {
            // Timeline time을 노드 시스템에 전달 (모듈화된 Timeline 사용)
            UpdateTime(timeline_.GetCurrentTime());

            // VideoEngine 업데이트 및 Preview 모드 VideoEngine만 Timeline에 등록
            {
                PERF_SCOPE("Video Update");
                if (node_manager_)
                {
                    // MovieFileInNode Update 호출 및 Preview 모드 VideoEngine만 Timeline에 등록
                    // 노드 시스템과 Timeline 완전 분리: NodeEditor에서만 연결
                    UpdateMovieFileInNodesAndSyncTimeline();
                }
            }
        }

        // The node editor window (now dockable)
        ImGui::Begin("Node Editor", NULL);

        ImNodes::BeginNodeEditor();

        // 노드 생성 메뉴 (개선된 UI/UX)
        {
            const bool open_popup = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
                                    ImNodes::IsEditorHovered() && ImGui::IsKeyReleased(ImGuiKey_A);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.f, 6.f));
            
            if (!ImGui::IsAnyItemHovered() && open_popup)
            {
                ImGui::OpenPopup("add node");
            }

            if (ImGui::BeginPopup("add node"))
            {
                const ImVec2 click_pos = ImGui::GetMousePosOnOpeningCurrentPopup();

                ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Add Node");
                ImGui::Separator();
                ImGui::Spacing();
                
                // NodeRegistry 기반 계층적 카테고리 메뉴
                if (node_manager_)
                {
                    const auto& category_index = NodeRegistry::Instance().GetCategoryIndex();
                    
                    // 카테고리를 그룹별로 정리 (PIX, SIG, IO 등)
                    std::map<std::string, std::map<std::string, std::vector<std::string>>> grouped_categories;
                    
                    for (const auto& [category, node_list] : category_index)
                    {
                        // 카테고리를 "메인 그룹/서브 카테고리"로 분할
                        size_t slash_pos = category.find('/');
                        std::string main_group = (slash_pos != std::string::npos) ? 
                                                category.substr(0, slash_pos) : category;
                        std::string sub_category = (slash_pos != std::string::npos) ? 
                                                   category.substr(slash_pos + 1) : "";
                        
                        // 서브 카테고리가 없으면 빈 문자열로
                        grouped_categories[main_group][sub_category].insert(
                            grouped_categories[main_group][sub_category].end(),
                            node_list.begin(), node_list.end()
                        );
                    }
                    
                    // 메인 그룹 순서 정의 - TouchDesigner Style
                    std::vector<std::string> group_order = {"TOP", "CHOP", "SOP", "DAT", "COMP", "MAT", "AI", "Output"};
                    
                    for (const auto& main_group : group_order)
                    {
                        // 그룹별 색상 및 레이블
                        const char* label = "";
                        ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                        
                        if (main_group == "TOP") {
                            label = "TOP (Texture Operators)";
                            color = ImVec4(0.8f, 0.5f, 1.0f, 1.0f); // Purple
                        } else if (main_group == "CHOP") {
                            label = "CHOP (Channel Operators)";
                            color = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // Green
                        } else if (main_group == "SOP") {
                            label = "SOP (Surface Operators)";
                            color = ImVec4(0.3f, 0.6f, 1.0f, 1.0f); // Blue
                        } else if (main_group == "DAT") {
                            label = "DAT (Data Operators)";
                            color = ImVec4(1.0f, 0.5f, 0.8f, 1.0f); // Pink
                        } else if (main_group == "COMP") {
                            label = "COMP (Components)";
                            color = ImVec4(0.3f, 0.3f, 0.3f, 1.0f); // Dark Grey/Black
                        } else if (main_group == "MAT") {
                            label = "MAT (Materials)";
                            color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f); // Yellow
                        } else if (main_group == "AI") {
                            label = "AI (Machine Learning)";
                            color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f); // Red
                        } else if (main_group == "Output") {
                            label = "Output";
                            color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                        }
                        
                        ImGui::PushStyleColor(ImGuiCol_Text, color);
                        
                        if (ImGui::BeginMenu(label))
                        {
                            ImGui::PopStyleColor();
                            
                            // 그룹에 노드가 있는지 확인
                            auto group_it = grouped_categories.find(main_group);
                            bool has_nodes = (group_it != grouped_categories.end() && !group_it->second.empty());
                            
                            if (has_nodes)
                            {
                                // 서브 카테고리별로 노드 표시
                                const auto& sub_categories = group_it->second;
                                bool first_category = true;
                                
                                for (const auto& [sub_cat, node_list] : sub_categories)
                                {
                                    // 서브 카테고리 헤더 표시 (있는 경우만)
                                    if (!sub_cat.empty() && sub_categories.size() > 1)
                                    {
                                        if (!first_category)
                                        {
                                            ImGui::Spacing();
                                            ImGui::Spacing();
                                        }
                                        
                                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
                                        ImGui::TextUnformatted(sub_cat.c_str());
                                        ImGui::PopStyleColor();
                                        ImGui::Separator();
                                        first_category = false;
                                    }
                                    
                                    // 노드 목록 표시
                                    for (const auto& full_name : node_list)
                                    {
                                        const NodeTypeInfo* info = NodeRegistry::Instance().GetNodeInfo(full_name);
                                        if (info)
                                        {
                                            if (ImGui::MenuItem(info->name.c_str()))
                                            {
                                                node_manager_->CreateNode(full_name, click_pos);
                                            }
                                        
                                        // 툴팁 표시
                                        if (ImGui::IsItemHovered() && !info->description.empty())
                                        {
                                            ImGui::BeginTooltip();
                                            ImGui::TextUnformatted(info->description.c_str());
                                            ImGui::EndTooltip();
                                        }
                                    }
                                }
                            }
                            }
                            else
                            {
                                // 노드가 없는 경우
                                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                                ImGui::TextUnformatted("Coming Soon...");
                                ImGui::PopStyleColor();
                            }
                            
                            ImGui::EndMenu();
                        }
                        else
                        {
                            ImGui::PopStyleColor();
                        }
                    }
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextDisabled("Press 'A' to open this menu");

                ImGui::EndPopup();
            }
            ImGui::PopStyleVar(2);
        }

        // 노드 시스템 렌더링
        if (node_manager_)
        {
            PERF_SCOPE("Node Rendering");
            node_manager_->RenderAllNodes();
        }

        // 레거시 PIX 노드 렌더링 제거됨 (마이그레이션 완료)
        id<MTLTexture> dedicated_output_preview_texture = nil;

        // NodeManager GPU 배치 처리
        if (!export_in_progress_ && node_manager_ && gpu_batch_processor_ && texture_pool_)
        {
            PERF_SCOPE("GPU Batch Processing");

            // RenderContext 생성 (Preview 모드)
            RenderContext preview_ctx;
            preview_ctx.mode = RenderMode::Preview;
            preview_ctx.allow_shallow_copy = true;   // Zero-copy 최적화 활성화
            preview_ctx.allow_texture_views = true;  // Texture View 최적화 활성화

            // 이전 프레임 stale 엔트리를 제거한 뒤 최신 텍스처 재등록
            texture_registry_.clear();
            node_manager_->UpdateTextureRegistry(texture_registry_);
            node_manager_->EnqueueGPUProcessing(*gpu_batch_processor_, *texture_pool_, texture_registry_, preview_ctx);
        }

        // 프레임 끝에서 모든 GPU 작업을 한 번에 실행
        {
            PERF_SCOPE("GPU Batch Flush");
            if (gpu_batch_processor_)
            {
                gpu_batch_processor_->flush();
            }
        }

        // Output Preview 전용: TOP/Output 체인만 Final(원본) 모드로 재평가
        if (!export_in_progress_ && node_manager_ && gpu_batch_processor_ && texture_pool_)
        {
            const int output_node_id = FindDedicatedOutputNodeId();
            if (output_node_id >= 0)
            {
                std::unordered_map<int, id<MTLTexture>> output_preview_registry;
                node_manager_->UpdateTextureRegistry(output_preview_registry);

                RenderContext output_preview_ctx;
                output_preview_ctx.mode = RenderMode::Final;
                output_preview_ctx.allow_shallow_copy = true;
                output_preview_ctx.allow_texture_views = true;
                output_preview_ctx.forced_output_node_id = output_node_id;

                node_manager_->EnqueueGPUProcessing(*gpu_batch_processor_, *texture_pool_, output_preview_registry, output_preview_ctx);
                gpu_batch_processor_->flush();
                dedicated_output_preview_texture = node_manager_->GetDedicatedOutputTexture();
            }
        }
        
        // 프레임 종료: 텍스처 풀 정리
        if (texture_pool_)
        {
            texture_pool_->end_frame();
        }

        // Render links
        for (const auto& edge : graph_.edges())
        {
            if (graph_.node(edge.from).type != NodeType::value)
                continue;

            ImNodes::Link(edge.id, edge.from, edge.to);
        }

        ImNodes::MiniMap(0.2f, minimap_location_);
        ImNodes::EndNodeEditor();

        // Handle new links
        {
            int start_attr, end_attr;
            if (ImNodes::IsLinkCreated(&start_attr, &end_attr))
            {
                // 타입 안전 연결 검증
                bool can_connect = true;
                
                if (node_manager_)
                {
                    can_connect = node_manager_->CanConnect(start_attr, end_attr);
                }
                
                // 기존 로직도 유지 (호환성)
                const NodeType start_type = graph_.node(start_attr).type;
                const NodeType end_type = graph_.node(end_attr).type;
                const bool valid_link = (start_type != end_type) || 
                                       (start_type == NodeType::value && end_type == NodeType::value);
                
                if (can_connect && valid_link)
                {
                    if (start_type != NodeType::value && end_type == NodeType::value)
                    {
                        std::swap(start_attr, end_attr);
                    }
                    graph_.insert_edge(start_attr, end_attr);
                }
                else if (!can_connect)
                {
                    // 연결 불가 피드백 (선택적)
                    ImGui::OpenPopup("Connection Error");
                }
            }
        }

        // 연결 오류 팝업
        if (ImGui::BeginPopup("Connection Error"))
        {
            ImGui::TextUnformatted("Cannot connect: Incompatible port types!");
            ImGui::TextDisabled("(e.g., PIX texture -> SIG float)");
            ImGui::EndPopup();
        }

        // Handle deleted links
        {
            int link_id;
            if (ImNodes::IsLinkDestroyed(&link_id))
            {
                graph_.erase_edge(link_id);
            }
        }

        {
            const int num_selected = ImNodes::NumSelectedLinks();
            if (num_selected > 0 && ImGui::IsKeyReleased(ImGuiKey_X))
            {
                static std::vector<int> selected_links;
                selected_links.resize(static_cast<size_t>(num_selected));
                ImNodes::GetSelectedLinks(selected_links.data());
                for (const int edge_id : selected_links)
                {
                    graph_.erase_edge(edge_id);
                }
            }
        }

        // 노드 삭제 (NodeManager 통합)
        {
            const int num_selected = ImNodes::NumSelectedNodes();
            if (num_selected > 0 && ImGui::IsKeyReleased(ImGuiKey_X))
            {
                static std::vector<int> selected_nodes;
                selected_nodes.resize(static_cast<size_t>(num_selected));
                ImNodes::GetSelectedNodes(selected_nodes.data());
                
                for (const int node_id : selected_nodes)
                {
                    // 삭제 전에 MovieFileInNode인지 확인하여 Timeline에서 해제
                    if (node_manager_)
                    {
                        NodeBase* node = node_manager_->GetNode(node_id);
                        if (node)
                        {
                            if (auto* video_node = dynamic_cast<nodes::MovieFileInNode*>(node))
                            {
                                auto* engine = video_node->GetVideoEngine();
                                if (engine)
                                {
                                    timeline_.UnregisterPreviewVideoEngine(engine);
                                }
                            }
                        }
                        
                        // NodeManager에서 삭제
                        node_manager_->DeleteNode(node_id);
                    }
                }
            }
        }

        ImGui::End();

        // 선택된 노드 추적
        {
            const int num_selected = ImNodes::NumSelectedNodes();
            if (num_selected == 1)
            {
                static std::vector<int> selected_nodes;
                selected_nodes.resize(1);
                ImNodes::GetSelectedNodes(selected_nodes.data());
                int node_id = selected_nodes[0];

                if (graph_.node_exists(node_id))
                {
                    selected_node_id_ = node_id;
                    selected_node_type_ = graph_.node(node_id).type;
                    
                    // NodeManager에도 선택 상태 전달
                    if (node_manager_)
                    {
                        node_manager_->SetSelectedNode(node_id);
                    }
                }
                else
                {
                    selected_node_id_ = -1;
                    selected_node_type_ = NodeType::value;
                }
            }
            else if (num_selected == 0)
            {
                selected_node_id_ = -1;
                selected_node_type_ = NodeType::value;
                
                if (node_manager_)
                {
                    node_manager_->SetSelectedNode(-1);
                }
            }
        }

        // Inspector Panel (NodeManager 통합!)
        ImGui::Begin("Inspector");
        if (selected_node_id_ != -1)
        {
            // NodeManager의 인스펙터 시도
            if (node_manager_ && node_manager_->GetNode(selected_node_id_))
            {
                node_manager_->RenderInspectorForNode(selected_node_id_);
            }
            else
            {
                // 기존 인스펙터 
                RenderInspectorForSelectedNode();
            }
        }
        else
        {
            ImGui::TextDisabled("No node selected");
        }
        
        // NodeManager 통계
        if (node_manager_)
        {
            ImGui::Separator();
            ImGui::Text("Nodes: %zu", node_manager_->GetNodeCount());
            ImGui::Text("- PIX: %zu", node_manager_->GetNodeCountByFamily(NodeFamily::TOP));
            ImGui::Text("- SIG: %zu", node_manager_->GetNodeCountByFamily(NodeFamily::CHOP));
        }
        
        // TexturePool 메모리 통계 (메모리 누수 모니터링)
        if (texture_pool_)
        {
            ImGui::Separator();
            ImGui::Text("Texture Pool:");
            ImGui::Text("- Total: %zu", texture_pool_->total_textures());
            ImGui::Text("- Available: %zu", texture_pool_->available_textures());
            ImGui::Text("- In Use: %zu", texture_pool_->in_use_textures());
            
            size_t memory_mb = texture_pool_->memory_usage_bytes() / 1024 / 1024;
            if (memory_mb > 1000)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
                ImGui::Text("- Memory: %zu MB (HIGH!)", memory_mb);
                ImGui::PopStyleColor();
            }
            else if (memory_mb > 500)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
                ImGui::Text("- Memory: %zu MB", memory_mb);
                ImGui::PopStyleColor();
            }
            else
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.5f, 1.0f));
                ImGui::Text("- Memory: %zu MB", memory_mb);
                ImGui::PopStyleColor();
            }
        }
        
        // 프리뷰 스케일 정보 표시
        {
            ImGui::Separator();
            auto& settings = PreviewSettings::Instance();
            ImGui::Text("Preview Scale: %s", settings.GetScaleName());
            float factor = settings.GetScaleFactor();
            if (factor < 1.0f) {
                ImGui::SameLine();
                ImGui::TextDisabled("(Active)");
            }
        }
        
        ImGui::End();

        // Properties Panel
        ImGui::Begin("Properties");
        ImGui::TextUnformatted("Node Properties");
        ImGui::Separator();
        if (ImGui::Checkbox("Emulate 3-Button Mouse", &emulate_three_button_mouse))
        {
            ImNodes::GetIO().EmulateThreeButtonMouse.Modifier =
                emulate_three_button_mouse ? &ImGui::GetIO().KeyAlt : NULL;
        }
        ImGui::End();

        // Preview Settings Panel
        ImGui::Begin("Preview Settings");
        ImGui::TextUnformatted("Preview Quality");
        ImGui::Separator();
        
        auto& preview_settings = PreviewSettings::Instance();
        
        // 현재 설정 표시
        ImGui::Text("Current: %s", preview_settings.GetScaleName());
        
        // 메모리 절약량 표시
        int savings = preview_settings.GetMemorySavings();
        if (savings > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.5f, 1.0f));
            ImGui::Text("Memory Savings: %d%%", savings);
            ImGui::PopStyleColor();
        } else {
            ImGui::TextDisabled("Memory Savings: 0%%");
        }
        
        // 성능 향상 표시
        float scale_factor = preview_settings.GetScaleFactor();
        if (scale_factor < 1.0f) {
            float speed_boost = 1.0f / (scale_factor * scale_factor);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("Speed Boost: %.1fx faster", speed_boost);
            ImGui::PopStyleColor();
        }
        
        ImGui::Spacing();
        
        // 스케일 선택
        int current_scale = static_cast<int>(preview_settings.GetScale());
        const char* items[] = { 
            "1/1 (Full Quality)", 
            "1/2 (Balanced) [Recommended]", 
            "1/4 (Fast)" 
        };
        
        if (ImGui::Combo("Preview Scale", &current_scale, items, IM_ARRAYSIZE(items)))
        {
            preview_settings.SetScale(static_cast<PreviewSettings::Scale>(current_scale));
            
            // 스케일 변경 시 모든 노드 캐시 무효화
            if (node_manager_)
            {
                node_manager_->InvalidateAllCaches();
            }
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // 설명
        ImGui::TextWrapped("Preview scale affects texture resolution for all processing nodes.");
        ImGui::Spacing();
        
        switch (preview_settings.GetScale())
        {
            case PreviewSettings::Scale::Full:
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
                ImGui::BulletText("Maximum quality");
                ImGui::BulletText("High memory usage");
                ImGui::BulletText("Slower performance");
                ImGui::PopStyleColor();
                break;
                
            case PreviewSettings::Scale::Half:
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.5f, 1.0f));
                ImGui::BulletText("Good quality");
                ImGui::BulletText("75%% less memory");
                ImGui::BulletText("4x faster processing");
                ImGui::BulletText("Recommended for most cases");
                ImGui::PopStyleColor();
                break;
                
            case PreviewSettings::Scale::Quarter:
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
                ImGui::BulletText("Lower quality");
                ImGui::BulletText("94%% less memory");
                ImGui::BulletText("16x faster processing");
                ImGui::BulletText("Best for complex graphs");
                ImGui::PopStyleColor();
                break;
        }
        
        ImGui::End();

        // Controls Panel
        ImGui::Begin("Controls");
        ImGui::TextUnformatted("Keyboard Shortcuts:");
        ImGui::Separator();
        ImGui::BulletText("A - Add new node");
        ImGui::BulletText("X - Delete selected node or link");
        ImGui::BulletText("Ctrl + Click - Detach link");
        ImGui::BulletText("Space - Play/Pause Timeline");
        ImGui::End();

        // Output Preview Window
        ImGui::Begin("Output Preview");
        ImVec2 preview_size = ImGui::GetContentRegionAvail();
        id<MTLTexture> output_texture = dedicated_output_preview_texture;
        if (output_texture == nil && node_manager_)
        {
            output_texture = node_manager_->GetDedicatedOutputTexture();
        }

        if (output_texture != nil && preview_size.x > 0.0f && preview_size.y > 0.0f)
        {
            const float tex_width = static_cast<float>([output_texture width]);
            const float tex_height = static_cast<float>([output_texture height]);

            if (tex_width > 0.0f && tex_height > 0.0f)
            {
                const float scale_x = preview_size.x / tex_width;
                const float scale_y = preview_size.y / tex_height;
                const float scale = std::min(scale_x, scale_y);
                const ImVec2 image_size(tex_width * scale, tex_height * scale);

                ImVec2 cursor = ImGui::GetCursorPos();
                const float offset_x = std::max(0.0f, (preview_size.x - image_size.x) * 0.5f);
                const float offset_y = std::max(0.0f, (preview_size.y - image_size.y) * 0.5f);
                ImGui::SetCursorPos(ImVec2(cursor.x + offset_x, cursor.y + offset_y));

                ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture;
                ImTextureRef texture_ref(texture_id);
                ImGui::Image(texture_ref, image_size);

                ImGui::SetCursorPos(ImVec2(cursor.x + 8.0f, cursor.y + 8.0f));
                ImGui::TextDisabled("%.0fx%.0f", tex_width, tex_height);
            }
            else
            {
                ImGui::TextDisabled("Invalid output texture");
            }
        }
        else
        {
            ImGui::TextDisabled("No dedicated output texture");
            ImGui::TextDisabled("Add a `TOP/Output` node and connect a TOP stream.");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextUnformatted("Offline Encode");

        if (ImGui::InputFloat("Start Time", &export_start_time_, 0.1f, 1.0f, "%.3f s"))
        {
            export_start_time_ = std::max(0.0f, export_start_time_);
            if (export_end_time_ < export_start_time_)
            {
                export_end_time_ = export_start_time_;
            }
        }

        if (ImGui::InputFloat("End Time", &export_end_time_, 0.1f, 1.0f, "%.3f s"))
        {
            export_end_time_ = std::max(export_start_time_, export_end_time_);
        }

        ImGui::Checkbox("Auto End (Source Duration)", &export_auto_end_);
        if (export_auto_end_)
        {
            ImGui::TextDisabled("Use connected Movie duration as export end.");
        }

        if (ImGui::InputFloat("Encode FPS", &export_fps_, 1.0f, 5.0f, "%.2f"))
        {
            export_fps_ = std::max(1.0f, export_fps_);
        }

        if (ImGui::Button("Select Export File"))
        {
            NSSavePanel* save_panel = [NSSavePanel savePanel];
            [save_panel setCanCreateDirectories:YES];
            [save_panel setNameFieldStringValue:@"output.mov"];
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            [save_panel setAllowedFileTypes:@[@"mov", @"mp4"]];
            #pragma clang diagnostic pop

            if ([save_panel runModal] == NSModalResponseOK)
            {
                NSURL* selected_url = [save_panel URL];
                if (selected_url)
                {
                    export_path_ = std::string([[[selected_url path] stringByStandardizingPath] UTF8String]);
                }
            }
        }

        if (!export_path_.empty())
        {
            ImGui::TextWrapped("Export File: %s", export_path_.c_str());
        }
        else
        {
            ImGui::TextDisabled("No export file selected");
        }

        bool can_encode = !export_in_progress_ && !export_path_.empty();
        if (!can_encode)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Encode Output"))
        {
            StartOfflineEncodeDedicatedOutput();
        }
        if (!can_encode)
        {
            ImGui::EndDisabled();
        }

        ImGui::TextWrapped("Status: %s", export_status_.c_str());
        ImGui::End();

        // Performance Monitor Panel
        g_performance_monitor.render_ui();
        g_performance_monitor.end_frame();
    }

private:
    // MovieFileInNode Update 호출 및 Preview 모드 VideoEngine만 Timeline에 동기화 (엄격한 제한)
    void UpdateMovieFileInNodesAndSyncTimeline()
    {
        if (export_in_progress_ || !node_manager_) return;
        
        // 모든 노드를 순회하여 MovieFileInNode 찾기
        // Preview 모드 VideoEngine만 Timeline에 등록
        node_manager_->ForEachNode([this](int node_id, NodeBase* node) {
            if (!node) return;
            
            // MovieFileInNode만 처리
            if (auto* video_node = dynamic_cast<nodes::MovieFileInNode*>(node))
            {
                // Update 호출 (VideoEngine 프레임 업데이트)
                video_node->Update();
                
                auto* engine = video_node->GetVideoEngine();
                if (!engine) return;
                
                // Preview 모드로 명시적으로 설정 (기본값이지만 확실히)
                video_node->SetEngineMode(rdo::core::media::video::VideoEngine::EngineMode::Preview);
                
                // Preview 모드인 경우만 Timeline에 등록
                if (engine->IsPreviewMode())
                {
                    timeline_.RegisterPreviewVideoEngine(engine);
                }
            }
        });
    }

    bool IsDedicatedOutputNode(NodeBase* node) const
    {
        if (!node) return false;
        const std::string category = node->GetCategory();
        const std::string type = node->GetTypeName();
        return (category == "TOP/Output") || (type == "Output");
    }

    int FindDedicatedOutputNodeId() const
    {
        if (!node_manager_)
        {
            return -1;
        }

        const int selected_id = node_manager_->GetSelectedNodeId();
        if (selected_id != -1)
        {
            NodeBase* selected = node_manager_->GetNode(selected_id);
            if (IsDedicatedOutputNode(selected))
            {
                return selected_id;
            }
        }

        int latest_output_node_id = -1;
        node_manager_->ForEachNode([&](int node_id, NodeBase* node) {
            if (!IsDedicatedOutputNode(node))
            {
                return;
            }
            if (node_id > latest_output_node_id)
            {
                latest_output_node_id = node_id;
            }
        });
        return latest_output_node_id;
    }

    std::unordered_set<int> CollectUpstreamTopNodeIds(int output_node_id) const
    {
        std::unordered_set<int> upstream_top_nodes;
        if (!node_manager_ || output_node_id < 0)
        {
            return upstream_top_nodes;
        }

        std::unordered_map<int, int> port_to_top_node;
        node_manager_->ForEachNode([&](int node_id, NodeBase* node) {
            auto* top_node = dynamic_cast<TOPNodeBase*>(node);
            if (!top_node)
            {
                return;
            }

            for (const auto& port : top_node->GetInputPorts())
            {
                if (port.family == NodeFamily::TOP)
                {
                    port_to_top_node[port.id] = node_id;
                }
            }
            for (const auto& port : top_node->GetOutputPorts())
            {
                if (port.family == NodeFamily::TOP)
                {
                    port_to_top_node[port.id] = node_id;
                }
            }
        });

        std::vector<int> stack;
        stack.push_back(output_node_id);

        while (!stack.empty())
        {
            const int node_id = stack.back();
            stack.pop_back();

            if (!upstream_top_nodes.insert(node_id).second)
            {
                continue;
            }

            NodeBase* base_node = node_manager_->GetNode(node_id);
            auto* top_node = dynamic_cast<TOPNodeBase*>(base_node);
            if (!top_node)
            {
                continue;
            }

            for (const auto& input_port : top_node->GetInputPorts())
            {
                if (input_port.family != NodeFamily::TOP)
                {
                    continue;
                }

                for (const auto& edge : graph_.edges())
                {
                    if (edge.to != input_port.id)
                    {
                        continue;
                    }

                    auto source_it = port_to_top_node.find(edge.from);
                    if (source_it != port_to_top_node.end())
                    {
                        stack.push_back(source_it->second);
                    }
                }
            }
        }

        return upstream_top_nodes;
    }

    void FinishOfflineEncode(bool failed, const std::string& reason)
    {
        if (export_encoder_)
        {
            if (export_encoder_->IsRecording())
            {
                export_encoder_->Stop();
            }

            if (!failed && export_encoder_->HasError())
            {
                failed = true;
                export_status_ = "Encode failed: " + export_encoder_->GetErrorMessage();
            }
            export_encoder_.reset();
        }

        export_movie_states_.clear();
        export_in_progress_ = false;
        export_output_node_id_ = -1;
        export_frame_seek_issued_ = false;
        export_auto_end_pending_ = false;

        if (failed)
        {
            if (export_status_.rfind("Encode failed:", 0) != 0)
            {
                export_status_ = "Encode failed: " + reason;
            }
        }
        else
        {
            export_status_ = "Encode complete: " + std::to_string(export_encoded_frames_) +
                             " / " + std::to_string(export_total_frames_) + " frames.";
        }
    }

    void StartOfflineEncodeDedicatedOutput()
    {
        if (export_in_progress_)
        {
            return;
        }

        if (!node_manager_ || !gpu_batch_processor_ || !texture_pool_ || !device_)
        {
            export_status_ = "Encode failed: encoder context is not ready.";
            return;
        }

        if (export_path_.empty())
        {
            export_status_ = "Encode failed: output file is not selected.";
            return;
        }

        const int output_node_id = FindDedicatedOutputNodeId();
        if (output_node_id < 0)
        {
            export_status_ = "Encode failed: TOP/Output node is missing.";
            return;
        }

        const float fps = std::max(1.0f, export_fps_);
        const float start_time = std::max(0.0f, export_start_time_);
        const auto upstream_top_nodes = CollectUpstreamTopNodeIds(output_node_id);

        std::vector<ExportMovieState> movie_states;
        float min_source_duration = std::numeric_limits<float>::max();
        bool has_valid_source_duration = false;
        node_manager_->ForEachNode([&](int node_id, NodeBase* node) {
            auto* movie_node = dynamic_cast<nodes::MovieFileInNode*>(node);
            if (!movie_node)
            {
                return;
            }

            if (upstream_top_nodes.find(node_id) == upstream_top_nodes.end())
            {
                return;
            }

            const std::string file_path = movie_node->GetFilePath();
            if (file_path.empty())
            {
                return;
            }

            ExportMovieState state;
            state.node = movie_node;

            for (const auto& port : movie_node->GetOutputPorts())
            {
                if (port.family == NodeFamily::TOP)
                {
                    state.output_port_ids.push_back(port.id);
                }
            }
            if (state.output_port_ids.empty())
            {
                return;
            }

            state.engine = std::make_unique<rdo::core::media::video::VideoEngine>(device_);
            state.engine->SetMode(rdo::core::media::video::VideoEngine::EngineMode::Export);
            state.engine->LoadVideoAsync(file_path);
            movie_states.push_back(std::move(state));

            auto* preview_engine = movie_node->GetVideoEngine();
            if (preview_engine && preview_engine->IsLoaded())
            {
                const float duration = preview_engine->GetDuration();
                if (duration > 0.0f)
                {
                    min_source_duration = std::min(min_source_duration, duration);
                    has_valid_source_duration = true;
                }
            }
        });

        float end_time = std::max(start_time, export_end_time_);
        bool auto_end_pending = false;
        if (export_auto_end_ && has_valid_source_duration)
        {
            end_time = std::max(start_time, min_source_duration);
        }
        else if (export_auto_end_ && !movie_states.empty())
        {
            auto_end_pending = true;
        }

        const float duration = end_time - start_time;
        const int total_frames = std::max(1, static_cast<int>(std::floor(duration * fps + 0.5f)) + 1);
        static constexpr int kMaxExportFrames = 600000;
        if (total_frames > kMaxExportFrames)
        {
            export_status_ = "Encode failed: frame count exceeds safety limit.";
            return;
        }

        export_encoder_ = std::make_unique<rdo::core::media::video::VideoFileEncoder>(device_);
        export_movie_states_ = std::move(movie_states);
        export_output_node_id_ = output_node_id;
        export_job_start_time_ = start_time;
        export_job_end_time_ = end_time;
        export_job_fps_ = fps;
        export_job_frame_tolerance_ = std::max(0.05f, 1.5f / fps);
        export_total_frames_ = total_frames;
        export_current_frame_ = 0;
        export_encoded_frames_ = 0;
        export_frame_seek_issued_ = false;
        export_auto_end_pending_ = auto_end_pending;
        export_in_progress_ = true;
        export_status_ = auto_end_pending
            ? "Encoding... loading source durations."
            : "Encoding... 0 / " + std::to_string(export_total_frames_);
    }

    void ProcessOfflineEncodeStep()
    {
        if (!export_in_progress_)
        {
            return;
        }

        if (!node_manager_ || !gpu_batch_processor_ || !texture_pool_ || !export_encoder_)
        {
            FinishOfflineEncode(true, "encoder context became unavailable.");
            return;
        }

        if (export_output_node_id_ < 0)
        {
            FinishOfflineEncode(true, "dedicated output node is missing.");
            return;
        }

        if (export_current_frame_ >= export_total_frames_)
        {
            FinishOfflineEncode(false, "");
            return;
        }

        bool all_sources_loaded = true;
        for (auto& state : export_movie_states_)
        {
            if (!state.engine)
            {
                continue;
            }

            if (state.engine->HasError())
            {
                FinishOfflineEncode(true, state.engine->GetErrorMessage());
                return;
            }

            if (!state.engine->IsLoaded())
            {
                all_sources_loaded = false;
            }
        }

        if (!all_sources_loaded)
        {
            export_status_ = "Encoding... loading sources.";
            return;
        }

        if (export_auto_end_pending_)
        {
            float min_duration = std::numeric_limits<float>::max();
            bool has_valid_duration = false;
            for (const auto& state : export_movie_states_)
            {
                if (!state.engine) continue;
                const float duration = state.engine->GetDuration();
                if (duration > 0.0f)
                {
                    min_duration = std::min(min_duration, duration);
                    has_valid_duration = true;
                }
            }

            if (has_valid_duration)
            {
                export_job_end_time_ = std::max(export_job_start_time_, min_duration);
                const float auto_duration = export_job_end_time_ - export_job_start_time_;
                const int auto_total_frames =
                    std::max(1, static_cast<int>(std::floor(auto_duration * export_job_fps_ + 0.5f)) + 1);
                static constexpr int kMaxExportFrames = 600000;
                if (auto_total_frames > kMaxExportFrames)
                {
                    FinishOfflineEncode(true, "frame count exceeds safety limit.");
                    return;
                }
                export_total_frames_ = auto_total_frames;
            }
            export_auto_end_pending_ = false;
        }

        const float t = export_job_start_time_ + (static_cast<float>(export_current_frame_) / export_job_fps_);
        UpdateTime(t);

        if (!export_frame_seek_issued_)
        {
            for (auto& state : export_movie_states_)
            {
                if (state.engine)
                {
                    state.engine->SeekToTime(t);
                }
            }
            export_frame_seek_issued_ = true;
        }

        bool all_ready = true;
        for (auto& state : export_movie_states_)
        {
            if (!state.engine) continue;
            state.engine->Update();
            id<MTLTexture> texture = state.engine->GetCurrentFrameTexture();
            const float engine_time = state.engine->GetCurrentTime();
            if (!texture || std::fabs(engine_time - t) > export_job_frame_tolerance_)
            {
                all_ready = false;
                break;
            }
        }

        if (!all_ready)
        {
            export_status_ = "Encoding... preparing frame " +
                             std::to_string(export_current_frame_ + 1) + " / " +
                             std::to_string(export_total_frames_);
            return;
        }

        texture_registry_.clear();
        for (const auto& state : export_movie_states_)
        {
            if (!state.engine) continue;
            id<MTLTexture> texture = state.engine->GetCurrentFrameTexture();
            if (!texture) continue;

            for (int output_port_id : state.output_port_ids)
            {
                texture_registry_[output_port_id] = texture;
            }
        }

        RenderContext export_ctx;
        export_ctx.mode = RenderMode::Final;
        export_ctx.allow_shallow_copy = true;
        export_ctx.allow_texture_views = true;
        export_ctx.skip_movie_file_in_nodes = true;
        export_ctx.forced_output_node_id = export_output_node_id_;

        node_manager_->EnqueueGPUProcessing(*gpu_batch_processor_, *texture_pool_, texture_registry_, export_ctx);
        gpu_batch_processor_->flush();

        id<MTLTexture> output_texture = node_manager_->GetDedicatedOutputTexture();
        if (!output_texture)
        {
            FinishOfflineEncode(true, "dedicated output texture is missing. Connect TOP stream to TOP/Output.");
            return;
        }

        if (!export_encoder_->IsRecording())
        {
            const int width = static_cast<int>([output_texture width]);
            const int height = static_cast<int>([output_texture height]);
            if (!export_encoder_->Start(export_path_, width, height, export_job_fps_))
            {
                FinishOfflineEncode(true, export_encoder_->GetErrorMessage());
                return;
            }
        }

        export_encoder_->EncodeFrame(output_texture);
        if (export_encoder_->HasError())
        {
            FinishOfflineEncode(true, export_encoder_->GetErrorMessage());
            return;
        }

        export_current_frame_++;
        export_encoded_frames_++;
        export_frame_seek_issued_ = false;
        export_status_ = "Encoding... " + std::to_string(export_current_frame_) + " / " + std::to_string(export_total_frames_);

        if (export_current_frame_ >= export_total_frames_)
        {
            FinishOfflineEncode(false, "");
        }
    }
    
    // Helper functions (기존 유지)
    bool RenderParameterSlider(const char* label, float* value, float min, float max, const char* format, int attribute_id)
    {
        bool is_connected = (attribute_id != -1) && (graph_.num_edges_from_node(attribute_id) > 0);

        if (is_connected)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.8f, 0.9f, 1.0f));
            ImGui::Text("● %s", label);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextDisabled("(Connected)");

            if (attribute_id != -1)
            {
                *value = graph_.node(attribute_id).value;
            }
        }

        if (is_connected)
        {
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.3f, 0.35f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.0f, 0.8f, 0.9f, 0.7f));
        }

        bool changed = ImGui::SliderFloat(is_connected ? ("##" + std::string(label)).c_str() : label,
                                          value, min, max, format);

        if (is_connected)
        {
            ImGui::PopStyleColor(2);
        }

        return changed;
    }

    bool RenderParameterInput(const char* label, int* value, int attribute_id)
    {
        bool is_connected = (attribute_id != -1) && (graph_.num_edges_from_node(attribute_id) > 0);

        if (is_connected)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.8f, 0.9f, 1.0f));
            ImGui::Text("● %s", label);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextDisabled("(Connected)");

            if (attribute_id != -1)
            {
                *value = static_cast<int>(graph_.node(attribute_id).value);
            }
        }

        if (is_connected)
        {
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.3f, 0.35f, 0.5f));
        }

        bool changed = ImGui::InputInt(is_connected ? ("##" + std::string(label)).c_str() : label, value);

        if (is_connected)
        {
            ImGui::PopStyleColor();
        }

        return changed;
    }

    // Inspector 렌더링
    void RenderInspectorForSelectedNode()
    {
        // NodeManager가 Inspector 렌더링 처리
        if (node_manager_)
        {
            node_manager_->RenderInspectorForNode(selected_node_id_);
        }
        else
        {
            ImGui::TextDisabled("No node selected");
        }
    }

    Graph<Node>            graph_;
    
    // NodeManager (핵심!)
    std::unique_ptr<NodeManager> node_manager_;
    
    int                    root_node_id_;
    ImNodesMiniMapLocation minimap_location_;
    id<MTLDevice>          device_;
    double                 last_time_;
    bool                   first_time_;

    // Texture registry
    std::unordered_map<int, id<MTLTexture>> texture_registry_;

    // GPU 배치 처리 시스템
    std::unique_ptr<GPUBatchProcessor> gpu_batch_processor_;

    // 텍스처 풀 시스템
    std::unique_ptr<TexturePool> texture_pool_;

    // 선택된 노드 추적
    int selected_node_id_;
    NodeType selected_node_type_;

    // Timeline (모듈화된 Timeline 클래스)
    Timeline timeline_;

    struct ExportMovieState
    {
        nodes::MovieFileInNode* node = nullptr;
        std::unique_ptr<rdo::core::media::video::VideoEngine> engine;
        std::vector<int> output_port_ids;
    };

    // 독립 오프라인 인코딩 설정 (타임라인과 분리)
    std::string export_path_;
    float export_fps_;
    float export_start_time_;
    float export_end_time_;
    bool export_auto_end_;
    bool export_in_progress_;
    std::string export_status_;
    std::unique_ptr<rdo::core::media::video::VideoFileEncoder> export_encoder_;
    std::vector<ExportMovieState> export_movie_states_;
    float export_job_start_time_ = 0.0f;
    float export_job_end_time_ = 0.0f;
    float export_job_fps_ = 30.0f;
    float export_job_frame_tolerance_ = 0.05f;
    int export_total_frames_ = 0;
    int export_current_frame_ = 0;
    int export_encoded_frames_ = 0;
    int export_output_node_id_ = -1;
    bool export_frame_seek_issued_ = false;
    bool export_auto_end_pending_ = false;
};

static ColorNodeEditor color_editor;
} // namespace

void NodeEditorInitialize()
{
    ImNodesIO& io = ImNodes::GetIO();
    io.LinkDetachWithModifierClick.Modifier = &ImGui::GetIO().KeyCtrl;
}

void NodeEditorSetDevice(id<MTLDevice> device)
{
    color_editor.SetDevice(device);
}

void NodeEditorShow() { color_editor.show(); }

void NodeEditorShutdown() {}
} // namespace example
