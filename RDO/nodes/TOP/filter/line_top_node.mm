#include "line_top_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../core/node_system/node_manager.h"
#include "../../CHOP/analysis/blob_track_info_node.h"
#include "../../CHOP/filter/trail_chop_node.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>
#include <mutex>
#include <map>
#include <algorithm>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Metal 셰이더 코드
static const char* lineTOPShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// 포인트 구조체 (16바이트 정렬)
struct Point {
    float2 pos;      // 8 bytes
    float alpha;     // 4 bytes
    float padding;   // 4 bytes
};

// Catmull-Rom spline 보간 함수
float2 catmullRomSpline(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    );
}

kernel void drawLinesWithStyle(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    device Point* points [[buffer(0)]],
    constant int &pointCount [[buffer(1)]],
    constant float &lineThickness [[buffer(2)]],
    constant float4 &lineColor [[buffer(3)]],
    constant float &fadeOut [[buffer(4)]],
    constant bool &useSmooth [[buffer(5)]],
    constant int &smoothSegments [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    
    float4 color = float4(0.0, 0.0, 0.0, 0.0);  // 투명 배경
    float px = float(gid.x);
    float py = float(gid.y);
    float lineThicknessSq = lineThickness * lineThickness;
    
    if (useSmooth && pointCount >= 4) {
        // Catmull-Rom spline으로 곡선 그리기
        int segments = smoothSegments;  // UI에서 조절 가능한 세그먼트 수
        
        for (int i = 0; i < pointCount - 1; i++) {
            // 4개 점 가져오기 (경계 처리)
            int i0 = max(0, i - 1);
            int i1 = i;
            int i2 = min(pointCount - 1, i + 1);
            int i3 = min(pointCount - 1, i + 2);
            
            Point p0 = points[i0];
            Point p1 = points[i1];
            Point p2 = points[i2];
            Point p3 = points[i3];
            
            // 각 세그먼트를 곡선으로 그리기
            for (int seg = 0; seg < segments; seg++) {
                float t1 = float(seg) / float(segments);
                float t2 = float(seg + 1) / float(segments);
                
                float2 pos1 = catmullRomSpline(p0.pos, p1.pos, p2.pos, p3.pos, t1);
                float2 pos2 = catmullRomSpline(p0.pos, p1.pos, p2.pos, p3.pos, t2);
                
                // AABB 체크
                float minX = min(pos1.x, pos2.x) - lineThickness;
                float maxX = max(pos1.x, pos2.x) + lineThickness;
                float minY = min(pos1.y, pos2.y) - lineThickness;
                float maxY = max(pos1.y, pos2.y) + lineThickness;
                
                if (px < minX || px > maxX || py < minY || py > maxY) continue;
                
                // 선분과 점의 거리 계산
                float2 line = pos2 - pos1;
                float lineLengthSq = dot(line, line);
                if (lineLengthSq < 0.001) continue;
                
                float2 toPoint = float2(px, py) - pos1;
                float t = dot(toPoint, line) / lineLengthSq;
                
                if (t < 0.0 || t > 1.0) continue;
                
                float2 closestPoint = pos1 + line * t;
                float2 diff = float2(px, py) - closestPoint;
                float distToLineSq = dot(diff, diff);
                
                if (distToLineSq < lineThicknessSq) {
                    // Alpha 보간 + fade out
                    float alpha1 = mix(p1.alpha, p2.alpha, t1);
                    float alpha2 = mix(p1.alpha, p2.alpha, t2);
                    float base_alpha = mix(alpha1, alpha2, t);
                    float fade_alpha = base_alpha * (1.0 - fadeOut * (1.0 - base_alpha));
                    float final_alpha = fade_alpha * lineColor.a;
                    float4 lineColorWithAlpha = float4(lineColor.rgb, final_alpha);
                    color = mix(color, lineColorWithAlpha, final_alpha);
                }
            }
        }
    } else {
        // 기존 방식: 직선으로 연결
        for (int i = 0; i < pointCount - 1; i++) {
            Point p1 = points[i];
            Point p2 = points[i + 1];
            
            float2 pos1 = p1.pos;
            float2 pos2 = p2.pos;
            
            // AABB 체크
            float minX = min(pos1.x, pos2.x) - lineThickness;
            float maxX = max(pos1.x, pos2.x) + lineThickness;
            float minY = min(pos1.y, pos2.y) - lineThickness;
            float maxY = max(pos1.y, pos2.y) + lineThickness;
            
            if (px < minX || px > maxX || py < minY || py > maxY) continue;
            
            // 선분과 점의 거리 계산
            float2 line = pos2 - pos1;
            float lineLengthSq = dot(line, line);
            if (lineLengthSq < 0.001) continue;
            
            float2 toPoint = float2(px, py) - pos1;
            float t = dot(toPoint, line) / lineLengthSq;
            
            if (t < 0.0 || t > 1.0) continue;
            
            float2 closestPoint = pos1 + line * t;
            float2 diff = float2(px, py) - closestPoint;
            float distToLineSq = dot(diff, diff);
            
            if (distToLineSq < lineThicknessSq) {
                // Alpha blending (두 점의 alpha 평균 + fade out)
                float base_alpha = (p1.alpha + p2.alpha) * 0.5;
                float fade_alpha = base_alpha * (1.0 - fadeOut * (1.0 - base_alpha));
                float final_alpha = fade_alpha * lineColor.a;
                float4 lineColorWithAlpha = float4(lineColor.rgb, final_alpha);
                color = mix(color, lineColorWithAlpha, final_alpha);
            }
        }
    }
    
    outputTexture.write(color, gid);
}
)";

LineTOPNode::LineTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , pipeline_(nil)
    , connected_chop_input_(nullptr)
    , last_chop_input_port_id_(-1)
    , graph_ref_(&graph)
    , is_trail_format_cached_(false)
    , format_cache_valid_(false)
    , line_thickness_(1.0f)
    , fade_out_(0.5f)
    , smooth_enabled_(false)  // 기본값: 비활성화
    , smooth_segments_(8)     // 기본값: 8 (성능과 품질의 균형)
    , width_(1920)
    , height_(1080)
{
    line_color_[0] = 1.0f;  // R
    line_color_[1] = 1.0f;  // G
    line_color_[2] = 1.0f;  // B
    line_color_[3] = 1.0f;  // A
    
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // TOP 입력 포트 (참조 텍스처)
    int ref_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(ref_input_id, NodeFamily::TOP, PortDirection::Input, "texture", "reference"));
    
    // CHOP 입력 포트 (Shape처럼 직접 CHOP 입력도 받을 수 있도록)
    int chop_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(chop_input_id, NodeFamily::CHOP, PortDirection::Input, "channels", "input"));
    
    // TOP 출력 포트
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

LineTOPNode::~LineTOPNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void LineTOPNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool LineTOPNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:lineTOPShaderSource]
                                                    options:nil
                                                      error:&error];
    if (!library || error) {
        NSLog(@"Error creating library: %@", error);
        return false;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"drawLinesWithStyle"];
    if (!function) {
        NSLog(@"Error finding function applyLineStyle");
        return false;
    }
    
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        NSLog(@"Error creating pipeline: %@", error);
        return false;
    }
    
    return true;
}

void LineTOPNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Line");
    ImNodes::EndNodeTitleBar();
    
    // TOP 입력 포트 (참조 텍스처)
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    // CHOP 입력 포트 (직접 CHOP 입력)
    if (input_ports_.size() > 1) {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 연결 상태 표시
    if (connected_chop_input_) {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Connected");
        ImGui::TextDisabled("Input: %s", connected_chop_input_->GetTypeName().c_str());
    } else {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Not Connected");
    }
    
    // Output
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize(port.name.c_str()).x;
        ImGui::Indent(node_width - label_width);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void LineTOPNode::RenderInspector()
{
    ImGui::Text("Line TOP");
    ImGui::Separator();
    
    // Common 파라미터 (터치디자이너 호환)
    if (ImGui::CollapsingHeader("Common", ImGuiTreeNodeFlags_DefaultOpen)) {
        int w = width_;
        int h = height_;
        if (ImGui::InputInt("Width", &w)) {
            SetWidth(w);
        }
        if (ImGui::InputInt("Height", &h)) {
            SetHeight(h);
        }
        ImGui::TextDisabled("Resolution (used when no input texture)");
    }
    
    ImGui::Spacing();
    
    ImGui::Text("Line Settings");
    ImGui::SliderFloat("Thickness", &line_thickness_, 0.5f, 10.0f, "%.1f");
    ImGui::SliderFloat("Fade Out", &fade_out_, 0.0f, 1.0f, "%.2f");
    ImGui::ColorEdit4("Line Color", line_color_);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Smooth Trail");
    
    if (ImGui::Checkbox("Enable Smooth", &smooth_enabled_)) {
        SetSmoothEnabled(smooth_enabled_);
    }
    if (smooth_enabled_) {
        int seg = smooth_segments_;
        if (ImGui::SliderInt("Smooth Segments", &seg, 2, 50)) {
            SetSmoothSegments(seg);
        }
        ImGui::TextDisabled("Higher = smoother curve, Lower = better performance");
        ImGui::TextDisabled("Uses Catmull-Rom spline interpolation");
    }
    
    ImGui::Spacing();
    ImGui::TextDisabled("(터치디자이너 Line TOP 호환)");
}

void LineTOPNode::UpdateConnectedCHOPNode(NodeManager* node_manager)
{
    if (!node_manager || !graph_ref_ || input_ports_.size() < 2) {
        if (connected_chop_input_ != nullptr) {
            connected_chop_input_ = nullptr;
            last_chop_input_port_id_ = -1;
            format_cache_valid_ = false;
        }
        return;
    }
    
    // CHOP 입력 포트는 두 번째 포트
    int input_port_id = input_ports_[1].id;
    
    // 연결이 변경되지 않았으면 스킵 (성능 최적화)
    if (input_port_id == last_chop_input_port_id_ && connected_chop_input_ != nullptr) {
        // 연결은 유지되지만 데이터는 업데이트
        static float eval_time = 0.0f;
        eval_time += 0.016f;
        connected_chop_input_->Evaluate({}, eval_time);
        
        const auto& output_channels = connected_chop_input_->GetOutputChannels();
        
        // 채널 이름은 노드 타입별로 처리
        std::vector<std::string> channel_names;
        if (auto* info_node = dynamic_cast<BlobTrackInfoNode*>(connected_chop_input_)) {
            channel_names = info_node->GetChannelNames();
        } else if (auto* trail_node = dynamic_cast<TrailCHOPNode*>(connected_chop_input_)) {
            channel_names = trail_node->GetChannelNames();
        } else {
            // 기본 채널 이름 생성
            channel_names.reserve(output_channels.size());
            for (size_t i = 0; i < output_channels.size(); i++) {
                channel_names.push_back("ch" + std::to_string(i));
            }
        }
        
        // 데이터 저장 및 캐시 무효화
        std::lock_guard<std::mutex> lock(channels_mutex_);
        bool size_changed = (input_channels_.size() != output_channels.size() || 
                            input_channel_names_.size() != channel_names.size());
        input_channels_ = output_channels;
        input_channel_names_ = channel_names;
        
        if (size_changed) {
            format_cache_valid_ = false;
        }
        
        return;
    }
    
    // 연결 변경 감지 - 새로 찾기
    last_chop_input_port_id_ = input_port_id;
    connected_chop_input_ = nullptr;
    format_cache_valid_ = false;
    
    // Graph에서 입력 포트에 연결된 엣지 찾기
    for (const auto& edge : graph_ref_->edges()) {
        if (edge.to == input_port_id) {
            int from_port_id = edge.from;
            
            NodeBase* connected_node = node_manager->GetNodeByPortId(from_port_id);
            if (!connected_node) {
                continue;
            }
            
            // AudioCHOPNodeBase로 캐스팅 (BlobTrackInfoNode, TrailCHOPNode 등 모두 포함)
            AudioCHOPNodeBase* chop_node = dynamic_cast<AudioCHOPNodeBase*>(connected_node);
            if (chop_node) {
                connected_chop_input_ = chop_node;
                
                // Evaluate 호출 (시간은 node_manager에서 관리)
                static float eval_time = 0.0f;
                eval_time += 0.016f;
                chop_node->Evaluate({}, eval_time);
                
                // 출력 채널 가져오기
                const auto& output_channels = chop_node->GetOutputChannels();
                
                // 채널 이름은 노드 타입별로 처리
                std::vector<std::string> channel_names;
                if (auto* info_node = dynamic_cast<BlobTrackInfoNode*>(chop_node)) {
                    channel_names = info_node->GetChannelNames();
                } else if (auto* trail_node = dynamic_cast<TrailCHOPNode*>(chop_node)) {
                    channel_names = trail_node->GetChannelNames();
                } else {
                    // 기본 채널 이름 생성
                    channel_names.reserve(output_channels.size());
                    for (size_t i = 0; i < output_channels.size(); i++) {
                        channel_names.push_back("ch" + std::to_string(i));
                    }
                }
                
                // 데이터 저장
                std::lock_guard<std::mutex> lock(channels_mutex_);
                input_channels_ = output_channels;
                input_channel_names_ = channel_names;
                
                return;
            }
        }
    }
}

void LineTOPNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    SetLastTexturePool(texture_pool);
    
    if (!texture_pool || !pipeline_) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    // 해상도 결정: 입력 텍스처가 있으면 그것을 사용, 없으면 파라미터 값 사용 (터치디자이너 호환)
    int width = width_;
    int height = height_;
    if (!inputs.empty() && inputs[0] != nil) {
        width = static_cast<int>([inputs[0] width]);
        height = static_cast<int>([inputs[0] height]);
    }
    
    // 출력 텍스처 할당
    if (!output_texture_ || [output_texture_ width] != width || [output_texture_ height] != height) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        output_texture_ = texture_pool->acquire_texture(width, height, MTLPixelFormatRGBA8Unorm);
    }
    
    if (!output_texture_) return;
    
    // CHOP 입력에서 포인트 데이터 생성
    std::vector<float> points_data;
    int point_count = 0;
    
    {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        
        if (input_channels_.empty() || input_channel_names_.empty()) {
            // 데이터 없음
        } else {
            // Trail 형식 확인 (캐싱)
            if (!format_cache_valid_) {
                is_trail_format_cached_ = false;
                // 첫 번째 채널 이름만 확인 (빠른 체크)
                if (!input_channel_names_.empty() && 
                    input_channel_names_[0].find("_t") != std::string::npos) {
                    is_trail_format_cached_ = true;
                }
                format_cache_valid_ = true;
            }
            
            if (is_trail_format_cached_) {
                // TrailCHOPNode 형식: ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...
                // 최적화: 인덱스 기반 접근으로 문자열 파싱 최소화
                
                // 채널 구조 파악: ch0_t0부터 시작하므로 인덱스로 접근 가능
                // 각 채널은 [ch_index]_[time_index] 형식
                // TrailCHOPNode는 채널별로 시간 순서대로 배치: ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...
                
                // 채널 개수와 시간 개수 파악
                size_t num_channels = 0;
                size_t num_times = 0;
                
                // 첫 번째 채널에서 시간 개수 파악
                if (!input_channel_names_.empty()) {
                    size_t first_t_pos = input_channel_names_[0].find("_t");
                    if (first_t_pos != std::string::npos) {
                        // ch0_t0 형식에서 시간 인덱스 추출
                        try {
                            int first_time = std::stoi(input_channel_names_[0].substr(first_t_pos + 2));
                            // 시간 인덱스는 0부터 시작하므로, 연속된 시간 인덱스 개수 확인
                            for (size_t i = 0; i < input_channel_names_.size(); i++) {
                                size_t t_pos = input_channel_names_[i].find("_t");
                                if (t_pos != std::string::npos) {
                                    int time_idx = std::stoi(input_channel_names_[i].substr(t_pos + 2));
                                    num_times = std::max(num_times, static_cast<size_t>(time_idx + 1));
                                }
                            }
                            
                            // 채널 개수 계산: 전체 채널 수 / 시간 개수
                            if (num_times > 0) {
                                num_channels = input_channel_names_.size() / num_times;
                            }
                        } catch (...) {
                            // 파싱 실패 시 기존 방식 사용
                            num_channels = 0;
                            num_times = 0;
                        }
                    }
                }
                
                if (num_channels > 0 && num_times > 0) {
                    // 인덱스 기반 접근으로 빠르게 처리
                    // BlobTrackInfoNode 형식: num_blobs(0), u1(1), v1(2), width1(3), height1(4), area1(5), id1(6), vx1(7), vy1(8), u2(9), v2(10), ...
                    // Trail을 거치면: ch0_t0, ch0_t1, ... (num_blobs), ch1_t0, ch1_t1, ... (u1), ch2_t0, ch2_t1, ... (v1), ...
                    
                    // num_blobs 채널 (ch0)에서 각 시간의 blob 개수 확인
                    std::vector<int> time_num_blobs(num_times, 0);
                    for (size_t t = 0; t < num_times; t++) {
                        size_t idx = t;  // ch0_t[t]
                        if (idx < input_channels_.size()) {
                            time_num_blobs[t] = static_cast<int>(input_channels_[idx]);
                        }
                    }
                    
                    // 각 blob의 u, v 추출
                    // blob N: ch(1+8*N) (u), ch(2+8*N) (v)
                    int max_blob = 0;
                    for (size_t t = 0; t < num_times; t++) {
                        max_blob = std::max(max_blob, time_num_blobs[t]);
                    }
                    
                    for (int blob_idx = 0; blob_idx < max_blob; blob_idx++) {
                        int u_ch_base = 1 + 8 * blob_idx;  // u 채널 시작 인덱스
                        int v_ch_base = 2 + 8 * blob_idx;  // v 채널 시작 인덱스
                        
                        if (u_ch_base * num_times >= input_channels_.size() ||
                            v_ch_base * num_times >= input_channels_.size()) {
                            continue;
                        }
                        
                        // 이 blob의 trail 포인트 수집
                        std::vector<std::pair<float, float>> blob_points;
                        for (size_t t = 0; t < num_times; t++) {
                            // 해당 시간에 blob이 존재하는지 확인
                            if (blob_idx >= time_num_blobs[t]) {
                                continue;
                            }
                            
                            size_t u_idx = u_ch_base * num_times + t;
                            size_t v_idx = v_ch_base * num_times + t;
                            
                            if (u_idx < input_channels_.size() && v_idx < input_channels_.size()) {
                                float u = input_channels_[u_idx];
                                float v = input_channels_[v_idx];
                                
                                if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                                    blob_points.push_back({u, v});
                                }
                            }
                        }
                        
                        // Trail 포인트 생성 (fade out 적용)
                        for (size_t i = 0; i < blob_points.size(); i++) {
                            float u = blob_points[i].first * width;
                            float v = blob_points[i].second * height;
                            
                            // 유효 범위 확인
                            if (u < 0 || u > width || v < 0 || v > height) continue;
                            
                            // Fade out 계산
                            float fade_t = blob_points.size() > 1 
                                ? static_cast<float>(i) / static_cast<float>(blob_points.size() - 1)
                                : 0.0f;
                            float alpha = 1.0f - fade_t * fade_out_;
                            
                            points_data.push_back(u);
                            points_data.push_back(v);
                            points_data.push_back(alpha);
                            points_data.push_back(0.0f);  // padding
                            point_count++;
                        }
                    }
                } else {
                    // 파싱 실패 시 간단한 fallback: 직접 연결 형식으로 처리
                    if (input_channels_.size() > 0) {
                        int num_blobs = static_cast<int>(input_channels_[0]);
                        for (int b = 0; b < num_blobs; b++) {
                            int channel_offset = 1 + b * 8;
                            if (channel_offset + 1 < static_cast<int>(input_channels_.size())) {
                                float u = input_channels_[channel_offset + 0];
                                float v = input_channels_[channel_offset + 1];
                                if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                                    points_data.push_back(u * width);
                                    points_data.push_back(v * height);
                                    points_data.push_back(1.0f);
                                    points_data.push_back(0.0f);
                                    point_count++;
                                }
                            }
                        }
                    }
                }
            } else {
                // BlobTrackInfoNode 직접 연결 형식: num_blobs, u1, v1, width1, height1, area1, id1, vx1, vy1, u2, v2, ...
                if (input_channels_.size() > 0) {
                    int num_blobs = static_cast<int>(input_channels_[0]);
                    
                    points_data.reserve(num_blobs * 4);  // 사전 할당
                    for (int b = 0; b < num_blobs; b++) {
                        int channel_offset = 1 + b * 8;
                        
                        if (channel_offset + 1 < static_cast<int>(input_channels_.size())) {
                            float u = input_channels_[channel_offset + 0];
                            float v = input_channels_[channel_offset + 1];
                            
                            if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                                points_data.push_back(u * width);
                                points_data.push_back(v * height);
                                points_data.push_back(1.0f);
                                points_data.push_back(0.0f);
                                point_count++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 포인트 데이터가 없으면 입력 텍스처 복사
    if (points_data.empty() || point_count == 0) {
        if (inputs.size() > 0 && inputs[0] != nil) {
            id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
            [blitEncoder copyFromTexture:inputs[0] toTexture:output_texture_];
            [blitEncoder endEncoding];
        }
        return;
    }
    
    // 포인트 데이터를 Point 구조체로 변환
    struct Point {
        float x, y;
        float alpha;
        float padding;
    };
    
    std::vector<Point> points;
    for (size_t i = 0; i + 3 < points_data.size(); i += 4) {
        points.push_back({
            points_data[i],
            points_data[i + 1],
            points_data[i + 2],
            points_data[i + 3]
        });
    }
    
    if (points.empty()) {
        if (inputs.size() > 0 && inputs[0] != nil) {
            id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
            [blitEncoder copyFromTexture:inputs[0] toTexture:output_texture_];
            [blitEncoder endEncoding];
        }
        return;
    }
    
    // GPU 버퍼에 포인트 데이터 전송
    id<MTLBuffer> points_buffer = [device_ newBufferWithBytes:points.data()
                                                       length:points.size() * sizeof(Point)
                                                      options:MTLResourceStorageModeShared];
    
    // 검은색으로 클리어
    if (inputs.size() > 0 && inputs[0] != nil) {
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:inputs[0] toTexture:output_texture_];
        [blitEncoder endEncoding];
    }
    
    // Line 스타일로 선 그리기
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:output_texture_ atIndex:0];
    [encoder setBuffer:points_buffer offset:0 atIndex:0];
    
    int point_count_int = static_cast<int>(points.size());
    [encoder setBytes:&point_count_int length:sizeof(int) atIndex:1];
    [encoder setBytes:&line_thickness_ length:sizeof(float) atIndex:2];
    
    struct { float r, g, b, a; } line_color = {
        line_color_[0],
        line_color_[1],
        line_color_[2],
        line_color_[3]
    };
    [encoder setBytes:&line_color length:sizeof(line_color) atIndex:3];
    [encoder setBytes:&fade_out_ length:sizeof(float) atIndex:4];
    
    // Line 노드 자체의 smooth 옵션 사용 (우선순위)
    // TrailCHOPNode의 smooth 옵션은 무시하고 Line 노드의 설정을 사용
    bool use_smooth = smooth_enabled_;
    int smooth_segments = smooth_segments_;
    
    [encoder setBytes:&use_smooth length:sizeof(bool) atIndex:5];
    [encoder setBytes:&smooth_segments length:sizeof(int) atIndex:6];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void LineTOPNode::GetLineColor(float color[4]) const
{
    color[0] = line_color_[0];
    color[1] = line_color_[1];
    color[2] = line_color_[2];
    color[3] = line_color_[3];
}

void LineTOPNode::SetLineColor(const float color[4])
{
    line_color_[0] = color[0];
    line_color_[1] = color[1];
    line_color_[2] = color[2];
    line_color_[3] = color[3];
}

std::unique_ptr<NodeBase> CreateLineTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<LineTOPNode>(graph, pos, device);
}

REGISTER_NODE(Line, "Line", "TOP/Composite", NodeFamily::TOP, CreateLineTOPNode, "Apply line style settings (TouchDesigner Line MAT compatible)");

} // namespace nodes
} // namespace example

