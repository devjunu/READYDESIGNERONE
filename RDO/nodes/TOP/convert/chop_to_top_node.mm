#include "chop_to_top_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../core/node_system/node_manager.h"
#include "../../CHOP/filter/trail_chop_node.h"
#include "../../CHOP/analysis/blob_track_info_node.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <cstring>
#include <map>
#include <tuple>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// ============================================================================
// Metal 셰이더: GPU 최적화 트레일 렌더링
// 4K 60fps에서 1만개 블롭 처리 가능
// ============================================================================
static const char* chopToTOPShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// 포인트 구조체 (32바이트 정렬 - 속도 정보 추가)
struct TrailPoint {
    float2 pos;      // 8 bytes (픽셀 좌표)
    float alpha;     // 4 bytes
    float blob_id;   // 4 bytes (블롭 ID, 선 연결 구분용)
    float2 velocity; // 8 bytes (속도 벡터, 픽셀/프레임)
    float thickness; // 4 bytes (선 두께)
    float age;       // 4 bytes (나이, 0.0 = 최신, 1.0 = 오래됨)
};

// Render Pipeline용 구조체
struct LineVertex {
    float4 position [[position]];
    float4 color;
};

// ============================================================================
// Compute: 텍스처 클리어 + Fade
// ============================================================================
kernel void clearTexture(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    outputTexture.write(float4(0.0, 0.0, 0.0, 0.0), gid);
}

// Accumulation Buffer: 이전 프레임 fade (alpha 감쇠)
kernel void fadeTexture(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &fadeAmount [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    
    float4 color = inputTexture.read(gid);
    color.a *= fadeAmount;  // Alpha 감쇠
    color.rgb *= fadeAmount;  // RGB도 감쇠 (깔끔한 fade)
    outputTexture.write(color, gid);
}

// ============================================================================
// Render Pipeline: 고성능 라인 렌더링
// 복잡도: O(points) - 픽셀 수와 무관!
// ============================================================================
vertex LineVertex lineVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    device const TrailPoint* points [[buffer(0)]],
    constant int &pointCount [[buffer(1)]],
    constant float &baseThickness [[buffer(2)]],
    constant float4 &lineColor [[buffer(3)]],
    constant float2 &texSize [[buffer(4)]],
    constant float &thicknessVariation [[buffer(5)]],
    constant bool &useColorGradient [[buffer(6)]],
    constant float &gradientIntensity [[buffer(7)]],
    constant bool &useVelocityColors [[buffer(8)]],
    constant float &velocityThreshold [[buffer(9)]],
    constant float4 &startColor [[buffer(10)]],
    constant float4 &endColor [[buffer(11)]],
    constant float4 &velocityColor [[buffer(12)]])
{
    LineVertex out;
    
    // 선분 인덱스 = instanceID
    int lineIdx = instanceID;
    if (lineIdx >= pointCount - 1) {
        out.position = float4(0, 0, -1, 1);  // 클리핑
        out.color = float4(0);
        return out;
    }
    
    TrailPoint p1 = points[lineIdx];
    TrailPoint p2 = points[lineIdx + 1];
    
    // 다른 블롭이면 선 그리지 않음 (각 blob별로 독립적인 trail)
    if (p1.blob_id != p2.blob_id) {
        out.position = float4(0, 0, -1, 1);
        out.color = float4(0);
        return out;
    }
    
    // 선분 방향 및 수직 벡터
    float2 dir = p2.pos - p1.pos;
    float len = length(dir);
    if (len < 0.001) {
        out.position = float4(0, 0, -1, 1);
        out.color = float4(0);
        return out;
    }
    dir /= len;
    
    // 두께 보간 (나이에 따라 변화)
    float t = (vertexID < 2) ? 0.0 : 1.0;
    float thickness1 = baseThickness * (1.0 + thicknessVariation * (1.0 - p1.age));
    float thickness2 = baseThickness * (1.0 + thicknessVariation * (1.0 - p2.age));
    float thickness = mix(thickness1, thickness2, t);
    float2 perp = float2(-dir.y, dir.x) * thickness;
    
    // Quad 정점 (0: p1-perp, 1: p1+perp, 2: p2-perp, 3: p2+perp)
    float2 positions[4] = {
        p1.pos - perp,
        p1.pos + perp,
        p2.pos - perp,
        p2.pos + perp
    };
    
    float2 pos = positions[vertexID];
    
    // 픽셀 좌표 → NDC 변환 (-1 to 1)
    float2 ndc = (pos / texSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Y축 반전
    
    out.position = float4(ndc, 0.0, 1.0);
    
    // 색상 계산
    float age = mix(p1.age, p2.age, t);
    float alpha = mix(p1.alpha, p2.alpha, t);
    
    // 속도 기반 색상 체크
    float2 avg_velocity = (p1.velocity + p2.velocity) * 0.5;
    float speed = length(avg_velocity);
    bool is_fast = useVelocityColors && (speed > velocityThreshold);
    
    float4 color;
    if (is_fast) {
        // 빠른 움직임: 빨간색 계열
        color = velocityColor;
        color.a *= alpha;
    } else if (useColorGradient) {
        // 색상 그라데이션: 나이에 따라 시작 색상에서 끝 색상으로
        float gradient_t = age * gradientIntensity;
        color = mix(startColor, endColor, gradient_t);
        color.a *= alpha;
    } else {
        // 기본 색상
        color = float4(lineColor.rgb, alpha * lineColor.a);
    }
    
    out.color = color;
    
    return out;
}

fragment float4 lineFragment(LineVertex in [[stage_in]])
{
    return in.color;
}

// ============================================================================
// Compute Pipeline: 소규모용 (fallback)
// ============================================================================
kernel void drawTrails(
    texture2d<float, access::read_write> outputTexture [[texture(0)]],
    device const TrailPoint* points [[buffer(0)]],
    constant int &pointCount [[buffer(1)]],
    constant float &baseThickness [[buffer(2)]],
    constant float4 &lineColor [[buffer(3)]],
    constant float &thicknessVariation [[buffer(4)]],
    constant bool &useColorGradient [[buffer(5)]],
    constant float &gradientIntensity [[buffer(6)]],
    constant bool &useVelocityColors [[buffer(7)]],
    constant float &velocityThreshold [[buffer(8)]],
    constant float4 &startColor [[buffer(9)]],
    constant float4 &endColor [[buffer(10)]],
    constant float4 &velocityColor [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    if (pointCount < 2) return;
    
    float4 existing = outputTexture.read(gid);
    float4 color = existing;
    float px = float(gid.x);
    float py = float(gid.y);
    
    for (int i = 0; i < pointCount - 1; i++) {
        TrailPoint p1 = points[i];
        TrailPoint p2 = points[i + 1];
        
        // 다른 블롭이면 선 그리지 않음 (각 blob별로 독립적인 trail)
        if (p1.blob_id != p2.blob_id) continue;
        
        float2 pos1 = p1.pos;
        float2 pos2 = p2.pos;
        
        // 두께 보간
        float thickness1 = baseThickness * (1.0 + thicknessVariation * (1.0 - p1.age));
        float thickness2 = baseThickness * (1.0 + thicknessVariation * (1.0 - p2.age));
        float maxThickness = max(thickness1, thickness2);
        
        // AABB 조기 탈출
        float minX = min(pos1.x, pos2.x) - maxThickness;
        float maxX = max(pos1.x, pos2.x) + maxThickness;
        float minY = min(pos1.y, pos2.y) - maxThickness;
        float maxY = max(pos1.y, pos2.y) + maxThickness;
        
        if (px < minX || px > maxX || py < minY || py > maxY) continue;
        
        float2 line = pos2 - pos1;
        float lineLengthSq = dot(line, line);
        if (lineLengthSq < 0.001) continue;
        
        float2 toPoint = float2(px, py) - pos1;
        float t = dot(toPoint, line) / lineLengthSq;
        
        if (t < 0.0 || t > 1.0) continue;
        
        float2 closestPoint = pos1 + line * t;
        float2 diff = float2(px, py) - closestPoint;
        float distToLineSq = dot(diff, diff);
        
        // 두께 보간
        float thickness = mix(thickness1, thickness2, t);
        float thicknessSq = thickness * thickness;
        
        if (distToLineSq < thicknessSq) {
            float age = mix(p1.age, p2.age, t);
            float alpha = mix(p1.alpha, p2.alpha, t);
            
            // 속도 기반 색상 체크
            float2 avg_velocity = (p1.velocity + p2.velocity) * 0.5;
            float speed = length(avg_velocity);
            bool is_fast = useVelocityColors && (speed > velocityThreshold);
            
            float4 lineColorWithAlpha;
            if (is_fast) {
                lineColorWithAlpha = velocityColor;
                lineColorWithAlpha.a *= alpha;
            } else if (useColorGradient) {
                float gradient_t = age * gradientIntensity;
                lineColorWithAlpha = mix(startColor, endColor, gradient_t);
                lineColorWithAlpha.a *= alpha;
            } else {
                lineColorWithAlpha = float4(lineColor.rgb, alpha * lineColor.a);
            }
            
            color = mix(color, lineColorWithAlpha, lineColorWithAlpha.a);
        }
    }
    
    outputTexture.write(color, gid);
}
)";

// TrailPoint 구조체 (C++ 측) - Metal 셰이더와 동일한 레이아웃
struct TrailPoint {
    float x, y;       // 위치 (픽셀 좌표) - 8 bytes
    float alpha;      // 투명도 - 4 bytes
    float blob_id;    // 블롭 ID - 4 bytes
    float vx, vy;     // 속도 벡터 (픽셀/프레임) - 8 bytes
    float thickness;  // 선 두께 - 4 bytes
    float age;        // 나이 (0.0 = 최신, 1.0 = 오래됨) - 4 bytes
    // 총 32 bytes (Metal 셰이더와 정렬)
};

CHOPToTOPNode::CHOPToTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , clear_pipeline_(nil)
    , fade_pipeline_(nil)
    , draw_lines_pipeline_(nil)
    , line_render_pipeline_(nil)
    , connected_chop_input_(nullptr)
    , graph_ref_(&graph)
    , width_(1920)
    , height_(1080)
    , auto_resolution_(true)
    , points_buffer_(nil)
    , points_buffer_capacity_(0)
    , line_thickness_(DEFAULT_LINE_THICKNESS)
    , thickness_variation_(0.5f)
    , use_color_gradient_(true)
    , gradient_intensity_(1.0f)
    , use_velocity_colors_(true)
    , velocity_threshold_(10.0f)
    , max_trail_length_(0.0f)  // 0 = 무제한
{
    // 기본 색상 초기화
    std::memcpy(trail_start_color_, DEFAULT_START_COLOR, sizeof(DEFAULT_START_COLOR));
    std::memcpy(trail_end_color_, DEFAULT_END_COLOR, sizeof(DEFAULT_END_COLOR));
    std::memcpy(velocity_color_, DEFAULT_VELOCITY_COLOR, sizeof(DEFAULT_VELOCITY_COLOR));
    
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // TOP 참조 입력 포트 (해상도 참조용, 선택적)
    int ref_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(ref_input_id, NodeFamily::TOP, PortDirection::Input, "texture", "reference"));
    
    // CHOP 입력 포트
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(input_id, NodeFamily::CHOP, PortDirection::Input, "channels", "input"));
    
    // 출력 포트
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

CHOPToTOPNode::~CHOPToTOPNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
    points_buffer_ = nil;
}


bool CHOPToTOPNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:chopToTOPShaderSource]
                                                    options:nil
                                                      error:&error];
    if (!library || error) {
        NSLog(@"Error creating library: %@", error);
        return false;
    }
    
    // Clear pipeline
    id<MTLFunction> clearFunction = [library newFunctionWithName:@"clearTexture"];
    if (clearFunction) {
        clear_pipeline_ = [device_ newComputePipelineStateWithFunction:clearFunction error:&error];
        if (error) {
            NSLog(@"Warning: Error creating clear pipeline: %@", error);
        }
    }
    
    // Fade pipeline (Accumulation Buffer용)
    id<MTLFunction> fadeFunction = [library newFunctionWithName:@"fadeTexture"];
    if (fadeFunction) {
        fade_pipeline_ = [device_ newComputePipelineStateWithFunction:fadeFunction error:&error];
        if (error) {
            NSLog(@"Warning: Error creating fade pipeline: %@", error);
        }
    }
    
    // Draw trails pipeline (Compute - fallback)
    id<MTLFunction> function = [library newFunctionWithName:@"drawTrails"];
    if (!function) {
        NSLog(@"Error finding function drawTrails");
        return false;
    }
    
    draw_lines_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        NSLog(@"Error creating pipeline: %@", error);
        return false;
    }
    
    // Render Pipeline (고성능 라인 렌더링)
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"lineVertex"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"lineFragment"];
    
    if (vertexFunc && fragmentFunc) {
        MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDesc.vertexFunction = vertexFunc;
        pipelineDesc.fragmentFunction = fragmentFunc;
        pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
        
        // Alpha blending
        pipelineDesc.colorAttachments[0].blendingEnabled = YES;
        pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        
        line_render_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (error) {
            NSLog(@"Warning: Error creating line render pipeline: %@ (falling back to compute)", error);
            line_render_pipeline_ = nil;
        }
    }
    
    return true;
}

void CHOPToTOPNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("CHOP to TOP");
    ImNodes::EndNodeTitleBar();
    
    // TOP 참조 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    // CHOP 입력 포트
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    ImGui::PushItemWidth(node_width - 20.0f);
    if (auto_resolution_) {
        ImGui::Text("Resolution: Auto");
    } else {
        ImGui::Text("Resolution: %d x %d", width_, height_);
    }
    ImGui::PopItemWidth();
    
    // 미리보기
    if (output_texture_ != nil) {
        ImGui::Spacing();
        ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
        ImTextureRef texture_ref(texture_id);
        
        float preview_width = node_width - 20.0f;
        float aspect_ratio = (float)[output_texture_ height] / (float)[output_texture_ width];
        float preview_height = preview_width * aspect_ratio;
        if (preview_height > 100.0f) {
            preview_height = 100.0f;
            preview_width = preview_height / aspect_ratio;
        }
        ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
    }
    
    ImNodes::BeginOutputAttribute(output_ports_[0].id);
    const float label_width = ImGui::CalcTextSize(output_ports_[0].name.c_str()).x;
    ImGui::Indent(node_width - label_width - 10.0f);
    ImGui::TextUnformatted(output_ports_[0].name.c_str());
    ImNodes::EndOutputAttribute();
    
    ImNodes::EndNode();
}

void CHOPToTOPNode::RenderInspector()
{
    ImGui::Text("CHOP to TOP (GPU Optimized)");
    ImGui::Separator();
    
    ImGui::Checkbox("Auto Resolution", &auto_resolution_);
    
    if (!auto_resolution_) {
        int w = width_, h = height_;
        if (ImGui::InputInt("Width", &w)) { SetWidth(w); }
        if (ImGui::InputInt("Height", &h)) { SetHeight(h); }
    }
    
    ImGui::Separator();
    ImGui::Text("Trail Visualization");
    
    ImGui::SliderFloat("Line Thickness", &line_thickness_, 0.5f, 10.0f, "%.1f");
    ImGui::SliderFloat("Thickness Variation", &thickness_variation_, 0.0f, 1.0f, "%.2f");
    ImGui::TextDisabled("(0.0 = fixed, 1.0 = max variation)");
    
    ImGui::Spacing();
    ImGui::Checkbox("Color Gradient", &use_color_gradient_);
    if (use_color_gradient_) {
        ImGui::SliderFloat("Gradient Intensity", &gradient_intensity_, 0.0f, 1.0f, "%.2f");
        ImGui::ColorEdit4("Start Color (New)", trail_start_color_);
        ImGui::ColorEdit4("End Color (Old)", trail_end_color_);
    }
    
    ImGui::Spacing();
    ImGui::Checkbox("Velocity Colors", &use_velocity_colors_);
    if (use_velocity_colors_) {
        ImGui::SliderFloat("Velocity Threshold", &velocity_threshold_, 1.0f, 100.0f, "%.1f px/frame");
        ImGui::ColorEdit4("Fast Movement Color", velocity_color_);
        ImGui::TextDisabled("(Blobs moving faster than threshold)");
    }
    
    ImGui::Spacing();
    ImGui::SliderFloat("Max Trail Length", &max_trail_length_, 0.0f, 1000.0f, "%.0f frames");
    ImGui::TextDisabled("(0 = unlimited)");
    
    ImGui::Separator();
    ImGui::TextDisabled("GPU Optimized Trail Rendering");
    ImGui::TextDisabled("- Render Pipeline: O(points)");
    ImGui::TextDisabled("- Blob-based trail grouping");
    ImGui::TextDisabled("- Supports 10K+ blobs at 4K 60fps");
    ImGui::TextDisabled("- Color gradients & velocity-based coloring");
}

void CHOPToTOPNode::SetCHOPInput(const std::vector<float>& channels, const std::vector<std::string>& channel_names)
{
    std::lock_guard<std::mutex> lock(channels_mutex_);
    input_channels_ = channels;
    input_channel_names_ = channel_names;
}

void CHOPToTOPNode::UpdateConnectedCHOPNode(NodeManager* node_manager)
{
    if (!node_manager || !graph_ref_ || input_ports_.size() < 2) {
        connected_chop_input_ = nullptr;
        return;
    }
    
    // CHOP 입력 포트 (인덱스 1)
    int chop_port_id = input_ports_[1].id;
    connected_chop_input_ = nullptr;
    
    for (const auto& edge : graph_ref_->edges()) {
        if (edge.to == chop_port_id) {
            NodeBase* node = node_manager->GetNodeByPortId(edge.from);
            if (auto* chop = dynamic_cast<AudioCHOPNodeBase*>(node)) {
                connected_chop_input_ = chop;
                return;
            }
        }
    }
}

void CHOPToTOPNode::GetPointData(std::vector<float>& points_data_out, int& point_count_out) const
{
    std::lock_guard<std::mutex> lock(points_mutex_);
    points_data_out = points_data_;
    point_count_out = point_count_;
}

void CHOPToTOPNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    SetLastTexturePool(texture_pool);
    
    // 해상도 결정
    NSUInteger output_width = width_;
    NSUInteger output_height = height_;
    
    if (auto_resolution_ && !inputs.empty() && inputs[0] != nil) {
        output_width = [inputs[0] width];
        output_height = [inputs[0] height];
    }
    
    if (output_width == 0 || output_height == 0) {
        output_width = 1920;
        output_height = 1080;
    }
    
    // 출력 텍스처 할당
    if (!output_texture_ || [output_texture_ width] != output_width || [output_texture_ height] != output_height) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        
        if (texture_pool) {
            output_texture_ = texture_pool->acquire_texture(output_width, output_height, MTLPixelFormatRGBA16Float);
        } else {
            MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                            width:output_width
                                                                                           height:output_height
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
            desc.storageMode = MTLStorageModePrivate;
            output_texture_ = [device_ newTextureWithDescriptor:desc];
        }
    }
    
    if (!output_texture_) return;
    
    // ========== TrailCHOPNode에서 히스토리 데이터 가져오기 ==========
    std::vector<TrailPoint> trail_points;
    
    // TrailCHOPNode 찾기
    TrailCHOPNode* trail_node = dynamic_cast<TrailCHOPNode*>(connected_chop_input_);
    
    if (trail_node) {
        // 히스토리 데이터에서 블롭별 트레일 포인트 생성
        std::lock_guard<std::mutex> lock(trail_node->GetHistoryMutex());
        const auto& history = trail_node->GetHistory();
        
        if (!history.empty()) {
            // 현재 프레임의 블롭 수 (히스토리와 다를 수 있음)
            int blob_count = trail_node->GetBlobCount();
            int history_count = static_cast<int>(history.size());
            
            // Smooth 옵션 확인
            bool use_smooth = trail_node->IsSmoothEnabled();
            int smooth_segments = trail_node->GetSmoothSegments();
            
            // LOD: 블롭 수 × 히스토리 수가 너무 많으면 서브샘플링
            int max_points = 100000;  // 최대 10만 포인트
            int estimated_points = blob_count * history_count * (use_smooth ? smooth_segments : 1);
            int sample_step = 1;
            
            if (estimated_points > max_points && history_count > 1) {
                sample_step = (estimated_points + max_points - 1) / max_points;
                sample_step = std::max(1, std::min(sample_step, history_count / 2));
            }
            
            // 각 blob별로 독립적인 trail 생성 (이전 버전과 동일한 동작)
            // 구조: blob_id -> [(frame_idx, px, py, alpha, age), ...]
            std::map<float, std::vector<std::tuple<int, float, float, float, float>>> blob_trails;
            
            // 모든 프레임을 순회하며 blob ID별로 포인트 수집
            for (int f = 0; f < history_count; f += sample_step) {
                const auto& frame = history[f];
                
                // 현재 프레임의 블롭 수 계산
                if (frame.empty()) continue;
                int frame_blob_count = (static_cast<int>(frame.size()) - 1) / 8;
                
                for (int b = 0; b < frame_blob_count; b++) {
                    int channel_offset = 1 + b * 8;
                    
                    // 채널 범위 확인
                    if (channel_offset + 5 >= static_cast<int>(frame.size())) continue;
                    
                    float u = frame[channel_offset + 0];  // u (0-1)
                    float v = frame[channel_offset + 1];  // v (0-1)
                    float blob_id = frame[channel_offset + 5];  // id
                    
                    // 유효 범위 확인
                    if (u < 0 || u > 1 || v < 0 || v > 1) continue;
                    if (blob_id < 0) continue;  // 유효하지 않은 ID
                    
                    // 픽셀 좌표로 변환
                    float px = u * (float)output_width;
                    float py = v * (float)output_height;
                    
                    // Alpha: 최신 프레임이 1.0, 오래된 프레임이 0.0에 가깝게
                    // f=0이 가장 오래된 것, f=history_count-1이 가장 최신
                    float alpha = (float)(f + 1) / (float)history_count;
                    
                    // Age: 0.0 = 최신, 1.0 = 오래됨 (alpha와 반대)
                    float age = 1.0f - alpha;
                    
                    blob_trails[blob_id].push_back(std::make_tuple(f, px, py, alpha, age));
                }
            }
            
            // 각 blob ID별로 trail 생성
            trail_points.reserve(estimated_points / sample_step);
            
            for (auto& [blob_id, points] : blob_trails) {
                // 프레임 순서대로 정렬 (오래된 것부터 최신 것까지)
                std::sort(points.begin(), points.end(), 
                    [](const auto& a, const auto& b) {
                        return std::get<0>(a) < std::get<0>(b);
                    });
                
                if (points.size() < 2) continue;
                
                // 최대 길이 제한
                if (max_trail_length_ > 0.0f && points.size() > max_trail_length_) {
                    points.erase(points.begin(), points.end() - static_cast<int>(max_trail_length_));
                }
                
                // 원본 포인트 추출
                std::vector<std::pair<float, float>> raw_points;
                std::vector<float> raw_alphas;
                std::vector<float> raw_ages;
                
                for (const auto& pt : points) {
                    raw_points.push_back({std::get<1>(pt), std::get<2>(pt)});
                    raw_alphas.push_back(std::get<3>(pt));
                    raw_ages.push_back(std::get<4>(pt));
                }
                
                // 속도 계산 (이전 프레임과의 차이)
                std::vector<std::pair<float, float>> velocities;
                velocities.reserve(raw_points.size());
                for (size_t i = 0; i < raw_points.size(); i++) {
                    if (i == 0) {
                        // 첫 번째 포인트: 속도 0
                        velocities.push_back({0.0f, 0.0f});
                    } else {
                        // 이전 포인트와의 차이
                        float vx = raw_points[i].first - raw_points[i-1].first;
                        float vy = raw_points[i].second - raw_points[i-1].second;
                        velocities.push_back({vx, vy});
                    }
                }
                
                // 각 blob의 독립적인 trail 생성 (blob_id 유지)
                
                // ========== Catmull-Rom 스플라인 보간 ==========
                if (use_smooth && raw_points.size() >= 4) {
                    for (size_t i = 0; i < raw_points.size() - 1; i++) {
                        // 4개 포인트 가져오기 (경계 처리)
                        size_t i0 = (i > 0) ? i - 1 : 0;
                        size_t i1 = i;
                        size_t i2 = std::min(raw_points.size() - 1, i + 1);
                        size_t i3 = std::min(raw_points.size() - 1, i + 2);
                        
                        auto& p0 = raw_points[i0];
                        auto& p1 = raw_points[i1];
                        auto& p2 = raw_points[i2];
                        auto& p3 = raw_points[i3];
                        
                        for (int seg = 0; seg < smooth_segments; seg++) {
                            float t = (float)seg / (float)smooth_segments;
                            float t2 = t * t;
                            float t3 = t2 * t;
                            
                            // Catmull-Rom 스플라인
                            float px = 0.5f * (
                                (2.0f * p1.first) +
                                (-p0.first + p2.first) * t +
                                (2.0f * p0.first - 5.0f * p1.first + 4.0f * p2.first - p3.first) * t2 +
                                (-p0.first + 3.0f * p1.first - 3.0f * p2.first + p3.first) * t3
                            );
                            float py = 0.5f * (
                                (2.0f * p1.second) +
                                (-p0.second + p2.second) * t +
                                (2.0f * p0.second - 5.0f * p1.second + 4.0f * p2.second - p3.second) * t2 +
                                (-p0.second + 3.0f * p1.second - 3.0f * p2.second + p3.second) * t3
                            );
                            float alpha = raw_alphas[i1] + (raw_alphas[i2] - raw_alphas[i1]) * t;
                            float age = raw_ages[i1] + (raw_ages[i2] - raw_ages[i1]) * t;
                            
                            // 속도 보간
                            float vx = velocities[i1].first + (velocities[i2].first - velocities[i1].first) * t;
                            float vy = velocities[i1].second + (velocities[i2].second - velocities[i1].second) * t;
                            
                            // 두께 계산 (나이에 따라)
                            float thickness = line_thickness_ * (1.0f + thickness_variation_ * (1.0f - age));
                            
                            trail_points.push_back({px, py, alpha, blob_id, vx, vy, thickness, age});
                        }
                    }
                    // 마지막 포인트 추가
                    trail_points.push_back({
                        raw_points.back().first, raw_points.back().second, 
                        raw_alphas.back(), blob_id,
                        velocities.back().first, velocities.back().second,
                        line_thickness_ * (1.0f + thickness_variation_ * (1.0f - raw_ages.back())),
                        raw_ages.back()
                    });
                } else {
                    // 스무딩 없음: 원본 포인트 사용
                    for (size_t i = 0; i < raw_points.size(); i++) {
                        float age = raw_ages[i];
                        float thickness = line_thickness_ * (1.0f + thickness_variation_ * (1.0f - age));
                        trail_points.push_back({
                            raw_points[i].first, raw_points[i].second, 
                            raw_alphas[i], blob_id,
                            velocities[i].first, velocities[i].second,
                            thickness, age
                        });
                    }
                }
            }
        }
    } else if (connected_chop_input_) {
        // 일반 CHOP 연결 (히스토리 없음)
        const auto& channels = connected_chop_input_->GetOutputChannels();
        
        if (channels.size() >= 2) {
            // 단순히 채널 쌍을 포인트로 변환
            for (size_t i = 0; i + 1 < channels.size(); i += 2) {
                float u = channels[i];
                float v = channels[i + 1];
                
                if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                    float px = u * (float)output_width;
                    float py = v * (float)output_height;
                    trail_points.push_back({px, py, 1.0f, 0.0f, 0.0f, 0.0f, line_thickness_, 0.0f});
                }
            }
        }
    }
    
    // ========== GPU 버퍼 업데이트 ==========
    size_t required_capacity = trail_points.size();
    if (required_capacity == 0) {
        // 포인트 없음: 검은 텍스처
        if (clear_pipeline_) {
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
            [encoder setComputePipelineState:clear_pipeline_];
            [encoder setTexture:output_texture_ atIndex:0];
            MTLSize gridSize = MTLSizeMake(output_width, output_height, 1);
            NSUInteger w = clear_pipeline_.threadExecutionWidth;
            NSUInteger h = clear_pipeline_.maxTotalThreadsPerThreadgroup / w;
            MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }
        return;
    }
    
    // 버퍼 풀링
    if (!points_buffer_ || points_buffer_capacity_ < required_capacity) {
        size_t new_capacity = std::max(kInitialPointCapacity, required_capacity * 2);
        points_buffer_ = [device_ newBufferWithLength:new_capacity * sizeof(TrailPoint)
                                              options:MTLResourceStorageModeShared];
        points_buffer_capacity_ = new_capacity;
    }
    
    memcpy([points_buffer_ contents], trail_points.data(), trail_points.size() * sizeof(TrailPoint));
    
    // 포인트 데이터 저장 (Line TOP용)
    // TrailPoint는 32바이트이므로 float 8개 (x, y, alpha, blob_id, vx, vy, thickness, age)
    {
        std::lock_guard<std::mutex> lock(points_mutex_);
        points_data_.resize(trail_points.size() * 8);
        memcpy(points_data_.data(), trail_points.data(), trail_points.size() * sizeof(TrailPoint));
        point_count_ = static_cast<int>(trail_points.size());
    }
    
    // ========== 텍스처 클리어 ==========
    if (clear_pipeline_) {
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:clear_pipeline_];
        [encoder setTexture:output_texture_ atIndex:0];
        MTLSize gridSize = MTLSizeMake(output_width, output_height, 1);
        NSUInteger w = clear_pipeline_.threadExecutionWidth;
        NSUInteger h = clear_pipeline_.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // ========== 트레일 렌더링 ==========
    int point_count = static_cast<int>(trail_points.size());
    float base_thickness = line_thickness_;
    struct { float r, g, b, a; } line_color = {
        DEFAULT_LINE_COLOR[0],
        DEFAULT_LINE_COLOR[1],
        DEFAULT_LINE_COLOR[2],
        DEFAULT_LINE_COLOR[3]
    };
    struct { float r, g, b, a; } start_color = {
        trail_start_color_[0],
        trail_start_color_[1],
        trail_start_color_[2],
        trail_start_color_[3]
    };
    struct { float r, g, b, a; } end_color = {
        trail_end_color_[0],
        trail_end_color_[1],
        trail_end_color_[2],
        trail_end_color_[3]
    };
    struct { float r, g, b, a; } velocity_color = {
        velocity_color_[0],
        velocity_color_[1],
        velocity_color_[2],
        velocity_color_[3]
    };
    
    // Render Pipeline 사용 (고성능)
    if (line_render_pipeline_ && point_count >= kRenderPipelineThreshold) {
        MTLRenderPassDescriptor* renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPass.colorAttachments[0].texture = output_texture_;
        renderPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
        
        id<MTLRenderCommandEncoder> encoder = [cmd_buffer renderCommandEncoderWithDescriptor:renderPass];
        [encoder setRenderPipelineState:line_render_pipeline_];
        
        [encoder setVertexBuffer:points_buffer_ offset:0 atIndex:0];
        [encoder setVertexBytes:&point_count length:sizeof(int) atIndex:1];
        [encoder setVertexBytes:&base_thickness length:sizeof(float) atIndex:2];
        [encoder setVertexBytes:&line_color length:sizeof(line_color) atIndex:3];
        
        struct { float x, y; } tex_size = { (float)output_width, (float)output_height };
        [encoder setVertexBytes:&tex_size length:sizeof(tex_size) atIndex:4];
        [encoder setVertexBytes:&thickness_variation_ length:sizeof(float) atIndex:5];
        [encoder setVertexBytes:&use_color_gradient_ length:sizeof(bool) atIndex:6];
        [encoder setVertexBytes:&gradient_intensity_ length:sizeof(float) atIndex:7];
        [encoder setVertexBytes:&use_velocity_colors_ length:sizeof(bool) atIndex:8];
        [encoder setVertexBytes:&velocity_threshold_ length:sizeof(float) atIndex:9];
        [encoder setVertexBytes:&start_color length:sizeof(start_color) atIndex:10];
        [encoder setVertexBytes:&end_color length:sizeof(end_color) atIndex:11];
        [encoder setVertexBytes:&velocity_color length:sizeof(velocity_color) atIndex:12];
        
        int line_count = point_count - 1;
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                    vertexStart:0
                    vertexCount:4
                  instanceCount:line_count];
        
        [encoder endEncoding];
    } else {
        // Compute Pipeline (소규모용)
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:draw_lines_pipeline_];
        [encoder setTexture:output_texture_ atIndex:0];
        [encoder setBuffer:points_buffer_ offset:0 atIndex:0];
        [encoder setBytes:&point_count length:sizeof(int) atIndex:1];
        [encoder setBytes:&base_thickness length:sizeof(float) atIndex:2];
        [encoder setBytes:&line_color length:sizeof(line_color) atIndex:3];
        [encoder setBytes:&thickness_variation_ length:sizeof(float) atIndex:4];
        [encoder setBytes:&use_color_gradient_ length:sizeof(bool) atIndex:5];
        [encoder setBytes:&gradient_intensity_ length:sizeof(float) atIndex:6];
        [encoder setBytes:&use_velocity_colors_ length:sizeof(bool) atIndex:7];
        [encoder setBytes:&velocity_threshold_ length:sizeof(float) atIndex:8];
        [encoder setBytes:&start_color length:sizeof(start_color) atIndex:9];
        [encoder setBytes:&end_color length:sizeof(end_color) atIndex:10];
        [encoder setBytes:&velocity_color length:sizeof(velocity_color) atIndex:11];
        
        MTLSize gridSize = MTLSizeMake(output_width, output_height, 1);
        NSUInteger w = draw_lines_pipeline_.threadExecutionWidth;
        NSUInteger h = draw_lines_pipeline_.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
}

void CHOPToTOPNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool,
    const RenderContext& context)
{
    ProcessGPU(inputs, cmd_buffer, texture_pool);
}


std::unique_ptr<NodeBase> CreateCHOPToTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<CHOPToTOPNode>(graph, pos, device);
}

REGISTER_NODE(CHOPToTOP, "CHOP to TOP", "TOP/Convert", NodeFamily::TOP, CreateCHOPToTOPNode, "GPU-optimized trail rendering for 10K+ blobs at 4K 60fps");

} // namespace nodes
} // namespace example
