#include "ramp_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>
#import <simd/simd.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Metal 셰이더 코드
static const char* rampShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void generateLinearRamp(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float4 &colorStart [[buffer(0)]],
    constant float4 &colorEnd [[buffer(1)]],
    constant float &angle [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    
    // 각도에 따라 회전
    float rad = angle * M_PI_F / 180.0;
    float2 dir = float2(cos(rad), sin(rad));
    
    float t = dot(uv - 0.5, dir) + 0.5;
    t = clamp(t, 0.0, 1.0);
    
    float4 color = mix(colorStart, colorEnd, t);
    outputTexture.write(color, gid);
}

kernel void generateRadialRamp(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float4 &colorStart [[buffer(0)]],
    constant float4 &colorEnd [[buffer(1)]],
    constant float2 &center [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    
    float dist = distance(uv, center);
    float t = clamp(dist / 0.707, 0.0, 1.0);  // 0.707 = sqrt(0.5^2 + 0.5^2)
    
    float4 color = mix(colorStart, colorEnd, t);
    outputTexture.write(color, gid);
}

kernel void generateAngleRamp(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float4 &colorStart [[buffer(0)]],
    constant float4 &colorEnd [[buffer(1)]],
    constant float2 &center [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 dir = uv - center;
    
    float angle = atan2(dir.y, dir.x);
    float t = (angle + M_PI_F) / (2.0 * M_PI_F);
    
    float4 color = mix(colorStart, colorEnd, t);
    outputTexture.write(color, gid);
}

kernel void generateBoxRamp(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float4 &colorStart [[buffer(0)]],
    constant float4 &colorEnd [[buffer(1)]],
    constant float2 &center [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 d = abs(uv - center);
    
    float dist = max(d.x, d.y);
    float t = clamp(dist / 0.5, 0.0, 1.0);
    
    float4 color = mix(colorStart, colorEnd, t);
    outputTexture.write(color, gid);
}
)";

RampNode::RampNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), ramp_type_(RampType::Linear), 
      center_x_(0.5f), center_y_(0.5f), angle_(0.0f),
      width_(1920), height_(1080)
{
    // 기본 색상: 검은색 -> 흰색
    color_start_[0] = 0.0f; color_start_[1] = 0.0f; color_start_[2] = 0.0f; color_start_[3] = 1.0f;
    color_end_[0] = 1.0f; color_end_[1] = 1.0f; color_end_[2] = 1.0f; color_end_[3] = 1.0f;
    
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 출력 포트만 있음 (Generator)
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

RampNode::~RampNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void RampNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool RampNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(rampShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Ramp shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> linearFunc = [library newFunctionWithName:@"generateLinearRamp"];
    id<MTLFunction> radialFunc = [library newFunctionWithName:@"generateRadialRamp"];
    id<MTLFunction> angleFunc = [library newFunctionWithName:@"generateAngleRamp"];
    id<MTLFunction> boxFunc = [library newFunctionWithName:@"generateBoxRamp"];
    
    linear_pipeline_ = [device_ newComputePipelineStateWithFunction:linearFunc error:&error];
    radial_pipeline_ = [device_ newComputePipelineStateWithFunction:radialFunc error:&error];
    angle_pipeline_ = [device_ newComputePipelineStateWithFunction:angleFunc error:&error];
    box_pipeline_ = [device_ newComputePipelineStateWithFunction:boxFunc error:&error];
    
    return (linear_pipeline_ != nil && radial_pipeline_ != nil && 
            angle_pipeline_ != nil && box_pipeline_ != nil);
}

void RampNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Ramp");
    ImNodes::EndNodeTitleBar();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    
    const char* ramp_types[] = { "Linear", "Radial", "Angle", "Box" };
    int current_type = (int)ramp_type_;
    if (ImGui::Combo("Type", &current_type, ramp_types, 4)) {
        ramp_type_ = (RampType)current_type;
    }
    
    ImGui::ColorEdit4("Start", color_start_);
    ImGui::ColorEdit4("End", color_end_);
    
    if (ramp_type_ == RampType::Linear) {
        ImGui::SliderFloat("Angle", &angle_, 0.0f, 360.0f);
    } else if (ramp_type_ != RampType::Linear) {
        ImGui::SliderFloat("Center X", &center_x_, 0.0f, 1.0f);
        ImGui::SliderFloat("Center Y", &center_y_, 0.0f, 1.0f);
    }
    
    ImGui::InputInt("Width", &width_);
    ImGui::InputInt("Height", &height_);
    
    // 해상도 제한
    if (width_ < 1) width_ = 1;
    if (height_ < 1) height_ = 1;
    if (width_ > 4096) width_ = 4096;
    if (height_ > 4096) height_ = 4096;
    
    ImGui::PopItemWidth();
    
    ImGui::Spacing();
    
    // 프리뷰
    if (output_texture_ != nil)
    {
        int tex_width = (int)[output_texture_ width];
        int tex_height = (int)[output_texture_ height];
        
        if (tex_width > 0 && tex_height > 0)
        {
            float preview_width = 180.0f;
            float aspect_ratio = (float)tex_height / (float)tex_width;
            float preview_height = preview_width * aspect_ratio;
            
            if (preview_height > 120.0f)
            {
                preview_height = 120.0f;
                preview_width = preview_height / aspect_ratio;
            }
            
            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    }
    
    ImGui::Spacing();
    
    // 출력 포트
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("output").x;
        ImGui::Indent(200.0f - label_width);
        ImGui::TextUnformatted("output");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void RampNode::RenderInspector()
{
    ImGui::Text("Ramp");
    ImGui::Separator();
    
    const char* ramp_types[] = { "Linear", "Radial", "Angle", "Box" };
    int current_type = (int)ramp_type_;
    if (ImGui::Combo("Type", &current_type, ramp_types, 4)) {
        ramp_type_ = (RampType)current_type;
    }
    
    ImGui::ColorEdit4("Start Color", color_start_);
    ImGui::ColorEdit4("End Color", color_end_);
    
    if (ramp_type_ == RampType::Linear) {
        ImGui::SliderFloat("Angle", &angle_, 0.0f, 360.0f);
    } else if (ramp_type_ != RampType::Linear) {
        ImGui::SliderFloat("Center X", &center_x_, 0.0f, 1.0f);
        ImGui::SliderFloat("Center Y", &center_y_, 0.0f, 1.0f);
    }
    
    ImGui::InputInt("Width", &width_);
    ImGui::InputInt("Height", &height_);
    
    // 해상도 제한
    if (width_ < 1) width_ = 1;
    if (height_ < 1) height_ = 1;
    if (width_ > 4096) width_ = 4096;
    if (height_ > 4096) height_ = 4096;
    
    ImGui::Spacing();
    
    if (output_texture_ != nil)
    {
        int tex_width = (int)[output_texture_ width];
        int tex_height = (int)[output_texture_ height];
        
        if (tex_width > 0 && tex_height > 0)
        {
            float preview_width = 300.0f;
            float aspect_ratio = (float)tex_height / (float)tex_width;
            float preview_height = preview_width * aspect_ratio;
            
            if (preview_height > 400.0f)
            {
                preview_height = 400.0f;
                preview_width = preview_height / aspect_ratio;
            }
            
            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    }
}

void RampNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);
    
    if (!texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    // 기존 output_texture 해제 (크기가 변경된 경우)
    if (output_texture_)
    {
        if ([output_texture_ width] != width_ || [output_texture_ height] != height_)
        {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
    }
    
    // 출력 텍스처 할당
    if (!output_texture_)
    {
        output_texture_ = texture_pool->acquire_texture(width_, height_, MTLPixelFormatRGBA8Unorm);
    }
    
    if (!output_texture_) {
        return;
    }
    
    // 타입에 따라 파이프라인 선택
    id<MTLComputePipelineState> pipeline = nil;
    switch (ramp_type_) {
        case RampType::Linear: pipeline = linear_pipeline_; break;
        case RampType::Radial: pipeline = radial_pipeline_; break;
        case RampType::Angle: pipeline = angle_pipeline_; break;
        case RampType::Box: pipeline = box_pipeline_; break;
    }
    
    if (!pipeline) return;
    
    // Ramp 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:output_texture_ atIndex:0];
    
    // 색상 전달
    simd_float4 colorStart = simd_make_float4(color_start_[0], color_start_[1], color_start_[2], color_start_[3]);
    simd_float4 colorEnd = simd_make_float4(color_end_[0], color_end_[1], color_end_[2], color_end_[3]);
    [encoder setBytes:&colorStart length:sizeof(simd_float4) atIndex:0];
    [encoder setBytes:&colorEnd length:sizeof(simd_float4) atIndex:1];
    
    // 타입별 파라미터
    if (ramp_type_ == RampType::Linear) {
        [encoder setBytes:&angle_ length:sizeof(float) atIndex:2];
    } else {
        simd_float2 center = simd_make_float2(center_x_, center_y_);
        [encoder setBytes:&center length:sizeof(simd_float2) atIndex:2];
    }
    
    MTLSize gridSize = MTLSizeMake(width_, height_, 1);
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateRampNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<RampNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Ramp, "Ramp", "TOP/Generator", NodeFamily::TOP, CreateRampNode, "Generate gradient ramp");

} // namespace nodes
} // namespace example
