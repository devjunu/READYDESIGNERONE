#include "hsv_adjust_node.h"
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
static const char* hsvShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// RGB to HSV conversion
float3 rgb2hsv(float3 rgb) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(rgb.bg, K.wz), float4(rgb.gb, K.xy), step(rgb.b, rgb.g));
    float4 q = mix(float4(p.xyw, rgb.r), float4(rgb.r, p.yzx), step(p.x, rgb.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB conversion
float3 hsv2rgb(float3 hsv) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

struct HSVParams {
    float hueShift;      // in degrees / 360
    float saturation;    // multiplier
    float value;         // multiplier
};

kernel void adjustHSV(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant HSVParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 color = inputTexture.read(gid);
    
    // Convert RGB to HSV
    float3 hsv = rgb2hsv(color.rgb);
    
    // Adjust Hue (with wrapping)
    hsv.x = fract(hsv.x + params.hueShift);
    
    // Adjust Saturation
    hsv.y = clamp(hsv.y * params.saturation, 0.0, 1.0);
    
    // Adjust Value (brightness)
    hsv.z = clamp(hsv.z * params.value, 0.0, 1.0);
    
    // Convert back to RGB
    color.rgb = hsv2rgb(hsv);
    
    outputTexture.write(color, gid);
}
)";

HSVAdjustNode::HSVAdjustNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil),
      hue_(0.0f), saturation_(1.0f), value_(1.0f)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 입력 및 출력 포트
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

HSVAdjustNode::~HSVAdjustNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void HSVAdjustNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool HSVAdjustNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(hsvShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"HSV Adjust shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> function = [library newFunctionWithName:@"adjustHSV"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void HSVAdjustNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("HSV Adjust");
    ImNodes::EndNodeTitleBar();
    
    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted("input");
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Hue", &hue_, -180.0f, 180.0f);
    ImGui::SliderFloat("Saturation", &saturation_, 0.0f, 2.0f);
    ImGui::SliderFloat("Value", &value_, 0.0f, 2.0f);
    ImGui::PopItemWidth();
    
    ImGui::Spacing();
    
    // 프리뷰
    if (output_texture_ != nil)
    {
        int width = (int)[output_texture_ width];
        int height = (int)[output_texture_ height];
        
        if (width > 0 && height > 0)
        {
            float preview_width = 180.0f;
            float aspect_ratio = (float)height / (float)width;
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
    else
    {
        ImGui::TextDisabled("No input connected");
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

void HSVAdjustNode::RenderInspector()
{
    ImGui::Text("HSV Adjust");
    ImGui::Separator();
    
    ImGui::SliderFloat("Hue Shift", &hue_, -180.0f, 180.0f);
    ImGui::SliderFloat("Saturation", &saturation_, 0.0f, 2.0f);
    ImGui::SliderFloat("Value (Brightness)", &value_, 0.0f, 2.0f);
    
    ImGui::Spacing();
    
    if (ImGui::Button("Reset")) {
        hue_ = 0.0f;
        saturation_ = 1.0f;
        value_ = 1.0f;
    }
    
    ImGui::Spacing();
    
    if (output_texture_ != nil)
    {
        int width = (int)[output_texture_ width];
        int height = (int)[output_texture_ height];
        
        if (width > 0 && height > 0)
        {
            float preview_width = 300.0f;
            float aspect_ratio = (float)height / (float)width;
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

void HSVAdjustNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);
    
    if (inputs.empty() || inputs[0] == nil || !texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    id<MTLTexture> input_texture = inputs[0];
    int width = (int)[input_texture width];
    int height = (int)[input_texture height];
    MTLPixelFormat format = [input_texture pixelFormat];
    
    // 기존 output_texture 해제 (크기가 변경된 경우)
    if (output_texture_)
    {
        if ([output_texture_ width] != width || [output_texture_ height] != height)
        {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
    }
    
    // 출력 텍스처 할당
    if (!output_texture_)
    {
        output_texture_ = texture_pool->acquire_texture(width, height, format);
    }
    
    if (!output_texture_) {
        return;
    }
    
    // HSV 조정
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    
    // 파라미터 구조체
    struct HSVParams {
        float hueShift;
        float saturation;
        float value;
    };
    
    HSVParams params = {
        hue_ / 360.0f,  // Convert degrees to 0-1 range
        saturation_,
        value_
    };
    
    [encoder setBytes:&params length:sizeof(HSVParams) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateHSVAdjustNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<HSVAdjustNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(HSVAdjust, "HSV Adjust", "TOP/Filter", NodeFamily::TOP, CreateHSVAdjustNode, "Adjust Hue, Saturation, and Value");

} // namespace nodes
} // namespace example
