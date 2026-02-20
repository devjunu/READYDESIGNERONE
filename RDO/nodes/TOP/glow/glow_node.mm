#include "glow_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Metal 셰이더 코드
static const char* glowShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gaussianBlurHorizontal(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &blurRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int radius = int(blurRadius);
    float4 color = float4(0.0);
    float totalWeight = 0.0;
    
    for (int x = -radius; x <= radius; x++) {
        int2 coord = int2(gid.x + x, gid.y);
        coord.x = clamp(coord.x, 0, int(inputTexture.get_width()) - 1);
        
        float weight = exp(-(x * x) / (2.0 * blurRadius * blurRadius));
        color += inputTexture.read(uint2(coord)) * weight;
        totalWeight += weight;
    }
    
    outputTexture.write(color / totalWeight, gid);
}

kernel void gaussianBlurVertical(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &blurRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int radius = int(blurRadius);
    float4 color = float4(0.0);
    float totalWeight = 0.0;
    
    for (int y = -radius; y <= radius; y++) {
        int2 coord = int2(gid.x, gid.y + y);
        coord.y = clamp(coord.y, 0, int(inputTexture.get_height()) - 1);
        
        float weight = exp(-(y * y) / (2.0 * blurRadius * blurRadius));
        color += inputTexture.read(uint2(coord)) * weight;
        totalWeight += weight;
    }
    
    outputTexture.write(color / totalWeight, gid);
}

kernel void compositeGlow(
    texture2d<float, access::read> originalTexture [[texture(0)]],
    texture2d<float, access::read> blurredTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &intensity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 original = originalTexture.read(gid);
    float4 blurred = blurredTexture.read(gid);
    
    float4 result = original + (blurred * intensity);
    result = clamp(result, 0.0, 1.0);
    
    outputTexture.write(result, gid);
}
)";

GlowNode::GlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), intensity_(1.0f), blur_size_(10.0f)
    , horizontal_blur_pipeline_(nil), vertical_blur_pipeline_(nil), composite_pipeline_(nil)
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

GlowNode::~GlowNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void GlowNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool GlowNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(glowShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Glow shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> hBlurFunc = [library newFunctionWithName:@"gaussianBlurHorizontal"];
    id<MTLFunction> vBlurFunc = [library newFunctionWithName:@"gaussianBlurVertical"];
    id<MTLFunction> compositeFunc = [library newFunctionWithName:@"compositeGlow"];
    
    horizontal_blur_pipeline_ = [device_ newComputePipelineStateWithFunction:hBlurFunc error:&error];
    vertical_blur_pipeline_ = [device_ newComputePipelineStateWithFunction:vBlurFunc error:&error];
    composite_pipeline_ = [device_ newComputePipelineStateWithFunction:compositeFunc error:&error];
    
    return (horizontal_blur_pipeline_ != nil && vertical_blur_pipeline_ != nil && composite_pipeline_ != nil);
}

void GlowNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Glow");
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
    ImGui::SliderFloat("Intensity", &intensity_, 0.0f, 2.0f);
    ImGui::SliderFloat("Blur Size", &blur_size_, 0.0f, 50.0f);
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

void GlowNode::RenderInspector()
{
    ImGui::Text("Glow");
    ImGui::Separator();
    
    ImGui::SliderFloat("Intensity", &intensity_, 0.0f, 2.0f);
    ImGui::SliderFloat("Blur Size", &blur_size_, 0.0f, 50.0f);
    
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

void GlowNode::ProcessGPU(
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
    
    // 중간 텍스처 생성
    MTLPixelFormat format = [input_texture pixelFormat];
    id<MTLTexture> intermediate_texture = texture_pool->acquire_texture(width, height, format);
    id<MTLTexture> blurred_texture = texture_pool->acquire_texture(width, height, format);
    
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
    
    if (!intermediate_texture || !blurred_texture || !output_texture_) {
        // 실패 시 할당된 텍스처 해제
        if (intermediate_texture) texture_pool->release_texture(intermediate_texture);
        if (blurred_texture) texture_pool->release_texture(blurred_texture);
        output_texture_ = nil;
        return;
    }
    
    // Horizontal Blur
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:horizontal_blur_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:intermediate_texture atIndex:1];
    [encoder setBytes:&blur_size_ length:sizeof(float) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = horizontal_blur_pipeline_.threadExecutionWidth;
    NSUInteger h = horizontal_blur_pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    // Vertical Blur
    encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:vertical_blur_pipeline_];
    [encoder setTexture:intermediate_texture atIndex:0];
    [encoder setTexture:blurred_texture atIndex:1];
    [encoder setBytes:&blur_size_ length:sizeof(float) atIndex:0];
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    // Composite
    encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:composite_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:blurred_texture atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&intensity_ length:sizeof(float) atIndex:0];
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    // 중간 텍스처 즉시 반환 (메모리 누수 방지)
    // 참고: 출력 텍스처는 노드가 보유하고 다음 프레임에서 재사용
    texture_pool->release_texture(intermediate_texture);
    texture_pool->release_texture(blurred_texture);
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateGlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<GlowNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Glow, "Glow", "TOP/Filter", NodeFamily::TOP, CreateGlowNode, "Glow effect");

} // namespace nodes
} // namespace example
