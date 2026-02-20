#include "edge_node.h"
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
static const char* edgeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Sobel Edge Detection
kernel void sobelEdge(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int width = inputTexture.get_width();
    int height = inputTexture.get_height();
    
    // Sobel kernels
    float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    float sumX = 0.0;
    float sumY = 0.0;
    
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 coord = int2(gid.x + i, gid.y + j);
            coord.x = clamp(coord.x, 0, width - 1);
            coord.y = clamp(coord.y, 0, height - 1);
            
            float4 pixel = inputTexture.read(uint2(coord));
            float gray = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
            
            int idx = (j + 1) * 3 + (i + 1);
            sumX += gray * Gx[idx];
            sumY += gray * Gy[idx];
        }
    }
    
    float magnitude = sqrt(sumX * sumX + sumY * sumY);
    magnitude = (magnitude > threshold) ? magnitude : 0.0;
    
    float4 result = float4(magnitude, magnitude, magnitude, 1.0);
    outputTexture.write(result, gid);
}

// Prewitt Edge Detection
kernel void prewittEdge(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int width = inputTexture.get_width();
    int height = inputTexture.get_height();
    
    // Prewitt kernels
    float Gx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    float Gy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    
    float sumX = 0.0;
    float sumY = 0.0;
    
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 coord = int2(gid.x + i, gid.y + j);
            coord.x = clamp(coord.x, 0, width - 1);
            coord.y = clamp(coord.y, 0, height - 1);
            
            float4 pixel = inputTexture.read(uint2(coord));
            float gray = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
            
            int idx = (j + 1) * 3 + (i + 1);
            sumX += gray * Gx[idx];
            sumY += gray * Gy[idx];
        }
    }
    
    float magnitude = sqrt(sumX * sumX + sumY * sumY);
    magnitude = (magnitude > threshold) ? magnitude : 0.0;
    
    float4 result = float4(magnitude, magnitude, magnitude, 1.0);
    outputTexture.write(result, gid);
}

// Simple Canny-like Edge Detection
kernel void cannyEdge(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int width = inputTexture.get_width();
    int height = inputTexture.get_height();
    
    // Sobel for gradient
    float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    float sumX = 0.0;
    float sumY = 0.0;
    
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 coord = int2(gid.x + i, gid.y + j);
            coord.x = clamp(coord.x, 0, width - 1);
            coord.y = clamp(coord.y, 0, height - 1);
            
            float4 pixel = inputTexture.read(uint2(coord));
            float gray = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
            
            int idx = (j + 1) * 3 + (i + 1);
            sumX += gray * Gx[idx];
            sumY += gray * Gy[idx];
        }
    }
    
    float magnitude = sqrt(sumX * sumX + sumY * sumY);
    
    // Non-maximum suppression approximation
    float angle = atan2(sumY, sumX);
    
    // Hysteresis thresholding
    float lowThreshold = threshold * 0.5;
    float highThreshold = threshold;
    
    float edge = 0.0;
    if (magnitude > highThreshold) {
        edge = 1.0;
    } else if (magnitude > lowThreshold) {
        edge = 0.5;  // Weak edge
    }
    
    float4 result = float4(edge, edge, edge, 1.0);
    outputTexture.write(result, gid);
}
)";

EdgeNode::EdgeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), edge_type_(EdgeType::Sobel), threshold_(0.1f)
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

EdgeNode::~EdgeNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void EdgeNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool EdgeNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(edgeShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Edge shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> sobelFunc = [library newFunctionWithName:@"sobelEdge"];
    id<MTLFunction> prewittFunc = [library newFunctionWithName:@"prewittEdge"];
    id<MTLFunction> cannyFunc = [library newFunctionWithName:@"cannyEdge"];
    
    sobel_pipeline_ = [device_ newComputePipelineStateWithFunction:sobelFunc error:&error];
    prewitt_pipeline_ = [device_ newComputePipelineStateWithFunction:prewittFunc error:&error];
    canny_pipeline_ = [device_ newComputePipelineStateWithFunction:cannyFunc error:&error];
    
    return (sobel_pipeline_ != nil && prewitt_pipeline_ != nil && canny_pipeline_ != nil);
}

void EdgeNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Edge");
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
    
    const char* edge_types[] = { "Sobel", "Prewitt", "Canny" };
    int current_type = (int)edge_type_;
    if (ImGui::Combo("Type", &current_type, edge_types, 3)) {
        edge_type_ = (EdgeType)current_type;
    }
    
    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f);
    
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

void EdgeNode::RenderInspector()
{
    ImGui::Text("Edge Detection");
    ImGui::Separator();
    
    const char* edge_types[] = { "Sobel", "Prewitt", "Canny" };
    int current_type = (int)edge_type_;
    if (ImGui::Combo("Detection Type", &current_type, edge_types, 3)) {
        edge_type_ = (EdgeType)current_type;
    }
    
    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f);
    
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

void EdgeNode::ProcessGPU(
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
    
    // 타입에 따라 파이프라인 선택
    id<MTLComputePipelineState> pipeline = nil;
    switch (edge_type_) {
        case EdgeType::Sobel: pipeline = sobel_pipeline_; break;
        case EdgeType::Prewitt: pipeline = prewitt_pipeline_; break;
        case EdgeType::Canny: pipeline = canny_pipeline_; break;
    }
    
    if (!pipeline) return;
    
    // Edge detection
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&threshold_ length:sizeof(float) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateEdgeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<EdgeNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Edge, "Edge", "TOP/Filter", NodeFamily::TOP, CreateEdgeNode, "Edge detection (Sobel, Prewitt, Canny)");

} // namespace nodes
} // namespace example
