#include "constant_node.h"
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
static const char* constantShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void generateConstant(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float4 &color [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    outputTexture.write(color, gid);
}
)";

ConstantNode::ConstantNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil), width_(1920), height_(1080)
{
    // 기본 색상: 흰색
    color_[0] = 1.0f;
    color_[1] = 1.0f;
    color_[2] = 1.0f;
    color_[3] = 1.0f;
    
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

ConstantNode::~ConstantNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void ConstantNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool ConstantNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(constantShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Constant shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> function = [library newFunctionWithName:@"generateConstant"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void ConstantNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Constant");
    ImNodes::EndNodeTitleBar();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    ImGui::ColorEdit4("Color", color_);
    
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

void ConstantNode::RenderInspector()
{
    ImGui::Text("Constant");
    ImGui::Separator();
    
    ImGui::ColorEdit4("Color", color_);
    
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

void ConstantNode::ProcessGPU(
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
    
    // Constant 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:output_texture_ atIndex:0];
    
    // RGBA 색상 전달
    simd_float4 color = simd_make_float4(color_[0], color_[1], color_[2], color_[3]);
    [encoder setBytes:&color length:sizeof(simd_float4) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width_, height_, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateConstantNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ConstantNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Constant, "Constant", "TOP/Generator", NodeFamily::TOP, CreateConstantNode, "Generate constant color texture");

} // namespace nodes
} // namespace example
