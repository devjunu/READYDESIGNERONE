#include "over_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Metal 셰이더 코드
static const char* overShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Alpha Over 공식: C = A + B * (1 - A.a)
kernel void alphaOver(
    texture2d<float, access::read> foregroundTexture [[texture(0)]],
    texture2d<float, access::read> backgroundTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &opacity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 fg = foregroundTexture.read(gid);
    float4 bg = backgroundTexture.read(gid);
    
    // Opacity 적용
    fg.a *= opacity;
    
    // Alpha Over compositing
    float4 result;
    result.rgb = fg.rgb + bg.rgb * (1.0 - fg.a);
    result.a = fg.a + bg.a * (1.0 - fg.a);
    
    outputTexture.write(result, gid);
}

// Pre-multiplied Alpha Over
kernel void alphaOverPremultiplied(
    texture2d<float, access::read> foregroundTexture [[texture(0)]],
    texture2d<float, access::read> backgroundTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &opacity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 fg = foregroundTexture.read(gid);
    float4 bg = backgroundTexture.read(gid);
    
    // Pre-multiplied alpha의 경우 RGB는 이미 알파와 곱해져 있음
    fg *= opacity;
    
    // Pre-multiplied Alpha Over
    float4 result;
    result.rgb = fg.rgb + bg.rgb * (1.0 - fg.a);
    result.a = fg.a + bg.a * (1.0 - fg.a);
    
    outputTexture.write(result, gid);
}
)";

OverNode::OverNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil), premult_pipeline_(nil),
      opacity_(1.0f), pre_multiplied_(false)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 입력 및 출력 포트 (foreground, background)
    int fg_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int bg_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(fg_input_id, NodeFamily::TOP, PortDirection::Input, "texture", "foreground"));
    AddInputPort(Port(bg_input_id, NodeFamily::TOP, PortDirection::Input, "texture", "background"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

OverNode::~OverNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void OverNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool OverNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(overShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Over shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> function = [library newFunctionWithName:@"alphaOver"];
    id<MTLFunction> premultFunc = [library newFunctionWithName:@"alphaOverPremultiplied"];
    
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    premult_pipeline_ = [device_ newComputePipelineStateWithFunction:premultFunc error:&error];
    
    return (pipeline_ != nil && premult_pipeline_ != nil);
}

void OverNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Over");
    ImNodes::EndNodeTitleBar();
    
    // 입력 포트 - Foreground
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted("foreground");
        ImNodes::EndInputAttribute();
    }
    
    // 입력 포트 - Background
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted("background");
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Opacity", &opacity_, 0.0f, 1.0f);
    ImGui::Checkbox("Pre-multiplied", &pre_multiplied_);
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
        ImGui::TextDisabled("No inputs connected");
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

void OverNode::RenderInspector()
{
    ImGui::Text("Over (Alpha Compositing)");
    ImGui::Separator();
    
    ImGui::SliderFloat("Opacity", &opacity_, 0.0f, 1.0f);
    ImGui::Checkbox("Pre-multiplied Alpha", &pre_multiplied_);
    
    ImGui::Spacing();
    ImGui::TextWrapped("Alpha Over: foreground is composited over background using alpha channel.");
    
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

void OverNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);
    
    if (inputs.size() < 2 || inputs[0] == nil || inputs[1] == nil || !texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    id<MTLTexture> fg_texture = inputs[0];
    id<MTLTexture> bg_texture = inputs[1];
    
    // 배경 텍스처 크기 사용
    int width = (int)[bg_texture width];
    int height = (int)[bg_texture height];
    MTLPixelFormat format = [bg_texture pixelFormat];
    
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
    
    // 파이프라인 선택
    id<MTLComputePipelineState> selected_pipeline = pre_multiplied_ ? premult_pipeline_ : pipeline_;
    
    // Over 합성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:selected_pipeline];
    [encoder setTexture:fg_texture atIndex:0];
    [encoder setTexture:bg_texture atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&opacity_ length:sizeof(float) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = selected_pipeline.threadExecutionWidth;
    NSUInteger h = selected_pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateOverNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<OverNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Over, "Over", "TOP/Composite", NodeFamily::TOP, CreateOverNode, "Alpha over compositing");

} // namespace nodes
} // namespace example
