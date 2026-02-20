#include "level_node.h"
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
static const char* levelShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct LevelParams {
    float inputMin;
    float inputMax;
    float gamma;
    float outputMin;
    float outputMax;
};

kernel void adjustLevels(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant LevelParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 color = inputTexture.read(gid);
    
    // Input levels: remap from [inputMin, inputMax] to [0, 1]
    color.rgb = (color.rgb - params.inputMin) / (params.inputMax - params.inputMin);
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    
    // Gamma correction (midtones)
    color.rgb = pow(color.rgb, float3(1.0 / params.gamma));
    
    // Output levels: remap from [0, 1] to [outputMin, outputMax]
    color.rgb = color.rgb * (params.outputMax - params.outputMin) + params.outputMin;
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    
    outputTexture.write(color, gid);
}
)";

LevelNode::LevelNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil),
      input_min_(0.0f), input_max_(1.0f), gamma_(1.0f),
      output_min_(0.0f), output_max_(1.0f)
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

LevelNode::~LevelNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void LevelNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool LevelNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(levelShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Level shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> function = [library newFunctionWithName:@"adjustLevels"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void LevelNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Level");
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
    
    ImGui::Text("Input Levels:");
    ImGui::SliderFloat("Min", &input_min_, 0.0f, 1.0f);
    ImGui::SliderFloat("Max", &input_max_, 0.0f, 1.0f);
    
    ImGui::Spacing();
    ImGui::SliderFloat("Gamma", &gamma_, 0.1f, 3.0f);
    
    ImGui::Spacing();
    ImGui::Text("Output Levels:");
    ImGui::SliderFloat("Out Min", &output_min_, 0.0f, 1.0f);
    ImGui::SliderFloat("Out Max", &output_max_, 0.0f, 1.0f);
    
    // 제약조건
    if (input_min_ > input_max_) input_min_ = input_max_;
    if (output_min_ > output_max_) output_min_ = output_max_;
    
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

void LevelNode::RenderInspector()
{
    ImGui::Text("Level");
    ImGui::Separator();
    
    ImGui::Text("Input Levels:");
    ImGui::SliderFloat("Input Min", &input_min_, 0.0f, 1.0f);
    ImGui::SliderFloat("Input Max", &input_max_, 0.0f, 1.0f);
    
    ImGui::Spacing();
    ImGui::SliderFloat("Gamma (Midtones)", &gamma_, 0.1f, 3.0f);
    
    ImGui::Spacing();
    ImGui::Text("Output Levels:");
    ImGui::SliderFloat("Output Min", &output_min_, 0.0f, 1.0f);
    ImGui::SliderFloat("Output Max", &output_max_, 0.0f, 1.0f);
    
    // 제약조건
    if (input_min_ > input_max_) input_min_ = input_max_;
    if (output_min_ > output_max_) output_min_ = output_max_;
    
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

void LevelNode::ProcessGPU(
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
    
    // Level 조정
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    
    // 파라미터 구조체
    struct LevelParams {
        float inputMin;
        float inputMax;
        float gamma;
        float outputMin;
        float outputMax;
    };
    
    LevelParams params = {
        input_min_,
        input_max_,
        gamma_,
        output_min_,
        output_max_
    };
    
    [encoder setBytes:&params length:sizeof(LevelParams) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateLevelNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<LevelNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Level, "Level", "TOP/Filter", NodeFamily::TOP, CreateLevelNode, "Adjust input/output levels and gamma");

} // namespace nodes
} // namespace example
