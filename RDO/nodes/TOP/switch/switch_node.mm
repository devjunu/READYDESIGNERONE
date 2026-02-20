#include "switch_node.h"
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

// Metal 셰이더 코드 (간단히 복사)
static const char* switchShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void copyTexture(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 color = inputTexture.read(gid);
    outputTexture.write(color, gid);
}
)";

SwitchNode::SwitchNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil), selected_index_(0), input_count_(4)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 초기 포트 생성 (4개 입력)
    for (int i = 0; i < input_count_; i++) {
        int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
        std::string name = "input " + std::to_string(i);
        AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", name));
    }
    
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

SwitchNode::~SwitchNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void SwitchNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void SwitchNode::SetInputCount(int count)
{
    if (count < 2) count = 2;
    if (count > 16) count = 16;
    
    if (count != input_count_) {
        input_count_ = count;
        // 포트는 외부에서 재구성 필요
    }
}

bool SwitchNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(switchShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Switch shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> function = [library newFunctionWithName:@"copyTexture"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void SwitchNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Switch");
    ImNodes::EndNodeTitleBar();
    
    // 입력 포트들
    for (size_t i = 0; i < input_ports_.size(); i++) {
        const Port& port = input_ports_[i];
        ImNodes::BeginInputAttribute(port.id);
        
        // 선택된 입력 강조
        if (i == selected_index_) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "* %s", port.name.c_str());
        } else {
            ImGui::Text("%s", port.name.c_str());
        }
        
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    
    int max_index = (int)input_ports_.size() - 1;
    ImGui::SliderInt("Select", &selected_index_, 0, max_index);
    
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

void SwitchNode::RenderInspector()
{
    ImGui::Text("Switch");
    ImGui::Separator();
    
    int max_index = (int)input_ports_.size() - 1;
    ImGui::SliderInt("Selected Input", &selected_index_, 0, max_index);
    
    ImGui::Spacing();
    ImGui::Text("Active Input: %d", selected_index_);
    
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

void SwitchNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);
    
    if (inputs.empty() || !texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    // 선택된 인덱스 검증
    if (selected_index_ < 0 || selected_index_ >= inputs.size() || inputs[selected_index_] == nil) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    id<MTLTexture> selected_texture = inputs[selected_index_];
    int width = (int)[selected_texture width];
    int height = (int)[selected_texture height];
    MTLPixelFormat format = [selected_texture pixelFormat];
    
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
    
    // 선택된 텍스처 복사
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:selected_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateSwitchNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<SwitchNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Switch, "Switch", "TOP/Misc", NodeFamily::TOP, CreateSwitchNode, "Select one input from multiple");

} // namespace nodes
} // namespace example
