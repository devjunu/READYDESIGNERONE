#include "composite_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>
#include <cmath>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Metal 셰이더 코드 (Composite)
static const char* compositeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void compositeImages(
    texture2d<float, access::read> input1Texture [[texture(0)]],
    texture2d<float, access::read> input2Texture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &opacity [[buffer(0)]],
    constant int &mode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float4 color1 = input1Texture.read(gid);
    float4 color2 = input2Texture.read(gid);

    float4 result;

    // Mode 0: Normal (Alpha blending)
    if (mode == 0) {
        result = mix(color1, color2, opacity * color2.a);
    }
    // Mode 1: Add
    else if (mode == 1) {
        result = color1 + color2 * opacity;
        result = clamp(result, 0.0, 1.0);
    }
    // Mode 2: Multiply
    else if (mode == 2) {
        result = mix(color1, color1 * color2, opacity);
    }
    // Mode 3: Screen
    else if (mode == 3) {
        float4 screen = 1.0 - (1.0 - color1) * (1.0 - color2);
        result = mix(color1, screen, opacity);
    }
    // Mode 4: Overlay
    else if (mode == 4) {
        float4 overlay;
        for (int i = 0; i < 3; i++) {
            if (color1[i] < 0.5) {
                overlay[i] = 2.0 * color1[i] * color2[i];
            } else {
                overlay[i] = 1.0 - 2.0 * (1.0 - color1[i]) * (1.0 - color2[i]);
            }
        }
        overlay.a = color1.a;
        result = mix(color1, overlay, opacity);
    }
    else {
        result = color1;
    }

    result.a = color1.a;
    outputTexture.write(result, gid);
}
)";

CompositeNode::CompositeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , composite_pipeline_(nil)
    , command_queue_(nil)
    , opacity_(1.0f)
    , mode_(CompositeMode::Normal)
    , last_input1_texture_(nil)
    , last_input2_texture_(nil)
    , last_opacity_(-1.0f)
    , last_mode_(CompositeMode::Normal)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input1_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int input2_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int opacity_id = graph.insert_node(Node(NodeType::value, opacity_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input1_id, NodeFamily::TOP, PortDirection::Input, "texture", "input1"));
    AddInputPort(Port(input2_id, NodeFamily::TOP, PortDirection::Input, "texture", "input2"));
    AddInputPort(Port(opacity_id, NodeFamily::CHOP, PortDirection::Input, "float", "opacity"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

CompositeNode::~CompositeNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
            output_texture_ = nil;
        }
    } else {
        // TexturePool 참조 없으면 ARC가 자동 관리
        output_texture_ = nil;
    }
}

void CompositeNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool CompositeNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:compositeShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> compositeFunction = [library newFunctionWithName:@"compositeImages"];

    composite_pipeline_ = [device_ newComputePipelineStateWithFunction:compositeFunction error:&error];
    if (error) {
        NSLog(@"Error creating composite pipeline: %@", error);
        return false;
    }

    return true;
}

void CompositeNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Composite");
    ImNodes::EndNodeTitleBar();

    // 첫 번째 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // 두 번째 입력 포트
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Composite Mode 드롭다운
    {
        ImGui::TextUnformatted("Mode");
        ImGui::SameLine();
        ImGui::PushItemWidth(120.0f);
        const char* mode_names[] = { "Normal", "Add", "Multiply", "Screen", "Overlay" };
        int current_mode = static_cast<int>(mode_);
        if (ImGui::Combo("##composite_mode", &current_mode, mode_names, 5))
        {
            mode_ = static_cast<CompositeMode>(current_mode);
        }
        ImGui::PopItemWidth();
    }

    ImGui::Spacing();

    // Opacity 파라미터 (연결 가능)
    {
        const Port& port = input_ports_[2];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            opacity_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##opacity", &opacity_, 0.0f, 1.0f);
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // 입력 연결 확인
    bool has_input1 = false;
    bool has_input2 = false;
    for (const auto& edge : graph.edges())
    {
        if (edge.to == input_ports_[0].id) has_input1 = true;
        if (edge.to == input_ports_[1].id) has_input2 = true;
    }

    // 프리뷰 표시 (노드 내부)
    if (output_texture_ != nil)
    {
        NSUInteger width = [output_texture_ width];
        NSUInteger height = [output_texture_ height];

        if (width > 0 && height > 0)
        {
            float preview_width = node_width - 20.0f;
            float aspect_ratio = static_cast<float>(height) / static_cast<float>(width);
            float preview_height = preview_width * aspect_ratio;

            const float max_preview_height = 120.0f;
            if (preview_height > max_preview_height)
            {
                preview_height = max_preview_height;
                preview_width = preview_height / aspect_ratio;
            }

            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    }
    else if (has_input1 && has_input2)
    {
        ImGui::TextDisabled("Processing...");
    }
    else
    {
        ImGui::TextDisabled("Connect both inputs");
    }

    ImGui::Spacing();

    // 출력 포트
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

void CompositeNode::RenderInspector()
{
    ImGui::Text("Composite");
    ImGui::Separator();

    // Composite Mode 드롭다운
    ImGui::Text("Mode");
    const char* mode_names[] = { "Normal", "Add", "Multiply", "Screen", "Overlay" };
    int current_mode = static_cast<int>(mode_);
    if (ImGui::Combo("##composite_mode_inspector", &current_mode, mode_names, 5))
    {
        mode_ = static_cast<CompositeMode>(current_mode);
    }

    ImGui::SliderFloat("Opacity", &opacity_, 0.0f, 1.0f, "%.2f");

    ImGui::Spacing();

    // 텍스처 프리뷰 (Inspector)
    if (output_texture_ != nil) {
        ImGui::Text("Output: %lux%lu",
                    [output_texture_ width],
                    [output_texture_ height]);

        ImGui::Spacing();

        NSUInteger width = [output_texture_ width];
        NSUInteger height = [output_texture_ height];

        if (width > 0 && height > 0)
        {
            // Metal 텍스처를 ImGui 텍스처로 표시
            float preview_width = ImGui::GetContentRegionAvail().x;
            float aspect_ratio = static_cast<float>(height) / static_cast<float>(width);
            float preview_height = preview_width * aspect_ratio;

            // 최대 높이 제한
            if (preview_height > 400.0f) {
                preview_height = 400.0f;
                preview_width = preview_height / aspect_ratio;
            }

            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    } else {
        ImGui::TextDisabled("No output texture");
        ImGui::TextDisabled("Connect both inputs to see preview");
    }
}

void CompositeNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장 (InvalidateCache에서 사용)
    SetLastTexturePool(texture_pool);

    if (inputs.size() < 2 || inputs[0] == nil || inputs[1] == nil) {
        // 입력 없음 - 기존 텍스처 정리
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!composite_pipeline_) return;

    id<MTLTexture> input1_texture = inputs[0];
    id<MTLTexture> input2_texture = inputs[1];

    // 텍스처 풀에서 출력 텍스처 할당
    NSUInteger width = [input1_texture width];
    NSUInteger height = [input1_texture height];
    NSUInteger pixelFormat = [input1_texture pixelFormat];

    // 기존 텍스처 해제 (크기가 변경된 경우만)
    if (output_texture_)
    {
        if ([output_texture_ width] != width || [output_texture_ height] != height)
        {
            if (texture_pool) {
                texture_pool->release_texture(output_texture_);
            }
            output_texture_ = nil;
        }
    }

    // 필요한 경우 새 텍스처 할당
    if (!output_texture_)
    {
        if (texture_pool) {
            output_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width
                                            height:height
                                         mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            output_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }

    if (!output_texture_) {
        return;
    }

    // Compute Encoder 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

    // Composite Pass
    [encoder setComputePipelineState:composite_pipeline_];
    [encoder setTexture:input1_texture atIndex:0];
    [encoder setTexture:input2_texture atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&opacity_ length:sizeof(float) atIndex:0];
    int mode_int = static_cast<int>(mode_);
    [encoder setBytes:&mode_int length:sizeof(int) atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + threadgroupSize.width - 1) / threadgroupSize.width,
        (height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];

    // 캐싱 업데이트
    last_input1_texture_ = input1_texture;
    last_input2_texture_ = input2_texture;
    last_opacity_ = opacity_;
    last_mode_ = mode_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateCompositeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<CompositeNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Composite, "Composite", "TOP/Composite", NodeFamily::TOP, CreateCompositeNode, "Composite two images with blend modes");

} // namespace nodes
} // namespace example
