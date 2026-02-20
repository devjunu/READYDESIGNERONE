#include "difference_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Metal 셰이더 코드 (Difference)
static const char* differenceShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyDifference(
    texture2d<float, access::read> inputTexture1 [[texture(0)]],
    texture2d<float, access::read> inputTexture2 [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &amplify [[buffer(0)]],
    constant bool &useAbsolute [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float4 color1 = inputTexture1.read(gid);
    float4 color2 = inputTexture2.read(gid);

    // Calculate difference
    float4 diff = color1 - color2;

    // Apply absolute value if requested
    if (useAbsolute) {
        diff = abs(diff);
    }

    // Amplify difference
    diff = diff * amplify;

    // Clamp to valid range
    diff = clamp(diff, 0.0, 1.0);

    // Preserve alpha from first input
    diff.a = color1.a;

    outputTexture.write(diff, gid);
}
)";

DifferenceNode::DifferenceNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , difference_pipeline_(nil)
    , command_queue_(nil)
    , amplify_(1.0f)
    , absolute_(true)
    , last_input1_texture_(nil)
    , last_input2_texture_(nil)
    , last_amplify_(-1.0f)
    , last_absolute_(false)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input1_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int input2_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input1_id, NodeFamily::TOP, PortDirection::Input, "texture", "input1"));
    AddInputPort(Port(input2_id, NodeFamily::TOP, PortDirection::Input, "texture", "input2"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

DifferenceNode::~DifferenceNode()
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

void DifferenceNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool DifferenceNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:differenceShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> differenceFunction = [library newFunctionWithName:@"applyDifference"];

    difference_pipeline_ = [device_ newComputePipelineStateWithFunction:differenceFunction error:&error];
    if (error) {
        NSLog(@"Error creating difference pipeline: %@", error);
        return false;
    }

    return true;
}

void DifferenceNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Difference");
    ImNodes::EndNodeTitleBar();

    // 입력 포트 1
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    // 입력 포트 2
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Amplify 파라미터
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Amplify", &amplify_, 1.0f, 10.0f);
    ImGui::Checkbox("Absolute", &absolute_);
    ImGui::PopItemWidth();

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

void DifferenceNode::RenderInspector()
{
    ImGui::Text("Difference");
    ImGui::Separator();

    ImGui::SliderFloat("Amplify", &amplify_, 1.0f, 10.0f, "%.1f");
    ImGui::Checkbox("Absolute Value", &absolute_);

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
        ImGui::TextDisabled("Connect both inputs");
    }
}

void DifferenceNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장 (InvalidateCache에서 사용)
    SetLastTexturePool(texture_pool);

    if (inputs.size() < 2 || inputs[0] == nil || inputs[1] == nil) {
        // 입력 부족 - 기존 텍스처 정리
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!difference_pipeline_) return;

    id<MTLTexture> input_texture1 = inputs[0];
    id<MTLTexture> input_texture2 = inputs[1];

    // 텍스처 풀에서 출력 텍스처 할당
    NSUInteger width = [input_texture1 width];
    NSUInteger height = [input_texture1 height];
    NSUInteger pixelFormat = [input_texture1 pixelFormat];

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

    // Difference Pass
    [encoder setComputePipelineState:difference_pipeline_];
    [encoder setTexture:input_texture1 atIndex:0];
    [encoder setTexture:input_texture2 atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&amplify_ length:sizeof(float) atIndex:0];
    [encoder setBytes:&absolute_ length:sizeof(bool) atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + threadgroupSize.width - 1) / threadgroupSize.width,
        (height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];

    // 캐싱 업데이트
    last_input1_texture_ = input_texture1;
    last_input2_texture_ = input_texture2;
    last_amplify_ = amplify_;
    last_absolute_ = absolute_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateDifferenceNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<DifferenceNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Difference, "Difference", "TOP/Composite", NodeFamily::TOP, CreateDifferenceNode, "Calculate difference between two images for motion detection");

} // namespace nodes
} // namespace example
