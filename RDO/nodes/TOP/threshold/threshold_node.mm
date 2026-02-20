#include "threshold_node.h"
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

// Metal 셰이더 코드 (Threshold)
static const char* thresholdShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyThreshold(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &thresholdValue [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(gid);

    // Calculate luminance (perceived brightness)
    float luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;

    // Apply threshold: if luminance >= threshold, white (1.0), else black (0.0)
    float result = (luminance >= thresholdValue) ? 1.0 : 0.0;

    // Output as grayscale, preserve alpha
    outputTexture.write(float4(result, result, result, color.a), gid);
}
)";

ThresholdNode::ThresholdNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , threshold_pipeline_(nil)
    , command_queue_(nil)
    , threshold_(0.5f)
    , last_input_texture_(nil)
    , last_threshold_(-1.0f)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int threshold_id = graph.insert_node(Node(NodeType::value, threshold_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(threshold_id, NodeFamily::CHOP, PortDirection::Input, "float", "threshold"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

ThresholdNode::~ThresholdNode()
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

void ThresholdNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool ThresholdNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:thresholdShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> thresholdFunction = [library newFunctionWithName:@"applyThreshold"];

    threshold_pipeline_ = [device_ newComputePipelineStateWithFunction:thresholdFunction error:&error];
    if (error) {
        NSLog(@"Error creating threshold pipeline: %@", error);
        return false;
    }

    return true;
}

void ThresholdNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Threshold");
    ImNodes::EndNodeTitleBar();

    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    // Threshold 파라미터 (연결 가능)
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            threshold_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##threshold", &threshold_, 0.0f, 1.0f);
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // 입력 연결 확인
    bool has_input_connection = false;
    for (const auto& edge : graph.edges())
    {
        if (edge.to == input_ports_[0].id)
        {
            has_input_connection = true;
            break;
        }
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
    else if (has_input_connection)
    {
        ImGui::TextDisabled("Processing...");
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
        const float label_width = ImGui::CalcTextSize(port.name.c_str()).x;
        ImGui::Indent(node_width - label_width);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndOutputAttribute();
    }

    ImNodes::EndNode();
}

void ThresholdNode::RenderInspector()
{
    ImGui::Text("Threshold");
    ImGui::Separator();

    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f, "%.2f");

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
        ImGui::TextDisabled("Connect input to see preview");
    }
}

void ThresholdNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장 (InvalidateCache에서 사용)
    SetLastTexturePool(texture_pool);

    if (inputs.empty() || inputs[0] == nil) {
        // 입력 없음 - 기존 텍스처 정리
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!threshold_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];

    // 텍스처 풀에서 출력 텍스처 할당
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];

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

    // Threshold Pass
    [encoder setComputePipelineState:threshold_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&threshold_ length:sizeof(float) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + threadgroupSize.width - 1) / threadgroupSize.width,
        (height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];

    // 캐싱 업데이트
    last_input_texture_ = input_texture;
    last_threshold_ = threshold_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateThresholdNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ThresholdNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Threshold, "Threshold", "TOP/Filter", NodeFamily::TOP, CreateThresholdNode, "Binary threshold effect");

} // namespace nodes
} // namespace example
