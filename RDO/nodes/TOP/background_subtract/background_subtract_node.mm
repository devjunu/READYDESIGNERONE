#include "background_subtract_node.h"
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

// Metal 셰이더 코드 (Background Subtract)
static const char* backgroundSubtractShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyBackgroundSubtract(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::read> backgroundTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float4 current = inputTexture.read(gid);
    float4 background = backgroundTexture.read(gid);

    // Calculate absolute difference
    float3 diff = abs(current.rgb - background.rgb);
    float diffMagnitude = (diff.r + diff.g + diff.b) / 3.0;

    // If difference is above threshold, keep original color
    // Otherwise, make it transparent
    float alpha = (diffMagnitude > threshold) ? 1.0 : 0.0;

    // Smooth alpha transition near threshold
    float smoothRange = threshold * 0.2;
    if (diffMagnitude > threshold - smoothRange && diffMagnitude < threshold + smoothRange) {
        alpha = smoothstep(threshold - smoothRange, threshold + smoothRange, diffMagnitude);
    }

    float4 result = float4(current.rgb, current.a * alpha);
    outputTexture.write(result, gid);
}
)";

BackgroundSubtractNode::BackgroundSubtractNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , background_subtract_pipeline_(nil)
    , command_queue_(nil)
    , threshold_(0.1f)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int background_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int threshold_id = graph.insert_node(Node(NodeType::value, threshold_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(background_id, NodeFamily::TOP, PortDirection::Input, "texture", "background"));
    AddInputPort(Port(threshold_id, NodeFamily::CHOP, PortDirection::Input, "float", "threshold"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

BackgroundSubtractNode::~BackgroundSubtractNode()
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

void BackgroundSubtractNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool BackgroundSubtractNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:backgroundSubtractShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> backgroundSubtractFunction = [library newFunctionWithName:@"applyBackgroundSubtract"];

    background_subtract_pipeline_ = [device_ newComputePipelineStateWithFunction:backgroundSubtractFunction error:&error];
    if (error) {
        NSLog(@"Error creating background subtract pipeline: %@", error);
        return false;
    }

    return true;
}

void BackgroundSubtractNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Background Subtract");
    ImNodes::EndNodeTitleBar();

    // 입력 포트 (현재 프레임)
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // 배경 포트
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Threshold 파라미터
    {
        const Port& port = input_ports_[2];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (!is_connected) {
            ImGui::Text("Threshold");
            if (ImGui::SliderFloat("##threshold", &threshold_, 0.0f, 1.0f)) {
                graph.node(port.id).value = threshold_;
            }
        } else {
            threshold_ = graph.node(port.id).value;
            ImGui::Text("Threshold: %.2f", threshold_);
        }

        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // 입력 연결 확인
    bool has_input = false;
    bool has_background = false;
    for (const auto& edge : graph.edges())
    {
        if (edge.to == input_ports_[0].id) has_input = true;
        if (edge.to == input_ports_[1].id) has_background = true;
    }

    // 프리뷰 표시
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
    else if (has_input && has_background)
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

void BackgroundSubtractNode::RenderInspector()
{
    ImGui::Text("Background Subtract");
    ImGui::Separator();

    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f);

    ImGui::Spacing();
    ImGui::TextWrapped("Subtracts background from input. Pixels with difference below threshold become transparent.");

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
            float preview_width = ImGui::GetContentRegionAvail().x;
            float aspect_ratio = static_cast<float>(height) / static_cast<float>(width);
            float preview_height = preview_width * aspect_ratio;

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

void BackgroundSubtractNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);

    // 이 노드는 2개의 입력이 필요함
    if (inputs.size() < 2 || inputs[0] == nil || inputs[1] == nil) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!background_subtract_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];
    id<MTLTexture> background_texture = inputs[1];
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];

    // 출력 텍스처 할당 (필요한 경우만)
    bool need_new_texture = false;
    if (output_texture_ == nil ||
        [output_texture_ width] != width ||
        [output_texture_ height] != height)
    {
        need_new_texture = true;
    }

    if (need_new_texture && texture_pool)
    {
        if (output_texture_ != nil)
        {
            texture_pool->release_texture(output_texture_);
        }

        output_texture_ = texture_pool->acquire_texture(width, height, (unsigned long)pixelFormat);
    }
    else if (need_new_texture)
    {
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                         width:width
                                        height:height
                                     mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        descriptor.storageMode = MTLStorageModePrivate;

        output_texture_ = [device_ newTextureWithDescriptor:descriptor];
    }

    if (!output_texture_) {
        return;
    }

    // Compute Encoder 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

    // Background Subtract Pass
    [encoder setComputePipelineState:background_subtract_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:background_texture atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&threshold_ length:sizeof(float) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + threadgroupSize.width - 1) / threadgroupSize.width,
        (height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateBackgroundSubtractNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<BackgroundSubtractNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(BackgroundSubtract, "Background Subtract", "TOP/Composite", NodeFamily::TOP, CreateBackgroundSubtractNode, "Subtract background from input texture");

} // namespace nodes
} // namespace example
