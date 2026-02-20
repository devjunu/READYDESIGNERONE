#include "resolution_node.h"
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

// Metal 셰이더 코드 (Resize)
static const char* resizeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyResize(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    uint inputWidth = inputTexture.get_width();
    uint inputHeight = inputTexture.get_height();
    uint outputWidth = outputTexture.get_width();
    uint outputHeight = outputTexture.get_height();

    // Calculate normalized coordinates
    float2 uv = float2(gid) / float2(outputWidth, outputHeight);

    // Map to input texture coordinates
    float2 inputCoord = uv * float2(inputWidth, inputHeight);

    // Bilinear interpolation
    if (inputCoord.x >= 0 && inputCoord.x < inputWidth - 1 &&
        inputCoord.y >= 0 && inputCoord.y < inputHeight - 1)
    {
        uint2 coord00 = uint2(floor(inputCoord.x), floor(inputCoord.y));
        uint2 coord10 = uint2(coord00.x + 1, coord00.y);
        uint2 coord01 = uint2(coord00.x, coord00.y + 1);
        uint2 coord11 = uint2(coord00.x + 1, coord00.y + 1);

        float2 frac = fract(inputCoord);

        float4 color00 = inputTexture.read(coord00);
        float4 color10 = inputTexture.read(coord10);
        float4 color01 = inputTexture.read(coord01);
        float4 color11 = inputTexture.read(coord11);

        float4 color0 = mix(color00, color10, frac.x);
        float4 color1 = mix(color01, color11, frac.x);
        float4 result = mix(color0, color1, frac.y);

        outputTexture.write(result, gid);
    }
    else
    {
        // Edge case: sample nearest
        uint2 coord = uint2(clamp(int(inputCoord.x), 0, int(inputWidth - 1)),
                            clamp(int(inputCoord.y), 0, int(inputHeight - 1)));
        outputTexture.write(inputTexture.read(coord), gid);
    }
}
)";

ResolutionNode::ResolutionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , resize_pipeline_(nil)
    , command_queue_(nil)
    , target_width_(1920)   // 128의 배수: 1920 = 128 * 15
    , target_height_(1024)  // 128의 배수: 1024 = 128 * 8
    , last_input_texture_(nil)
    , last_target_width_(-1)
    , last_target_height_(-1)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int width_id = graph.insert_node(Node(NodeType::value, (float)target_width_));
    int height_id = graph.insert_node(Node(NodeType::value, (float)target_height_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(width_id, NodeFamily::CHOP, PortDirection::Input, "float", "width"));
    AddInputPort(Port(height_id, NodeFamily::CHOP, PortDirection::Input, "float", "height"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

ResolutionNode::~ResolutionNode()
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

void ResolutionNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool ResolutionNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:resizeShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> resizeFunction = [library newFunctionWithName:@"applyResize"];

    resize_pipeline_ = [device_ newComputePipelineStateWithFunction:resizeFunction error:&error];
    if (error) {
        NSLog(@"Error creating resize pipeline: %@", error);
        return false;
    }

    return true;
}

void ResolutionNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Resolution");
    ImNodes::EndNodeTitleBar();

    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Width 파라미터
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            // 값을 128픽셀 단위로 스냅 (메모리 최적화 강화)
            int new_width = (int)(graph.node(port.id).value + 0.5f);
            new_width = ((new_width + 64) / 128) * 128;  // 128픽셀 단위로 반올림
            new_width = std::max(128, std::min(new_width, 8192));

            // 크기가 실제로 변경된 경우만 업데이트
            if (new_width != target_width_) {
                target_width_ = new_width;
            }
        }

        // SliderInt 사용으로 변경 (정확한 제어)
        ImGui::Text("W:");
        ImGui::SameLine();
        int width_int = target_width_;
        if (ImGui::SliderInt("##width", &width_int, 128, 4096)) {
            // 128픽셀 단위로 스냅
            target_width_ = ((width_int + 64) / 128) * 128;
            target_width_ = std::max(128, std::min(target_width_, 8192));
        }
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Height 파라미터
    {
        const Port& port = input_ports_[2];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            // 값을 128픽셀 단위로 스냅 (메모리 최적화 강화)
            int new_height = (int)(graph.node(port.id).value + 0.5f);
            new_height = ((new_height + 64) / 128) * 128;  // 128픽셀 단위로 반올림
            new_height = std::max(128, std::min(new_height, 8192));

            // 크기가 실제로 변경된 경우만 업데이트
            if (new_height != target_height_) {
                target_height_ = new_height;
            }
        }

        // SliderInt 사용으로 변경 (정확한 제어)
        ImGui::Text("H:");
        ImGui::SameLine();
        int height_int = target_height_;
        if (ImGui::SliderInt("##height", &height_int, 128, 4096)) {
            // 128픽셀 단위로 스냅
            target_height_ = ((height_int + 64) / 128) * 128;
            target_height_ = std::max(128, std::min(target_height_, 8192));
        }
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

void ResolutionNode::RenderInspector()
{
    ImGui::Text("Resolution");
    ImGui::Separator();

    int width_int = target_width_;
    int height_int = target_height_;

    if (ImGui::SliderInt("Width", &width_int, 128, 4096)) {
        // 128픽셀 단위로 스냅하여 메모리 최적화
        target_width_ = ((width_int + 64) / 128) * 128;
        target_width_ = std::max(128, std::min(target_width_, 8192));
    }
    if (ImGui::SliderInt("Height", &height_int, 128, 4096)) {
        // 128픽셀 단위로 스냅하여 메모리 최적화
        target_height_ = ((height_int + 64) / 128) * 128;
        target_height_ = std::max(128, std::min(target_height_, 8192));
    }

    ImGui::Text("Snapped to 128px grid for memory efficiency");

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
        ImGui::TextDisabled("Connect input to see preview");
    }
}

void ResolutionNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);

    if (inputs.empty() || inputs[0] == nil) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!resize_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];
    NSUInteger pixelFormat = [input_texture pixelFormat];

    // 타겟 크기를 128픽셀 단위로 스냅 (메모리 최적화 강화)
    int safe_width = ((target_width_ + 64) / 128) * 128;
    int safe_height = ((target_height_ + 64) / 128) * 128;
    safe_width = std::max(128, std::min(safe_width, 8192));
    safe_height = std::max(128, std::min(safe_height, 8192));

    // 캐싱 체크: 크기가 변경되지 않았으면 재사용
    bool size_changed = (last_target_width_ != safe_width || last_target_height_ != safe_height);

    // 기존 텍스처 해제 (크기가 변경된 경우만)
    // IMPORTANT: ResolutionNode는 TexturePool을 사용하지 않고 직접 관리 (메모리 누수 방지)
    if (output_texture_ && size_changed)
    {
        // 즉시 해제 (풀에 보관하지 않음)
        output_texture_ = nil;
    }

    // 필요한 경우 새 텍스처 할당 (크기가 변경되었거나 텍스처가 없을 때만)
    if (!output_texture_)
    {
        // TexturePool 대신 직접 생성 (메모리 제어 강화)
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                         width:safe_width
                                        height:safe_height
                                     mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        descriptor.storageMode = MTLStorageModePrivate;
        output_texture_ = [device_ newTextureWithDescriptor:descriptor];

        // 새 텍스처 할당 시에만 캐싱 업데이트
        last_target_width_ = safe_width;
        last_target_height_ = safe_height;
    }

    if (!output_texture_) {
        return;
    }

    // Compute Encoder 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

    // Resize Pass
    [encoder setComputePipelineState:resize_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (safe_width + threadgroupSize.width - 1) / threadgroupSize.width,
        (safe_height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];

    // 캐싱 업데이트 (입력 텍스처만)
    last_input_texture_ = input_texture;
}

// RenderContext를 받는 새로운 ProcessGPU (Preview/Final 모드 지원)
void ResolutionNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool,
    const RenderContext& context)
{
    // Resolution 노드는 명시적 해상도 제어용
    // Preview 설정에 관계없이 항상 사용자 지정 해상도로 출력
    // (Preview scaling은 display 시점에서만 적용됨)
    ProcessGPU(inputs, cmd_buffer, texture_pool);
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateResolutionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ResolutionNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Resolution, "Resolution", "TOP/Transform", NodeFamily::TOP, CreateResolutionNode, "Resize texture to target resolution");

} // namespace nodes
} // namespace example
