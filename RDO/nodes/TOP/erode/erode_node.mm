#include "erode_node.h"
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

// Metal 셰이더 코드 (Erode)
static const char* erodeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyErode(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant int &kernelSize [[buffer(0)]],
    constant int &kernelShape [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    int halfKernel = kernelSize / 2;
    float4 minColor = float4(1.0, 1.0, 1.0, 1.0);

    // Iterate over kernel
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            // Check kernel shape
            bool inKernel = true;

            if (kernelShape == 1) { // Ellipse
                float dist = sqrt(float(dx*dx + dy*dy));
                inKernel = (dist <= float(halfKernel));
            } else if (kernelShape == 2) { // Cross
                inKernel = (dx == 0 || dy == 0);
            }
            // kernelShape == 0 (Rect) is always true

            if (!inKernel) continue;

            int2 samplePos = int2(gid) + int2(dx, dy);

            // Clamp to texture bounds
            samplePos.x = clamp(samplePos.x, 0, int(inputTexture.get_width()) - 1);
            samplePos.y = clamp(samplePos.y, 0, int(inputTexture.get_height()) - 1);

            float4 sampleColor = inputTexture.read(uint2(samplePos));

            // Take minimum value (erosion)
            minColor = min(minColor, sampleColor);
        }
    }

    outputTexture.write(minColor, gid);
}
)";

ErodeNode::ErodeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , erode_pipeline_(nil)
    , command_queue_(nil)
    , kernel_size_(3)
    , iterations_(1)
    , kernel_shape_(ErosionKernelShape::Rect)
    , last_input_texture_(nil)
    , last_kernel_size_(-1)
    , last_iterations_(-1)
    , last_kernel_shape_(ErosionKernelShape::Rect)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

ErodeNode::~ErodeNode()
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

void ErodeNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool ErodeNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:erodeShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> erodeFunction = [library newFunctionWithName:@"applyErode"];

    erode_pipeline_ = [device_ newComputePipelineStateWithFunction:erodeFunction error:&error];
    if (error) {
        NSLog(@"Error creating erode pipeline: %@", error);
        return false;
    }

    return true;
}

void ErodeNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Erode");
    ImNodes::EndNodeTitleBar();

    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Kernel Size 파라미터
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderInt("Size", &kernel_size_, 3, 15);
    if (kernel_size_ % 2 == 0) kernel_size_++; // 홀수만 허용

    ImGui::SliderInt("Iterations", &iterations_, 1, 5);

    // Kernel Shape
    const char* shapes[] = { "Rect", "Ellipse", "Cross" };
    int shape_idx = static_cast<int>(kernel_shape_);
    if (ImGui::Combo("Shape", &shape_idx, shapes, 3)) {
        kernel_shape_ = static_cast<ErosionKernelShape>(shape_idx);
    }
    ImGui::PopItemWidth();

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

void ErodeNode::RenderInspector()
{
    ImGui::Text("Erode");
    ImGui::Separator();

    ImGui::SliderInt("Kernel Size", &kernel_size_, 3, 15);
    if (kernel_size_ % 2 == 0) kernel_size_++;

    ImGui::SliderInt("Iterations", &iterations_, 1, 5);

    const char* shapes[] = { "Rect", "Ellipse", "Cross" };
    int shape_idx = static_cast<int>(kernel_shape_);
    if (ImGui::Combo("Kernel Shape", &shape_idx, shapes, 3)) {
        kernel_shape_ = static_cast<ErosionKernelShape>(shape_idx);
    }

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

void ErodeNode::ProcessGPU(
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
    if (!erode_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];

    // 텍스처 풀에서 출력 텍스처 할당
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];

    // 다중 반복을 위한 임시 텍스처
    id<MTLTexture> temp_texture = nil;
    id<MTLTexture> current_input = input_texture;

    for (int iter = 0; iter < iterations_; iter++) {
        // 출력 텍스처 할당
        id<MTLTexture> current_output;

        if (iter == iterations_ - 1) {
            // 마지막 반복: output_texture_ 사용
            if (!output_texture_ || [output_texture_ width] != width || [output_texture_ height] != height) {
                if (output_texture_ && texture_pool) {
                    texture_pool->release_texture(output_texture_);
                }

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
            current_output = output_texture_;
        } else {
            // 중간 반복: 임시 텍스처 사용
            if (!temp_texture) {
                if (texture_pool) {
                    temp_texture = texture_pool->acquire_texture(width, height, pixelFormat);
                } else {
                    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                                     width:width
                                                    height:height
                                                 mipmapped:NO];
                    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                    temp_texture = [device_ newTextureWithDescriptor:descriptor];
                }
            }
            current_output = temp_texture;
        }

        if (!current_output) {
            if (temp_texture && texture_pool) {
                texture_pool->release_texture(temp_texture);
            }
            return;
        }

        // Compute Encoder 생성
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        // Erode Pass
        [encoder setComputePipelineState:erode_pipeline_];
        [encoder setTexture:current_input atIndex:0];
        [encoder setTexture:current_output atIndex:1];

        int shape_int = static_cast<int>(kernel_shape_);
        [encoder setBytes:&kernel_size_ length:sizeof(int) atIndex:0];
        [encoder setBytes:&shape_int length:sizeof(int) atIndex:1];

        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroups = MTLSizeMake(
            (width + threadgroupSize.width - 1) / threadgroupSize.width,
            (height + threadgroupSize.height - 1) / threadgroupSize.height,
            1
        );
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];

        // 다음 반복을 위해 입력 교체
        current_input = current_output;
    }

    // 임시 텍스처 정리
    if (temp_texture && texture_pool) {
        texture_pool->release_texture(temp_texture);
    }

    // 캐싱 업데이트
    last_input_texture_ = input_texture;
    last_kernel_size_ = kernel_size_;
    last_iterations_ = iterations_;
    last_kernel_shape_ = kernel_shape_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateErodeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ErodeNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Erode, "Erode", "TOP/Filter", NodeFamily::TOP, CreateErodeNode, "Erode pixels (morphological operation)");

} // namespace nodes
} // namespace example
