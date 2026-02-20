#include "analyze_node.h"
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

// GPU Analyze 결과 구조체 (CPU side)
struct GPUAnalyzeResult {
    float sum[4];        // R, G, B, A 합계
    float min_value[4];  // 최소값
    float max_value[4];  // 최대값
    int min_pos[2];      // 최소값 위치
    int max_pos[2];      // 최대값 위치
    uint32_t pixel_count;    // 픽셀 수
    uint32_t _padding[3];    // 정렬
};

// Metal 셰이더 (GPU Parallel Reduction)
static const char* analyzeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct AnalyzeResult {
    float4 sum;
    float4 min_value;
    float4 max_value;
    int2 min_pos;
    int2 max_pos;
    uint pixel_count;
    uint _padding[3];
};

// Parallel Reduction for Min/Max/Sum
kernel void analyzeImage(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    device AnalyzeResult* result [[buffer(0)]],
    constant int &analyzeChannel [[buffer(1)]],
    threadgroup float4* local_sum [[threadgroup(0)]],
    threadgroup float4* local_min [[threadgroup(1)]],
    threadgroup float4* local_max [[threadgroup(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    uint width = inputTexture.get_width();
    uint height = inputTexture.get_height();

    // 초기값
    float4 pixel_sum = float4(0.0);
    float4 pixel_min = float4(1.0);
    float4 pixel_max = float4(0.0);

    // 각 스레드가 픽셀 읽기
    if (gid.x < width && gid.y < height) {
        float4 color = inputTexture.read(gid);
        pixel_sum = color;
        pixel_min = color;
        pixel_max = color;
    }

    // Threadgroup shared memory에 저장
    uint local_index = tid.y * tg_size.x + tid.x;
    local_sum[local_index] = pixel_sum;
    local_min[local_index] = pixel_min;
    local_max[local_index] = pixel_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction (단순화된 버전)
    // 실제로는 더 효율적인 reduction 알고리즘 필요
    if (local_index == 0) {
        float4 total_sum = float4(0.0);
        float4 total_min = float4(1.0);
        float4 total_max = float4(0.0);

        for (uint i = 0; i < tg_size.x * tg_size.y; i++) {
            total_sum += local_sum[i];
            total_min = min(total_min, local_min[i]);
            total_max = max(total_max, local_max[i]);
        }

        // Atomic 업데이트 (간단한 버전 - 실제로는 더 정교한 구현 필요)
        result[0].sum = total_sum;  // 실제로는 atomic add 필요
        result[0].min_value = total_min;
        result[0].max_value = total_max;
        result[0].pixel_count = width * height;
    }
}
)";

AnalyzeNode::AnalyzeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , analyze_pipeline_(nil)
    , command_queue_(nil)
    , result_buffer_(nil)
    , mode_(AnalyzeMode::Average)
    , channel_(AnalyzeChannel::Luminance)
{
    output_texture_ = nil;
    result_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0};

    node_id_ = graph.insert_node(Node(NodeType::value));

    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

AnalyzeNode::~AnalyzeNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
        output_texture_ = nil;
    }
}

void AnalyzeNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool AnalyzeNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    // 결과 버퍼 생성 (GPU → CPU 작은 데이터만)
    result_buffer_ = [device_ newBufferWithLength:sizeof(GPUAnalyzeResult)
                                           options:MTLResourceStorageModeShared];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:analyzeShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode options:nil error:&error];
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    analyze_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"analyzeImage"] error:&error];
    if (error) {
        NSLog(@"Error creating analyze pipeline: %@", error);
        return false;
    }

    return true;
}

void AnalyzeNode::Render(Graph<Node>& graph)
{
    const float node_width = 220.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Analyze");
    ImNodes::EndNodeTitleBar();

    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    ImGui::PushItemWidth(150.0f);
    const char* modes[] = { "Average", "Min Pixel", "Max Pixel", "Count Pixels", "Sum" };
    int mode_idx = static_cast<int>(mode_);
    if (ImGui::Combo("Mode", &mode_idx, modes, 5)) {
        mode_ = static_cast<AnalyzeMode>(mode_idx);
    }

    const char* channels[] = { "Luminance", "Red", "Green", "Blue", "Alpha", "RGB Average" };
    int channel_idx = static_cast<int>(channel_);
    if (ImGui::Combo("Channel", &channel_idx, channels, 6)) {
        channel_ = static_cast<AnalyzeChannel>(channel_idx);
    }
    ImGui::PopItemWidth();

    ImGui::Spacing();

    // 분석 결과 표시
    if (result_.value != 0.0f || result_.r != 0.0f) {
        ImGui::Text("Result:");
        if (mode_ == AnalyzeMode::Average || mode_ == AnalyzeMode::Sum) {
            ImGui::Text(" R: %.3f", result_.r);
            ImGui::Text(" G: %.3f", result_.g);
            ImGui::Text(" B: %.3f", result_.b);
        } else if (mode_ == AnalyzeMode::CountPixels) {
            ImGui::Text(" Count: %.0f", result_.value);
        }
    }

    ImGui::Spacing();

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

void AnalyzeNode::RenderInspector()
{
    ImGui::Text("Analyze (GPU Zero-Copy)");
    ImGui::Separator();

    const char* modes[] = { "Average", "Min Pixel", "Max Pixel", "Count Pixels", "Sum" };
    int mode_idx = static_cast<int>(mode_);
    if (ImGui::Combo("Analysis Mode", &mode_idx, modes, 5)) {
        mode_ = static_cast<AnalyzeMode>(mode_idx);
    }

    const char* channels[] = { "Luminance", "Red", "Green", "Blue", "Alpha", "RGB Average" };
    int channel_idx = static_cast<int>(channel_);
    if (ImGui::Combo("Channel", &channel_idx, channels, 6)) {
        channel_ = static_cast<AnalyzeChannel>(channel_idx);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Analysis Results:");

    if (result_.value != 0.0f || result_.r != 0.0f) {
        if (mode_ == AnalyzeMode::Average || mode_ == AnalyzeMode::Sum) {
            ImGui::Text("Red:   %.4f", result_.r);
            ImGui::Text("Green: %.4f", result_.g);
            ImGui::Text("Blue:  %.4f", result_.b);
            ImGui::Text("Alpha: %.4f", result_.a);
        } else if (mode_ == AnalyzeMode::CountPixels) {
            ImGui::Text("Pixel Count: %.0f", result_.value);
        }
    } else {
        ImGui::TextDisabled("No data");
    }

    ImGui::Spacing();
    ImGui::TextDisabled("GPU parallel reduction - zero copy");
}

void AnalyzeNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    SetLastTexturePool(texture_pool);

    if (inputs.empty() || inputs[0] == nil) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }

    id<MTLTexture> input_texture = inputs[0];
    output_texture_ = input_texture;  // Pass-through (zero-copy!)

    if (!analyze_pipeline_ || !result_buffer_) return;

    // GPU에서 분석 수행 (parallel reduction)
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

    [encoder setComputePipelineState:analyze_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setBuffer:result_buffer_ offset:0 atIndex:0];

    int channel_int = static_cast<int>(channel_);
    [encoder setBytes:&channel_int length:sizeof(int) atIndex:1];

    // Threadgroup 설정
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        ([input_texture width] + 15) / 16,
        ([input_texture height] + 15) / 16,
        1
    );

    NSUInteger threadgroupMemoryLength = threadgroupSize.width * threadgroupSize.height * sizeof(float) * 4;
    [encoder setThreadgroupMemoryLength:threadgroupMemoryLength atIndex:0];
    [encoder setThreadgroupMemoryLength:threadgroupMemoryLength atIndex:1];
    [encoder setThreadgroupMemoryLength:threadgroupMemoryLength atIndex:2];

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    // ✅ Zero-Copy: 비동기로 결과 읽기
    // waitUntilCompleted 없음!
    // addCompletedHandler로 나중에 결과 읽기 가능 (현재는 간단하게 pass-through만)

    // 참고: 실제 구현에서는 addCompletedHandler를 사용해서
    // GPU 작업 완료 후 작은 버퍼만 비동기로 읽어야 함
}

std::unique_ptr<NodeBase> CreateAnalyzeNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<AnalyzeNode>(graph, pos, device);
}

REGISTER_NODE(Analyze, "Analyze", "TOP/Analysis", NodeFamily::TOP, CreateAnalyzeNode, "GPU-accelerated image analysis (zero-copy)");

} // namespace nodes
} // namespace example
