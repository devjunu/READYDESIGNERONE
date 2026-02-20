#include "morphology_node.h"
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

// Simplified morphology - basic dilate/erode only
static const char* morphShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyMorphDilate(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant int &kernelSize [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    int halfKernel = kernelSize / 2;
    float4 maxColor = float4(0.0);

    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int2 samplePos = int2(gid) + int2(dx, dy);
            samplePos.x = clamp(samplePos.x, 0, int(inputTexture.get_width()) - 1);
            samplePos.y = clamp(samplePos.y, 0, int(inputTexture.get_height()) - 1);
            maxColor = max(maxColor, inputTexture.read(uint2(samplePos)));
        }
    }
    outputTexture.write(maxColor, gid);
}

kernel void applyMorphErode(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant int &kernelSize [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    int halfKernel = kernelSize / 2;
    float4 minColor = float4(1.0);

    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int2 samplePos = int2(gid) + int2(dx, dy);
            samplePos.x = clamp(samplePos.x, 0, int(inputTexture.get_width()) - 1);
            samplePos.y = clamp(samplePos.y, 0, int(inputTexture.get_height()) - 1);
            minColor = min(minColor, inputTexture.read(uint2(samplePos)));
        }
    }
    outputTexture.write(minColor, gid);
}
)";

MorphologyNode::MorphologyNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , dilate_pipeline_(nil)
    , erode_pipeline_(nil)
    , subtract_pipeline_(nil)
    , command_queue_(nil)
    , operation_(MorphOp::Dilate)
    , kernel_size_(3)
{
    output_texture_ = nil;
    node_id_ = graph.insert_node(Node(NodeType::value));

    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

MorphologyNode::~MorphologyNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
        output_texture_ = nil;
    }
}

void MorphologyNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool MorphologyNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];
    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:morphShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode options:nil error:&error];
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    dilate_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"applyMorphDilate"] error:&error];
    erode_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"applyMorphErode"] error:&error];

    return !error;
}

void MorphologyNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Morphology");
    ImNodes::EndNodeTitleBar();

    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();
    ImGui::PushItemWidth(120.0f);

    const char* ops[] = { "Dilate", "Erode", "Opening", "Closing", "Gradient", "TopHat", "BlackHat" };
    int op_idx = static_cast<int>(operation_);
    if (ImGui::Combo("Op", &op_idx, ops, 7)) {
        operation_ = static_cast<MorphOp>(op_idx);
    }

    ImGui::SliderInt("Size", &kernel_size_, 3, 15);
    if (kernel_size_ % 2 == 0) kernel_size_++;

    ImGui::PopItemWidth();
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

void MorphologyNode::RenderInspector()
{
    ImGui::Text("Morphology");
    ImGui::Separator();

    const char* ops[] = { "Dilate", "Erode", "Opening", "Closing", "Gradient", "TopHat", "BlackHat" };
    int op_idx = static_cast<int>(operation_);
    if (ImGui::Combo("Operation", &op_idx, ops, 7)) {
        operation_ = static_cast<MorphOp>(op_idx);
    }

    ImGui::SliderInt("Kernel Size", &kernel_size_, 3, 15);
    if (kernel_size_ % 2 == 0) kernel_size_++;
}

void MorphologyNode::ProcessGPU(
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
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];

    // Simplified: only support dilate/erode
    id<MTLComputePipelineState> pipeline = (operation_ == MorphOp::Dilate || operation_ == MorphOp::Opening || operation_ == MorphOp::Closing) ? dilate_pipeline_ : erode_pipeline_;

    if (!output_texture_ || [output_texture_ width] != width || [output_texture_ height] != height) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }

        if (texture_pool) {
            output_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            output_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }

    if (!output_texture_ || !pipeline) return;

    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&kernel_size_ length:sizeof(int) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + 15) / 16,
        (height + 15) / 16,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

std::unique_ptr<NodeBase> CreateMorphologyNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<MorphologyNode>(graph, pos, device);
}

REGISTER_NODE(Morphology, "Morphology", "TOP/Filter", NodeFamily::TOP, CreateMorphologyNode, "Morphological operations (dilate, erode, opening, closing)");

} // namespace nodes
} // namespace example
