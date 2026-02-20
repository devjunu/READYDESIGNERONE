#include "optical_flow_node.h"
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

// Simplified optical flow using frame differencing
static const char* flowShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void computeOpticalFlow(
    texture2d<float, access::read> currentFrame [[texture(0)]],
    texture2d<float, access::read> previousFrame [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &amplify [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    float4 current = currentFrame.read(gid);
    float4 previous = previousFrame.read(gid);

    // Simple frame difference for motion detection
    float4 diff = abs(current - previous) * amplify;

    // X motion in R, Y motion in G (simplified)
    float motion = length(diff.rgb);
    float4 output = float4(motion, motion * 0.5, 0.0, 1.0);

    outputTexture.write(clamp(output, 0.0, 1.0), gid);
}
)";

OpticalFlowNode::OpticalFlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , flow_pipeline_(nil)
    , command_queue_(nil)
    , amplify_(2.0f)
    , previous_frame_(nil)
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

OpticalFlowNode::~OpticalFlowNode()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
            output_texture_ = nil;
        }
        if (previous_frame_) {
            last_texture_pool_->release_texture(previous_frame_);
            previous_frame_ = nil;
        }
    }
}

void OpticalFlowNode::InvalidateCache()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        if (previous_frame_) {
            last_texture_pool_->release_texture(previous_frame_);
        }
    }
    output_texture_ = nil;
    previous_frame_ = nil;
}

bool OpticalFlowNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];
    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:flowShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode options:nil error:&error];
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    flow_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"computeOpticalFlow"] error:&error];
    return !error;
}

void OpticalFlowNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Optical Flow");
    ImNodes::EndNodeTitleBar();

    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Amplify", &amplify_, 1.0f, 10.0f);
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

void OpticalFlowNode::RenderInspector()
{
    ImGui::Text("Optical Flow");
    ImGui::Separator();
    ImGui::SliderFloat("Amplify", &amplify_, 1.0f, 10.0f, "%.1f");
    ImGui::Text("Note: Simplified frame-difference based");
}

void OpticalFlowNode::ProcessGPU(
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

    // First frame: just store it
    if (!previous_frame_) {
        if (texture_pool) {
            previous_frame_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            previous_frame_ = [device_ newTextureWithDescriptor:descriptor];
        }

        // Copy current to previous
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:input_texture toTexture:previous_frame_];
        [blitEncoder endEncoding];

        output_texture_ = input_texture;
        return;
    }

    // Allocate output
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

    if (!output_texture_ || !flow_pipeline_) return;

    // Compute optical flow
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:flow_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:previous_frame_ atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&amplify_ length:sizeof(float) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    // Copy current to previous for next frame
    id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
    [blitEncoder copyFromTexture:input_texture toTexture:previous_frame_];
    [blitEncoder endEncoding];
}

std::unique_ptr<NodeBase> CreateOpticalFlowNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<OpticalFlowNode>(graph, pos, device);
}

REGISTER_NODE(OpticalFlow, "Optical Flow", "TOP/Analysis", NodeFamily::TOP, CreateOpticalFlowNode, "Detect motion flow patterns (simplified)");

} // namespace nodes
} // namespace example
