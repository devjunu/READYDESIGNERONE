#include "crop_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>

#import <Metal/Metal.h>
#import <simd/simd.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

static const char* cropShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct CropParams {
    float left;
    float right;
    float top;
    float bottom;
};

kernel void cropTexture(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant CropParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int inputWidth = inputTexture.get_width();
    int inputHeight = inputTexture.get_height();
    
    // Calculate crop region
    int cropLeft = int(params.left * inputWidth);
    int cropTop = int(params.top * inputHeight);
    
    // Map output coordinate to input coordinate
    uint2 inputCoord;
    inputCoord.x = gid.x + cropLeft;
    inputCoord.y = gid.y + cropTop;
    
    // Clamp to input bounds
    inputCoord.x = clamp(inputCoord.x, 0u, uint(inputWidth - 1));
    inputCoord.y = clamp(inputCoord.y, 0u, uint(inputHeight - 1));
    
    float4 color = inputTexture.read(inputCoord);
    outputTexture.write(color, gid);
}
)";

CropNode::CropNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil),
      left_(0.0f), right_(0.0f), top_(0.0f), bottom_(0.0f)
{
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

CropNode::~CropNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void CropNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool CropNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(cropShaderSource)
                                                   options:nil error:&error];
    
    if (!library) {
        NSLog(@"Crop shader compilation failed: %@", error);
        return false;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"cropTexture"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void CropNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Crop");
    ImNodes::EndNodeTitleBar();
    
    const Port& port = input_ports_[0];
    ImNodes::BeginInputAttribute(port.id);
    ImGui::TextUnformatted("input");
    ImNodes::EndInputAttribute();
    
    ImGui::Spacing();
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Left", &left_, 0.0f, 1.0f);
    ImGui::SliderFloat("Right", &right_, 0.0f, 1.0f);
    ImGui::SliderFloat("Top", &top_, 0.0f, 1.0f);
    ImGui::SliderFloat("Bottom", &bottom_, 0.0f, 1.0f);
    ImGui::PopItemWidth();
    ImGui::Spacing();
    
    if (output_texture_ != nil) {
        int width = (int)[output_texture_ width];
        int height = (int)[output_texture_ height];
        if (width > 0 && height > 0) {
            float preview_width = 180.0f;
            float aspect_ratio = (float)height / (float)width;
            float preview_height = preview_width * aspect_ratio;
            if (preview_height > 120.0f) {
                preview_height = 120.0f;
                preview_width = preview_height / aspect_ratio;
            }
            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    } else {
        ImGui::TextDisabled("No input connected");
    }
    
    ImGui::Spacing();
    
    const Port& out_port = output_ports_[0];
    ImNodes::BeginOutputAttribute(out_port.id);
    const float label_width = ImGui::CalcTextSize("output").x;
    ImGui::Indent(200.0f - label_width);
    ImGui::TextUnformatted("output");
    ImNodes::EndOutputAttribute();
    
    ImNodes::EndNode();
}

void CropNode::RenderInspector()
{
    ImGui::Text("Crop");
    ImGui::Separator();
    ImGui::SliderFloat("Left", &left_, 0.0f, 1.0f);
    ImGui::SliderFloat("Right", &right_, 0.0f, 1.0f);
    ImGui::SliderFloat("Top", &top_, 0.0f, 1.0f);
    ImGui::SliderFloat("Bottom", &bottom_, 0.0f, 1.0f);
}

void CropNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    SetLastTexturePool(texture_pool);
    
    if (inputs.empty() || inputs[0] == nil || !texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    id<MTLTexture> input_texture = inputs[0];
    int input_width = (int)[input_texture width];
    int input_height = (int)[input_texture height];
    MTLPixelFormat format = [input_texture pixelFormat];
    
    // Calculate output size
    int crop_width = int((1.0f - left_ - right_) * input_width);
    int crop_height = int((1.0f - top_ - bottom_) * input_height);
    
    if (crop_width < 1) crop_width = 1;
    if (crop_height < 1) crop_height = 1;
    
    if (output_texture_) {
        if ([output_texture_ width] != crop_width || [output_texture_ height] != crop_height) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
    }
    
    if (!output_texture_) {
        output_texture_ = texture_pool->acquire_texture(crop_width, crop_height, format);
    }
    
    if (!output_texture_) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    
    struct CropParams {
        float left;
        float right;
        float top;
        float bottom;
    };
    
    CropParams params = { left_, right_, top_, bottom_ };
    [encoder setBytes:&params length:sizeof(CropParams) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(crop_width, crop_height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

std::unique_ptr<NodeBase> CreateCropNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<CropNode>(graph, pos, device);
}

REGISTER_NODE(Crop, "Crop", "TOP/Transform", NodeFamily::TOP, CreateCropNode, "Crop texture to region");

} // namespace nodes
} // namespace example
