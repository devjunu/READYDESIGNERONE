#include "lookup_node.h"
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

static const char* lookupShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyLookup(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::read> lutTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant float &mixAmount [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 color = inputTexture.read(gid);
    
    // Sample LUT (assuming 1D LUT stored as width=256, height=1)
    if (lutTexture.get_width() > 0 && lutTexture.get_height() > 0) {
        uint lutX_r = uint(clamp(color.r, 0.0f, 1.0f) * float(lutTexture.get_width() - 1));
        uint lutX_g = uint(clamp(color.g, 0.0f, 1.0f) * float(lutTexture.get_width() - 1));
        uint lutX_b = uint(clamp(color.b, 0.0f, 1.0f) * float(lutTexture.get_width() - 1));
        
        float4 lut_r = lutTexture.read(uint2(lutX_r, 0));
        float4 lut_g = lutTexture.read(uint2(lutX_g, 1 % lutTexture.get_height()));
        float4 lut_b = lutTexture.read(uint2(lutX_b, 2 % lutTexture.get_height()));
        
        float4 lutColor = float4(lut_r.r, lut_g.g, lut_b.b, color.a);
        color = mix(color, lutColor, mixAmount);
    }
    
    outputTexture.write(color, gid);
}
)";

LookupNode::LookupNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil), mix_amount_(1.0f)
{
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int lut_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(lut_id, NodeFamily::TOP, PortDirection::Input, "texture", "lut"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

LookupNode::~LookupNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void LookupNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool LookupNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(lookupShaderSource)
                                                   options:nil error:&error];
    
    if (!library) {
        NSLog(@"Lookup shader compilation failed: %@", error);
        return false;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"applyLookup"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void LookupNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Lookup");
    ImNodes::EndNodeTitleBar();
    
    for (const auto& port : input_ports_) {
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    ImGui::PushItemWidth(120.0f);
    ImGui::SliderFloat("Mix", &mix_amount_, 0.0f, 1.0f);
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
    
    const Port& port = output_ports_[0];
    ImNodes::BeginOutputAttribute(port.id);
    const float label_width = ImGui::CalcTextSize("output").x;
    ImGui::Indent(200.0f - label_width);
    ImGui::TextUnformatted("output");
    ImNodes::EndOutputAttribute();
    
    ImNodes::EndNode();
}

void LookupNode::RenderInspector()
{
    ImGui::Text("Lookup (LUT)");
    ImGui::Separator();
    ImGui::SliderFloat("Mix Amount", &mix_amount_, 0.0f, 1.0f);
}

void LookupNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    SetLastTexturePool(texture_pool);
    
    if (inputs.size() < 2 || inputs[0] == nil || inputs[1] == nil || !texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    id<MTLTexture> input_texture = inputs[0];
    id<MTLTexture> lut_texture = inputs[1];
    int width = (int)[input_texture width];
    int height = (int)[input_texture height];
    MTLPixelFormat format = [input_texture pixelFormat];
    
    if (output_texture_) {
        if ([output_texture_ width] != width || [output_texture_ height] != height) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
    }
    
    if (!output_texture_) {
        output_texture_ = texture_pool->acquire_texture(width, height, format);
    }
    
    if (!output_texture_) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:lut_texture atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    [encoder setBytes:&mix_amount_ length:sizeof(float) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

std::unique_ptr<NodeBase> CreateLookupNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<LookupNode>(graph, pos, device);
}

REGISTER_NODE(Lookup, "Lookup", "TOP/Filter", NodeFamily::TOP, CreateLookupNode, "Apply LUT (Look-Up Table)");

} // namespace nodes
} // namespace example
