#include "math_node.h"
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

static const char* mathShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void mathOperation(
    texture2d<float, access::read> inputA [[texture(0)]],
    texture2d<float, access::read> inputB [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant int &operation [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 a = inputA.read(gid);
    float4 b = inputB.read(gid);
    float4 result;
    
    switch(operation) {
        case 0: result = a + b; break;           // Add
        case 1: result = a - b; break;           // Subtract
        case 2: result = a * b; break;           // Multiply
        case 3: result = a / (b + 0.0001); break; // Divide
        case 4: result = max(a, b); break;       // Max
        case 5: result = min(a, b); break;       // Min
        default: result = a; break;
    }
    
    result = clamp(result, 0.0, 1.0);
    output.write(result, gid);
}
)";

MathNode::MathNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), pipeline_(nil), operation_(MathOp::Add)
{
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    int inputA_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int inputB_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    AddInputPort(Port(inputA_id, NodeFamily::TOP, PortDirection::Input, "texture", "A"));
    AddInputPort(Port(inputB_id, NodeFamily::TOP, PortDirection::Input, "texture", "B"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

MathNode::~MathNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void MathNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool MathNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(mathShaderSource)
                                                   options:nil error:&error];
    
    if (!library) {
        NSLog(@"Math shader compilation failed: %@", error);
        return false;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"mathOperation"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    return (pipeline_ != nil);
}

void MathNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Math");
    ImNodes::EndNodeTitleBar();
    
    for (const auto& port : input_ports_) {
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    ImGui::PushItemWidth(120.0f);
    const char* ops[] = { "Add", "Subtract", "Multiply", "Divide", "Max", "Min" };
    int current_op = (int)operation_;
    if (ImGui::Combo("Op", &current_op, ops, 6)) {
        operation_ = (MathOp)current_op;
    }
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
        ImGui::TextDisabled("No inputs connected");
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

void MathNode::RenderInspector()
{
    ImGui::Text("Math");
    ImGui::Separator();
    const char* ops[] = { "Add", "Subtract", "Multiply", "Divide", "Max", "Min" };
    int current_op = (int)operation_;
    if (ImGui::Combo("Operation", &current_op, ops, 6)) {
        operation_ = (MathOp)current_op;
    }
}

void MathNode::ProcessGPU(
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
    
    id<MTLTexture> inputA = inputs[0];
    id<MTLTexture> inputB = inputs[1];
    int width = (int)[inputA width];
    int height = (int)[inputA height];
    MTLPixelFormat format = [inputA pixelFormat];
    
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
    [encoder setTexture:inputA atIndex:0];
    [encoder setTexture:inputB atIndex:1];
    [encoder setTexture:output_texture_ atIndex:2];
    
    int op = (int)operation_;
    [encoder setBytes:&op length:sizeof(int) atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline_.threadExecutionWidth;
    NSUInteger h = pipeline_.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

std::unique_ptr<NodeBase> CreateMathNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<MathNode>(graph, pos, device);
}

REGISTER_NODE(Math, "Math", "TOP/Math", NodeFamily::TOP, CreateMathNode, "Mathematical operations on textures");

} // namespace nodes
} // namespace example
