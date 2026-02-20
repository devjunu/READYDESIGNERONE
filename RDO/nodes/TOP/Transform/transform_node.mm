#include "transform_node.h"
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

// Metal 셰이더 코드 (Transform)
static const char* transformShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void applyTransform(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &rotation [[buffer(0)]],
    constant float &scaleX [[buffer(1)]],
    constant float &scaleY [[buffer(2)]],
    constant float &translateX [[buffer(3)]],
    constant float &translateY [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    uint width = inputTexture.get_width();
    uint height = inputTexture.get_height();

    // Normalize coordinates to [-1, 1]
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    uv = uv * 2.0 - 1.0;

    // Apply translation
    uv.x -= translateX;
    uv.y -= translateY;

    // Apply scale
    uv.x /= scaleX;
    uv.y /= scaleY;

    // Apply rotation
    float angle = radians(rotation);
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float2 rotated;
    rotated.x = uv.x * cosAngle - uv.y * sinAngle;
    rotated.y = uv.x * sinAngle + uv.y * cosAngle;

    // Convert back to [0, 1] and then to texture coordinates
    rotated = (rotated + 1.0) * 0.5;
    float2 texCoord = rotated * float2(width, height);

    // Bilinear sampling
    if (texCoord.x >= 0 && texCoord.x < width - 1 &&
        texCoord.y >= 0 && texCoord.y < height - 1)
    {
        uint2 coord00 = uint2(floor(texCoord.x), floor(texCoord.y));
        uint2 coord10 = uint2(coord00.x + 1, coord00.y);
        uint2 coord01 = uint2(coord00.x, coord00.y + 1);
        uint2 coord11 = uint2(coord00.x + 1, coord00.y + 1);

        float2 frac = fract(texCoord);

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
        outputTexture.write(float4(0.0, 0.0, 0.0, 0.0), gid);
    }
}
)";

TransformNode::TransformNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , transform_pipeline_(nil)
    , command_queue_(nil)
    , rotation_(0.0f)
    , scale_x_(1.0f)
    , scale_y_(1.0f)
    , translate_x_(0.0f)
    , translate_y_(0.0f)
{
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int rotation_id = graph.insert_node(Node(NodeType::value, rotation_));
    int scale_x_id = graph.insert_node(Node(NodeType::value, scale_x_));
    int scale_y_id = graph.insert_node(Node(NodeType::value, scale_y_));
    int translate_x_id = graph.insert_node(Node(NodeType::value, translate_x_));
    int translate_y_id = graph.insert_node(Node(NodeType::value, translate_y_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(rotation_id, NodeFamily::CHOP, PortDirection::Input, "float", "rotation"));
    AddInputPort(Port(scale_x_id, NodeFamily::CHOP, PortDirection::Input, "float", "scale_x"));
    AddInputPort(Port(scale_y_id, NodeFamily::CHOP, PortDirection::Input, "float", "scale_y"));
    AddInputPort(Port(translate_x_id, NodeFamily::CHOP, PortDirection::Input, "float", "translate_x"));
    AddInputPort(Port(translate_y_id, NodeFamily::CHOP, PortDirection::Input, "float", "translate_y"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

TransformNode::~TransformNode()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
            output_texture_ = nil;
        }
    } else {
        output_texture_ = nil;
    }
}

void TransformNode::InvalidateCache()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }
    output_texture_ = nil;
}

bool TransformNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:transformShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> transformFunction = [library newFunctionWithName:@"applyTransform"];

    transform_pipeline_ = [device_ newComputePipelineStateWithFunction:transformFunction error:&error];
    if (error) {
        NSLog(@"Error creating transform pipeline: %@", error);
        return false;
    }

    return true;
}

void TransformNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Transform");
    ImNodes::EndNodeTitleBar();

    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Rotation 파라미터
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            rotation_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##rotation", &rotation_, 0.0f, 360.0f, "%.1f°");
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Scale X 파라미터
    {
        const Port& port = input_ports_[2];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            scale_x_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##scale_x", &scale_x_, 0.1f, 5.0f, "SX: %.2f");
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Scale Y 파라미터
    {
        const Port& port = input_ports_[3];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            scale_y_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##scale_y", &scale_y_, 0.1f, 5.0f, "SY: %.2f");
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Translate X 파라미터
    {
        const Port& port = input_ports_[4];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            translate_x_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##translate_x", &translate_x_, -1.0f, 1.0f, "TX: %.2f");
        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Translate Y 파라미터
    {
        const Port& port = input_ports_[5];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            translate_y_ = graph.node(port.id).value;
        }

        ImGui::SliderFloat("##translate_y", &translate_y_, -1.0f, 1.0f, "TY: %.2f");
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

void TransformNode::RenderInspector()
{
    ImGui::Text("Transform");
    ImGui::Separator();

    ImGui::SliderFloat("Rotation", &rotation_, 0.0f, 360.0f, "%.1f°");
    ImGui::SliderFloat("Scale X", &scale_x_, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Scale Y", &scale_y_, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Translate X", &translate_x_, -1.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Translate Y", &translate_y_, -1.0f, 1.0f, "%.2f");

    ImGui::Spacing();

    // 텍스처 프리뷰
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

void TransformNode::ProcessGPU(
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
    if (!transform_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];
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

    // Transform Pass
    [encoder setComputePipelineState:transform_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&rotation_ length:sizeof(float) atIndex:0];
    [encoder setBytes:&scale_x_ length:sizeof(float) atIndex:1];
    [encoder setBytes:&scale_y_ length:sizeof(float) atIndex:2];
    [encoder setBytes:&translate_x_ length:sizeof(float) atIndex:3];
    [encoder setBytes:&translate_y_ length:sizeof(float) atIndex:4];

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
std::unique_ptr<NodeBase> CreateTransformNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<TransformNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Transform, "Transform", "TOP/Transform", NodeFamily::TOP, CreateTransformNode, "2D transform with rotation, scale, and translation");

} // namespace nodes
} // namespace example
