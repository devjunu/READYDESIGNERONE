#include "color_correction_node.h"
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

// Metal 셰이더 코드 (Color Correction)
static const char* colorCorrectionShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// RGB to HSV conversion
float3 rgb2hsv(float3 rgb)
{
    float cmax = max(max(rgb.r, rgb.g), rgb.b);
    float cmin = min(min(rgb.r, rgb.g), rgb.b);
    float delta = cmax - cmin;

    float3 hsv;

    // Hue
    if (delta < 0.00001) {
        hsv.x = 0.0;
    } else if (cmax == rgb.r) {
        hsv.x = 60.0 * fmod((rgb.g - rgb.b) / delta, 6.0);
    } else if (cmax == rgb.g) {
        hsv.x = 60.0 * ((rgb.b - rgb.r) / delta + 2.0);
    } else {
        hsv.x = 60.0 * ((rgb.r - rgb.g) / delta + 4.0);
    }

    if (hsv.x < 0.0) {
        hsv.x += 360.0;
    }

    // Saturation
    hsv.y = (cmax < 0.00001) ? 0.0 : (delta / cmax);

    // Value
    hsv.z = cmax;

    return hsv;
}

// HSV to RGB conversion
float3 hsv2rgb(float3 hsv)
{
    float c = hsv.z * hsv.y;
    float x = c * (1.0 - abs(fmod(hsv.x / 60.0, 2.0) - 1.0));
    float m = hsv.z - c;

    float3 rgb;
    if (hsv.x < 60.0) {
        rgb = float3(c, x, 0.0);
    } else if (hsv.x < 120.0) {
        rgb = float3(x, c, 0.0);
    } else if (hsv.x < 180.0) {
        rgb = float3(0.0, c, x);
    } else if (hsv.x < 240.0) {
        rgb = float3(0.0, x, c);
    } else if (hsv.x < 300.0) {
        rgb = float3(x, 0.0, c);
    } else {
        rgb = float3(c, 0.0, x);
    }

    return rgb + m;
}

kernel void applyColorCorrection(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &brightness [[buffer(0)]],
    constant float &contrast [[buffer(1)]],
    constant float &saturation [[buffer(2)]],
    constant float &hueShift [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(gid);

    // Apply brightness
    color.rgb += brightness;

    // Apply contrast
    color.rgb = (color.rgb - 0.5) * contrast + 0.5;

    // Convert to HSV for hue and saturation adjustment
    float3 hsv = rgb2hsv(color.rgb);

    // Apply hue shift
    hsv.x = fmod(hsv.x + hueShift + 360.0, 360.0);

    // Apply saturation
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);

    // Convert back to RGB
    color.rgb = hsv2rgb(hsv);

    // Clamp to valid range
    color = clamp(color, 0.0, 1.0);

    outputTexture.write(color, gid);
}
)";

ColorCorrectionNode::ColorCorrectionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , color_correction_pipeline_(nil)
    , command_queue_(nil)
    , brightness_(0.0f)
    , contrast_(1.0f)
    , saturation_(1.0f)
    , hue_(0.0f)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;

    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));

    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int brightness_id = graph.insert_node(Node(NodeType::value, brightness_));
    int contrast_id = graph.insert_node(Node(NodeType::value, contrast_));
    int saturation_id = graph.insert_node(Node(NodeType::value, saturation_));
    int hue_id = graph.insert_node(Node(NodeType::value, hue_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(brightness_id, NodeFamily::CHOP, PortDirection::Input, "float", "brightness"));
    AddInputPort(Port(contrast_id, NodeFamily::CHOP, PortDirection::Input, "float", "contrast"));
    AddInputPort(Port(saturation_id, NodeFamily::CHOP, PortDirection::Input, "float", "saturation"));
    AddInputPort(Port(hue_id, NodeFamily::CHOP, PortDirection::Input, "float", "hue"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    // Metal 초기화
    InitializeMetal();

    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

ColorCorrectionNode::~ColorCorrectionNode()
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

void ColorCorrectionNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }

    output_texture_ = nil;
}

bool ColorCorrectionNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:colorCorrectionShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];

    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    id<MTLFunction> colorCorrectionFunction = [library newFunctionWithName:@"applyColorCorrection"];

    color_correction_pipeline_ = [device_ newComputePipelineStateWithFunction:colorCorrectionFunction error:&error];
    if (error) {
        NSLog(@"Error creating color correction pipeline: %@", error);
        return false;
    }

    return true;
}

void ColorCorrectionNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Color Correction");
    ImNodes::EndNodeTitleBar();

    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    // Brightness 파라미터
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (!is_connected) {
            ImGui::Text("Brightness");
            if (ImGui::SliderFloat("##brightness", &brightness_, -1.0f, 1.0f)) {
                graph.node(port.id).value = brightness_;
            }
        } else {
            brightness_ = graph.node(port.id).value;
            ImGui::Text("Brightness: %.2f", brightness_);
        }

        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Contrast 파라미터
    {
        const Port& port = input_ports_[2];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (!is_connected) {
            ImGui::Text("Contrast");
            if (ImGui::SliderFloat("##contrast", &contrast_, 0.0f, 2.0f)) {
                graph.node(port.id).value = contrast_;
            }
        } else {
            contrast_ = graph.node(port.id).value;
            ImGui::Text("Contrast: %.2f", contrast_);
        }

        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Saturation 파라미터
    {
        const Port& port = input_ports_[3];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (!is_connected) {
            ImGui::Text("Saturation");
            if (ImGui::SliderFloat("##saturation", &saturation_, 0.0f, 2.0f)) {
                graph.node(port.id).value = saturation_;
            }
        } else {
            saturation_ = graph.node(port.id).value;
            ImGui::Text("Saturation: %.2f", saturation_);
        }

        ImGui::PopItemWidth();
        ImNodes::EndInputAttribute();
    }

    // Hue 파라미터
    {
        const Port& port = input_ports_[4];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);

        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (!is_connected) {
            ImGui::Text("Hue Shift");
            if (ImGui::SliderFloat("##hue", &hue_, -180.0f, 180.0f)) {
                graph.node(port.id).value = hue_;
            }
        } else {
            hue_ = graph.node(port.id).value;
            ImGui::Text("Hue: %.1f°", hue_);
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

void ColorCorrectionNode::RenderInspector()
{
    ImGui::Text("Color Correction");
    ImGui::Separator();

    ImGui::SliderFloat("Brightness", &brightness_, -1.0f, 1.0f);
    ImGui::SliderFloat("Contrast", &contrast_, 0.0f, 2.0f);
    ImGui::SliderFloat("Saturation", &saturation_, 0.0f, 2.0f);
    ImGui::SliderFloat("Hue Shift", &hue_, -180.0f, 180.0f);

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

void ColorCorrectionNode::ProcessGPU(
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
    if (!color_correction_pipeline_) return;

    id<MTLTexture> input_texture = inputs[0];
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

    // Color Correction Pass
    [encoder setComputePipelineState:color_correction_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&brightness_ length:sizeof(float) atIndex:0];
    [encoder setBytes:&contrast_ length:sizeof(float) atIndex:1];
    [encoder setBytes:&saturation_ length:sizeof(float) atIndex:2];
    [encoder setBytes:&hue_ length:sizeof(float) atIndex:3];

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
std::unique_ptr<NodeBase> CreateColorCorrectionNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ColorCorrectionNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(ColorCorrection, "Color Correction", "TOP/Filter", NodeFamily::TOP, CreateColorCorrectionNode, "Adjust brightness, contrast, saturation, and hue");

} // namespace nodes
} // namespace example
