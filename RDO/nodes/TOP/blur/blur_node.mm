#include "blur_node.h"
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

// Metal 셰이더 코드 (Gaussian Blur)
static const char* blurShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gaussianBlurHorizontal(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &blurRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int radius = int(blurRadius);
    float4 color = float4(0.0);
    float totalWeight = 0.0;
    
    // Gaussian weights 계산
    for (int x = -radius; x <= radius; x++) {
        int2 coord = int2(gid.x + x, gid.y);
        coord.x = clamp(coord.x, 0, int(inputTexture.get_width()) - 1);
        
        float weight = exp(-(x * x) / (2.0 * blurRadius * blurRadius));
        color += inputTexture.read(uint2(coord)) * weight;
        totalWeight += weight;
    }
    
    outputTexture.write(color / totalWeight, gid);
}

kernel void gaussianBlurVertical(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float &blurRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    int radius = int(blurRadius);
    float4 color = float4(0.0);
    float totalWeight = 0.0;
    
    // Gaussian weights 계산
    for (int y = -radius; y <= radius; y++) {
        int2 coord = int2(gid.x, gid.y + y);
        coord.y = clamp(coord.y, 0, int(inputTexture.get_height()) - 1);
        
        float weight = exp(-(y * y) / (2.0 * blurRadius * blurRadius));
        color += inputTexture.read(uint2(coord)) * weight;
        totalWeight += weight;
    }
    
    outputTexture.write(color / totalWeight, gid);
}
)";

BlurNode::BlurNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , horizontal_blur_pipeline_(nil)
    , vertical_blur_pipeline_(nil)
    , command_queue_(nil)
    , intermediate_texture_(nil)
    , blur_radius_(5.0f)
    , last_input_texture_(nil)
    , last_blur_radius_(-1.0f)
{
    // output_texture_는 TOPNodeBase의 멤버이므로 여기서 초기화
    output_texture_ = nil;
    
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 포트 생성
    int input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int blur_amount_id = graph.insert_node(Node(NodeType::value, blur_radius_));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    
    // 포트 추가
    AddInputPort(Port(input_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(blur_amount_id, NodeFamily::CHOP, PortDirection::Input, "float", "radius"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

BlurNode::~BlurNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_) {
        if (intermediate_texture_) {
            last_texture_pool_->release_texture(intermediate_texture_);
            intermediate_texture_ = nil;
        }
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
            output_texture_ = nil;
        }
    } else {
        // TexturePool 참조 없으면 ARC가 자동 관리
        intermediate_texture_ = nil;
        output_texture_ = nil;
    }
}

void BlurNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_) {
        if (intermediate_texture_) {
            last_texture_pool_->release_texture(intermediate_texture_);
        }
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
    }
    
    intermediate_texture_ = nil;
    output_texture_ = nil;
}

bool BlurNode::InitializeMetal()
{
    if (!device_) return false;
    
    command_queue_ = [device_ newCommandQueue];
    
    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:blurShaderSource];
    
    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode
                                                   options:nil
                                                     error:&error];
    
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }
    
    id<MTLFunction> horizontalBlurFunction = [library newFunctionWithName:@"gaussianBlurHorizontal"];
    id<MTLFunction> verticalBlurFunction = [library newFunctionWithName:@"gaussianBlurVertical"];
    
    horizontal_blur_pipeline_ = [device_ newComputePipelineStateWithFunction:horizontalBlurFunction error:&error];
    if (error) {
        NSLog(@"Error creating horizontal blur pipeline: %@", error);
        return false;
    }
    
    vertical_blur_pipeline_ = [device_ newComputePipelineStateWithFunction:verticalBlurFunction error:&error];
    if (error) {
        NSLog(@"Error creating vertical blur pipeline: %@", error);
        return false;
    }
    
    return true;
}

void BlurNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Blur");
    ImNodes::EndNodeTitleBar();
    
    // 입력 포트
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    // Blur Radius 파라미터 (연결 가능)
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::PushItemWidth(120.0f);
        
        bool is_connected = (graph.num_edges_from_node(port.id) > 0);
        if (is_connected) {
            blur_radius_ = graph.node(port.id).value;
        }
        
        ImGui::SliderFloat("##radius", &blur_radius_, 0.0f, 50.0f);
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

void BlurNode::RenderInspector()
{
    ImGui::Text("Blur");
    ImGui::Separator();
    
    ImGui::SliderFloat("Blur Radius", &blur_radius_, 0.0f, 50.0f, "%.1f");
    
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

void BlurNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    // TexturePool 참조 저장 (InvalidateCache에서 사용)
    SetLastTexturePool(texture_pool);
    
    if (inputs.empty() || inputs[0] == nil) {
        // 입력 없음 - 기존 텍스처 정리
        if (intermediate_texture_ && texture_pool) {
            texture_pool->release_texture(intermediate_texture_);
            intermediate_texture_ = nil;
        }
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
        return;
    }
    if (!horizontal_blur_pipeline_ || !vertical_blur_pipeline_) return;
    
    id<MTLTexture> input_texture = inputs[0];
    
    // 텍스처 풀에서 중간/출력 텍스처 할당
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];
    
    // 기존 텍스처 해제 (크기가 변경된 경우만)
    if (intermediate_texture_)
    {
        if ([intermediate_texture_ width] != width || [intermediate_texture_ height] != height)
        {
            if (texture_pool) {
                texture_pool->release_texture(intermediate_texture_);
            }
            intermediate_texture_ = nil;
        }
    }
    
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
    if (!intermediate_texture_)
    {
        if (texture_pool) {
            intermediate_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor 
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width
                                            height:height
                                         mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            intermediate_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }
    
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
    
    if (!intermediate_texture_ || !output_texture_) {
        return;
    }
    
    // Compute Encoder 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    // Horizontal Pass
    [encoder setComputePipelineState:horizontal_blur_pipeline_];
    [encoder setTexture:input_texture atIndex:0];
    [encoder setTexture:intermediate_texture_ atIndex:1];
    [encoder setBytes:&blur_radius_ length:sizeof(float) atIndex:0];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + threadgroupSize.width - 1) / threadgroupSize.width,
        (height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    
    // Vertical Pass
    [encoder setComputePipelineState:vertical_blur_pipeline_];
    [encoder setTexture:intermediate_texture_ atIndex:0];
    [encoder setTexture:output_texture_ atIndex:1];
    [encoder setBytes:&blur_radius_ length:sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    
    // 캐싱 업데이트
    last_input_texture_ = input_texture;
    last_blur_radius_ = blur_radius_;
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateBlurNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<BlurNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Blur, "Blur", "TOP/Filter", NodeFamily::TOP, CreateBlurNode, "Gaussian blur effect");

} // namespace nodes
} // namespace example
