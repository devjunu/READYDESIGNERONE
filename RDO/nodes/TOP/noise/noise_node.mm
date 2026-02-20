#include "noise_node.h"
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

// Metal 셰이더 코드
static const char* noiseShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Hash 함수
float hash(float2 p, float seed) {
    float3 p3 = fract(float3(p.x, p.y, seed) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// White Noise
kernel void generateWhiteNoise(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float &seed [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 p = float2(gid);
    float value = hash(p, seed);
    
    float4 color = float4(value, value, value, 1.0);
    outputTexture.write(color, gid);
}

// Perlin Noise (간단한 구현)
float2 random2(float2 p, float seed) {
    float3 p3 = fract(float3(p.x, p.y, seed) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

float perlinNoise(float2 p, float seed) {
    float2 i = floor(p);
    float2 f = fract(p);
    
    float a = hash(i, seed);
    float b = hash(i + float2(1.0, 0.0), seed);
    float c = hash(i + float2(0.0, 1.0), seed);
    float d = hash(i + float2(1.0, 1.0), seed);
    
    float2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

kernel void generatePerlinNoise(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float &scale [[buffer(0)]],
    constant float &octaves [[buffer(1)]],
    constant float &persistence [[buffer(2)]],
    constant float &seed [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 p = uv * scale;
    
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;
    
    int oct = int(octaves);
    for (int i = 0; i < oct; i++) {
        value += perlinNoise(p * frequency, seed + float(i)) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    value /= maxValue;
    
    float4 color = float4(value, value, value, 1.0);
    outputTexture.write(color, gid);
}

// Simplex Noise (간단한 2D 구현)
float simplexNoise(float2 p, float seed) {
    const float F2 = 0.366025403;
    const float G2 = 0.211324865;
    
    float2 s = floor(p + dot(p, float2(F2)));
    float2 x = p - s + dot(s, float2(G2));
    
    float2 i1 = (x.x > x.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    
    float2 x1 = x - i1 + G2;
    float2 x2 = x - 1.0 + 2.0 * G2;
    
    float n0 = max(0.5 - dot(x, x), 0.0);
    n0 = n0 * n0 * n0 * n0 * (hash(s, seed) * 2.0 - 1.0);
    
    float n1 = max(0.5 - dot(x1, x1), 0.0);
    n1 = n1 * n1 * n1 * n1 * (hash(s + i1, seed) * 2.0 - 1.0);
    
    float n2 = max(0.5 - dot(x2, x2), 0.0);
    n2 = n2 * n2 * n2 * n2 * (hash(s + 1.0, seed) * 2.0 - 1.0);
    
    return 0.5 + 0.5 * (n0 + n1 + n2);
}

kernel void generateSimplexNoise(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float &scale [[buffer(0)]],
    constant float &octaves [[buffer(1)]],
    constant float &persistence [[buffer(2)]],
    constant float &seed [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 p = uv * scale;
    
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;
    
    int oct = int(octaves);
    for (int i = 0; i < oct; i++) {
        value += simplexNoise(p * frequency, seed + float(i)) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    value /= maxValue;
    
    float4 color = float4(value, value, value, 1.0);
    outputTexture.write(color, gid);
}

// Cellular Noise (Worley/Voronoi)
kernel void generateCellularNoise(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float &scale [[buffer(0)]],
    constant float &seed [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float2 uv = float2(gid) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 p = uv * scale;
    
    float2 i = floor(p);
    float2 f = fract(p);
    
    float minDist = 1.0;
    
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 neighbor = float2(float(x), float(y));
            float2 point = random2(i + neighbor, seed);
            float2 diff = neighbor + point - f;
            float dist = length(diff);
            minDist = min(minDist, dist);
        }
    }
    
    float value = minDist;
    
    float4 color = float4(value, value, value, 1.0);
    outputTexture.write(color, gid);
}
)";

NoiseNode::NoiseNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device), noise_type_(NoiseType::Perlin),
      scale_(10.0f), octaves_(4.0f), persistence_(0.5f), seed_(0.0f),
      width_(1920), height_(1080)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 출력 포트만 있음 (Generator)
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // Metal 초기화
    InitializeMetal();
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

NoiseNode::~NoiseNode()
{
    // TexturePool에 텍스처 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

void NoiseNode::InvalidateCache()
{
    // 텍스처를 TexturePool에 반환
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool NoiseNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    
    // 셰이더 컴파일
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(noiseShaderSource)
                                                   options:nil
                                                     error:&error];
    
    if (!library) {
        NSLog(@"Noise shader compilation failed: %@", error);
        return false;
    }
    
    // Pipeline 생성
    id<MTLFunction> whiteFunc = [library newFunctionWithName:@"generateWhiteNoise"];
    id<MTLFunction> perlinFunc = [library newFunctionWithName:@"generatePerlinNoise"];
    id<MTLFunction> simplexFunc = [library newFunctionWithName:@"generateSimplexNoise"];
    id<MTLFunction> cellularFunc = [library newFunctionWithName:@"generateCellularNoise"];
    
    white_pipeline_ = [device_ newComputePipelineStateWithFunction:whiteFunc error:&error];
    perlin_pipeline_ = [device_ newComputePipelineStateWithFunction:perlinFunc error:&error];
    simplex_pipeline_ = [device_ newComputePipelineStateWithFunction:simplexFunc error:&error];
    cellular_pipeline_ = [device_ newComputePipelineStateWithFunction:cellularFunc error:&error];
    
    return (white_pipeline_ != nil && perlin_pipeline_ != nil && 
            simplex_pipeline_ != nil && cellular_pipeline_ != nil);
}

void NoiseNode::Render(Graph<Node>& graph)
{
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Noise");
    ImNodes::EndNodeTitleBar();
    
    // 파라미터
    ImGui::PushItemWidth(120.0f);
    
    const char* noise_types[] = { "White", "Perlin", "Simplex", "Cellular" };
    int current_type = (int)noise_type_;
    if (ImGui::Combo("Type", &current_type, noise_types, 4)) {
        noise_type_ = (NoiseType)current_type;
    }
    
    if (noise_type_ != NoiseType::White) {
        ImGui::SliderFloat("Scale", &scale_, 1.0f, 50.0f);
    }
    
    if (noise_type_ == NoiseType::Perlin || noise_type_ == NoiseType::Simplex) {
        ImGui::SliderFloat("Octaves", &octaves_, 1.0f, 8.0f);
        ImGui::SliderFloat("Persistence", &persistence_, 0.1f, 1.0f);
    }
    
    ImGui::SliderFloat("Seed", &seed_, 0.0f, 100.0f);
    
    ImGui::InputInt("Width", &width_);
    ImGui::InputInt("Height", &height_);
    
    // 해상도 제한
    if (width_ < 1) width_ = 1;
    if (height_ < 1) height_ = 1;
    if (width_ > 4096) width_ = 4096;
    if (height_ > 4096) height_ = 4096;
    
    ImGui::PopItemWidth();
    
    ImGui::Spacing();
    
    // 프리뷰
    if (output_texture_ != nil)
    {
        int tex_width = (int)[output_texture_ width];
        int tex_height = (int)[output_texture_ height];
        
        if (tex_width > 0 && tex_height > 0)
        {
            float preview_width = 180.0f;
            float aspect_ratio = (float)tex_height / (float)tex_width;
            float preview_height = preview_width * aspect_ratio;
            
            if (preview_height > 120.0f)
            {
                preview_height = 120.0f;
                preview_width = preview_height / aspect_ratio;
            }
            
            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    }
    
    ImGui::Spacing();
    
    // 출력 포트
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("output").x;
        ImGui::Indent(200.0f - label_width);
        ImGui::TextUnformatted("output");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void NoiseNode::RenderInspector()
{
    ImGui::Text("Noise");
    ImGui::Separator();
    
    const char* noise_types[] = { "White", "Perlin", "Simplex", "Cellular" };
    int current_type = (int)noise_type_;
    if (ImGui::Combo("Type", &current_type, noise_types, 4)) {
        noise_type_ = (NoiseType)current_type;
    }
    
    if (noise_type_ != NoiseType::White) {
        ImGui::SliderFloat("Scale", &scale_, 1.0f, 50.0f);
    }
    
    if (noise_type_ == NoiseType::Perlin || noise_type_ == NoiseType::Simplex) {
        ImGui::SliderFloat("Octaves", &octaves_, 1.0f, 8.0f);
        ImGui::SliderFloat("Persistence", &persistence_, 0.1f, 1.0f);
    }
    
    ImGui::SliderFloat("Seed", &seed_, 0.0f, 100.0f);
    
    ImGui::InputInt("Width", &width_);
    ImGui::InputInt("Height", &height_);
    
    // 해상도 제한
    if (width_ < 1) width_ = 1;
    if (height_ < 1) height_ = 1;
    if (width_ > 4096) width_ = 4096;
    if (height_ > 4096) height_ = 4096;
    
    ImGui::Spacing();
    
    if (output_texture_ != nil)
    {
        int tex_width = (int)[output_texture_ width];
        int tex_height = (int)[output_texture_ height];
        
        if (tex_width > 0 && tex_height > 0)
        {
            float preview_width = 300.0f;
            float aspect_ratio = (float)tex_height / (float)tex_width;
            float preview_height = preview_width * aspect_ratio;
            
            if (preview_height > 400.0f)
            {
                preview_height = 400.0f;
                preview_width = preview_height / aspect_ratio;
            }
            
            ImTextureID texture_id = (ImTextureID)(__bridge void*)output_texture_;
            ImTextureRef texture_ref(texture_id);
            ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
        }
    }
}

void NoiseNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);
    
    if (!texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    // 기존 output_texture 해제 (크기가 변경된 경우)
    if (output_texture_)
    {
        if ([output_texture_ width] != width_ || [output_texture_ height] != height_)
        {
            texture_pool->release_texture(output_texture_);
            output_texture_ = nil;
        }
    }
    
    // 출력 텍스처 할당
    if (!output_texture_)
    {
        output_texture_ = texture_pool->acquire_texture(width_, height_, MTLPixelFormatRGBA8Unorm);
    }
    
    if (!output_texture_) {
        return;
    }
    
    // 타입에 따라 파이프라인 선택
    id<MTLComputePipelineState> pipeline = nil;
    switch (noise_type_) {
        case NoiseType::White: pipeline = white_pipeline_; break;
        case NoiseType::Perlin: pipeline = perlin_pipeline_; break;
        case NoiseType::Simplex: pipeline = simplex_pipeline_; break;
        case NoiseType::Cellular: pipeline = cellular_pipeline_; break;
    }
    
    if (!pipeline) return;
    
    // Noise 생성
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:output_texture_ atIndex:0];
    
    // 타입별 파라미터
    int bufferIndex = 0;
    if (noise_type_ == NoiseType::White) {
        [encoder setBytes:&seed_ length:sizeof(float) atIndex:bufferIndex++];
    } else if (noise_type_ == NoiseType::Perlin || noise_type_ == NoiseType::Simplex) {
        [encoder setBytes:&scale_ length:sizeof(float) atIndex:bufferIndex++];
        [encoder setBytes:&octaves_ length:sizeof(float) atIndex:bufferIndex++];
        [encoder setBytes:&persistence_ length:sizeof(float) atIndex:bufferIndex++];
        [encoder setBytes:&seed_ length:sizeof(float) atIndex:bufferIndex++];
    } else if (noise_type_ == NoiseType::Cellular) {
        [encoder setBytes:&scale_ length:sizeof(float) atIndex:bufferIndex++];
        [encoder setBytes:&seed_ length:sizeof(float) atIndex:bufferIndex++];
    }
    
    MTLSize gridSize = MTLSizeMake(width_, height_, 1);
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateNoiseNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<NoiseNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(Noise, "Noise", "TOP/Generator", NodeFamily::TOP, CreateNoiseNode, "Generate procedural noise");

} // namespace nodes
} // namespace example
