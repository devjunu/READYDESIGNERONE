#include "shape_top_node.h"
#include "../../../core/node_system/node_base.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../core/node_system/node_manager.h"
#include "../../../texture_pool.h"
#include "../../CHOP/analysis/blob_track_info_node.h"
#include "../blob_track/blob_track_node.h"  // GPU Indirect Draw 지원
#include <imgui.h>
#include <imnodes.h>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <simd/simd.h>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Metal 셰이더 코드
static const char* shapeTOPShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// 텍스처 클리어
kernel void clearTexture(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    outputTexture.write(float4(0.0, 0.0, 0.0, 0.0), gid);
}

// 박스 그리기
kernel void drawBoxes(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    device float4* boxes [[buffer(0)]],  // x, y, width, height
    constant int &boxCount [[buffer(1)]],
    constant float4 &fillColor [[buffer(2)]],
    constant float4 &strokeColor [[buffer(3)]],
    constant float &strokeWidth [[buffer(4)]],
    constant bool &fillEnabled [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    
    float4 color = inputTexture.read(gid);
    float px = float(gid.x);
    float py = float(gid.y);
    
    for (int i = 0; i < boxCount; i++) {
        float4 box = boxes[i];
        float x = box.x;
        float y = box.y;
        float w = box.z;
        float h = box.w;
        
        float left = x - w * 0.5;
        float right = x + w * 0.5;
        float top = y - h * 0.5;
        float bottom = y + h * 0.5;
        
        bool inside = (px >= left && px <= right && py >= top && py <= bottom);
        
        if (inside) {
            // 채우기
            if (fillEnabled) {
                color = mix(color, fillColor, fillColor.a);
            }
            
            // 테두리 (stroke)
            if (strokeWidth > 0.0) {
                bool onEdge = (px <= left + strokeWidth || px >= right - strokeWidth ||
                              py <= top + strokeWidth || py >= bottom - strokeWidth);
                if (onEdge) {
                    color = mix(color, strokeColor, strokeColor.a);
                }
            }
        }
    }
    
    outputTexture.write(color, gid);
}

// 원 그리기
kernel void drawCircles(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    device float4* circles [[buffer(0)]],  // x, y, radius, unused
    constant int &circleCount [[buffer(1)]],
    constant float4 &fillColor [[buffer(2)]],
    constant float4 &strokeColor [[buffer(3)]],
    constant float &strokeWidth [[buffer(4)]],
    constant bool &fillEnabled [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    
    float4 color = inputTexture.read(gid);
    float px = float(gid.x);
    float py = float(gid.y);
    
    for (int i = 0; i < circleCount; i++) {
        float4 circle = circles[i];
        float cx = circle.x;
        float cy = circle.y;
        float radius = circle.z;
        
        float2 center = float2(cx, cy);
        float2 pos = float2(px, py);
        float dist = distance(pos, center);
        
        bool inside = (dist <= radius);
        
        if (inside) {
            // 채우기
            if (fillEnabled) {
                color = mix(color, fillColor, fillColor.a);
            }
            
            // 테두리
            if (strokeWidth > 0.0 && dist >= radius - strokeWidth) {
                color = mix(color, strokeColor, strokeColor.a);
            }
        }
    }
    
    outputTexture.write(color, gid);
}

// ================= Render pipeline (instanced quad) =================
struct ShapeInstance {
    float2 center;   // pixel space
    float2 size;     // pixel space (width,height) or (radius,radius)
};

struct ShapeUniforms {
    float2 invTexSize;
    float4 fillColor;
    float4 strokeColor;
    float  strokeWidth;  // pixels
    int    shapeType;    // 0=box, 1=circle
    bool   fillEnabled;
};

struct VSOut {
    float4 position [[position]];
    float2 local;    // -1..1 quad local coords
    float2 halfSize; // pixel half size
};

// fullscreen-aligned quad per instance
vertex VSOut shapeVertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    device const ShapeInstance* instances [[buffer(1)]],
    constant ShapeUniforms& u [[buffer(2)]])
{
    // quad corners in local space (-1,-1) to (1,1)
    float2 corners[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 local = corners[vid];

    ShapeInstance inst = instances[iid];
    float2 halfSize = inst.size * 0.5;
    float2 pos_px = inst.center + local * halfSize;

    float2 ndc = float2(pos_px.x * u.invTexSize.x * 2.0 - 1.0,
                        1.0 - pos_px.y * u.invTexSize.y * 2.0); // flip Y

    VSOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.local = local;
    out.halfSize = halfSize;
    return out;
}

fragment float4 shapeFragment(
    VSOut in [[stage_in]],
    constant ShapeUniforms& u [[buffer(2)]])
{
    float4 dst = float4(0.0);
    float2 absLocal = abs(in.local);

    if (u.shapeType == 0) {
        // box: stroke near edges
        float2 pxDist = (in.halfSize * (1.0 - absLocal)); // remaining pixels to edge
        bool inside = (absLocal.x <= 1.0 && absLocal.y <= 1.0);
        bool onStroke = inside && (pxDist.x <= u.strokeWidth || pxDist.y <= u.strokeWidth);

        if (inside && u.fillEnabled) {
            dst = u.fillColor;
        }
        if (onStroke && u.strokeWidth > 0.0) {
            dst = u.strokeColor;
        }
    } else {
        // circle: radius = max(halfSize)
        float radius = max(in.halfSize.x, in.halfSize.y);
        float dist = length(in.local) * radius;
        bool inside = dist <= radius;
        bool onStroke = inside && (dist >= radius - u.strokeWidth);

        if (inside && u.fillEnabled) {
            dst = u.fillColor;
        }
        if (onStroke && u.strokeWidth > 0.0) {
            dst = u.strokeColor;
        }
    }
    return dst;
}

// ================= GPU Indirect Draw (BlobTrackNode 버퍼 직접 사용) =================
// GPUBlobData 구조체 (blob_track_node.mm과 동일)
struct GPUBlobData {
    float x, y;           // 중심점
    float width, height;  // 크기
    float area;           // 면적
    int id;               // ID
    int _padding[2];      // 16바이트 정렬
};

struct IndirectUniforms {
    float2 invTexSize;
    float4 fillColor;
    float4 strokeColor;
    float  strokeWidth;
    int    shapeType;
    bool   fillEnabled;
};

struct IndirectVSOut {
    float4 position [[position]];
    float2 local;
    float2 halfSize;
};

// GPU Indirect Draw: GPUBlobData 버퍼에서 직접 읽기
vertex IndirectVSOut indirectShapeVertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    device const GPUBlobData* blobs [[buffer(0)]],
    constant uint& blobCount [[buffer(1)]],
    constant IndirectUniforms& u [[buffer(2)]])
{
    float2 corners[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 local = corners[vid];
    
    IndirectVSOut out;
    if (iid >= blobCount) {
        out.position = float4(0, 0, -1, 1);  // 클리핑
        out.local = float2(0);
        out.halfSize = float2(0);
        return out;
    }

    GPUBlobData blob = blobs[iid];
    float2 center = float2(blob.x, blob.y);
    float2 size = float2(blob.width, blob.height);
    float2 halfSize = size * 0.5;
    float2 pos_px = center + local * halfSize;

    float2 ndc = float2(pos_px.x * u.invTexSize.x * 2.0 - 1.0,
                        1.0 - pos_px.y * u.invTexSize.y * 2.0);

    out.position = float4(ndc, 0.0, 1.0);
    out.local = local;
    out.halfSize = halfSize;
    return out;
}

fragment float4 indirectShapeFragment(
    IndirectVSOut in [[stage_in]],
    constant IndirectUniforms& u [[buffer(2)]])
{
    float4 dst = float4(0.0);
    float2 absLocal = abs(in.local);

    if (u.shapeType == 0) {
        // box
        float2 pxDist = (in.halfSize * (1.0 - absLocal));
        bool inside = (absLocal.x <= 1.0 && absLocal.y <= 1.0);
        bool onStroke = inside && (pxDist.x <= u.strokeWidth || pxDist.y <= u.strokeWidth);

        if (inside && u.fillEnabled) {
            dst = u.fillColor;
        }
        if (onStroke && u.strokeWidth > 0.0) {
            dst = u.strokeColor;
        }
    } else {
        // circle
        float radius = max(in.halfSize.x, in.halfSize.y);
        float dist = length(in.local) * radius;
        bool inside = dist <= radius;
        bool onStroke = inside && (dist >= radius - u.strokeWidth);

        if (inside && u.fillEnabled) {
            dst = u.fillColor;
        }
        if (onStroke && u.strokeWidth > 0.0) {
            dst = u.strokeColor;
        }
    }
    return dst;
}
)";

ShapeTOPNode::ShapeTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , box_pipeline_(nil)
    , circle_pipeline_(nil)
    , render_pipeline_(nil)
    , connected_chop_input_ptr_(nullptr)
    , graph_ref_(&graph)
    , shape_type_(ShapeType::Box)
    , stroke_width_(2.0f)
    , fill_enabled_(false)
    , width_(1920)
    , height_(1080)
    , shapes_buffer_(nil)
    , shapes_buffer_capacity_(0)
    , indirect_render_pipeline_(nil)
    , connected_blob_track_ptr_(nullptr)
    , use_gpu_indirect_(true)  // 기본값: GPU Indirect Draw 활성화
{
    fill_color_[0] = 1.0f;  // R
    fill_color_[1] = 0.0f;  // G
    fill_color_[2] = 0.0f;  // B
    fill_color_[3] = 0.5f;  // A
    
    stroke_color_[0] = 0.0f;  // R
    stroke_color_[1] = 1.0f;  // G
    stroke_color_[2] = 0.0f;  // B
    stroke_color_[3] = 1.0f;  // A
    
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // TOP 입력 포트 (참조 텍스처)
    int ref_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(ref_input_id, NodeFamily::TOP, PortDirection::Input, "texture", "reference"));
    
    // CHOP 입력 포트
    int chop_input_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddInputPort(Port(chop_input_id, NodeFamily::CHOP, PortDirection::Input, "channels", "input"));
    
    // TOP 출력 포트
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

ShapeTOPNode::~ShapeTOPNode()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
    shapes_buffer_ = nil;  // ARC가 해제
}

void ShapeTOPNode::InvalidateCache()
{
    if (last_texture_pool_ && output_texture_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
}

bool ShapeTOPNode::InitializeMetal()
{
    if (!device_) return false;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:shapeTOPShaderSource]
                                                    options:nil
                                                      error:&error];
    if (!library || error) {
        NSLog(@"Error creating library: %@", error);
        return false;
    }
    
    id<MTLFunction> boxFunction = [library newFunctionWithName:@"drawBoxes"];
    if (boxFunction) {
        box_pipeline_ = [device_ newComputePipelineStateWithFunction:boxFunction error:&error];
        if (error) {
            NSLog(@"Error creating box pipeline: %@", error);
        }
    }
    
    id<MTLFunction> circleFunction = [library newFunctionWithName:@"drawCircles"];
    if (circleFunction) {
        circle_pipeline_ = [device_ newComputePipelineStateWithFunction:circleFunction error:&error];
        if (error) {
            NSLog(@"Error creating circle pipeline: %@", error);
        }
    }
    
    // Clear pipeline
    id<MTLFunction> clearFunction = [library newFunctionWithName:@"clearTexture"];
    if (clearFunction) {
        clear_pipeline_ = [device_ newComputePipelineStateWithFunction:clearFunction error:&error];
        if (error) {
            NSLog(@"Warning: Error creating clear pipeline: %@", error);
        }
    }

    // Render pipeline (instanced quad for box/circle)
    id<MTLFunction> vs = [library newFunctionWithName:@"shapeVertex"];
    id<MTLFunction> fs = [library newFunctionWithName:@"shapeFragment"];
    if (vs && fs) {
        MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
        rpd.label = @"ShapeTOP Render Pipeline";
        rpd.vertexFunction = vs;
        rpd.fragmentFunction = fs;
        rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        rpd.colorAttachments[0].blendingEnabled = YES;
        rpd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        rpd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        rpd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        rpd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        rpd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        rpd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        render_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:rpd error:&error];
        if (error) {
            NSLog(@"Error creating render pipeline: %@", error);
        }
    }
    
    // GPU Indirect Draw 파이프라인 (BlobTrackNode 버퍼 직접 사용)
    id<MTLFunction> indirectVS = [library newFunctionWithName:@"indirectShapeVertex"];
    id<MTLFunction> indirectFS = [library newFunctionWithName:@"indirectShapeFragment"];
    if (indirectVS && indirectFS) {
        MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
        rpd.label = @"ShapeTOP Indirect Render Pipeline";
        rpd.vertexFunction = indirectVS;
        rpd.fragmentFunction = indirectFS;
        rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        rpd.colorAttachments[0].blendingEnabled = YES;
        rpd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        rpd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        rpd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        rpd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        rpd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        rpd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        indirect_render_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:rpd error:&error];
        if (error) {
            NSLog(@"Error creating indirect render pipeline: %@", error);
        }
    }

    return (box_pipeline_ != nil || circle_pipeline_ != nil || render_pipeline_ != nil);
}

void ShapeTOPNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Shape");
    ImNodes::EndNodeTitleBar();
    
    // TOP 참조 입력
    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // CHOP 입력
    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }
    
    ImGui::Spacing();
    
    // 연결 상태
    if (connected_chop_input_ptr_) {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "CHOP Connected");
    } else {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "CHOP Not Connected");
    }
    
    // Output
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

void ShapeTOPNode::RenderInspector()
{
    ImGui::Text("Shape TOP");
    ImGui::Separator();
    
    // Common 파라미터 (터치디자이너 호환)
    if (ImGui::CollapsingHeader("Common", ImGuiTreeNodeFlags_DefaultOpen)) {
        int w = width_;
        int h = height_;
        if (ImGui::InputInt("Width", &w)) {
            SetWidth(w);
        }
        if (ImGui::InputInt("Height", &h)) {
            SetHeight(h);
        }
        ImGui::TextDisabled("Resolution (used when no input texture)");
    }
    
    ImGui::Spacing();
    
    // Shape Type
    ImGui::Text("Shape Type");
    const char* shape_names[] = { "Box", "Circle" };
    int current_shape = static_cast<int>(shape_type_);
    if (ImGui::Combo("##shape_type", &current_shape, shape_names, 2)) {
        shape_type_ = static_cast<ShapeType>(current_shape);
    }
    
    ImGui::Spacing();
    
    // Fill
    ImGui::Checkbox("Fill", &fill_enabled_);
    if (fill_enabled_) {
        ImGui::ColorEdit4("Fill Color", fill_color_);
    }
    
    ImGui::Spacing();
    
    // Stroke
    ImGui::SliderFloat("Stroke Width", &stroke_width_, 0.0f, 10.0f, "%.1f");
    ImGui::ColorEdit4("Stroke Color", stroke_color_);
    
    ImGui::Spacing();
    ImGui::TextDisabled("(터치디자이너 Shape TOP 호환)");
}

void ShapeTOPNode::UpdateConnectedCHOPNode(NodeManager* node_manager)
{
    if (!node_manager || !graph_ref_ || input_ports_.size() < 2) {
        connected_chop_input_ptr_ = nullptr;
        connected_blob_track_ptr_ = nullptr;
        return;
    }
    
    connected_chop_input_ptr_ = nullptr;
    connected_blob_track_ptr_ = nullptr;
    
    // 1. TOP 참조 입력 (첫 번째 포트)에서 BlobTrackNode 확인 (GPU Indirect Draw용)
    if (use_gpu_indirect_) {
        int ref_port_id = input_ports_[0].id;
        for (const auto& edge : graph_ref_->edges()) {
            if (edge.to == ref_port_id) {
                NodeBase* connected_node = node_manager->GetNodeByPortId(edge.from);
                if (connected_node) {
                    BlobTrackNode* blob_track = dynamic_cast<BlobTrackNode*>(connected_node);
                    if (blob_track) {
                        connected_blob_track_ptr_ = blob_track;
                        // GPU Indirect Draw 경로 사용 - CHOP 데이터 불필요
                        return;
                    }
                }
            }
        }
    }
    
    // 2. CHOP 입력 포트 (두 번째 포트)에서 BlobTrackInfoNode 확인
    int input_port_id = input_ports_[1].id;
    
    for (const auto& edge : graph_ref_->edges()) {
        if (edge.to == input_port_id) {
            int from_port_id = edge.from;
            
            NodeBase* connected_node = node_manager->GetNodeByPortId(from_port_id);
            if (!connected_node) {
                continue;
            }
            
            // BlobTrackInfoNode로 직접 캐스팅
            BlobTrackInfoNode* info_node = dynamic_cast<BlobTrackInfoNode*>(connected_node);
            if (info_node) {
                connected_chop_input_ptr_ = info_node;
                
                // Evaluate 호출
                static float eval_time = 0.0f;
                eval_time += 0.016f;
                info_node->Evaluate({}, eval_time);
                
                // 출력 채널 가져오기
                const auto& output_channels = info_node->GetOutputChannels();
                const auto& channel_names = info_node->GetChannelNames();
                
                // 데이터 저장
                std::lock_guard<std::mutex> lock(channels_mutex_);
                input_channels_ = output_channels;
                input_channel_names_ = channel_names;
                
                return;
            }
        }
    }
}

void ShapeTOPNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool)
{
    SetLastTexturePool(texture_pool);
    
    if (!texture_pool) {
        if (output_texture_ && last_texture_pool_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        output_texture_ = nil;
        return;
    }
    
    // 해상도 결정: 입력 텍스처가 있으면 그것을 사용, 없으면 파라미터 값 사용 (터치디자이너 호환)
    int output_width = width_;
    int output_height = height_;
    if (!inputs.empty() && inputs[0] != nil) {
        output_width = static_cast<int>([inputs[0] width]);
        output_height = static_cast<int>([inputs[0] height]);
    }
    
    // 출력 텍스처 할당 (RenderTarget 필요)
    if (!output_texture_ || [output_texture_ width] != output_width || [output_texture_ height] != output_height
        || !([output_texture_ usage] & MTLTextureUsageRenderTarget)) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }

        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                         width:output_width
                                                                                        height:output_height
                                                                                     mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
        desc.storageMode = MTLStorageModePrivate;
        output_texture_ = [device_ newTextureWithDescriptor:desc];
    }
    
    if (!output_texture_) return;
    
    // 입력 텍스처가 있으면 복사, 없으면 검은색으로 클리어
    if (inputs.size() > 0 && inputs[0] != nil) {
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:inputs[0] toTexture:output_texture_];
        [blitEncoder endEncoding];
    } else {
        // 입력 텍스처가 없으면 검은색으로 클리어 (누적 방지)
        if (clear_pipeline_) {
            id<MTLComputeCommandEncoder> clearEncoder = [cmd_buffer computeCommandEncoder];
            [clearEncoder setComputePipelineState:clear_pipeline_];
            [clearEncoder setTexture:output_texture_ atIndex:0];
            MTLSize gridSize = MTLSizeMake(output_width, output_height, 1);
            NSUInteger w = clear_pipeline_.threadExecutionWidth;
            NSUInteger h = clear_pipeline_.maxTotalThreadsPerThreadgroup / w;
            MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
            [clearEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [clearEncoder endEncoding];
        }
    }
    
    // ========== GPU Indirect Draw 경로 (BlobTrackNode 직접 연결 시) ==========
    if (connected_blob_track_ptr_ && indirect_render_pipeline_) {
        BlobTrackNode* blob_track = static_cast<BlobTrackNode*>(connected_blob_track_ptr_);
        id<MTLBuffer> gpu_blob_buffer = blob_track->GetGPUBlobBuffer();
        id<MTLBuffer> gpu_count_buffer = blob_track->GetGPUBlobCountBuffer();
        
        if (gpu_blob_buffer && gpu_count_buffer) {
            uint32_t* count_ptr = (uint32_t*)[gpu_count_buffer contents];
            uint32_t blob_count = count_ptr ? *count_ptr : 0;
            blob_count = std::min(blob_count, (uint32_t)blob_track->GetMaxBlobs());
            
            if (blob_count > 0) {
                // GPU Indirect Draw: GPU 버퍼에서 직접 렌더링 (CPU 복사 없음!)
                MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
                rpd.colorAttachments[0].texture = output_texture_;
                rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
                rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
                
                id<MTLRenderCommandEncoder> renc = [cmd_buffer renderCommandEncoderWithDescriptor:rpd];
                [renc setRenderPipelineState:indirect_render_pipeline_];
                
                // IndirectUniforms 구조체 (GPU 셰이더와 동일)
                struct IndirectUniformsCPU {
                    simd::float2 invTexSize;
                    simd::float4 fillColor;
                    simd::float4 strokeColor;
                    float strokeWidth;
                    int shapeType;
                    bool fillEnabled;
                } uniforms;
                uniforms.invTexSize = simd::float2{1.0f / output_width, 1.0f / output_height};
                uniforms.fillColor = simd::float4{fill_color_[0], fill_color_[1], fill_color_[2], fill_color_[3]};
                uniforms.strokeColor = simd::float4{stroke_color_[0], stroke_color_[1], stroke_color_[2], stroke_color_[3]};
                uniforms.strokeWidth = stroke_width_;
                uniforms.shapeType = static_cast<int>(shape_type_);
                uniforms.fillEnabled = fill_enabled_;
                
                [renc setVertexBuffer:gpu_blob_buffer offset:0 atIndex:0];
                [renc setVertexBytes:&blob_count length:sizeof(uint32_t) atIndex:1];
                [renc setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:2];
                [renc setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:2];
                
                // Instanced draw: 4 vertices per quad, blob_count instances
                [renc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:blob_count];
                [renc endEncoding];
            }
            return;  // GPU Indirect Draw 완료
        }
    }
    
    // ========== CHOP 데이터 경로 (폴백) ==========
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    if (input_channels_.empty() || input_channel_names_.empty()) {
        return;  // 데이터 없으면 그냥 입력 텍스처만 출력
    }
    
    // Blob Track Info 형식 파싱: num_blobs, u1, v1, width1, height1, area1, id1, ...
    int num_blobs = static_cast<int>(input_channels_[0]);
    
    if (num_blobs <= 0) return;
    
    // Shape 데이터 생성 (Metal float4와 호환되는 구조체)
    struct ShapeData {
        float x, y, z, w;
    };
    std::vector<ShapeData> shapes;
    
    for (int i = 1; i <= num_blobs; i++) {
        // 각 blob당 8개 채널: u, v, width, height, area, id, vx, vy
        int base_idx = 1 + (i - 1) * 8;  // num_blobs(1) + (i-1) * 8채널
        
        if (base_idx + 3 >= input_channels_.size()) break;
        
        float u = input_channels_[base_idx + 0];      // u (정규화)
        float v = input_channels_[base_idx + 1];      // v (정규화)
        float width = input_channels_[base_idx + 2];   // width (픽셀)
        float height = input_channels_[base_idx + 3];  // height (픽셀)
        
        // 픽셀 좌표로 변환
        float x = u * output_width;
        float y = v * output_height;
        
        if (shape_type_ == ShapeType::Box) {
            shapes.push_back({x, y, width, height});
        } else if (shape_type_ == ShapeType::Circle) {
            float radius = std::max(width, height) * 0.5f;
            shapes.push_back({x, y, radius, 0.0f});
        }
    }
    
    if (shapes.empty()) return;
    
    // 버퍼 풀링: 용량이 부족하면 재할당, 아니면 재사용
    size_t required_capacity = shapes.size();
    if (!shapes_buffer_ || shapes_buffer_capacity_ < required_capacity) {
        // 2배 성장 전략으로 재할당 빈도 최소화
        size_t new_capacity = std::max(kInitialShapeCapacity, required_capacity * 2);
        shapes_buffer_ = [device_ newBufferWithLength:new_capacity * sizeof(ShapeData)
                                              options:MTLResourceStorageModeShared];
        shapes_buffer_capacity_ = new_capacity;
    }
    
    // 데이터 복사 (버퍼 재생성 없이)
    memcpy([shapes_buffer_ contents], shapes.data(), shapes.size() * sizeof(ShapeData));
    
    if (!render_pipeline_) return;
    
    // Render pass: 기존 출력 유지 후 오버레이(알파 블렌딩)
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = output_texture_;
    rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    
    id<MTLRenderCommandEncoder> renc = [cmd_buffer renderCommandEncoderWithDescriptor:rpd];
    [renc setRenderPipelineState:render_pipeline_];
    
    struct ShapeUniformsCPU {
        simd::float2 invTexSize;
        simd::float4 fillColor;
        simd::float4 strokeColor;
        float  strokeWidth;
        int    shapeType;
        bool   fillEnabled;
    } uniforms;
    uniforms.invTexSize = simd::float2{1.0f / output_width, 1.0f / output_height};
    uniforms.fillColor = simd::float4{fill_color_[0], fill_color_[1], fill_color_[2], fill_color_[3]};
    uniforms.strokeColor = simd::float4{stroke_color_[0], stroke_color_[1], stroke_color_[2], stroke_color_[3]};
    uniforms.strokeWidth = stroke_width_;
    uniforms.shapeType = static_cast<int>(shape_type_);
    uniforms.fillEnabled = fill_enabled_;

    [renc setVertexBuffer:shapes_buffer_ offset:0 atIndex:1];
    [renc setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:2];
    [renc setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:2];

    int shape_count = static_cast<int>(shapes.size());
    if (shape_count > 0) {
        [renc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:shape_count];
    }
    [renc endEncoding];
}

void ShapeTOPNode::GetFillColor(float color[4]) const
{
    color[0] = fill_color_[0];
    color[1] = fill_color_[1];
    color[2] = fill_color_[2];
    color[3] = fill_color_[3];
}

void ShapeTOPNode::SetFillColor(const float color[4])
{
    fill_color_[0] = color[0];
    fill_color_[1] = color[1];
    fill_color_[2] = color[2];
    fill_color_[3] = color[3];
}

void ShapeTOPNode::GetStrokeColor(float color[4]) const
{
    color[0] = stroke_color_[0];
    color[1] = stroke_color_[1];
    color[2] = stroke_color_[2];
    color[3] = stroke_color_[3];
}

void ShapeTOPNode::SetStrokeColor(const float color[4])
{
    stroke_color_[0] = color[0];
    stroke_color_[1] = color[1];
    stroke_color_[2] = color[2];
    stroke_color_[3] = color[3];
}

std::unique_ptr<NodeBase> CreateShapeTOPNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<ShapeTOPNode>(graph, pos, device);
}

REGISTER_NODE(Shape, "Shape", "TOP/Composite", NodeFamily::TOP, CreateShapeTOPNode, "Draw shapes from CHOP data (TouchDesigner Shape TOP compatible)");

} // namespace nodes
} // namespace example
