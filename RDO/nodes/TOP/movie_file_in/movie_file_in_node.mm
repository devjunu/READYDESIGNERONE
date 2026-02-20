#include "movie_file_in_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../preview_settings.h"
#include <imgui.h>
#include <imnodes.h>
#include <iostream>

#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

MovieFileInNode::MovieFileInNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : file_path_(""), device_(device)
{
    // 노드 생성
    node_id_ = graph.insert_node(Node(NodeType::value));
    
    // 출력 포트 생성
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));
    
    // VideoEngine 초기화
    if (device_)
    {
        engine_ = std::make_unique<rdo::core::media::video::VideoEngine>(device_);
    }
    
    // 노드 위치 설정
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

MovieFileInNode::~MovieFileInNode()
{
    std::cerr << "MovieFileInNode: Destructor called" << std::endl;
    
    // VideoEngine을 먼저 정리 (Stop 호출로 스레드 정리)
    if (engine_)
    {
        std::cerr << "MovieFileInNode: Destroying engine" << std::endl;
        engine_.reset();  // 명시적으로 소멸자 호출
        std::cerr << "MovieFileInNode: Engine destroyed" << std::endl;
    }
    
    // TexturePool 소유 텍스처는 여기서 반환하지 않음
    // (texture_pool_이 이미 소멸되었을 수 있음)
    // Metal의 ARC가 자동으로 정리함
    output_texture_ = nil;
    is_texture_from_pool_ = false;
    
    std::cerr << "MovieFileInNode: Destructor complete" << std::endl;
}

void MovieFileInNode::Render(Graph<Node>& graph)
{
    const float node_width = 200.0f;
    
    ImNodes::BeginNode(node_id_);
    
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Movie File In");
    ImNodes::EndNodeTitleBar();
    
    ImGui::Spacing();
    
    // 파일 선택 버튼
    if (ImGui::Button("Select Video File", ImVec2(node_width - 20.0f, 0.0f)))
    {
        NSOpenPanel* openPanel = [NSOpenPanel openPanel];
        [openPanel setCanChooseFiles:YES];
        [openPanel setCanChooseDirectories:NO];
        [openPanel setAllowsMultipleSelection:NO];
        
        if (@available(macOS 12.0, *))
        {
            NSArray<UTType*>* contentTypes = @[
                [UTType typeWithFilenameExtension:@"mov"],
                [UTType typeWithFilenameExtension:@"mp4"],
                [UTType typeWithFilenameExtension:@"avi"],
                [UTType typeWithFilenameExtension:@"mkv"],
                [UTType typeWithFilenameExtension:@"m4v"],
                [UTType typeWithFilenameExtension:@"mpg"],
                [UTType typeWithFilenameExtension:@"mpeg"],
                [UTType typeWithFilenameExtension:@"wmv"],
                [UTType typeWithFilenameExtension:@"flv"],
                [UTType typeWithFilenameExtension:@"webm"]
            ];
            [openPanel setAllowedContentTypes:contentTypes];
        }
        else
        {
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            [openPanel setAllowedFileTypes:@[@"mov", @"mp4", @"avi", @"mkv", @"m4v", @"mpg", @"mpeg", @"wmv", @"flv", @"webm"]];
            #pragma clang diagnostic pop
        }
        
        NSWindow* mainWindow = [NSApp mainWindow];
        if (!mainWindow)
        {
            mainWindow = [NSApp keyWindow];
        }
        
        MovieFileInNode* node_ptr = this;
        
        if (mainWindow)
        {
            [openPanel beginSheetModalForWindow:mainWindow completionHandler:^(NSModalResponse result) {
                if (result == NSModalResponseOK)
                {
                    NSURL* selectedURL = [[openPanel URLs] firstObject];
                    if (selectedURL)
                    {
                        NSString* path = [selectedURL path];
                        const char* cPath = [path UTF8String];
                        node_ptr->LoadVideo(std::string(cPath));
                    }
                }
            }];
        }
        else
        {
            [openPanel beginWithCompletionHandler:^(NSModalResponse result) {
                if (result == NSModalResponseOK)
                {
                    NSURL* selectedURL = [[openPanel URLs] firstObject];
                    if (selectedURL)
                    {
                        NSString* path = [selectedURL path];
                        const char* cPath = [path UTF8String];
                        node_ptr->LoadVideo(std::string(cPath));
                    }
                }
            }];
        }
    }
    
    ImGui::Spacing();
    
    // 로딩 상태 표시
    if (engine_)
    {
        if (engine_->IsLoading())
        {
            static float loading_time = 0.0f;
            loading_time += ImGui::GetIO().DeltaTime;
            int dots = (int)(loading_time * 2.0f) % 4;
            std::string loading_text = "Loading";
            for (int i = 0; i < dots; i++) loading_text += ".";
            
            ImGui::BeginChildFrame(ImGui::GetID("loading_frame"), ImVec2(node_width - 20.0f, 100.0f), ImGuiWindowFlags_NoScrollbar);
            ImGui::SetCursorPos(ImVec2((node_width - 20.0f - ImGui::CalcTextSize(loading_text.c_str()).x) * 0.5f, 40.0f));
            ImGui::Text("%s", loading_text.c_str());
            ImGui::EndChildFrame();
        }
        else if (engine_->HasError())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::TextWrapped("Error: %s", engine_->GetErrorMessage().c_str());
            ImGui::PopStyleColor();
        }
        else if (engine_->IsLoaded())
        {
            // 영상 정보
            ImGui::Text("%dx%d @ %.1ffps", 
                        engine_->GetWidth(), 
                        engine_->GetHeight(),
                        engine_->GetFPS());
            
            ImGui::Spacing();
            
            // 영상 프리뷰 (노드 내)
            id<MTLTexture> texture = engine_->GetCurrentFrameTexture();
            if (texture)
            {
                int video_width = engine_->GetWidth();
                int video_height = engine_->GetHeight();
                
                if (video_width > 0 && video_height > 0)
                {
                    float preview_width = node_width - 20.0f;
                    float aspect_ratio = static_cast<float>(video_height) / static_cast<float>(video_width);
                    float preview_height = preview_width * aspect_ratio;
                    
                    const float max_preview_height = 120.0f;
                    if (preview_height > max_preview_height)
                    {
                        preview_height = max_preview_height;
                        preview_width = preview_height / aspect_ratio;
                    }
                    
                    ImTextureID texture_id = (ImTextureID)(__bridge void*)texture;
                    ImTextureRef texture_ref(texture_id);
                    ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
                }
            }
        }
    }
    
    ImGui::Spacing();
    
    // 파일 경로 표시
    if (!file_path_.empty())
    {
        size_t lastSlash = file_path_.find_last_of('/');
        std::string filename = (lastSlash != std::string::npos) ? 
                               file_path_.substr(lastSlash + 1) : 
                               file_path_;
        ImGui::TextWrapped("%s", filename.c_str());
    }
    else
    {
        ImGui::TextDisabled("No file selected");
    }
    
    ImGui::Spacing();
    
    // 출력 포트
    {
        const Port& port = output_ports_[0];
        ImNodes::BeginOutputAttribute(port.id);
        const float label_width = ImGui::CalcTextSize("output").x;
        ImGui::Indent(node_width - label_width);
        ImGui::TextUnformatted("output");
        ImNodes::EndOutputAttribute();
    }
    
    ImNodes::EndNode();
}

void MovieFileInNode::RenderInspector()
{
    ImGui::Text("Movie File In");
    ImGui::Separator();
    
    if (!file_path_.empty())
    {
        ImGui::Text("File:");
        size_t lastSlash = file_path_.find_last_of('/');
        std::string filename = (lastSlash != std::string::npos) ? 
                               file_path_.substr(lastSlash + 1) : 
                               file_path_;
        ImGui::TextWrapped("%s", filename.c_str());
        
        if (engine_ && engine_->IsLoaded())
        {
            ImGui::Spacing();
            
            // 원본 해상도 표시
            int original_width = engine_->GetWidth();
            int original_height = engine_->GetHeight();
            ImGui::Text("Original: %dx%d", original_width, original_height);
            
            // 출력 해상도 표시 (프리뷰 스케일 적용됨)
            if (output_texture_) {
                int output_width = (int)[output_texture_ width];
                int output_height = (int)[output_texture_ height];
                
                // 스케일 정보 표시
                if (output_width != original_width || output_height != original_height) {
                    float scale = (float)output_width / (float)original_width;
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.5f, 1.0f));
                    ImGui::Text("Output: %dx%d (%.0f%%)", output_width, output_height, scale * 100.0f);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::Text("Output: %dx%d", output_width, output_height);
                }
            }
            
            ImGui::Text("FPS: %.2f", engine_->GetFPS());
            ImGui::Text("Duration: %.2fs", engine_->GetDuration());
            ImGui::Text("Current Time: %.2fs", engine_->GetCurrentTime());
            
            ImGui::Spacing();
            
            // Inspector에서 더 큰 프리뷰
            id<MTLTexture> texture = output_texture_ ? output_texture_ : engine_->GetCurrentFrameTexture();
            if (texture)
            {
                int video_width = (int)[texture width];
                int video_height = (int)[texture height];
                
                if (video_width > 0 && video_height > 0)
                {
                    float preview_width = 300.0f;
                    float aspect_ratio = static_cast<float>(video_height) / static_cast<float>(video_width);
                    float preview_height = preview_width * aspect_ratio;
                    
                    const float max_preview_height = 300.0f;
                    if (preview_height > max_preview_height)
                    {
                        preview_height = max_preview_height;
                        preview_width = preview_height / aspect_ratio;
                    }
                    
                    ImTextureID texture_id = (ImTextureID)(__bridge void*)texture;
                    ImTextureRef texture_ref(texture_id);
                    ImGui::Image(texture_ref, ImVec2(preview_width, preview_height));
                }
            }
        }
    }
    else
    {
        ImGui::TextDisabled("No file loaded");
    }
}

void MovieFileInNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool
)
{
    // TexturePool 참조 저장
    SetLastTexturePool(texture_pool);

    // VideoEngine에서 현재 프레임의 텍스처 가져오기
    if (engine_ && engine_->IsLoaded())
    {
        id<MTLTexture> current_frame = engine_->GetCurrentFrameTexture();

        if (!current_frame) {
            // 이전 텍스처가 TexturePool 소유인 경우만 반환
            if (output_texture_ && is_texture_from_pool_ && texture_pool) {
                texture_pool->release_texture(output_texture_);
            }
            output_texture_ = nil;
            is_texture_from_pool_ = false;
            return;
        }

        // 이전에 할당된 텍스처가 있었다면 반환 (TexturePool 소유인 경우만)
        if (output_texture_ && is_texture_from_pool_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }

        // VideoEngine의 텍스처를 원본 해상도 그대로 사용 (Preview 설정 무시)
        // Preview scaling은 display 시점에서만 적용됨
        output_texture_ = current_frame;
        is_texture_from_pool_ = false;  // VideoEngine 소유
    }
    else
    {
        // VideoEngine이 로드되지 않음 - 이전 텍스처 반환 (TexturePool 소유인 경우만)
        if (output_texture_ && is_texture_from_pool_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        output_texture_ = nil;
        is_texture_from_pool_ = false;
    }
}

// RenderContext를 받는 새로운 ProcessGPU (Preview/Final 모드 지원)
void MovieFileInNode::ProcessGPU(
    const std::vector<id<MTLTexture>>& inputs,
    id<MTLCommandBuffer> cmd_buffer,
    TexturePool* texture_pool,
    const RenderContext& context
)
{
    SetLastTexturePool(texture_pool);

    if (!engine_ || !engine_->IsLoaded()) {
        if (output_texture_ && is_texture_from_pool_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        output_texture_ = nil;
        is_texture_from_pool_ = false;
        return;
    }

    id<MTLTexture> current_frame = engine_->GetCurrentFrameTexture();
    if (!current_frame) {
        if (output_texture_ && is_texture_from_pool_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        output_texture_ = nil;
        is_texture_from_pool_ = false;
        return;
    }

    int original_width = (int)[current_frame width];
    int original_height = (int)[current_frame height];
    int target_width = original_width;
    int target_height = original_height;

    // Preview 모드: PreviewSettings 적용 (성능 최적화)
    // Final 모드: 원본 해상도 유지
    if (context.mode == RenderMode::Preview) {
        auto& preview_settings = PreviewSettings::Instance();
        preview_settings.ApplyScale(original_width, original_height, target_width, target_height);
    }

    // 원본 해상도와 동일하면 그대로 사용 (zero-copy)
    // 단, 프레임이 실제로 변경되었을 때만 업데이트하여 프레임 튐 방지
    if (target_width == original_width && target_height == original_height) {
        float current_time = engine_->GetCurrentTime();
        
        // 프레임이 변경되었거나 텍스처가 다른 경우에만 업데이트
        // (동일한 프레임을 여러 번 처리하는 것을 방지하여 안정성 확보)
        if (current_time != last_frame_time_ || output_texture_ != current_frame) {
            if (output_texture_ && is_texture_from_pool_ && texture_pool) {
                texture_pool->release_texture(output_texture_);
            }
            output_texture_ = current_frame;
            is_texture_from_pool_ = false;
            last_frame_time_ = current_time;
        }
        return;
    }

    // Preview 다운스케일 필요
    MTLPixelFormat format = [current_frame pixelFormat];

    // 출력 텍스처 크기 확인 및 재할당
    bool need_new_texture = false;
    if (!output_texture_) {
        need_new_texture = true;
    } else if ([output_texture_ width] != target_width || [output_texture_ height] != target_height) {
        if (is_texture_from_pool_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }
        output_texture_ = nil;
        is_texture_from_pool_ = false;
        need_new_texture = true;
    }

    if (need_new_texture && texture_pool) {
        output_texture_ = texture_pool->acquire_texture(target_width, target_height, format);
        is_texture_from_pool_ = true;
    }

    if (!output_texture_) return;

    // Metal Performance Shaders로 고품질 다운스케일
    MPSImageLanczosScale* scaler = [[MPSImageLanczosScale alloc] initWithDevice:device_];
    [scaler encodeToCommandBuffer:cmd_buffer
                    sourceTexture:current_frame
               destinationTexture:output_texture_];
}

void MovieFileInNode::LoadVideo(const std::string& file_path)
{
    if (file_path_ != file_path)
    {
        file_path_ = file_path;

        if (engine_ && !file_path.empty())
        {
            engine_->LoadVideoAsync(file_path);
            
            // 비동기 로딩이 완료되면 Timeline에 자동으로 등록됨
            // (Timeline::RegisterVideoEngine에서 duration 확인)
        }
    }
}

void MovieFileInNode::InvalidateCache()
{
    // TexturePool 소유 텍스처만 반환
    if (output_texture_ && is_texture_from_pool_ && last_texture_pool_) {
        last_texture_pool_->release_texture(output_texture_);
    }
    output_texture_ = nil;
    is_texture_from_pool_ = false;
    last_frame_time_ = -1.0f;  // 프레임 시간 리셋 (다음 프레임에서 업데이트 보장)
}

void MovieFileInNode::Update()
{
    if (engine_)
    {
        engine_->Update();
    }
}

void MovieFileInNode::SetEngineMode(rdo::core::media::video::VideoEngine::EngineMode mode)
{
    if (engine_)
    {
        engine_->SetMode(mode);
    }
}

// 팩토리 함수
std::unique_ptr<NodeBase> CreateMovieFileInNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<MovieFileInNode>(graph, pos, device);
}

// 자동 등록
REGISTER_NODE(MovieFileIn, "Movie File In", "TOP/IO", NodeFamily::TOP, CreateMovieFileInNode, "Load video files");

} // namespace nodes
} // namespace example
