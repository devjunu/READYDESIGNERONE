#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../core/media/video/video_engine.h"
#include "../../../texture_pool.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <memory>
#include <string>

struct ImVec2;

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// Video File Loader 노드 (PIX/IO)
class MovieFileInNode : public TOPNodeBase
{
public:
    MovieFileInNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~MovieFileInNode();
    
    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "Movie File In"; }
    std::string GetCategory() const override { return "TOP/IO"; }
    
    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // RenderContext 오버로드 (Preview/Final 모드 지원)
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool,
        const RenderContext& context
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;
    
    // Video 관련 메서드
    void LoadVideo(const std::string& file_path);
    void Update();
    
    std::string GetFilePath() const { return file_path_; }
    
    // VideoEngine 접근 (NodeEditor에서 Timeline 등록용)
    rdo::core::media::video::VideoEngine* GetVideoEngine() const { return engine_.get(); }
    
    // 엔진 모드 설정 (Preview/Export 분리)
    void SetEngineMode(rdo::core::media::video::VideoEngine::EngineMode mode);
    
private:
    std::string file_path_;
    std::unique_ptr<rdo::core::media::video::VideoEngine> engine_;
    id<MTLDevice> device_;
    
    // 텍스처 소유권 추적 (TexturePool vs VideoEngine)
    bool is_texture_from_pool_ = false;
    
    // 프레임 변경 추적 (1/1 설정 시 프레임 튐 방지)
    float last_frame_time_ = -1.0f;
};

// 팩토리 함수
std::unique_ptr<NodeBase> CreateMovieFileInNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
