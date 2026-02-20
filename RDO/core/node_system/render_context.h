#pragma once

namespace example {

// 렌더링 모드 (미리보기 vs 최종 렌더링)
enum class RenderMode {
    Preview,  // 미리보기 (PreviewSettings 적용, 성능 최적화)
    Final     // 최종 렌더링 (원본 해상도, 고품질)
};

// 렌더링 컨텍스트 (모든 노드에 전달)
struct RenderContext {
    RenderMode mode = RenderMode::Preview;

    // Coordinate Space: 좌표계 기준 해상도 (항상 원본, Normalized 좌표 기준)
    // 모든 좌표 기반 연산은 이 해상도를 기준으로 함
    // 예: Shape 노드, CHOP to TOP 등
    int coordinate_width = 1920;
    int coordinate_height = 1080;

    // Render Resolution: 실제 렌더링 해상도 (Preview 설정 적용됨)
    // GPU 연산과 텍스처 크기는 이 해상도를 사용
    // Preview 모드: PreviewSettings 적용 (예: 960x540)
    // Final 모드: coordinate와 동일 (1920x1080)
    int render_width = 1920;
    int render_height = 1080;

    // Final 모드에서 사용할 타겟 해상도 (deprecated: render_width/height 사용)
    int target_width = 1920;
    int target_height = 1080;

    // 성능 최적화 플래그
    bool allow_shallow_copy = true;    // Passthrough 노드에서 텍스처 참조만 복사 (zero-copy)
    bool allow_texture_views = true;   // 포맷 변환 시 Texture View 사용 (zero-copy)
    bool skip_movie_file_in_nodes = false; // 외부 소스 텍스처 주입 시 MovieFileIn 처리 생략
    int forced_output_node_id = -1;    // 지정된 TOP 출력 노드 체인만 실행 (-1이면 자동 선택)

    // Normalized 좌표를 Render 좌표로 변환 (0.0~1.0 → 픽셀)
    inline float ToRenderX(float normalized_x) const {
        return normalized_x * render_width;
    }

    inline float ToRenderY(float normalized_y) const {
        return normalized_y * render_height;
    }

    // Coordinate Space 좌표를 Render 좌표로 변환
    inline float CoordToRenderX(float coord_x) const {
        return coord_x * (float)render_width / (float)coordinate_width;
    }

    inline float CoordToRenderY(float coord_y) const {
        return coord_y * (float)render_height / (float)coordinate_height;
    }
};

} // namespace example
