#pragma once

#include "../../../core/node_system/node_base.h"
#include "../../../graph.h"
#include "../../node_types.h"
#include <vector>
#include <memory>

struct ImVec2;

@protocol MTLDevice;
@protocol MTLTexture;
@protocol MTLComputePipelineState;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;

namespace example { class TexturePool; }

namespace example
{
namespace nodes
{

// nodes 네임스페이스의 Node를 사용
using ::example::nodes::Node;

// Blob 정보 구조체 (Event-Driven Architecture)
struct BlobInfo {
    int id;
    float x, y;           // 중심점
    float width, height;  // 크기
    float area;           // 면적
    float vx, vy;         // 속도 (velocity)
    float ax, ay;         // 가속도 (acceleration) - Kalman Filter용
    float predicted_x, predicted_y;  // 예측 위치
    float confidence;     // 신뢰도 (0.0 ~ 1.0)
    float last_seen_time; // 마지막으로 감지된 시간 (초)
    bool is_lost;         // 사라진 blob인지 여부
    bool is_expired;      // 만료된 blob인지 여부
};

// Mono Source (분석할 채널) - 터치디자이너와 동일 (5가지)
enum class MonoSource {
    Luminance = 0,
    Red = 1,
    Green = 2,
    Blue = 3,
    Alpha = 4
};

// BlobTrack 노드 - Blob 감지 및 추적
class BlobTrackNode : public TOPNodeBase
{
public:
    BlobTrackNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);
    virtual ~BlobTrackNode();

    // NodeBase 인터페이스 구현
    void Render(Graph<Node>& graph) override;
    void RenderInspector() override;
    std::string GetTypeName() const override { return "BlobTrack"; }
    std::string GetCategory() const override { return "TOP/Analysis"; }

    // TOPNodeBase 인터페이스 구현
    void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) override;

    // 캐시 무효화 오버라이드
    void InvalidateCache() override;

    // BlobTrack 파라미터
    MonoSource GetMonoSource() const { return mono_source_; }
    void SetMonoSource(MonoSource source) { mono_source_ = source; }

    float GetThreshold() const { return threshold_; }
    void SetThreshold(float threshold) { threshold_ = threshold; }

    int GetMinBlobSize() const { return min_blob_size_; }
    void SetMinBlobSize(int size) { min_blob_size_ = size; }

    int GetMaxBlobSize() const { return max_blob_size_; }
    void SetMaxBlobSize(int size) { max_blob_size_ = size; }

    // Max Blobs (터치디자이너 호환 - 동적 설정 가능)
    int GetMaxBlobs() const { return max_blobs_; }
    void SetMaxBlobs(int max_blobs);

    float GetMaxMoveDistance() const { return max_move_distance_; }
    void SetMaxMoveDistance(float dist) { max_move_distance_ = dist; }

    bool GetDrawBlobBounds() const { return draw_blob_bounds_; }
    void SetDrawBlobBounds(bool draw) { draw_blob_bounds_ = draw; }

    // Blob Bound Color (터치디자이너 호환)
    void GetBlobBoundColor(float color[4]) const;
    void SetBlobBoundColor(const float color[4]);

    // Draw Lines Between Blobs (터치디자이너 호환)
    bool GetDrawLines() const { return draw_lines_; }
    void SetDrawLines(bool draw) { draw_lines_ = draw; }
    float GetLineDistance() const { return line_distance_; }
    void SetLineDistance(float dist) { line_distance_ = dist; }
    void GetLineColor(float color[4]) const;
    void SetLineColor(const float color[4]);

    // Delete Nearby Blobs (터치디자이너 호환)
    bool GetDeleteNearby() const { return delete_nearby_; }
    void SetDeleteNearby(bool enable) { delete_nearby_ = enable; }
    float GetDeleteDistance() const { return delete_distance_; }
    void SetDeleteDistance(float dist) { delete_distance_ = dist; }
    float GetDeleteAreaTolerance() const { return delete_area_tolerance_; }
    void SetDeleteAreaTolerance(float tol) { delete_area_tolerance_ = tol; }

    // Delete Overlapping Blobs (터치디자이너 호환)
    bool GetDeleteOverlapping() const { return delete_overlapping_; }
    void SetDeleteOverlapping(bool enable) { delete_overlapping_ = enable; }
    float GetDeleteOverlapTolerance() const { return delete_overlap_tolerance_; }
    void SetDeleteOverlapTolerance(float tol) { delete_overlap_tolerance_ = std::max(0.0f, std::min(1.0f, tol)); }

    // Revive Blobs (터치디자이너 호환)
    bool GetReviveBlobs() const { return revive_blobs_; }
    void SetReviveBlobs(bool enable) { revive_blobs_ = enable; }
    float GetReviveTime() const { return revive_time_; }
    void SetReviveTime(float time) { revive_time_ = std::max(0.0f, time); }
    float GetReviveAreaDifference() const { return revive_area_difference_; }
    void SetReviveAreaDifference(float diff) { revive_area_difference_ = std::max(0.0f, std::min(1.0f, diff)); }
    float GetReviveDistance() const { return revive_distance_; }
    void SetReviveDistance(float dist) { revive_distance_ = std::max(0.0f, dist); }
    
    // Include Lost/Expired Blobs in Table (터치디자이너 호환)
    bool GetIncludeLostBlobsInTable() const { return include_lost_blobs_in_table_; }
    void SetIncludeLostBlobsInTable(bool include) { include_lost_blobs_in_table_ = include; }
    bool GetIncludeExpiredBlobsInTable() const { return include_expired_blobs_in_table_; }
    void SetIncludeExpiredBlobsInTable(bool include) { include_expired_blobs_in_table_ = include; }
    float GetExpiredTime() const { return expired_time_; }
    void SetExpiredTime(float time) { expired_time_ = std::max(0.0f, time); }

    // Reset (터치디자이너 호환)
    void Reset();
    
    // Event-Driven Algorithm Options (사용자 설정)
    bool GetUsePrediction() const { return use_prediction_; }
    void SetUsePrediction(bool enable) { use_prediction_ = enable; }
    
    bool GetUseHungarian() const { return use_hungarian_; }
    void SetUseHungarian(bool enable) { use_hungarian_ = enable; }
    
    int GetHungarianThreshold() const { return hungarian_threshold_; }
    void SetHungarianThreshold(int threshold) { hungarian_threshold_ = std::max(10, std::min(1000, threshold)); }
    
    float GetEventThreshold() const { return event_threshold_; }
    void SetEventThreshold(float threshold) { event_threshold_ = std::max(0.01f, std::min(1.0f, threshold)); }
    
    bool GetUseEventDetection() const { return use_event_detection_; }
    void SetUseEventDetection(bool enable) { use_event_detection_ = enable; }
    
    float GetVelocitySmoothing() const { return velocity_smoothing_; }
    void SetVelocitySmoothing(float smoothing) { velocity_smoothing_ = std::max(0.0f, std::min(1.0f, smoothing)); }

    // Blob 정보 접근
    const std::vector<BlobInfo>& GetBlobs() const { return blobs_; }
    const std::vector<BlobInfo>& GetLostBlobs() const { return lost_blobs_; }
    int GetBlobCount() const { return static_cast<int>(blobs_.size()); }
    
    // GPU Indirect Draw 지원: GPU 버퍼 직접 접근
    id<MTLBuffer> GetGPUBlobBuffer() const {
        if (render_buffer_index_ >= 0) {
            return use_filtered_buffer_[render_buffer_index_] 
                ? filtered_track_buffers_[render_buffer_index_]
                : track_buffers_[render_buffer_index_];
        }
        return nil;
    }
    id<MTLBuffer> GetGPUBlobCountBuffer() const {
        if (render_buffer_index_ >= 0) {
            return use_filtered_buffer_[render_buffer_index_]
                ? filtered_track_counter_buffers_[render_buffer_index_]
                : track_counter_buffers_[render_buffer_index_];
        }
        return nil;
    }

private:
    // Metal 리소스
    id<MTLDevice> device_;
    id<MTLComputePipelineState> threshold_pipeline_;
    id<MTLComputePipelineState> background_subtraction_pipeline_;
    id<MTLComputePipelineState> detect_pipeline_;
    id<MTLComputePipelineState> draw_bounds_pipeline_;
    id<MTLComputePipelineState> draw_lines_pipeline_;
    id<MTLComputePipelineState> match_pipeline_;        // GPU 매칭
    id<MTLComputePipelineState> zero_counts_pipeline_;  // 그리드 카운트 초기화
    id<MTLComputePipelineState> build_bins_pipeline_;   // 그리드 binning
    id<MTLComputePipelineState> filter_delete_pipeline_; // GPU 삭제/겹침 필터
    id<MTLComputePipelineState> compact_pipeline_;       // 삭제 플래그 압축
    
    // Event-Driven Architecture: GPU Compute Pipelines
    id<MTLComputePipelineState> event_detection_pipeline_;   // Pixel Change Event 생성
    id<MTLComputePipelineState> event_clustering_pipeline_;  // Local Region Activation
    id<MTLComputePipelineState> temporal_prediction_pipeline_;  // Kalman Filter 예측
    id<MTLCommandQueue> command_queue_;

    // GPU Blob detection buffers (GPU Zero-Copy, double-buffered)
    id<MTLBuffer> blob_counter_buffers_[2];  // atomic counter (double buffer)
    id<MTLBuffer> blob_data_buffers_[2];     // blob data array (double buffer)
    int current_buffer_index_ = 0;           // GPU write 대상
    int ready_buffer_index_ = -1;            // GPU 완료 후 CPU가 읽을 인덱스
    int render_buffer_index_ = -1;           // GPU draw 시 사용할 인덱스
    bool has_ready_buffer_ = false;          // GPU 완료된 버퍼 존재 여부

    // GPU Tracking buffers (GPU only, double-buffered)
    id<MTLBuffer> track_buffers_[2];          // 트랙 결과 (x,y,w,h,area,id)
    id<MTLBuffer> track_counter_buffers_[2];  // 트랙 개수
    id<MTLBuffer> filtered_track_buffers_[2]; // 필터링된 트랙 결과
    id<MTLBuffer> filtered_track_counter_buffers_[2]; // 필터링된 개수
    id<MTLBuffer> delete_flags_buffers_[2];   // 삭제 플래그 (uint8)
    bool use_filtered_buffer_[2] = {false, false}; // 렌더 시 필터 적용 여부
    id<MTLBuffer> track_id_counter_buffer_;   // 글로벌 ID 카운터
    id<MTLBuffer> cell_counts_buffer_;        // grid counts (prev track or filter)
    id<MTLBuffer> cell_bins_buffer_;          // grid bins (indices)
    int cell_capacity_ = 0;                   // grid_w * grid_h

    id<MTLBuffer> event_buffer_;         // pixel change events (GPU only)
    id<MTLBuffer> event_counter_buffer_; // event counter
    id<MTLBuffer> cost_matrix_buffer_;   // Hungarian algorithm cost matrix
    
    // Event-Driven Architecture: Previous Frame (GPU Zero-Copy)
    id<MTLTexture> previous_frame_texture_;  // 이전 프레임 (이벤트 감지용)
    
    // Event-Driven Algorithm Options (사용자 설정)
    bool use_prediction_;       // Kalman Filter 예측 사용 여부
    bool use_hungarian_;        // Hungarian Algorithm 사용 여부
    int hungarian_threshold_;   // Hungarian 알고리즘 활성화 임계값 (blob 개수)
    float event_threshold_;     // 픽셀 변화 임계값 (Δv) - Event Detection용
    bool use_event_detection_;  // Event Detection 사용 여부
    float velocity_smoothing_;  // 속도 스무딩 계수 (0.0 ~ 1.0)

    // 파라미터
    MonoSource mono_source_;
    float threshold_;
    int min_blob_size_;
    int max_blob_size_;
    int max_blobs_;  // 최대 blob 개수 (터치디자이너 호환 - 동적 설정)
    float max_move_distance_;
    bool draw_blob_bounds_;
    float blob_bound_color_[4];  // Blob 경계선 색상 (RGBA)

    // Draw Lines 옵션
    bool draw_lines_;
    float line_distance_;  // 선을 그릴 최대 거리
    float line_color_[4];  // 선 색상 (RGBA)

    // Delete Nearby/Overlapping 옵션
    bool delete_nearby_;
    float delete_distance_;
    float delete_area_tolerance_;
    bool delete_overlapping_;
    float delete_overlap_tolerance_;  // 터치디자이너 호환

    // Revive Blobs 옵션 (터치디자이너 호환)
    bool revive_blobs_;
    float revive_time_;              // 초 단위
    float revive_area_difference_;  // 0.0 ~ 1.0
    float revive_distance_;         // 픽셀 단위
    
    // Include Lost/Expired Blobs 옵션 (터치디자이너 호환)
    bool include_lost_blobs_in_table_;
    bool include_expired_blobs_in_table_;
    float expired_time_;  // 초 단위

    // Blob 데이터 (Event-Driven State Graph)
    std::vector<BlobInfo> blobs_;
    std::vector<BlobInfo> lost_blobs_;  // 사라진 blob들 (복구 가능)
    int next_blob_id_;
    float current_time_;  // 현재 시간 (초)
    
    // Event-Driven Architecture: Temporal Prediction
    float delta_time_;  // 프레임 간 시간 차이
    int frame_width_ = 0;
    int frame_height_ = 0;

    // NOTE: CPU 매칭용 그리드 버퍼 제거됨 - GPU 그리드 사용
    // cell_counts_buffer_, cell_bins_buffer_가 GPU에서 동일 역할 수행

    // 임시 텍스처 (GPU Zero-Copy)
    id<MTLTexture> binary_texture_;
    id<MTLTexture> event_texture_;  // 이벤트 맵 (시각화용)

    // Metal 리소스 초기화
    bool InitializeMetal();

    // GPU-only 파이프라인 사용 - CPU 추적 함수 제거됨
    // 모든 추적/필터링은 GPU 커널에서 수행 (matchBlobs, filterDelete, compactBlobs)
    void TemporalPrediction(float delta_time);  // Kalman Filter 예측 (GPU 전용)
    void HungarianAssignment(const std::vector<BlobInfo>& new_blobs, std::vector<BlobInfo>& tracked_blobs);
};

// 팩토리 함수 (레지스트리 등록용)
std::unique_ptr<NodeBase> CreateBlobTrackNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device);

} // namespace nodes
} // namespace example
