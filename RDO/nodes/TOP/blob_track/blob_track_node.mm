#include "blob_track_node.h"
#include "../../../core/node_system/node_registry.h"
#include "../../../texture_pool.h"
#include <imgui.h>
#include <imnodes.h>
#include <cmath>
#include <algorithm>
#include <simd/simd.h>

#import <Metal/Metal.h>

namespace example
{
namespace nodes
{

using ::example::nodes::Node;

// GPU Blob 데이터 구조체 (Metal buffer용)
struct GPUBlobData {
    float x, y;           // 중심점
    float width, height;  // 크기
    float area;           // 면적
    int id;               // ID (CPU에서 할당)
    int _padding[2];      // 16바이트 정렬
};


// Metal 셰이더 코드 (GPU Zero-Copy)
static const char* blobTrackShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Blob 데이터 구조체
struct BlobData {
    float x, y;
    float width, height;
    float area;
    int id;
    int _padding[2];
};

struct ShapeInstance {
    float2 center;
    float2 size;
};

struct ShapeUniforms {
    float2 invTexSize;
    float4 fillColor;
    float4 strokeColor;
    float  strokeWidth;
    int    shapeType;
    bool   fillEnabled;
};

// GPU-only blob matching with spatial bins
// MAX_PER_CELL는 호스트와 동일하게 256으로 가정
constant int kMaxPerCell = 256;
kernel void matchBlobs(
    device const BlobData* newBlobs [[buffer(0)]],
    device const BlobData* prevBlobs [[buffer(1)]],
    device const atomic_uint* cellCounts [[buffer(2)]],
    device const uint* cellBins [[buffer(3)]],
    device BlobData* outBlobs [[buffer(4)]],
    device atomic_uint* outCount [[buffer(5)]],
    device atomic_uint* idCounter [[buffer(6)]],
    device atomic_uint* newCountBuf [[buffer(7)]],
    device atomic_uint* prevCountBuf [[buffer(8)]],
    constant float& maxMoveDistSq [[buffer(9)]],
    constant int& gridW [[buffer(10)]],
    constant int& gridH [[buffer(11)]],
    constant float& cellSizeInv [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint newCount = atomic_load_explicit(newCountBuf, memory_order_relaxed);
    uint prevCount = atomic_load_explicit(prevCountBuf, memory_order_relaxed);
    if (gid >= newCount) return;

    BlobData nb = newBlobs[gid];
    float bestDist = maxMoveDistSq;
    int bestIdx = -1;

    int cgx = clamp((int)(nb.x * cellSizeInv), 0, gridW - 1);
    int cgy = clamp((int)(nb.y * cellSizeInv), 0, gridH - 1);

    for (int dy = -1; dy <= 1; ++dy) {
        int gy = cgy + dy;
        if (gy < 0 || gy >= gridH) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int gx = cgx + dx;
            if (gx < 0 || gx >= gridW) continue;
            int cell = gy * gridW + gx;
            uint count = atomic_load_explicit(&cellCounts[cell], memory_order_relaxed);
            count = min(count, (uint)kMaxPerCell);
            int base = cell * kMaxPerCell;
            for (uint k = 0; k < count; ++k) {
                uint idx = cellBins[base + k];
                if (idx >= prevCount) continue;
                BlobData pb = prevBlobs[idx];
                float dxv = nb.x - pb.x;
                float dyv = nb.y - pb.y;
                float d2 = dxv * dxv + dyv * dyv;
                if (d2 < bestDist) {
                    bestDist = d2;
                    bestIdx = (int)idx;
                    if (d2 < 0.01f) break;
                }
            }
        }
    }

    uint outIdx = atomic_fetch_add_explicit(outCount, 1, memory_order_relaxed);
    BlobData ob;
    ob.x = nb.x;
    ob.y = nb.y;
    ob.width = nb.width;
    ob.height = nb.height;
    ob.area = nb.area;
    if (bestIdx >= 0) {
        ob.id = prevBlobs[bestIdx].id;
    } else {
        ob.id = (int)atomic_fetch_add_explicit(idCounter, 1, memory_order_relaxed);
    }
    ob._padding[0] = 0;
    ob._padding[1] = 0;
    outBlobs[outIdx] = ob;
}

// zero cell counts
kernel void zeroCounts(
    device atomic_uint* counts [[buffer(0)]],
    constant uint& total [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < total) {
        atomic_store_explicit(&counts[gid], 0, memory_order_relaxed);
    }
}

// bin prev tracked blobs into grid cells
kernel void buildBins(
    device const BlobData* prevBlobs [[buffer(0)]],
    device atomic_uint* cellCounts [[buffer(1)]],
    device uint* cellBins [[buffer(2)]],
    device atomic_uint* prevCountBuf [[buffer(3)]],
    constant int& gridW [[buffer(4)]],
    constant int& gridH [[buffer(5)]],
    constant float& cellSizeInv [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint prevCount = atomic_load_explicit(prevCountBuf, memory_order_relaxed);
    if (gid >= prevCount) return;
    BlobData pb = prevBlobs[gid];
    int gx = clamp((int)(pb.x * cellSizeInv), 0, gridW - 1);
    int gy = clamp((int)(pb.y * cellSizeInv), 0, gridH - 1);
    int cell = gy * gridW + gx;
    uint idx = atomic_fetch_add_explicit(&cellCounts[cell], 1, memory_order_relaxed);
    if (idx < (uint)kMaxPerCell) {
        cellBins[cell * kMaxPerCell + idx] = gid;
    }
}

// Pass: 삭제/겹침 필터 (GPU 옵션 처리)
kernel void filterDelete(
    device const BlobData* blobs [[buffer(0)]],
    device atomic_uint* deleteFlags [[buffer(1)]], // uint8처럼 사용 (0/1)
    device const atomic_uint* cellCounts [[buffer(2)]],
    device const uint* cellBins [[buffer(3)]],
    device const atomic_uint* countBuf [[buffer(4)]],
    constant float& deleteDist [[buffer(5)]],
    constant float& areaTol [[buffer(6)]],
    constant float& overlapTol [[buffer(7)]],
    constant int& gridW [[buffer(8)]],
    constant int& gridH [[buffer(9)]],
    constant float& cellSizeInv [[buffer(10)]],
    constant bool& useNear [[buffer(11)]],
    constant bool& useOverlap [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = atomic_load_explicit(countBuf, memory_order_relaxed);
    if (gid >= count) return;

    BlobData self = blobs[gid];
    float deleteDistSq = deleteDist * deleteDist;

    int cgx = clamp((int)(self.x * cellSizeInv), 0, gridW - 1);
    int cgy = clamp((int)(self.y * cellSizeInv), 0, gridH - 1);

    bool markDelete = false;

    for (int dy = -1; dy <= 1 && !markDelete; ++dy) {
        int gy = cgy + dy;
        if (gy < 0 || gy >= gridH) continue;
        for (int dx = -1; dx <= 1 && !markDelete; ++dx) {
            int gx = cgx + dx;
            if (gx < 0 || gx >= gridW) continue;
            int cell = gy * gridW + gx;
            uint localCount = atomic_load_explicit(&cellCounts[cell], memory_order_relaxed);
            localCount = min(localCount, (uint)kMaxPerCell);
            int base = cell * kMaxPerCell;
            for (uint k = 0; k < localCount; ++k) {
                uint idx = cellBins[base + k];
                if (idx >= count || idx == gid) continue;
                BlobData other = blobs[idx];

                // 근접 삭제
                if (useNear) {
                    float dxv = self.x - other.x;
                    float dyv = self.y - other.y;
                    float d2 = dxv * dxv + dyv * dyv;
                    if (d2 < deleteDistSq) {
                        float areaRatio = min(self.area, other.area) / max(self.area, other.area);
                        if (areaRatio >= areaTol) {
                            // 더 작은 쪽 삭제
                            if (self.area < other.area) markDelete = true;
                        }
                    }
                }

                // 겹침 삭제
                if (!markDelete && useOverlap) {
                    float self_left = self.x - self.width * 0.5f;
                    float self_right = self.x + self.width * 0.5f;
                    float self_top = self.y - self.height * 0.5f;
                    float self_bottom = self.y + self.height * 0.5f;

                    float o_left = other.x - other.width * 0.5f;
                    float o_right = other.x + other.width * 0.5f;
                    float o_top = other.y - other.height * 0.5f;
                    float o_bottom = other.y + other.height * 0.5f;

                    float overlap_left = max(self_left, o_left);
                    float overlap_right = min(self_right, o_right);
                    float overlap_top = max(self_top, o_top);
                    float overlap_bottom = min(self_bottom, o_bottom);

                    if (overlap_left < overlap_right && overlap_top < overlap_bottom) {
                        float overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top);
                        float min_area = min(self.width * self.height, other.width * other.height);
                        float overlap_ratio = overlap_area / min_area;
                        if (overlap_ratio >= overlapTol) {
                            if (self.area < other.area) markDelete = true;
                        }
                    }
                }

                if (markDelete) break;
            }
        }
    }

    if (markDelete) {
        atomic_store_explicit(&deleteFlags[gid], 1, memory_order_relaxed);
    } else {
        atomic_store_explicit(&deleteFlags[gid], 0, memory_order_relaxed);
    }
}

// Pass: 삭제 플래그를 적용해 압축
kernel void compactBlobs(
    device const BlobData* inBlobs [[buffer(0)]],
    device const atomic_uint* deleteFlags [[buffer(1)]],
    device BlobData* outBlobs [[buffer(2)]],
    device atomic_uint* outCount [[buffer(3)]],
    device const atomic_uint* countBuf [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = atomic_load_explicit(countBuf, memory_order_relaxed);
    if (gid >= count) return;
    uint flag = atomic_load_explicit(&deleteFlags[gid], memory_order_relaxed);
    if (flag == 0) {
        uint dst = atomic_fetch_add_explicit(outCount, 1, memory_order_relaxed);
        outBlobs[dst] = inBlobs[gid];
    }
}

// Pass 1: Threshold + Mono Source (SimpleBlobDetector 모드)
kernel void computeThreshold(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant int &monoSource [[buffer(0)]],
    constant float &threshold [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    float4 color = inputTexture.read(gid);

    // Mono source 계산 (터치디자이너와 동일 - 5가지만)
    float value = 0.0f;
    switch (monoSource) {
        case 0: // Luminance
            value = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
            break;
        case 1: value = color.r; break;  // Red
        case 2: value = color.g; break;  // Green
        case 3: value = color.b; break;  // Blue
        case 4: value = color.a; break;  // Alpha
    }

    // Threshold
    float result = (value >= threshold) ? 1.0 : 0.0;
    outputTexture.write(float4(result, result, result, 1.0), gid);
}

// Pass 1b: Background Subtraction (BackgroundSubtraction 모드)
kernel void computeBackgroundSubtraction(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::read> backgroundTexture [[texture(1)]],
    texture2d<float, access::write> outputTexture [[texture(2)]],
    constant int &monoSource [[buffer(0)]],
    constant float &threshold [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    float4 inputColor = inputTexture.read(gid);
    float4 bgColor = backgroundTexture.read(gid);

    // Mono source 계산 (터치디자이너와 동일 - 5가지만)
    float inputValue = 0.0f;
    float bgValue = 0.0f;
    switch (monoSource) {
        case 0: // Luminance
            inputValue = 0.299 * inputColor.r + 0.587 * inputColor.g + 0.114 * inputColor.b;
            bgValue = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
            break;
        case 1:
            inputValue = inputColor.r;
            bgValue = bgColor.r;
            break;
        case 2:
            inputValue = inputColor.g;
            bgValue = bgColor.g;
            break;
        case 3:
            inputValue = inputColor.b;
            bgValue = bgColor.b;
            break;
        case 4:
            inputValue = inputColor.a;
            bgValue = bgColor.a;
            break;
    }

    // Background subtraction
    float diff = abs(inputValue - bgValue);
    float result = (diff >= threshold) ? 1.0 : 0.0;
    outputTexture.write(float4(result, result, result, 1.0), gid);
}

// Pass 2: Optimized Blob Detection (4x4 다운샘플 + 균등 분포)
// 4K 60fps에서 1000+ blobs 처리를 위한 최적화
// 핵심 수정: dispatch 크기를 (width/4, height/4)로 축소하여 모든 스레드 유효
constant int kSampleStep = 4;

kernel void detectBlobs(
    texture2d<float, access::read> binaryTexture [[texture(0)]],
    device atomic_uint* blobCounter [[buffer(0)]],
    device BlobData* blobs [[buffer(1)]],
    constant int &minSize [[buffer(2)]],
    constant int &maxSize [[buffer(3)]],
    constant int &maxBlobs [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    int width = int(binaryTexture.get_width());
    int height = int(binaryTexture.get_height());

    // gid는 이제 다운샘플된 좌표 (dispatch가 width/4 x height/4)
    // 실제 이미지 좌표로 변환
    int2 samplePos = int2(gid) * kSampleStep + kSampleStep / 2;
    
    if (samplePos.x >= width || samplePos.y >= height) return;

    // 4x4 블록에서 최소 1개 픽셀이 흰색이면 blob 후보
    bool hasWhite = false;
    for (int dy = 0; dy < kSampleStep && !hasWhite; dy++) {
        for (int dx = 0; dx < kSampleStep && !hasWhite; dx++) {
            int2 checkPos = int2(gid) * kSampleStep + int2(dx, dy);
            if (checkPos.x < width && checkPos.y < height) {
                if (binaryTexture.read(uint2(checkPos)).r >= 0.5) {
                    hasWhite = true;
                }
            }
        }
    }
    
    if (!hasWhite) return;

    // Local centroid 찾기 (32x32 스캔 - 더 넓은 영역)
    int scanRange = 32;
    
    int area = 0;
    int minX = width, maxX = 0;
    int minY = height, maxY = 0;
    float sumX = 0.0f, sumY = 0.0f;
    int sumCount = 0;

    // 2픽셀 스텝으로 빠른 탐색
    for (int dy = -scanRange; dy <= scanRange; dy += 2) {
        for (int dx = -scanRange; dx <= scanRange; dx += 2) {
            int2 pos = samplePos + int2(dx, dy);
            if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) continue;

            if (binaryTexture.read(uint2(pos)).r >= 0.5) {
                area += 4;  // 2x2 블록으로 카운트
                minX = min(minX, pos.x);
                maxX = max(maxX, pos.x);
                minY = min(minY, pos.y);
                maxY = max(maxY, pos.y);
                sumX += float(pos.x);
                sumY += float(pos.y);
                sumCount++;
            }
        }
    }

    // 크기 필터링
    if (area < minSize || area > maxSize) return;
    if (sumCount == 0) return;

    // 무게중심 계산
    float centerX = sumX / float(sumCount);
    float centerY = sumY / float(sumCount);

    // 중복 blob 방지: 무게중심이 현재 샘플 셀 내에 있는 경우에만 추가
    // 이렇게 하면 각 blob이 정확히 하나의 셀에서만 등록됨
    int cellX = int(centerX) / kSampleStep;
    int cellY = int(centerY) / kSampleStep;
    
    if (cellX != int(gid.x) || cellY != int(gid.y)) return;

    // Blob 추가 (atomic)
    uint index = atomic_fetch_add_explicit(blobCounter, 1, memory_order_relaxed);
    if (index >= uint(maxBlobs)) return;

    blobs[index].x = centerX;
    blobs[index].y = centerY;
    blobs[index].width = float(maxX - minX + 1);
    blobs[index].height = float(maxY - minY + 1);  // 수정: minX → minY
    blobs[index].area = float(area);
    blobs[index].id = -1;  // GPU matchBlobs에서 할당
}

// Pass 3: Draw Blob Bounds
kernel void drawBlobBounds(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    device BlobData* blobs [[buffer(0)]],
    constant int &blobCount [[buffer(1)]],
    constant float4 &boundColor [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    float4 color = inputTexture.read(gid);

    for (int i = 0; i < blobCount; i++) {
        BlobData blob = blobs[i];

        float left = blob.x - blob.width / 2.0;
        float right = blob.x + blob.width / 2.0;
        float top = blob.y - blob.height / 2.0;
        float bottom = blob.y + blob.height / 2.0;

        float px = float(gid.x);
        float py = float(gid.y);

        // 2픽셀 두께 경계선
        if ((px >= left - 1 && px <= left + 1 && py >= top && py <= bottom) ||
            (px >= right - 1 && px <= right + 1 && py >= top && py <= bottom) ||
            (py >= top - 1 && py <= top + 1 && px >= left && px <= right) ||
            (py >= bottom - 1 && py <= bottom + 1 && px >= left && px <= right))
        {
            color = boundColor;  // 사용자 설정 색상
        }
    }

    outputTexture.write(color, gid);
}

// ============================================================================
// EVENT-DRIVEN BLOB TRACKING (자체 알고리즘)
// ============================================================================

// Event-Driven Pass 1: Pixel Change Event Detection (GPU Zero-Copy)
// 이전 프레임과 현재 프레임의 차이를 계산하여 이벤트 생성
kernel void detectPixelChangeEvents(
    texture2d<float, access::read> currentFrame [[texture(0)]],
    texture2d<float, access::read> previousFrame [[texture(1)]],
    texture2d<float, access::write> eventTexture [[texture(2)]],
    device atomic_uint* eventCounter [[buffer(0)]],
    constant float &eventThreshold [[buffer(1)]],
    constant int &monoSource [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= currentFrame.get_width() || gid.y >= currentFrame.get_height()) return;

    float4 current = currentFrame.read(gid);
    float4 previous = previousFrame.read(gid);

    // Mono source 계산
    float currentValue = 0.0f;
    float previousValue = 0.0f;
    switch (monoSource) {
        case 0: // Luminance
            currentValue = 0.299 * current.r + 0.587 * current.g + 0.114 * current.b;
            previousValue = 0.299 * previous.r + 0.587 * previous.g + 0.114 * previous.b;
            break;
        case 1:
            currentValue = current.r;
            previousValue = previous.r;
            break;
        case 2:
            currentValue = current.g;
            previousValue = previous.g;
            break;
        case 3:
            currentValue = current.b;
            previousValue = previous.b;
            break;
        case 4:
            currentValue = current.a;
            previousValue = previous.a;
            break;
    }

    // 픽셀 변화량 계산 (Δv)
    float delta = abs(currentValue - previousValue);

    // 이벤트 발생 조건: |I_t(x,y) - I_{t-Δt}(x,y)| > θ
    if (delta > eventThreshold) {
        // 이벤트 카운터 증가
        atomic_fetch_add_explicit(eventCounter, 1, memory_order_relaxed);
        
        // 이벤트 텍스처에 표시 (시각화 + clustering용)
        eventTexture.write(float4(delta, delta, delta, 1.0), gid);
    } else {
        eventTexture.write(float4(0.0, 0.0, 0.0, 1.0), gid);
    }
}

// Event-Driven Pass 2: Event Clustering (Local Region Activation)
// 공간-시간적으로 가까운 이벤트들을 클러스터링하여 blob 생성
// dispatch 크기: (width/4, height/4) - detectBlobs와 동일
kernel void clusterEventsToBlobs(
    texture2d<float, access::read> eventTexture [[texture(0)]],
    device atomic_uint* blobCounter [[buffer(0)]],
    device BlobData* blobs [[buffer(1)]],
    constant int &minSize [[buffer(2)]],
    constant int &maxSize [[buffer(3)]],
    constant int &maxBlobs [[buffer(4)]],
    constant float &clusterRadius [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int width = int(eventTexture.get_width());
    int height = int(eventTexture.get_height());

    // gid는 다운샘플된 좌표 (dispatch가 width/4 x height/4)
    int2 samplePos = int2(gid) * kSampleStep + kSampleStep / 2;
    
    if (samplePos.x >= width || samplePos.y >= height) return;

    float eventValue = eventTexture.read(uint2(samplePos)).r;
    if (eventValue < 0.01) return;  // 이벤트가 없으면 스킵

    // 클러스터링: 인접한 이벤트들을 묶어서 blob으로 만듦
    int clusterRadiusInt = int(clusterRadius);
    int area = 0;
    int minX = width, maxX = 0;
    int minY = height, maxY = 0;
    float sumX = 0.0f, sumY = 0.0f;
    int sumCount = 0;

    // 2픽셀 스텝으로 빠른 탐색
    for (int dy = -clusterRadiusInt; dy <= clusterRadiusInt; dy += 2) {
        for (int dx = -clusterRadiusInt; dx <= clusterRadiusInt; dx += 2) {
            int2 pos = samplePos + int2(dx, dy);
            if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) continue;

            float ev = eventTexture.read(uint2(pos)).r;
            if (ev >= 0.01) {
                area += 4;  // 2x2 블록
                minX = min(minX, pos.x);
                maxX = max(maxX, pos.x);
                minY = min(minY, pos.y);
                maxY = max(maxY, pos.y);
                sumX += float(pos.x);
                sumY += float(pos.y);
                sumCount++;
            }
        }
    }

    // 크기 필터링
    if (area < minSize || area > maxSize) return;
    if (sumCount == 0) return;

    // 무게중심 계산
    float centerX = sumX / float(sumCount);
    float centerY = sumY / float(sumCount);

    // 중복 blob 방지: 무게중심이 현재 샘플 셀 내에 있는 경우에만 추가
    int cellX = int(centerX) / kSampleStep;
    int cellY = int(centerY) / kSampleStep;
    
    if (cellX != int(gid.x) || cellY != int(gid.y)) return;

    // Blob 추가 (atomic)
    uint index = atomic_fetch_add_explicit(blobCounter, 1, memory_order_relaxed);
    if (index >= uint(maxBlobs)) return;

    blobs[index].x = centerX;
    blobs[index].y = centerY;
    blobs[index].width = float(maxX - minX + 1);
    blobs[index].height = float(maxY - minY + 1);
    blobs[index].area = float(area);
    blobs[index].id = -1;  // GPU matchBlobs에서 할당
}

// Pass 3b: Draw Lines Between Blobs (터치디자이너 호환 - 최적화)
kernel void drawBlobLines(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    device BlobData* blobs [[buffer(0)]],
    constant int &blobCount [[buffer(1)]],
    constant float &maxDistance [[buffer(2)]],
    constant float4 &lineColor [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    float4 color = inputTexture.read(gid);
    float px = float(gid.x);
    float py = float(gid.y);
    
    // 최대 거리 제곱 (sqrt 제거로 성능 향상)
    float maxDistSq = maxDistance * maxDistance;
    float lineThickness = 1.5;  // 선 두께
    float lineThicknessSq = lineThickness * lineThickness;

    // 모든 blob 쌍에 대해 선 그리기 (최적화된 버전)
    for (int i = 0; i < blobCount; i++) {
        for (int j = i + 1; j < blobCount; j++) {
            BlobData blob1 = blobs[i];
            BlobData blob2 = blobs[j];

            float2 p1 = float2(blob1.x, blob1.y);
            float2 p2 = float2(blob2.x, blob2.y);
            float2 p = float2(px, py);

            // 1. 거리 제곱으로 빠른 체크 (sqrt 제거)
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float distSq = dx * dx + dy * dy;
            
            if (distSq > maxDistSq) continue;  // 거리 초과 시 스킵
            if (distSq < 0.001) continue;  // 너무 가까운 blob 스킵

            // 2. AABB 체크로 선분이 픽셀 근처에 있는지 빠르게 확인
            float minX = min(p1.x, p2.x) - lineThickness;
            float maxX = max(p1.x, p2.x) + lineThickness;
            float minY = min(p1.y, p2.y) - lineThickness;
            float maxY = max(p1.y, p2.y) + lineThickness;
            
            if (px < minX || px > maxX || py < minY || py > maxY) continue;

            // 3. 선분과 점의 최단 거리 계산 (제곱 거리로 비교)
            float2 line = p2 - p1;
            float lineLengthSq = distSq;
            float2 toPoint = p - p1;
            float t = dot(toPoint, line) / lineLengthSq;
            
            // 선분 범위 밖이면 스킵
            if (t < 0.0 || t > 1.0) continue;

            // 가장 가까운 점 계산
            float2 closestPoint = p1 + line * t;
            float2 diff = p - closestPoint;
            float distToLineSq = dot(diff, diff);

            // 선 두께 내에 있는지 확인
            if (distToLineSq < lineThicknessSq) {
                // Alpha blending으로 선 그리기 (여러 선이 겹칠 수 있음)
                color = mix(color, lineColor, lineColor.a);
            }
        }
    }

    outputTexture.write(color, gid);
}
)";

BlobTrackNode::BlobTrackNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
    : device_(device)
    , threshold_pipeline_(nil)
    , background_subtraction_pipeline_(nil)
    , detect_pipeline_(nil)
    , draw_bounds_pipeline_(nil)
    , draw_lines_pipeline_(nil)
    , match_pipeline_(nil)
    , event_detection_pipeline_(nil)
    , event_clustering_pipeline_(nil)
    , temporal_prediction_pipeline_(nil)
    , command_queue_(nil)
    , blob_counter_buffers_{nil, nil}
    , blob_data_buffers_{nil, nil}
    , current_buffer_index_(0)
    , ready_buffer_index_(-1)
    , render_buffer_index_(-1)
    , has_ready_buffer_(false)
    , track_buffers_{nil, nil}
    , track_counter_buffers_{nil, nil}
    , track_id_counter_buffer_(nil)
    , event_buffer_(nil)
    , event_counter_buffer_(nil)
    , cost_matrix_buffer_(nil)
    , previous_frame_texture_(nil)
    , use_prediction_(true)         // 기본: 예측 활성화
    , use_hungarian_(true)          // 기본: Hungarian 활성화
    , hungarian_threshold_(100)     // 기본: 100개 이상에서 Hungarian 사용
    , event_threshold_(0.05f)       // 기본: 5% 픽셀 변화
    , use_event_detection_(false)   // 기본: Event Detection 비활성화 (선택적 기능)
    , velocity_smoothing_(0.7f)     // 기본: 70% 새 값, 30% 이전 값
    , mono_source_(MonoSource::Luminance)
    , threshold_(0.5f)
    , min_blob_size_(10)
    , max_blob_size_(10000)
    , max_blobs_(256)  // 기본값 256 (터치디자이너 호환)
    , max_move_distance_(50.0f)
    , draw_blob_bounds_(true)
    , blob_bound_color_{0.0f, 1.0f, 0.0f, 1.0f}  // 기본 녹색
    , draw_lines_(false)
    , line_distance_(200.0f)  // 기본 최대 거리
    , line_color_{1.0f, 1.0f, 1.0f, 0.5f}  // 기본 흰색 반투명
    , delete_nearby_(false)
    , delete_distance_(10.0f)
    , delete_area_tolerance_(1.0f)
    , delete_overlapping_(false)
    , delete_overlap_tolerance_(1.0f)  // 기본값: 완전 겹침만 삭제
    , revive_blobs_(false)
    , revive_time_(1.0f)  // 기본값: 1초
    , revive_area_difference_(0.5f)  // 기본값: 50% 면적 차이 허용
    , revive_distance_(50.0f)  // 기본값: 50픽셀
    , include_lost_blobs_in_table_(false)
    , include_expired_blobs_in_table_(false)
    , expired_time_(5.0f)  // 기본값: 5초
    , next_blob_id_(0)
    , current_time_(0.0f)
    , delta_time_(0.016f)  // 60fps 기본값
    , binary_texture_(nil)
    , event_texture_(nil)
{
    output_texture_ = nil;
    node_id_ = graph.insert_node(Node(NodeType::value));

    int input1_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int input2_id = graph.insert_node(Node(NodeType::value, 0.0f));
    int output_id = graph.insert_node(Node(NodeType::value, 0.0f));

    AddInputPort(Port(input1_id, NodeFamily::TOP, PortDirection::Input, "texture", "input"));
    AddInputPort(Port(input2_id, NodeFamily::TOP, PortDirection::Input, "texture", "background"));
    AddOutputPort(Port(output_id, NodeFamily::TOP, PortDirection::Output, "texture", "output"));

    InitializeMetal();
    ImNodes::SetNodeScreenSpacePos(node_id_, pos);
}

BlobTrackNode::~BlobTrackNode()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
            output_texture_ = nil;
        }
        if (binary_texture_) {
            last_texture_pool_->release_texture(binary_texture_);
            binary_texture_ = nil;
        }
    }
}

void BlobTrackNode::InvalidateCache()
{
    if (last_texture_pool_) {
        if (output_texture_) {
            last_texture_pool_->release_texture(output_texture_);
        }
        if (binary_texture_) {
            last_texture_pool_->release_texture(binary_texture_);
        }
    }
    output_texture_ = nil;
    binary_texture_ = nil;
}

bool BlobTrackNode::InitializeMetal()
{
    if (!device_) return false;

    command_queue_ = [device_ newCommandQueue];

    NSError* error = nil;
    NSString* shaderCode = [NSString stringWithUTF8String:blobTrackShaderSource];

    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderCode options:nil error:&error];
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
        return false;
    }

    threshold_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"computeThreshold"] error:&error];
    if (error) {
        NSLog(@"Error creating threshold pipeline: %@", error);
        return false;
    }

    background_subtraction_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"computeBackgroundSubtraction"] error:&error];
    if (error) {
        NSLog(@"Error creating background subtraction pipeline: %@", error);
        return false;
    }

    detect_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"detectBlobs"] error:&error];
    if (error) {
        NSLog(@"Error creating detect pipeline: %@", error);
        return false;
    }

    draw_bounds_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"drawBlobBounds"] error:&error];
    if (error) {
        NSLog(@"Error creating draw bounds pipeline: %@", error);
        return false;
    }

    draw_lines_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"drawBlobLines"] error:&error];
    if (error) {
        NSLog(@"Error creating draw lines pipeline: %@", error);
        return false;
    }

    match_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"matchBlobs"] error:&error];
    if (error) {
        NSLog(@"Error creating match pipeline: %@", error);
        return false;
    }

    zero_counts_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"zeroCounts"] error:&error];
    if (error) {
        NSLog(@"Error creating zeroCounts pipeline: %@", error);
        return false;
    }

    build_bins_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"buildBins"] error:&error];
    if (error) {
        NSLog(@"Error creating buildBins pipeline: %@", error);
        return false;
    }

    filter_delete_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"filterDelete"] error:&error];
    if (error) {
        NSLog(@"Error creating filterDelete pipeline: %@", error);
        return false;
    }

    compact_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"compactBlobs"] error:&error];
    if (error) {
        NSLog(@"Error creating compact pipeline: %@", error);
        return false;
    }

    // Event-Driven Architecture: Initialize GPU Pipelines
    event_detection_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"detectPixelChangeEvents"] error:&error];
    if (error) {
        NSLog(@"Error creating event detection pipeline: %@", error);
        return false;
    }

    event_clustering_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"clusterEventsToBlobs"] error:&error];
    if (error) {
        NSLog(@"Error creating event clustering pipeline: %@", error);
        return false;
    }

    // GPU blob detection + tracking buffers (zero-copy, double-buffered)
    for (int i = 0; i < 2; ++i) {
        blob_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        blob_data_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                      options:MTLResourceStorageModeShared];

        track_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                  options:MTLResourceStorageModeShared];
        track_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        filtered_track_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                           options:MTLResourceStorageModeShared];
        filtered_track_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        delete_flags_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t) * max_blobs_
                                                        options:MTLResourceStorageModeShared];

        uint32_t* c1 = (uint32_t*)[blob_counter_buffers_[i] contents];
        uint32_t* c2 = (uint32_t*)[track_counter_buffers_[i] contents];
        uint32_t* c3 = (uint32_t*)[filtered_track_counter_buffers_[i] contents];
        if (c1) *c1 = 0;
        if (c2) *c2 = 0;
        if (c3) *c3 = 0;
        use_filtered_buffer_[i] = false;
    }
    track_id_counter_buffer_ = [device_ newBufferWithLength:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
    if (track_id_counter_buffer_) {
        uint32_t* idp = (uint32_t*)[track_id_counter_buffer_ contents];
        *idp = 0;
    }
    cell_counts_buffer_ = nil;
    cell_bins_buffer_ = nil;
    cell_capacity_ = 0;
    current_buffer_index_ = 0;
    ready_buffer_index_ = -1;
    render_buffer_index_ = -1;
    has_ready_buffer_ = false;

    // Event-Driven Architecture: Event buffers (GPU Zero-Copy)
    event_counter_buffer_ = [device_ newBufferWithLength:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    
    // Cost matrix buffer for Hungarian Algorithm (작은 버퍼 - 최대 500x500)
    cost_matrix_buffer_ = [device_ newBufferWithLength:sizeof(float) * 500 * 500
                                                options:MTLResourceStorageModeShared];

    return true;
}

void BlobTrackNode::SetMaxBlobs(int max_blobs)
{
    // 범위 제한 (1 ~ 10000)
    max_blobs = std::max(1, std::min(10000, max_blobs));
    
    if (max_blobs == max_blobs_) {
        return;  // 변경 없음
    }

    max_blobs_ = max_blobs;

    // GPU 버퍼 재할당 (값이 변경된 경우에만)
    if (device_) {
        for (int i = 0; i < 2; ++i) {
            blob_data_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                          options:MTLResourceStorageModeShared];
            blob_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
            track_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                      options:MTLResourceStorageModeShared];
            track_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
            filtered_track_buffers_[i] = [device_ newBufferWithLength:sizeof(GPUBlobData) * max_blobs_
                                                           options:MTLResourceStorageModeShared];
            filtered_track_counter_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
            delete_flags_buffers_[i] = [device_ newBufferWithLength:sizeof(uint32_t) * max_blobs_
                                                        options:MTLResourceStorageModeShared];
            uint32_t* c1 = (uint32_t*)[blob_counter_buffers_[i] contents];
            uint32_t* c2 = (uint32_t*)[track_counter_buffers_[i] contents];
            uint32_t* c3 = (uint32_t*)[filtered_track_counter_buffers_[i] contents];
            if (c1) *c1 = 0;
            if (c2) *c2 = 0;
            if (c3) *c3 = 0;
            use_filtered_buffer_[i] = false;
        }
    }
}

void BlobTrackNode::Render(Graph<Node>& graph)
{
    const float node_width = 220.0f;
    ImNodes::BeginNode(node_id_);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Blob Track");
    ImNodes::EndNodeTitleBar();

    {
        const Port& port = input_ports_[0];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    {
        const Port& port = input_ports_[1];
        ImNodes::BeginInputAttribute(port.id);
        ImGui::TextUnformatted(port.name.c_str());
        ImNodes::EndInputAttribute();
    }

    ImGui::Spacing();

    ImGui::PushItemWidth(150.0f);

    const char* sources[] = { "Luminance", "Red", "Green", "Blue", "Alpha" };
    int source_idx = static_cast<int>(mono_source_);
    if (ImGui::Combo("Source", &source_idx, sources, 5)) {
        mono_source_ = static_cast<MonoSource>(source_idx);
    }

    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f);
    ImGui::SliderInt("Min Size", &min_blob_size_, 1, 500);
    ImGui::SliderInt("Max Size", &max_blob_size_, 100, 50000);
    ImGui::Checkbox("Draw Bounds", &draw_blob_bounds_);
    ImGui::Checkbox("Draw Lines", &draw_lines_);

    ImGui::PopItemWidth();
    ImGui::Spacing();

    ImGui::Text("Blobs: %d", GetBlobCount());
    ImGui::Spacing();

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

    ImGui::Spacing();

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

void BlobTrackNode::RenderInspector()
{
    ImGui::Text("Blob Track (GPU Detection)");
    ImGui::Separator();

    const char* sources[] = { "Luminance", "Red", "Green", "Blue", "Alpha" };
    int source_idx = static_cast<int>(mono_source_);
    if (ImGui::Combo("Mono Source", &source_idx, sources, 5)) {
        mono_source_ = static_cast<MonoSource>(source_idx);
    }

    ImGui::SliderFloat("Threshold", &threshold_, 0.0f, 1.0f, "%.2f");
    ImGui::SliderInt("Min Blob Size", &min_blob_size_, 1, 500);
    ImGui::SliderInt("Max Blob Size", &max_blob_size_, 100, 50000);
    
    // Max Blobs (터치디자이너 호환 - 동적 설정)
    int max_blobs_temp = max_blobs_;
    if (ImGui::SliderInt("Max Blobs", &max_blobs_temp, 1, 10000)) {
        SetMaxBlobs(max_blobs_temp);  // 버퍼 재할당
    }
    
    ImGui::SliderFloat("Max Move Dist", &max_move_distance_, 10.0f, 500.0f, "%.0f");
    ImGui::Checkbox("Draw Blob Bounds", &draw_blob_bounds_);
    
    if (draw_blob_bounds_) {
        ImGui::ColorEdit4("Blob Bound Color", blob_bound_color_);
    }

    ImGui::Checkbox("Draw Lines", &draw_lines_);
    if (draw_lines_) {
        ImGui::SliderFloat("Line Distance", &line_distance_, 10.0f, 1000.0f, "%.0f");
        ImGui::ColorEdit4("Line Color", line_color_);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Constraints");
    
    ImGui::Checkbox("Delete Nearby Blobs", &delete_nearby_);
    if (delete_nearby_) {
        ImGui::SliderFloat("Min Distance", &delete_distance_, 1.0f, 100.0f, "%.0f");
        ImGui::SliderFloat("Area Tolerance", &delete_area_tolerance_, 0.0f, 1.0f, "%.2f");
    }
    
    ImGui::Checkbox("Delete Overlapping", &delete_overlapping_);
    if (delete_overlapping_) {
        ImGui::SliderFloat("Overlap Tolerance", &delete_overlap_tolerance_, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("(1.0 = complete overlap only)");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Revival");
    
    ImGui::Checkbox("Revive Blobs", &revive_blobs_);
    if (revive_blobs_) {
        ImGui::SliderFloat("Revive Time (s)", &revive_time_, 0.1f, 10.0f, "%.1f");
        ImGui::SliderFloat("Revive Area Diff", &revive_area_difference_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Revive Distance", &revive_distance_, 10.0f, 500.0f, "%.0f");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Info Table");
    
    ImGui::Checkbox("Include Lost Blobs", &include_lost_blobs_in_table_);
    ImGui::Checkbox("Include Expired Blobs", &include_expired_blobs_in_table_);
    if (include_expired_blobs_in_table_) {
        ImGui::SliderFloat("Expired Time (s)", &expired_time_, 1.0f, 30.0f, "%.1f");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Tracking Algorithm");
    ImGui::TextDisabled("(Event-Driven Architecture)");
    
    ImGui::Checkbox("Use Prediction", &use_prediction_);
    if (use_prediction_) {
        ImGui::SliderFloat("Velocity Smoothing", &velocity_smoothing_, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("(Higher = more responsive)");
    }
    
    ImGui::Checkbox("Use Hungarian Algorithm", &use_hungarian_);
    if (use_hungarian_) {
        ImGui::SliderInt("Hungarian Threshold", &hungarian_threshold_, 10, 1000);
        ImGui::TextDisabled("(Activate when blobs >= threshold)");
    }
    
    ImGui::Checkbox("Use Event Detection", &use_event_detection_);
    if (use_event_detection_) {
        ImGui::SliderFloat("Event Threshold", &event_threshold_, 0.01f, 1.0f, "%.3f");
        ImGui::TextDisabled("(GPU Zero-Copy pixel change detection)");
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::Button("Reset")) {
        Reset();
    }

    ImGui::Spacing();
    ImGui::Separator();
    
    // Algorithm Info
    const char* current_algorithm = "Greedy (Fast)";
    if (use_hungarian_ && GetBlobCount() >= hungarian_threshold_) {
        current_algorithm = "Hungarian (Optimal)";
    }
    ImGui::Text("Current Algorithm: %s", current_algorithm);
    ImGui::Text("Detected Blobs: %d", GetBlobCount());
    
    // 디버깅 정보
    if (GetBlobCount() == 0) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No blobs detected!");
        ImGui::TextDisabled("Troubleshooting:");
        ImGui::TextDisabled("- Check input texture");
        ImGui::TextDisabled("- Adjust threshold (%.2f)", threshold_);
        ImGui::TextDisabled("- Check min/max size");
        ImGui::TextDisabled("- Try lower threshold (0.1-0.3)");
    }

    if (!blobs_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Blob Details:");
        for (size_t i = 0; i < blobs_.size() && i < 10; i++) {
            const auto& blob = blobs_[i];
            ImGui::Text("  [%d] pos:(%.0f,%.0f) size:(%.0f,%.0f) area:%.0f",
                blob.id, blob.x, blob.y, blob.width, blob.height, blob.area);
        }
        if (blobs_.size() > 10) {
            ImGui::TextDisabled("  ... and %zu more", blobs_.size() - 10);
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("GPU detection with tracking");
}

void BlobTrackNode::Reset()
{
    blobs_.clear();
    lost_blobs_.clear();
    next_blob_id_ = 0;
    current_time_ = 0.0f;
}

void BlobTrackNode::GetBlobBoundColor(float color[4]) const
{
    color[0] = blob_bound_color_[0];
    color[1] = blob_bound_color_[1];
    color[2] = blob_bound_color_[2];
    color[3] = blob_bound_color_[3];
}

void BlobTrackNode::SetBlobBoundColor(const float color[4])
{
    blob_bound_color_[0] = color[0];
    blob_bound_color_[1] = color[1];
    blob_bound_color_[2] = color[2];
    blob_bound_color_[3] = color[3];
}

void BlobTrackNode::GetLineColor(float color[4]) const
{
    color[0] = line_color_[0];
    color[1] = line_color_[1];
    color[2] = line_color_[2];
    color[3] = line_color_[3];
}

void BlobTrackNode::SetLineColor(const float color[4])
{
    line_color_[0] = color[0];
    line_color_[1] = color[1];
    line_color_[2] = color[2];
    line_color_[3] = color[3];
}

// ============================================================================
// EVENT-DRIVEN BLOB TRACKING IMPLEMENTATION
// ============================================================================

// Temporal Prediction using Simple Kalman Filter
void BlobTrackNode::TemporalPrediction(float delta_time)
{
    // 모든 활성 blob에 대해 예측 수행
    for (auto& blob : blobs_) {
        // 등속도 모델: predicted_position = current_position + velocity * delta_time
        blob.predicted_x = blob.x + blob.vx * delta_time * 60.0f;  // 60fps 정규화
        blob.predicted_y = blob.y + blob.vy * delta_time * 60.0f;
        
        // 가속도 모델 (선택적)
        // blob.predicted_x += 0.5f * blob.ax * delta_time * delta_time;
        // blob.predicted_y += 0.5f * blob.ay * delta_time * delta_time;
        
        // 신뢰도 감소 (시간이 지날수록 예측 불확실성 증가)
        blob.confidence *= 0.98f;
    }
    
    // Lost blob도 예측 (revive 가능성)
    for (auto& lost : lost_blobs_) {
        lost.predicted_x = lost.x + lost.vx * delta_time * 60.0f;
        lost.predicted_y = lost.y + lost.vy * delta_time * 60.0f;
        lost.confidence *= 0.95f;  // 더 빠르게 감소
    }
}

// Hungarian Algorithm for Optimal Assignment
void BlobTrackNode::HungarianAssignment(const std::vector<BlobInfo>& new_blobs, std::vector<BlobInfo>& tracked_blobs)
{
    // 간단한 Hungarian Algorithm 구현 (100+ blobs에서만 사용)
    // Cost Matrix: distance between predicted position and detected position
    
    size_t n = blobs_.size();
    size_t m = new_blobs.size();
    
    if (n == 0 || m == 0) {
        // 빈 경우 처리
        for (const auto& new_blob : new_blobs) {
            BlobInfo tracked = new_blob;
            tracked.id = next_blob_id_++;
            tracked.vx = 0.0f;
            tracked.vy = 0.0f;
            tracked.ax = 0.0f;
            tracked.ay = 0.0f;
            tracked.predicted_x = new_blob.x;
            tracked.predicted_y = new_blob.y;
            tracked.confidence = 1.0f;
            tracked.last_seen_time = current_time_;
            tracked.is_lost = false;
            tracked.is_expired = false;
            tracked_blobs.push_back(tracked);
        }
        return;
    }
    
    // Cost matrix 생성 (예측 위치와의 거리)
    float* cost_matrix = (float*)cost_matrix_buffer_.contents;
    float max_distance_sq = max_move_distance_ * max_move_distance_;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float dx = new_blobs[i].x - blobs_[j].predicted_x;  // 예측 위치 사용
            float dy = new_blobs[i].y - blobs_[j].predicted_y;
            float distance_sq = dx * dx + dy * dy;
            
            // 거리가 너무 멀면 매칭 불가 (큰 비용)
            cost_matrix[i * n + j] = (distance_sq < max_distance_sq) ? distance_sq : 1e9f;
        }
    }
    
    // 간단한 Greedy Hungarian (O(N*M) - 전역 최적은 아니지만 빠름)
    // 실제 Hungarian은 O(N^3)이지만 복잡하므로 간소화
    std::vector<bool> matched_new(m, false);
    std::vector<bool> matched_old(n, false);
    std::vector<int> assignment(m, -1);  // new_blob[i] -> old_blob[assignment[i]]
    
    // 비용이 작은 순서대로 매칭
    for (int iter = 0; iter < int(std::min(n, m)); iter++) {
        float min_cost = 1e9f;
        int best_i = -1, best_j = -1;
        
        for (size_t i = 0; i < m; i++) {
            if (matched_new[i]) continue;
            for (size_t j = 0; j < n; j++) {
                if (matched_old[j]) continue;
                float cost = cost_matrix[i * n + j];
                if (cost < min_cost) {
                    min_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if (best_i >= 0 && best_j >= 0 && min_cost < max_distance_sq) {
            assignment[best_i] = best_j;
            matched_new[best_i] = true;
            matched_old[best_j] = true;
        } else {
            break;  // 더 이상 매칭 불가
        }
    }
    
    // 매칭 결과 적용
    for (size_t i = 0; i < m; i++) {
        BlobInfo tracked = new_blobs[i];
        
        if (assignment[i] >= 0) {
            // 매칭 성공: ID 유지, 속도/가속도 업데이트
            int old_idx = assignment[i];
            tracked.id = blobs_[old_idx].id;
            
            // 속도 업데이트 (현재 - 이전)
            float new_vx = new_blobs[i].x - blobs_[old_idx].x;
            float new_vy = new_blobs[i].y - blobs_[old_idx].y;
            
            // 가속도 업데이트 (현재 속도 - 이전 속도)
            tracked.ax = new_vx - blobs_[old_idx].vx;
            tracked.ay = new_vy - blobs_[old_idx].vy;
            
            // 속도 스무딩 (사용자 설정)
            tracked.vx = velocity_smoothing_ * new_vx + (1.0f - velocity_smoothing_) * blobs_[old_idx].vx;
            tracked.vy = velocity_smoothing_ * new_vy + (1.0f - velocity_smoothing_) * blobs_[old_idx].vy;
            
            tracked.predicted_x = tracked.x;
            tracked.predicted_y = tracked.y;
            tracked.confidence = std::min(1.0f, blobs_[old_idx].confidence + 0.1f);  // 신뢰도 회복
        } else {
            // 새 blob: ID 할당
            tracked.id = next_blob_id_++;
            tracked.vx = 0.0f;
            tracked.vy = 0.0f;
            tracked.ax = 0.0f;
            tracked.ay = 0.0f;
            tracked.predicted_x = tracked.x;
            tracked.predicted_y = tracked.y;
            tracked.confidence = 0.5f;  // 초기 신뢰도 낮음
        }
        
        tracked.last_seen_time = current_time_;
        tracked.is_lost = false;
        tracked.is_expired = false;
        tracked_blobs.push_back(tracked);
    }
    
    // 매칭되지 않은 기존 blob들을 lost로 이동
    for (size_t j = 0; j < n; j++) {
        if (!matched_old[j]) {
            BlobInfo lost = blobs_[j];
            lost.is_lost = true;
            lost.is_expired = false;
            lost_blobs_.push_back(lost);
        }
    }
}

// NOTE: CPU 추적 함수들 제거됨 - GPU-only 파이프라인 사용
// TrackBlobs, GreedyAssignment, DeleteNearbyBlobs, DeleteOverlappingBlobs는
// 모두 GPU 커널 (matchBlobs, filterDelete, compactBlobs)로 대체됨

void BlobTrackNode::ProcessGPU(
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
        if (binary_texture_ && texture_pool) {
            texture_pool->release_texture(binary_texture_);
            binary_texture_ = nil;
        }
        return;
    }

    id<MTLTexture> input_texture = inputs[0];
    NSUInteger width = [input_texture width];
    NSUInteger height = [input_texture height];
    NSUInteger pixelFormat = [input_texture pixelFormat];
    frame_width_ = static_cast<int>(width);
    frame_height_ = static_cast<int>(height);

    // GPU Zero-Copy: 모든 텍스처를 GPU 메모리에만 유지 (MTLStorageModePrivate)

    // Binary 텍스처 할당 (GPU only)
    if (!binary_texture_ || [binary_texture_ width] != width || [binary_texture_ height] != height) {
        if (binary_texture_ && texture_pool) {
            texture_pool->release_texture(binary_texture_);
        }

        if (texture_pool) {
            binary_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            descriptor.storageMode = MTLStorageModePrivate;  // GPU only!
            binary_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }
    
    // Event-Driven Architecture: Previous Frame 텍스처 할당 (GPU Zero-Copy)
    if (!previous_frame_texture_ || [previous_frame_texture_ width] != width || [previous_frame_texture_ height] != height) {
        if (previous_frame_texture_ && texture_pool) {
            texture_pool->release_texture(previous_frame_texture_);
        }

        if (texture_pool) {
            previous_frame_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            descriptor.storageMode = MTLStorageModePrivate;  // GPU only!
            previous_frame_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }
    
    // Event Texture 할당 (시각화 + clustering용)
    if (!event_texture_ || [event_texture_ width] != width || [event_texture_ height] != height) {
        if (event_texture_ && texture_pool) {
            texture_pool->release_texture(event_texture_);
        }

        if (texture_pool) {
            event_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            descriptor.storageMode = MTLStorageModePrivate;  // GPU only!
            event_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }

    // Output 텍스처 할당 (GPU only)
    if (!output_texture_ || [output_texture_ width] != width || [output_texture_ height] != height) {
        if (output_texture_ && texture_pool) {
            texture_pool->release_texture(output_texture_);
        }

        if (texture_pool) {
            output_texture_ = texture_pool->acquire_texture(width, height, pixelFormat);
        } else {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixelFormat
                                             width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            descriptor.storageMode = MTLStorageModePrivate;  // GPU only!
            output_texture_ = [device_ newTextureWithDescriptor:descriptor];
        }
    }

    if (!binary_texture_ || !output_texture_ || !threshold_pipeline_) return;

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake(
        (width + 15) / 16,
        (height + 15) / 16,
        1
    );

    // ===== Grid setup for GPU matching (binning) =====
    const int MAX_PER_CELL = 256;
    float cell_size = std::max(1.0f, max_move_distance_);
    float cell_size_inv = 1.0f / cell_size;
    int grid_w = std::max(1, (int)std::ceil((float)width / cell_size));
    int grid_h = std::max(1, (int)std::ceil((float)height / cell_size));
    int cell_cap = grid_w * grid_h;
    if (cell_cap != cell_capacity_) {
        cell_counts_buffer_ = [device_ newBufferWithLength:sizeof(uint32_t) * cell_cap
                                                   options:MTLResourceStorageModeShared];
        cell_bins_buffer_ = [device_ newBufferWithLength:sizeof(uint32_t) * cell_cap * MAX_PER_CELL
                                                 options:MTLResourceStorageModeShared];
        cell_capacity_ = cell_cap;
    }

    // ========== Consume last finished GPU tracking buffer ==========
    // zero-copy: GPU writes shared buffers, CPU reads/modifies in-place (no extra copies)
    if (has_ready_buffer_) {
        render_buffer_index_ = ready_buffer_index_;
        ready_buffer_index_ = -1;
        has_ready_buffer_ = false;

        if (render_buffer_index_ >= 0) {
            bool filtered = use_filtered_buffer_[render_buffer_index_];
            id<MTLBuffer> ready_counter_buffer = filtered
                ? filtered_track_counter_buffers_[render_buffer_index_]
                : track_counter_buffers_[render_buffer_index_];
            id<MTLBuffer> ready_blob_buffer = filtered
                ? filtered_track_buffers_[render_buffer_index_]
                : track_buffers_[render_buffer_index_];

            uint32_t* ready_count_ptr = (uint32_t*)[ready_counter_buffer contents];
            uint32_t ready_count = ready_count_ptr ? *ready_count_ptr : 0;
            ready_count = std::min<uint32_t>(ready_count, static_cast<uint32_t>(max_blobs_));

            GPUBlobData* ready_blobs = (GPUBlobData*)[ready_blob_buffer contents];

            // GPU → CPU mirror (최적화: 벡터 재할당 최소화)
            // reserve로 용량 확보 후 resize로 크기만 조정 (재할당 없음)
            if (blobs_.capacity() < ready_count) {
                blobs_.reserve(std::max(size_t(ready_count * 2), size_t(1024)));
            }
            blobs_.resize(ready_count);
            
            // 구조체 레이아웃이 다르므로 필드별 복사 필요
            // SIMD 최적화된 루프 사용
            for (uint32_t i = 0; i < ready_count; ++i) {
                blobs_[i].id = ready_blobs[i].id;
                blobs_[i].x = ready_blobs[i].x;
                blobs_[i].y = ready_blobs[i].y;
                blobs_[i].width = ready_blobs[i].width;
                blobs_[i].height = ready_blobs[i].height;
                blobs_[i].area = ready_blobs[i].area;
                blobs_[i].vx = 0.0f;
                blobs_[i].vy = 0.0f;
                blobs_[i].ax = 0.0f;
                blobs_[i].ay = 0.0f;
                blobs_[i].predicted_x = ready_blobs[i].x;
                blobs_[i].predicted_y = ready_blobs[i].y;
                blobs_[i].confidence = 1.0f;
                blobs_[i].last_seen_time = current_time_;
                blobs_[i].is_lost = false;
                blobs_[i].is_expired = false;
            }
            // GPU-only 경로에서는 lost/expired를 사용하지 않으므로 비움
            lost_blobs_.clear();
        }
    }

    int tracked_count = 0;

    // ========== Prepare write buffers for the next GPU detection ==========
    int write_buffer_index = current_buffer_index_;
    current_buffer_index_ = (current_buffer_index_ + 1) % 2;

    id<MTLBuffer> write_counter_buffer = blob_counter_buffers_[write_buffer_index];
    id<MTLBuffer> write_blob_buffer = blob_data_buffers_[write_buffer_index];
    id<MTLBuffer> write_track_buffer = track_buffers_[write_buffer_index];
    id<MTLBuffer> write_track_counter_buffer = track_counter_buffers_[write_buffer_index];

    uint32_t* counter = (uint32_t*)[write_counter_buffer contents];
    *counter = 0;
    uint32_t* trackCounterPtr = (uint32_t*)[write_track_counter_buffer contents];
    if (trackCounterPtr) *trackCounterPtr = 0;

    // ========== Create our own command buffer for blob detection ==========
    // We need to commit/wait for blob data, so we can't use the passed cmd_buffer
    id<MTLCommandBuffer> detect_cmd_buffer = [command_queue_ commandBuffer];

    bool use_event_path = use_event_detection_ && previous_frame_texture_ != nil && event_detection_pipeline_ != nil;
    id<MTLComputeCommandEncoder> encoder = nil;

    if (use_event_path) {
        // ========== Event-Driven Path ==========
        // Reset counters
        uint32_t* ev_counter = (uint32_t*)[event_counter_buffer_ contents];
        uint32_t* blob_counter = (uint32_t*)[write_counter_buffer contents];
        counter = blob_counter;
        *ev_counter = 0;
        *blob_counter = 0;

        // Pass 1: 픽셀 변화 이벤트 감지
        encoder = [detect_cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:event_detection_pipeline_];
        [encoder setTexture:input_texture atIndex:0];
        [encoder setTexture:previous_frame_texture_ atIndex:1];
        [encoder setTexture:event_texture_ atIndex:2];
        [encoder setBuffer:event_counter_buffer_ offset:0 atIndex:0];
        [encoder setBytes:&event_threshold_ length:sizeof(float) atIndex:1];
        int mono_source_int = static_cast<int>(mono_source_);
        [encoder setBytes:&mono_source_int length:sizeof(int) atIndex:2];
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Pass 2: 이벤트를 blob으로 클러스터링
        // 다운샘플된 dispatch 크기 사용 (detectBlobs와 동일)
        encoder = [detect_cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:event_clustering_pipeline_];
        [encoder setTexture:event_texture_ atIndex:0];
        [encoder setBuffer:write_counter_buffer offset:0 atIndex:0];
        [encoder setBuffer:write_blob_buffer offset:0 atIndex:1];
        [encoder setBytes:&min_blob_size_ length:sizeof(int) atIndex:2];
        [encoder setBytes:&max_blob_size_ length:sizeof(int) atIndex:3];
        int max_blobs = max_blobs_;
        [encoder setBytes:&max_blobs length:sizeof(int) atIndex:4];
        float cluster_radius = max_move_distance_;
        [encoder setBytes:&cluster_radius length:sizeof(float) atIndex:5];
        
        // 4x4 다운샘플에 맞게 dispatch 크기 조정
        const int eventSampleStep = 4;
        MTLSize eventThreadgroups = MTLSizeMake(
            ((width / eventSampleStep) + 15) / 16,
            ((height / eventSampleStep) + 15) / 16,
            1
        );
        [encoder dispatchThreadgroups:eventThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Pass 3: 현재 프레임을 이전 프레임 텍스처로 복사 (다음 이벤트 비교용)
        id<MTLBlitCommandEncoder> prevBlit = [detect_cmd_buffer blitCommandEncoder];
        [prevBlit copyFromTexture:input_texture toTexture:previous_frame_texture_];
        [prevBlit endEncoding];
    } else {
        // ========== Frame-based Path (기존) ==========
        encoder = [detect_cmd_buffer computeCommandEncoder];
        
        // Background Subtraction 모드 체크 (두 번째 입력이 연결되어 있는지)
        bool use_background_subtraction = (inputs.size() > 1 && inputs[1] != nil);
        
        if (use_background_subtraction) {
            [encoder setComputePipelineState:background_subtraction_pipeline_];
            [encoder setTexture:input_texture atIndex:0];
            [encoder setTexture:inputs[1] atIndex:1];
            [encoder setTexture:binary_texture_ atIndex:2];
        } else {
            [encoder setComputePipelineState:threshold_pipeline_];
            [encoder setTexture:input_texture atIndex:0];
            [encoder setTexture:binary_texture_ atIndex:1];
        }

        int mono_source_int = static_cast<int>(mono_source_);
        [encoder setBytes:&mono_source_int length:sizeof(int) atIndex:0];
        [encoder setBytes:&threshold_ length:sizeof(float) atIndex:1];

        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // ========== Pass 2: Blob Detection (GPU) ==========
        // Reset blob counter
        *counter = 0;

        encoder = [detect_cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:detect_pipeline_];
        [encoder setTexture:binary_texture_ atIndex:0];
        [encoder setBuffer:write_counter_buffer offset:0 atIndex:0];
        [encoder setBuffer:write_blob_buffer offset:0 atIndex:1];
        [encoder setBytes:&min_blob_size_ length:sizeof(int) atIndex:2];
        [encoder setBytes:&max_blob_size_ length:sizeof(int) atIndex:3];

        int max_blobs = max_blobs_;
        [encoder setBytes:&max_blobs length:sizeof(int) atIndex:4];

        // 핵심 수정: 4x4 다운샘플에 맞게 dispatch 크기 조정
        // 기존: 전체 이미지 크기 dispatch → 대부분 조기 종료
        // 수정: (width/4, height/4) 크기만 dispatch → 모든 스레드 유효
        const int sampleStep = 4;
        MTLSize detectThreadgroups = MTLSizeMake(
            ((width / sampleStep) + 15) / 16,
            ((height / sampleStep) + 15) / 16,
            1
        );
        [encoder dispatchThreadgroups:detectThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }

    // ========== Pass 3: GPU Matching (new blobs -> tracked) ==========
    int prev_track_index = (render_buffer_index_ >= 0) ? render_buffer_index_ : write_buffer_index;
    id<MTLBuffer> prev_track_buffer = track_buffers_[prev_track_index];
    id<MTLBuffer> prev_track_counter_buffer = track_counter_buffers_[prev_track_index];

    // 카운트 가져오기 (디스패치 크기 최소화)
    uint32_t prevCountHost = 0;
    uint32_t* prevCountPtrHost = (uint32_t*)[prev_track_counter_buffer contents];
    if (prevCountPtrHost) prevCountHost = std::min(*prevCountPtrHost, static_cast<uint32_t>(max_blobs_));

    // Zero cell counts
    if (cell_counts_buffer_ && zero_counts_pipeline_) {
        id<MTLComputeCommandEncoder> encZero = [detect_cmd_buffer computeCommandEncoder];
        [encZero setComputePipelineState:zero_counts_pipeline_];
        uint totalCells = static_cast<uint>(cell_capacity_);
        [encZero setBuffer:cell_counts_buffer_ offset:0 atIndex:0];
        [encZero setBytes:&totalCells length:sizeof(uint) atIndex:1];
        NSUInteger wzc = zero_counts_pipeline_.threadExecutionWidth;
        NSUInteger tgZero = (totalCells + wzc - 1) / wzc;
        [encZero dispatchThreadgroups:MTLSizeMake(tgZero, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(wzc, 1, 1)];
        [encZero endEncoding];
    }

    // Build bins from previous tracked blobs
    if (build_bins_pipeline_ && prev_track_buffer && cell_counts_buffer_ && cell_bins_buffer_ && prevCountHost > 0) {
        id<MTLComputeCommandEncoder> encBin = [detect_cmd_buffer computeCommandEncoder];
        [encBin setComputePipelineState:build_bins_pipeline_];
        [encBin setBuffer:prev_track_buffer offset:0 atIndex:0];
        [encBin setBuffer:cell_counts_buffer_ offset:0 atIndex:1];
        [encBin setBuffer:cell_bins_buffer_ offset:0 atIndex:2];
        [encBin setBuffer:prev_track_counter_buffer offset:0 atIndex:3];
        [encBin setBytes:&grid_w length:sizeof(int) atIndex:4];
        [encBin setBytes:&grid_h length:sizeof(int) atIndex:5];
        [encBin setBytes:&cell_size_inv length:sizeof(float) atIndex:6];
        NSUInteger wb = build_bins_pipeline_.threadExecutionWidth;
        NSUInteger tgBin = (prevCountHost + wb - 1) / wb;
        [encBin dispatchThreadgroups:MTLSizeMake(tgBin, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(wb, 1, 1)];
        [encBin endEncoding];
    }

    encoder = [detect_cmd_buffer computeCommandEncoder];
    [encoder setComputePipelineState:match_pipeline_];
    [encoder setBuffer:write_blob_buffer offset:0 atIndex:0];              // new blobs
    [encoder setBuffer:prev_track_buffer offset:0 atIndex:1];              // prev tracked
    [encoder setBuffer:cell_counts_buffer_ offset:0 atIndex:2];            // cell counts
    [encoder setBuffer:cell_bins_buffer_ offset:0 atIndex:3];              // cell bins
    [encoder setBuffer:write_track_buffer offset:0 atIndex:4];             // out tracked
    [encoder setBuffer:write_track_counter_buffer offset:0 atIndex:5];     // out count
    [encoder setBuffer:track_id_counter_buffer_ offset:0 atIndex:6];       // id counter
    [encoder setBuffer:write_counter_buffer offset:0 atIndex:7];           // newCount pointer
    [encoder setBuffer:prev_track_counter_buffer offset:0 atIndex:8];      // prevCount pointer
    float max_move_sq = max_move_distance_ * max_move_distance_;
    [encoder setBytes:&max_move_sq length:sizeof(float) atIndex:9];
    [encoder setBytes:&grid_w length:sizeof(int) atIndex:10];
    [encoder setBytes:&grid_h length:sizeof(int) atIndex:11];
    [encoder setBytes:&cell_size_inv length:sizeof(float) atIndex:12];

    // 쓰레드 그룹: new blob 수에 맞추어 (최대 max_blobs_)
    NSUInteger threadsPerGroup = match_pipeline_.threadExecutionWidth;
    uint32_t maxDispatch = static_cast<uint32_t>(max_blobs_); // detect 커널 결과는 match 내부에서 카운트 읽음
    NSUInteger tgCount = (maxDispatch + threadsPerGroup - 1) / threadsPerGroup;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize grid = MTLSizeMake(tgCount * threadsPerGroup, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tgSize];
    [encoder endEncoding];

    // ========== Optional GPU post-filter (delete/overlap) ==========
    bool need_filter = (delete_nearby_ || delete_overlapping_);
    use_filtered_buffer_[write_buffer_index] = need_filter;
    if (need_filter) {
        // zero delete flags / counters
        uint32_t* filt_count = (uint32_t*)[filtered_track_counter_buffers_[write_buffer_index] contents];
        if (filt_count) *filt_count = 0;

        // reuse grid buffers for current tracked blobs
        if (cell_capacity_ > 0 && cell_counts_buffer_ != nil) {
            id<MTLComputeCommandEncoder> encZero = [detect_cmd_buffer computeCommandEncoder];
            [encZero setComputePipelineState:zero_counts_pipeline_];
            uint32_t totalCells = static_cast<uint32_t>(cell_capacity_);
            [encZero setBuffer:cell_counts_buffer_ offset:0 atIndex:0];
            [encZero setBytes:&totalCells length:sizeof(uint32_t) atIndex:1];
            NSUInteger wb = zero_counts_pipeline_.threadExecutionWidth;
            NSUInteger tg = (totalCells + wb - 1) / wb;
            [encZero dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(wb, 1, 1)];
            [encZero endEncoding];
        }

        // build bins for current tracked blobs
        id<MTLComputeCommandEncoder> encBinCur = [detect_cmd_buffer computeCommandEncoder];
        [encBinCur setComputePipelineState:build_bins_pipeline_];
        [encBinCur setBuffer:write_track_buffer offset:0 atIndex:0];
        [encBinCur setBuffer:cell_counts_buffer_ offset:0 atIndex:1];
        [encBinCur setBuffer:cell_bins_buffer_ offset:0 atIndex:2];
        [encBinCur setBuffer:write_track_counter_buffer offset:0 atIndex:3];
        [encBinCur setBytes:&grid_w length:sizeof(int) atIndex:4];
        [encBinCur setBytes:&grid_h length:sizeof(int) atIndex:5];
        [encBinCur setBytes:&cell_size_inv length:sizeof(float) atIndex:6];
        NSUInteger wbCur = build_bins_pipeline_.threadExecutionWidth;
        NSUInteger tgCur = (max_blobs_ + wbCur - 1) / wbCur;
        [encBinCur dispatchThreadgroups:MTLSizeMake(tgCur, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(wbCur, 1, 1)];
        [encBinCur endEncoding];

        // filter delete/overlap
        id<MTLComputeCommandEncoder> encFilter = [detect_cmd_buffer computeCommandEncoder];
        [encFilter setComputePipelineState:filter_delete_pipeline_];
        [encFilter setBuffer:write_track_buffer offset:0 atIndex:0];
        [encFilter setBuffer:delete_flags_buffers_[write_buffer_index] offset:0 atIndex:1];
        [encFilter setBuffer:cell_counts_buffer_ offset:0 atIndex:2];
        [encFilter setBuffer:cell_bins_buffer_ offset:0 atIndex:3];
        [encFilter setBuffer:write_track_counter_buffer offset:0 atIndex:4];
        [encFilter setBytes:&delete_distance_ length:sizeof(float) atIndex:5];
        [encFilter setBytes:&delete_area_tolerance_ length:sizeof(float) atIndex:6];
        [encFilter setBytes:&delete_overlap_tolerance_ length:sizeof(float) atIndex:7];
        [encFilter setBytes:&grid_w length:sizeof(int) atIndex:8];
        [encFilter setBytes:&grid_h length:sizeof(int) atIndex:9];
        [encFilter setBytes:&cell_size_inv length:sizeof(float) atIndex:10];
        bool useNear = delete_nearby_;
        bool useOverlap = delete_overlapping_;
        [encFilter setBytes:&useNear length:sizeof(bool) atIndex:11];
        [encFilter setBytes:&useOverlap length:sizeof(bool) atIndex:12];
        NSUInteger wf = filter_delete_pipeline_.threadExecutionWidth;
        NSUInteger tgf = (max_blobs_ + wf - 1) / wf;
        [encFilter dispatchThreadgroups:MTLSizeMake(tgf, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(wf, 1, 1)];
        [encFilter endEncoding];

        // compact
        id<MTLComputeCommandEncoder> encCompact = [detect_cmd_buffer computeCommandEncoder];
        [encCompact setComputePipelineState:compact_pipeline_];
        [encCompact setBuffer:write_track_buffer offset:0 atIndex:0];
        [encCompact setBuffer:delete_flags_buffers_[write_buffer_index] offset:0 atIndex:1];
        [encCompact setBuffer:filtered_track_buffers_[write_buffer_index] offset:0 atIndex:2];
        [encCompact setBuffer:filtered_track_counter_buffers_[write_buffer_index] offset:0 atIndex:3];
        [encCompact setBuffer:write_track_counter_buffer offset:0 atIndex:4];
        NSUInteger wc = compact_pipeline_.threadExecutionWidth;
        NSUInteger tgc = (max_blobs_ + wc - 1) / wc;
        [encCompact dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(wc, 1, 1)];
        [encCompact endEncoding];
    }

    // Commit OUR buffer asynchronously; completion marks buffer as ready without CPU stall
    int completed_index = write_buffer_index;
    bool completed_filtered = need_filter;
    BlobTrackNode* self_ptr = this;
    [detect_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        self_ptr->ready_buffer_index_ = completed_index;
        self_ptr->use_filtered_buffer_[completed_index] = completed_filtered;
        self_ptr->has_ready_buffer_ = true;
    }];
    [detect_cmd_buffer commit];

    // ========== Determine buffer for rendering ==========
    int draw_buffer_index = (render_buffer_index_ >= 0) ? render_buffer_index_ : -1;
    if (draw_buffer_index < 0) {
        // 아직 준비된 버퍼가 없으면 입력만 출력으로 복사하고 종료
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:input_texture toTexture:output_texture_];
        [blitEncoder endEncoding];
        return;
    }
    bool draw_filtered = use_filtered_buffer_[draw_buffer_index];
    id<MTLBuffer> draw_blob_buffer = draw_filtered ? filtered_track_buffers_[draw_buffer_index] : track_buffers_[draw_buffer_index];
    id<MTLBuffer> draw_count_buffer = draw_filtered ? filtered_track_counter_buffers_[draw_buffer_index] : track_counter_buffers_[draw_buffer_index];
    uint32_t* draw_count_ptr = (uint32_t*)[draw_count_buffer contents];
    tracked_count = draw_count_ptr ? static_cast<int>(*draw_count_ptr) : 0;

    // ========== Pass 4: Draw Output (use PASSED cmd_buffer) ==========
    // Now use the passed cmd_buffer for final output
    if ((draw_blob_bounds_ || draw_lines_) && tracked_count > 0) {
        // 먼저 입력을 출력으로 복사
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:input_texture toTexture:output_texture_];
        [blitEncoder endEncoding];

        // Blob bounds 그리기
        if (draw_blob_bounds_) {
        encoder = [cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:draw_bounds_pipeline_];
            [encoder setTexture:output_texture_ atIndex:0];
            [encoder setTexture:output_texture_ atIndex:1];  // 읽고 쓰기
        [encoder setBuffer:draw_blob_buffer offset:0 atIndex:0];

        int tracked_count_local = tracked_count;
        [encoder setBytes:&tracked_count_local length:sizeof(int) atIndex:1];
            
            // Blob bound color 전달
            struct { float r, g, b, a; } bound_color = {
                blob_bound_color_[0],
                blob_bound_color_[1],
                blob_bound_color_[2],
                blob_bound_color_[3]
            };
            [encoder setBytes:&bound_color length:sizeof(bound_color) atIndex:2];

            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        // Blob 간 선 그리기
        if (draw_lines_ && tracked_count > 1) {
            encoder = [cmd_buffer computeCommandEncoder];
            [encoder setComputePipelineState:draw_lines_pipeline_];
            [encoder setTexture:output_texture_ atIndex:0];
            [encoder setTexture:output_texture_ atIndex:1];  // 읽고 쓰기
            [encoder setBuffer:draw_blob_buffer offset:0 atIndex:0];

            int tracked_count_local2 = tracked_count;
            [encoder setBytes:&tracked_count_local2 length:sizeof(int) atIndex:1];
            [encoder setBytes:&line_distance_ length:sizeof(float) atIndex:2];
            
            // Line color 전달
            struct { float r, g, b, a; } line_color = {
                line_color_[0],
                line_color_[1],
                line_color_[2],
                line_color_[3]
            };
            [encoder setBytes:&line_color length:sizeof(line_color) atIndex:3];

        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        }
    } else {
        // Just copy input to output
        id<MTLBlitCommandEncoder> blitEncoder = [cmd_buffer blitCommandEncoder];
        [blitEncoder copyFromTexture:input_texture toTexture:output_texture_];
        [blitEncoder endEncoding];
    }

    // Don't commit the passed cmd_buffer - let the caller handle it
    // Detection command buffer is committed asynchronously (no CPU wait) for zero-copy flow
}

std::unique_ptr<NodeBase> CreateBlobTrackNode(Graph<Node>& graph, const ImVec2& pos, id<MTLDevice> device)
{
    return std::make_unique<BlobTrackNode>(graph, pos, device);
}

REGISTER_NODE(BlobTrack, "Blob Track", "TOP/Analysis", NodeFamily::TOP, CreateBlobTrackNode, "GPU-accelerated blob tracking (zero-copy)");

} // namespace nodes
} // namespace example
