#pragma once

#import <Metal/Metal.h>

namespace example {

// Metal Texture View 유틸리티 (Zero-Copy 최적화)
// Texture View는 기존 텍스처의 메모리를 다른 포맷으로 재해석하는 기능
// 메모리 복사 없이 포맷 변환 가능 (호환 포맷에 한정)

// 두 픽셀 포맷이 Texture View로 변환 가능한지 확인
// 예: RGBA8Unorm <-> BGRA8Unorm (바이트 순서만 다름)
bool IsPixelFormatViewCompatible(MTLPixelFormat source, MTLPixelFormat target);

// Texture View 생성 (zero-copy)
// 호환 포맷이면 View 반환, 아니면 nil 반환 (복사 필요)
id<MTLTexture> CreateTextureViewIfCompatible(
    id<MTLTexture> source,
    MTLPixelFormat target_format
);

} // namespace example
