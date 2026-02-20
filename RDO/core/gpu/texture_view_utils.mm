#include "texture_view_utils.h"
#import <Metal/Metal.h>

namespace example {

// 픽셀 포맷 호환성 테이블
// Metal에서 newTextureViewWithPixelFormat으로 변환 가능한 포맷들
bool IsPixelFormatViewCompatible(MTLPixelFormat source, MTLPixelFormat target)
{
    if (source == target) {
        return true;  // 같은 포맷은 항상 호환
    }

    // RGBA8 <-> BGRA8 (8비트 언팩 포맷 그룹)
    if ((source == MTLPixelFormatRGBA8Unorm && target == MTLPixelFormatBGRA8Unorm) ||
        (source == MTLPixelFormatBGRA8Unorm && target == MTLPixelFormatRGBA8Unorm)) {
        return true;
    }

    if ((source == MTLPixelFormatRGBA8Unorm_sRGB && target == MTLPixelFormatBGRA8Unorm_sRGB) ||
        (source == MTLPixelFormatBGRA8Unorm_sRGB && target == MTLPixelFormatRGBA8Unorm_sRGB)) {
        return true;
    }

    // Linear <-> sRGB (같은 채널 레이아웃)
    if ((source == MTLPixelFormatRGBA8Unorm && target == MTLPixelFormatRGBA8Unorm_sRGB) ||
        (source == MTLPixelFormatRGBA8Unorm_sRGB && target == MTLPixelFormatRGBA8Unorm)) {
        return true;
    }

    if ((source == MTLPixelFormatBGRA8Unorm && target == MTLPixelFormatBGRA8Unorm_sRGB) ||
        (source == MTLPixelFormatBGRA8Unorm_sRGB && target == MTLPixelFormatBGRA8Unorm)) {
        return true;
    }

    // RGBA16Float <-> RGBA16Unorm
    if ((source == MTLPixelFormatRGBA16Float && target == MTLPixelFormatRGBA16Unorm) ||
        (source == MTLPixelFormatRGBA16Unorm && target == MTLPixelFormatRGBA16Float)) {
        return true;
    }

    // RGBA32Float <-> RGBA32Uint/Sint
    if (source == MTLPixelFormatRGBA32Float &&
        (target == MTLPixelFormatRGBA32Uint || target == MTLPixelFormatRGBA32Sint)) {
        return true;
    }

    if ((source == MTLPixelFormatRGBA32Uint || source == MTLPixelFormatRGBA32Sint) &&
        target == MTLPixelFormatRGBA32Float) {
        return true;
    }

    // 추가 호환 포맷들을 여기에 추가 가능

    return false;
}

// Texture View 생성 (zero-copy)
id<MTLTexture> CreateTextureViewIfCompatible(
    id<MTLTexture> source,
    MTLPixelFormat target_format)
{
    if (!source) {
        return nil;
    }

    // 이미 같은 포맷이면 원본 그대로 반환
    if ([source pixelFormat] == target_format) {
        return source;
    }

    // 호환성 체크
    if (!IsPixelFormatViewCompatible([source pixelFormat], target_format)) {
        return nil;  // 호환 안 됨, 복사 필요
    }

    // Texture View 생성 (zero-copy!)
    @try {
        id<MTLTexture> view = [source newTextureViewWithPixelFormat:target_format];
        return view;
    }
    @catch (NSException *exception) {
        // Metal이 예상치 못한 이유로 실패한 경우
        NSLog(@"Failed to create texture view: %@ -> %lu: %@",
              @([source pixelFormat]),
              (unsigned long)target_format,
              exception.reason);
        return nil;
    }
}

} // namespace example
