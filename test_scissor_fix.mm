// 디버깅용: 모든 ImGui::Image 호출 전에 텍스처 크기 검증
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

void ValidateTextureBeforeImGui(id<MTLTexture> texture, const char* location) {
    if (!texture) {
        NSLog(@"[%s] WARNING: nil texture!", location);
        return;
    }
    
    NSUInteger width = [texture width];
    NSUInteger height = [texture height];
    
    NSLog(@"[%s] Texture: %lux%lu, Format: %lu", 
          location, width, height, (unsigned long)[texture pixelFormat]);
    
    // 의심스러운 크기 경고
    if (width > 4096 || height > 4096) {
        NSLog(@"[%s] WARNING: Unusually large texture!", location);
    }
    if (width < 64 || height < 64) {
        NSLog(@"[%s] WARNING: Unusually small texture!", location);
    }
}
