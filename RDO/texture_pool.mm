#include "texture_pool.h"
#import <Metal/Metal.h>
#include <iostream>

namespace example
{

TexturePool::TexturePool(id<MTLDevice> device)
    : device_(device)
    , available_textures_()
    , in_use_textures_()
    , total_memory_bytes_(0)
    , texture_heap_(nil)
    , heap_size_bytes_(0)
    , heap_used_bytes_(0)
{
}

TexturePool::~TexturePool()
{
    // 모든 텍스처 해제
    for (auto& pair : available_textures_)
    {
        for (id<MTLTexture> texture : pair.second)
        {
            // ARC가 자동으로 해제
            (void)texture;
        }
    }

    for (auto& pair : in_use_textures_)
    {
        for (id<MTLTexture> texture : pair.second)
        {
            // ARC가 자동으로 해제
            (void)texture;
        }
    }

    // Metal Heap 해제
    if (texture_heap_ != nil)
    {
        texture_heap_ = nil;  // ARC가 자동으로 해제
    }
}

id<MTLTexture> TexturePool::acquire_texture(unsigned long width, unsigned long height, unsigned long pixel_format)
{
    TextureKey key = {width, height, pixel_format};
    
    // 풀에 사용 가능한 텍스처가 있는지 확인
    auto& available = available_textures_[key];
    if (!available.empty())
    {
        // 재사용
        id<MTLTexture> texture = available.back();
        available.pop_back();
        
        // in_use로 이동
        in_use_textures_[key].push_back(texture);
        
        return texture;
    }
    
    // 새 텍스처 생성
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor 
        texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixel_format
                                     width:width
                                    height:height
                                 mipmapped:NO];
    // RenderTarget 플래그 추가: Render Pipeline에서 렌더 타겟으로 사용 가능
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    
    id<MTLTexture> texture = [device_ newTextureWithDescriptor:descriptor];
    
    if (texture)
    {
        // 메모리 사용량 추적
        size_t bytes_per_pixel = 4;  // RGBA
        size_t texture_bytes = width * height * bytes_per_pixel;
        total_memory_bytes_ += texture_bytes;
        
        // in_use로 추가
        in_use_textures_[key].push_back(texture);
    }
    
    return texture;
}

void TexturePool::release_texture(id<MTLTexture> texture)
{
    if (texture == nil)
    {
        return;
    }
    
    TextureKey key = {
        [texture width],
        [texture height],
        (unsigned long)[texture pixelFormat]
    };
    
    // in_use에서 제거
    auto& in_use = in_use_textures_[key];
    auto it = std::find(in_use.begin(), in_use.end(), texture);
    if (it != in_use.end())
    {
        in_use.erase(it);
    }
    
    // available로 이동
    available_textures_[key].push_back(texture);
}

void TexturePool::end_frame()
{
    // 현재는 텍스처를 계속 보관
    // 향후 메모리 압박이 있을 경우 오래된 텍스처 정리 로직 추가 가능
    
    // 예: 풀이 너무 크면 일부 텍스처 해제
    const size_t MAX_POOL_SIZE = 50;  // 100 -> 50으로 줄여서 메모리 사용 제한
    size_t total = 0;
    
    for (auto& pair : available_textures_)
    {
        total += pair.second.size();
    }
    
    if (total > MAX_POOL_SIZE)
    {
        // 가장 많이 쌓인 텍스처 타입부터 제거
        for (auto& pair : available_textures_)
        {
            while (pair.second.size() > 1 && total > MAX_POOL_SIZE)  // 최소 1개는 유지
            {
                // 텍스처 제거 (ARC가 자동으로 해제)
                id<MTLTexture> texture = pair.second.back();
                pair.second.pop_back();
                
                // 메모리 사용량 감소
                size_t bytes_per_pixel = 4;
                size_t texture_bytes = pair.first.width * pair.first.height * bytes_per_pixel;
                if (total_memory_bytes_ >= texture_bytes)
                {
                    total_memory_bytes_ -= texture_bytes;
                }
                
                total--;
            }
        }
    }
}

size_t TexturePool::total_textures() const
{
    size_t count = 0;
    for (const auto& pair : available_textures_)
    {
        count += pair.second.size();
    }
    for (const auto& pair : in_use_textures_)
    {
        count += pair.second.size();
    }
    return count;
}

size_t TexturePool::available_textures() const
{
    size_t count = 0;
    for (const auto& pair : available_textures_)
    {
        count += pair.second.size();
    }
    return count;
}

size_t TexturePool::in_use_textures() const
{
    size_t count = 0;
    for (const auto& pair : in_use_textures_)
    {
        count += pair.second.size();
    }
    return count;
}

size_t TexturePool::memory_usage_bytes() const
{
    return total_memory_bytes_;
}

void TexturePool::print_stats() const
{
    size_t total = total_textures();
    size_t available = available_textures();
    size_t in_use = in_use_textures();

    std::cout << "=== TexturePool Stats ===" << std::endl;
    std::cout << "Total textures: " << total << std::endl;
    std::cout << "  - Available: " << available << std::endl;
    std::cout << "  - In use: " << in_use << std::endl;
    std::cout << "Memory usage: " << (total_memory_bytes_ / 1024 / 1024) << " MB" << std::endl;

    if (texture_heap_ != nil)
    {
        std::cout << "Metal Heap:" << std::endl;
        std::cout << "  - Size: " << (heap_size_bytes_ / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - Used: " << (heap_used_bytes_ / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - Available: " << ((heap_size_bytes_ - heap_used_bytes_) / 1024 / 1024) << " MB" << std::endl;
    }

    std::cout << "Texture types:" << std::endl;
    for (const auto& pair : available_textures_)
    {
        size_t count = pair.second.size();
        if (count > 0)
        {
            std::cout << "  - " << pair.first.width << "x" << pair.first.height
                     << " format:" << pair.first.pixel_format
                     << " available:" << count << std::endl;
        }
    }

    for (const auto& pair : in_use_textures_)
    {
        size_t count = pair.second.size();
        if (count > 0)
        {
            std::cout << "  - " << pair.first.width << "x" << pair.first.height
                     << " format:" << pair.first.pixel_format
                     << " in_use:" << count << std::endl;
        }
    }
}

// Metal Heap 초기화
void TexturePool::initialize_heap(size_t heap_size_mb)
{
    if (texture_heap_ != nil)
    {
        std::cout << "⚠️ TexturePool: Heap already initialized" << std::endl;
        return;
    }

    heap_size_bytes_ = heap_size_mb * 1024 * 1024;

    MTLHeapDescriptor* heapDescriptor = [[MTLHeapDescriptor alloc] init];
    heapDescriptor.size = heap_size_bytes_;
    heapDescriptor.storageMode = MTLStorageModePrivate;
    heapDescriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    heapDescriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;

    texture_heap_ = [device_ newHeapWithDescriptor:heapDescriptor];

    if (texture_heap_ != nil)
    {
        std::cout << "✅ TexturePool: Metal Heap initialized (" << heap_size_mb << " MB)" << std::endl;
    }
    else
    {
        std::cerr << "❌ TexturePool: Failed to create Metal Heap" << std::endl;
    }
}

// Aliased 텍스처 할당
id<MTLTexture> TexturePool::acquire_aliased_texture(
    unsigned long width,
    unsigned long height,
    unsigned long pixel_format,
    id<MTLTexture> reuse_from
)
{
    // Heap이 초기화되지 않았으면 일반 할당으로 fallback
    if (texture_heap_ == nil)
    {
        return acquire_texture(width, height, pixel_format);
    }

    // reuse_from 텍스처를 aliasable로 표시
    if (reuse_from != nil && [reuse_from heap] == texture_heap_)
    {
        [reuse_from makeAliasable];
    }

    // 텍스처 디스크립터 생성
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixel_format
                                     width:width
                                    height:height
                                 mipmapped:NO];
    // RenderTarget 플래그 추가: Render Pipeline에서 렌더 타겟으로 사용 가능
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;

    // Heap에서 텍스처 할당
    id<MTLTexture> texture = [texture_heap_ newTextureWithDescriptor:descriptor];

    if (texture != nil)
    {
        // 메모리 사용량 추적
        size_t bytes_per_pixel = 4;  // RGBA
        size_t texture_bytes = width * height * bytes_per_pixel;
        heap_used_bytes_ += texture_bytes;

        std::cout << "📦 TexturePool: Aliased texture created " << width << "x" << height
                  << " (Heap: " << (heap_used_bytes_ / 1024 / 1024) << "/"
                  << (heap_size_bytes_ / 1024 / 1024) << " MB)" << std::endl;
    }
    else
    {
        std::cerr << "❌ TexturePool: Failed to allocate aliased texture from heap" << std::endl;
        // Fallback to regular allocation
        return acquire_texture(width, height, pixel_format);
    }

    return texture;
}

// 텍스처를 aliasable로 표시
void TexturePool::make_aliasable(id<MTLTexture> texture)
{
    if (texture == nil || texture_heap_ == nil)
    {
        return;
    }

    // 텍스처가 이 heap에서 할당되었는지 확인
    if ([texture heap] == texture_heap_)
    {
        [texture makeAliasable];

        // 메모리 사용량 감소
        size_t bytes_per_pixel = 4;
        size_t texture_bytes = [texture width] * [texture height] * bytes_per_pixel;
        if (heap_used_bytes_ >= texture_bytes)
        {
            heap_used_bytes_ -= texture_bytes;
        }

        std::cout << "♻️ TexturePool: Texture made aliasable " << [texture width] << "x" << [texture height]
                  << " (Heap: " << (heap_used_bytes_ / 1024 / 1024) << "/"
                  << (heap_size_bytes_ / 1024 / 1024) << " MB)" << std::endl;
    }
}

size_t TexturePool::heap_used_bytes() const
{
    return heap_used_bytes_;
}

size_t TexturePool::heap_size_bytes() const
{
    return heap_size_bytes_;
}

} // namespace example
