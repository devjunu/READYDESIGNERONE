#pragma once

#include <vector>
#include <unordered_map>

@protocol MTLDevice;
@protocol MTLTexture;
@protocol MTLHeap;

namespace example
{

// 텍스처 풀: GPU 메모리 재사용으로 할당 오버헤드 최소화
class TexturePool
{
public:
    struct TextureKey
    {
        unsigned long width;
        unsigned long height;
        unsigned long pixel_format;
        
        bool operator==(const TextureKey& other) const
        {
            return width == other.width && 
                   height == other.height && 
                   pixel_format == other.pixel_format;
        }
    };
    
    struct TextureKeyHash
    {
        size_t operator()(const TextureKey& key) const
        {
            return ((key.width << 16) ^ key.height) ^ (key.pixel_format << 8);
        }
    };
    
    TexturePool(id<MTLDevice> device);
    ~TexturePool();
    
    // 텍스처 요청 (풀에서 재사용 또는 새로 생성)
    id<MTLTexture> acquire_texture(unsigned long width, unsigned long height, unsigned long pixel_format);

    // Metal Heap 기반 Aliased 텍스처 요청 (메모리 재사용)
    // reuse_from: 이 텍스처가 더 이상 사용되지 않으면 그 메모리를 새 텍스처에 재사용
    // 반환값: 새로 할당된 텍스처 (reuse_from과 같은 메모리 공간 사용 가능)
    id<MTLTexture> acquire_aliased_texture(
        unsigned long width,
        unsigned long height,
        unsigned long pixel_format,
        id<MTLTexture> reuse_from  // nil이면 일반 할당
    );

    // 텍스처를 aliasable로 표시 (다른 텍스처가 이 메모리를 재사용 가능하도록)
    void make_aliasable(id<MTLTexture> texture);

    // 텍스처 반환 (풀에 다시 넣음)
    void release_texture(id<MTLTexture> texture);

    // 프레임 종료 시 호출 (사용하지 않는 텍스처 정리)
    void end_frame();

    // Metal Heap 초기화 (메모리 aliasing 활성화)
    // heap_size_mb: Heap 크기 (MB 단위), 기본값 256MB
    void initialize_heap(size_t heap_size_mb = 256);

    // 통계 정보
    size_t total_textures() const;
    size_t available_textures() const;
    size_t in_use_textures() const;
    size_t memory_usage_bytes() const;
    size_t heap_used_bytes() const;
    size_t heap_size_bytes() const;

    // 디버깅용: 상세 통계
    void print_stats() const;

private:
    id<MTLDevice> device_;
    std::unordered_map<TextureKey, std::vector<id<MTLTexture>>, TextureKeyHash> available_textures_;
    std::unordered_map<TextureKey, std::vector<id<MTLTexture>>, TextureKeyHash> in_use_textures_;
    size_t total_memory_bytes_;

    // Metal Heap for memory aliasing
    id<MTLHeap> texture_heap_;
    size_t heap_size_bytes_;
    size_t heap_used_bytes_;
};

} // namespace example
