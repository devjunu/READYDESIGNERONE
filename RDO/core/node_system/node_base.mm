#include "node_base.h"
#include "../../texture_pool.h"

namespace example
{

// TOPNodeBase 소멸자 구현
TOPNodeBase::~TOPNodeBase()
{
    ReleaseOutputTexture();
}

// 캐시 무효화 구현
void TOPNodeBase::InvalidateCache()
{
    ReleaseOutputTexture();
}

// 출력 텍스처 반환 헬퍼 함수 구현
void TOPNodeBase::ReleaseOutputTexture()
{
    if (output_texture_ != nil && last_texture_pool_ != nullptr)
    {
        last_texture_pool_->release_texture(output_texture_);
        output_texture_ = nil;
    }
}

} // namespace example
