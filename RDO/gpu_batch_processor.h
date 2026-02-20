#pragma once

#include <vector>
#include <functional>

@protocol MTLDevice;
@protocol MTLCommandBuffer;
@protocol MTLCommandQueue;

namespace example
{

// GPU 배치 처리 시스템: 여러 노드의 GPU 작업을 하나의 CommandBuffer로 통합
class GPUBatchProcessor
{
public:
    GPUBatchProcessor(id<MTLDevice> device);
    ~GPUBatchProcessor();
    
    // GPU 작업 등록 (실제 실행은 flush()에서 수행)
    using GPUWork = std::function<void(id<MTLCommandBuffer>)>;
    void enqueue(GPUWork work);
    
    // 등록된 모든 GPU 작업을 하나의 CommandBuffer로 실행
    void flush();
    
    // 현재 등록된 작업 수
    size_t pending_count() const { return pending_work_.size(); }
    
    // 배치 처리 활성화/비활성화 (디버깅용)
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    std::vector<GPUWork> pending_work_;
    bool enabled_;
};

} // namespace example
