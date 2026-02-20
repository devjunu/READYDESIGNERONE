#pragma once

#import <Metal/Metal.h>
#include "port.h"
#include "../../graph.h"
#include "../../nodes/node_types.h"
#include "render_context.h"
#include <vector>
#include <string>
#include <memory>

namespace example
{

class TexturePool;
class GPUBatchProcessor;

// nodes 네임스페이스의 Node를 사용
using nodes::Node;

// 모든 노드의 베이스 클래스
class NodeBase
{
public:
    virtual ~NodeBase() = default;
    
    // ============ 필수 인터페이스 ============
    
    // 노드 UI 렌더링
    virtual void Render(Graph<Node>& graph) = 0;
    
    // Inspector 패널에서 속성 렌더링
    virtual void RenderInspector() = 0;
    
    // 노드가 속한 패밀리 반환
    virtual NodeFamily GetFamily() const = 0;
    
    // 노드 타입 이름 (예: "Blur", "ColorCorrection")
    virtual std::string GetTypeName() const = 0;
    
    // 노드 카테고리 (예: "TOP/Filter", "CHOP/Math")
    virtual std::string GetCategory() const = 0;
    
    // ============ 공통 속성 접근 ============
    
    int GetNodeId() const { return node_id_; }
    const std::vector<Port>& GetInputPorts() const { return input_ports_; }
    const std::vector<Port>& GetOutputPorts() const { return output_ports_; }
    
    // 특정 포트 검색
    const Port* GetInputPort(int index) const {
        return (index >= 0 && index < input_ports_.size()) ? &input_ports_[index] : nullptr;
    }
    
    const Port* GetOutputPort(int index) const {
        return (index >= 0 && index < output_ports_.size()) ? &output_ports_[index] : nullptr;
    }
    
    // 포트 ID로 검색
    const Port* FindPortById(int port_id) const {
        for (const auto& port : input_ports_) {
            if (port.id == port_id) return &port;
        }
        for (const auto& port : output_ports_) {
            if (port.id == port_id) return &port;
        }
        return nullptr;
    }
    
protected:
    int node_id_;                        // 노드의 고유 ID
    std::vector<Port> input_ports_;      // 입력 포트들
    std::vector<Port> output_ports_;     // 출력 포트들
    
    // 파생 클래스에서 포트 추가용 헬퍼 함수
    void AddInputPort(const Port& port) { input_ports_.push_back(port); }
    void AddOutputPort(const Port& port) { output_ports_.push_back(port); }
};

// ============ TOP (Texture Operators) 노드 베이스 ============
class TOPNodeBase : public NodeBase
{
public:
    // 가상 소멸자: TexturePool에 텍스처 자동 반환
    virtual ~TOPNodeBase();
    
    NodeFamily GetFamily() const override { return NodeFamily::TOP; }

    // GPU 처리 인터페이스 (배치 처리용) - 기존 시그니처 (역호환성)
    virtual void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool
    ) = 0;

    // GPU 처리 인터페이스 (RenderContext 포함) - 새 시그니처
    // 기본 구현: 기존 메서드 호출 (역호환성 유지)
    virtual void ProcessGPU(
        const std::vector<id<MTLTexture>>& inputs,
        id<MTLCommandBuffer> cmd_buffer,
        TexturePool* texture_pool,
        const RenderContext& context
    ) {
        // 기본 동작: RenderContext 무시하고 기존 메서드 호출
        ProcessGPU(inputs, cmd_buffer, texture_pool);
    }
    
    // 출력 텍스처 접근
    id<MTLTexture> GetOutputTexture() const { return output_texture_; }
    
    // 캐시 무효화 (프리뷰 스케일 변경 시)
    virtual void InvalidateCache();
    
    // TexturePool 참조 설정 (ProcessGPU 호출 시 자동 저장)
    void SetLastTexturePool(TexturePool* pool) { last_texture_pool_ = pool; }
    
protected:
    id<MTLTexture> output_texture_ = nil;
    TexturePool* last_texture_pool_ = nullptr;  // 마지막 사용 TexturePool
    
    // 출력 텍스처 반환 헬퍼 함수 (구현은 .mm 파일에)
    void ReleaseOutputTexture();
};

// ============ CHOP (Channel Operators) 노드 베이스 - 단일 값 ============
class CHOPNodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::CHOP; }
    
    // 신호 평가 인터페이스
    virtual float Evaluate(const std::vector<float>& inputs, float time) = 0;
    
    // 현재 출력 값
    float GetOutputValue() const { return output_value_; }
    
protected:
    float output_value_;
};

// ============ Audio CHOP (Channel Operators) 노드 베이스 - 멀티 채널 ============
// TouchDesigner에서는 모두 CHOP이지만, 구현상 최적화를 위해 분리
class AudioCHOPNodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::CHOP; }
    
    // 채널 데이터 평가 인터페이스
    virtual void Evaluate(const std::vector<std::vector<float>>& inputs, float time) = 0;
    
    // 출력 채널 데이터
    const std::vector<float>& GetOutputChannels() const { return output_channels_; }
    
protected:
    std::vector<float> output_channels_;
};

// ============ DAT (Data Operators) 노드 베이스 ============
class DATNodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::DAT; }
    
    // 데이터 처리 인터페이스
    virtual void Process() = 0;
    
    // 출력 데이터
    const std::string& GetOutputData() const { return output_data_; }
    
protected:
    std::string output_data_;
};

// ============ COMP (Component Operators) 노드 베이스 ============
class COMPNodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::COMP; }
    
    // 실행 인터페이스
    virtual void Execute() = 0;
};

// ============ SOP (Surface Operators) 노드 베이스 ============
class SOPNodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::SOP; }
    
    // 지오메트리 처리 인터페이스
    virtual void ProcessGeometry() = 0;
};

// ============ AI (머신러닝) 노드 베이스 ============
class AINodeBase : public NodeBase
{
public:
    NodeFamily GetFamily() const override { return NodeFamily::AI; }
    
    // AI 처리 인터페이스
    virtual void ProcessAI() = 0;
};

} // namespace example
