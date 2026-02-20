#pragma once

#include <string>

namespace example
{

// 노드 패밀리 (데이터 타입 카테고리) - TouchDesigner 대응
enum class NodeFamily
{
    TOP,    // Texture Operators (2D 이미지/비디오)
    CHOP,   // Channel Operators (신호/오디오/모션/로직)
    SOP,    // Surface Operators (3D 지오메트리)
    DAT,    // Data Operators (텍스트/테이블/JSON/스크립트)
    COMP,   // Component Operators (컨테이너/UI/실행흐름)
    MAT,    // Material Operators (쉐이더/재질)
    AI      // AI/ML Operators (커스텀 확장)
};

// 포트 방향
enum class PortDirection
{
    Input,
    Output
};

// 노드 포트 정의
struct Port
{
    int id;                        // 포트의 고유 ID (graph에서 attribute ID)
    NodeFamily family;             // 데이터 패밀리
    PortDirection direction;       // 입력/출력
    std::string data_type;         // 세부 데이터 타입 ("texture", "float", "vec3" 등)
    std::string name;              // 포트 이름 (UI 표시용)
    
    Port()
        : id(-1)
        , family(NodeFamily::TOP)
        , direction(PortDirection::Input)
        , data_type("unknown")
        , name("")
    {}
    
    Port(int _id, NodeFamily _family, PortDirection _direction, 
         const std::string& _data_type, const std::string& _name)
        : id(_id)
        , family(_family)
        , direction(_direction)
        , data_type(_data_type)
        , name(_name)
    {}
};

// 포트 간 연결 가능 여부 체크 (TouchDesigner 규칙 준수)
inline bool CanConnect(const Port& from_port, const Port& to_port)
{
    // 출력 -> 입력만 연결 가능
    if (from_port.direction != PortDirection::Output) return false;
    if (to_port.direction != PortDirection::Input) return false;
    
    // 같은 패밀리만 연결 가능 (TouchDesigner 규칙)
    // 단, Info CHOP은 TOP 입력을 받을 수 있음 (터치디자이너 호환)
    if (from_port.family == NodeFamily::TOP && to_port.family == NodeFamily::CHOP) {
        // TOP -> CHOP 연결 허용 (Info CHOP 등)
        return true;
    }
    
    if (from_port.family != to_port.family) return false;
    
    return true;
}

// NodeFamily를 문자열로 변환 (디버깅/UI용)
inline const char* NodeFamilyToString(NodeFamily family)
{
    switch (family)
    {
        case NodeFamily::TOP:    return "TOP";
        case NodeFamily::CHOP:   return "CHOP";
        case NodeFamily::SOP:    return "SOP";
        case NodeFamily::DAT:    return "DAT";
        case NodeFamily::COMP:   return "COMP";
        case NodeFamily::MAT:    return "MAT";
        case NodeFamily::AI:     return "AI";
        default: return "UNKNOWN";
    }
}

} // namespace example
