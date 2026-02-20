#pragma once

#import <Metal/Metal.h>
#include "node_base.h"
#include <functional>
#include <map>
#include <string>
#include <memory>

struct ImVec2;

namespace example
{

// nodes 네임스페이스의 Node를 사용
using nodes::Node;

// 노드 생성자 타입
using NodeCreator = std::function<std::unique_ptr<NodeBase>(Graph<Node>&, const ImVec2&, id<MTLDevice>)>;

// 노드 타입 정보
struct NodeTypeInfo
{
    std::string name;           // 노드 이름 (예: "Blur")
    std::string category;       // 카테고리 (예: "TOP/Filter")
    NodeFamily family;          // 패밀리
    NodeCreator creator;        // 생성 함수
    std::string description;    // 설명 (툴팁용)
    
    NodeTypeInfo() : family(NodeFamily::TOP) {}
    
    NodeTypeInfo(const std::string& _name, const std::string& _category, 
                 NodeFamily _family, NodeCreator _creator, const std::string& _desc = "")
        : name(_name), category(_category), family(_family), creator(_creator), description(_desc)
    {}
};

// 노드 레지스트리 (싱글톤)
class NodeRegistry
{
public:
    // 싱글톤 인스턴스
    static NodeRegistry& Instance()
    {
        static NodeRegistry instance;
        return instance;
    }
    
    // 노드 타입 등록
    void RegisterNodeType(const NodeTypeInfo& info)
    {
        std::string full_name = info.category + "/" + info.name;
        nodes_[full_name] = info;
        
        // 카테고리별 인덱스
        category_index_[info.category].push_back(full_name);
    }
    
    // 노드 생성
    std::unique_ptr<NodeBase> CreateNode(const std::string& full_name,
                                         Graph<Node>& graph,
                                         const ImVec2& pos,
                                         id<MTLDevice> device)
    {
        auto it = nodes_.find(full_name);
        if (it == nodes_.end()) return nullptr;
        
        return it->second.creator(graph, pos, device);
    }
    
    // 카테고리별 노드 목록 반환 (UI 메뉴 생성용)
    const std::map<std::string, std::vector<std::string>>& GetCategoryIndex() const
    {
        return category_index_;
    }
    
    // 특정 카테고리의 노드 목록
    const std::vector<std::string>* GetNodesInCategory(const std::string& category) const
    {
        auto it = category_index_.find(category);
        return (it != category_index_.end()) ? &it->second : nullptr;
    }
    
    // 노드 정보 조회
    const NodeTypeInfo* GetNodeInfo(const std::string& full_name) const
    {
        auto it = nodes_.find(full_name);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }
    
    // 모든 노드 정보
    const std::map<std::string, NodeTypeInfo>& GetAllNodes() const
    {
        return nodes_;
    }
    
    // 등록된 노드 개수
    size_t GetNodeCount() const { return nodes_.size(); }
    
private:
    NodeRegistry() = default;
    NodeRegistry(const NodeRegistry&) = delete;
    NodeRegistry& operator=(const NodeRegistry&) = delete;
    
    // full_name (category/name) -> NodeTypeInfo
    std::map<std::string, NodeTypeInfo> nodes_;
    
    // category -> [full_name1, full_name2, ...]
    std::map<std::string, std::vector<std::string>> category_index_;
};

// 자동 등록 헬퍼 클래스
class NodeRegistrar
{
public:
    NodeRegistrar(const NodeTypeInfo& info)
    {
        NodeRegistry::Instance().RegisterNodeType(info);
    }
};

// 자동 등록 매크로
// 자동 등록 매크로
#define REGISTER_NODE(ClassName, DisplayName, Category, Family, Creator, Description) \
    static NodeRegistrar g_registrar_##ClassName##_(NodeTypeInfo(DisplayName, Category, Family, Creator, Description))

} // namespace example
