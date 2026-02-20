#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <stack>
#include <stddef.h>
#include <utility>
#include <vector>
#include <unordered_map>

namespace example
{
template<typename ElementType>
struct Span
{
    using iterator = ElementType*;

    template<typename Container>
    Span(Container& c) : begin_(c.data()), end_(begin_ + c.size())
    {
    }

    iterator begin() const { return begin_; }
    iterator end() const { return end_; }

private:
    iterator begin_;
    iterator end_;
};

// 성능 최적화: Binary Search O(log n) → Hash Table O(1)
template<typename ElementType>
class IdMap
{
public:
    using iterator = typename std::vector<ElementType>::iterator;
    using const_iterator = typename std::vector<ElementType>::const_iterator;

    // Iterators

    const_iterator begin() const { return elements_.begin(); }
    const_iterator end() const { return elements_.end(); }

    // Element access

    Span<const ElementType> elements() const { return elements_; }

    // Capacity

    bool   empty() const { return map_.empty(); }
    size_t size() const { return map_.size(); }

    // Modifiers

    std::pair<iterator, bool> insert(int id, const ElementType& element);
    std::pair<iterator, bool> insert(int id, ElementType&& element);
    size_t                    erase(int id);
    void                      clear();

    // Lookup

    iterator       find(int id);
    const_iterator find(int id) const;
    bool           contains(int id) const;

private:
    std::vector<ElementType> elements_;
    std::unordered_map<int, size_t> map_;  // id -> index in elements_
};

template<typename ElementType>
std::pair<typename IdMap<ElementType>::iterator, bool> IdMap<ElementType>::insert(
    const int          id,
    const ElementType& element)
{
    auto it = map_.find(id);
    if (it != map_.end())
    {
        return std::make_pair(std::next(elements_.begin(), it->second), false);
    }

    size_t index = elements_.size();
    map_[id] = index;
    elements_.push_back(element);
    return std::make_pair(std::prev(elements_.end()), true);
}

template<typename ElementType>
std::pair<typename IdMap<ElementType>::iterator, bool> IdMap<ElementType>::insert(
    const int     id,
    ElementType&& element)
{
    auto it = map_.find(id);
    if (it != map_.end())
    {
        return std::make_pair(std::next(elements_.begin(), it->second), false);
    }

    size_t index = elements_.size();
    map_[id] = index;
    elements_.push_back(std::move(element));
    return std::make_pair(std::prev(elements_.end()), true);
}

template<typename ElementType>
size_t IdMap<ElementType>::erase(const int id)
{
    auto it = map_.find(id);
    if (it == map_.end())
    {
        return 0ull;
    }

    size_t index = it->second;
    
    // Swap with last element and update indices
    if (index < elements_.size() - 1)
    {
        std::swap(elements_[index], elements_.back());
        // Update the swapped element's index in map
        for (auto& pair : map_)
        {
            if (pair.second == elements_.size() - 1)
            {
                pair.second = index;
                break;
            }
        }
    }
    
    elements_.pop_back();
    map_.erase(it);

    return 1ull;
}

template<typename ElementType>
void IdMap<ElementType>::clear()
{
    elements_.clear();
    map_.clear();
}

template<typename ElementType>
typename IdMap<ElementType>::iterator IdMap<ElementType>::find(const int id)
{
    auto it = map_.find(id);
    return (it == map_.end()) ? elements_.end() : std::next(elements_.begin(), it->second);
}

template<typename ElementType>
typename IdMap<ElementType>::const_iterator IdMap<ElementType>::find(const int id) const
{
    auto it = map_.find(id);
    return (it == map_.end()) ? elements_.cend() : std::next(elements_.cbegin(), it->second);
}

template<typename ElementType>
bool IdMap<ElementType>::contains(const int id) const
{
    return map_.find(id) != map_.end();
}

// a very simple directional graph
template<typename NodeType>
class Graph
{
public:
    Graph() : current_id_(0), nodes_(), edges_from_node_(), node_neighbors_(), edges_(), dirty_(true) {}

    struct Edge
    {
        int id;
        int from, to;

        Edge() = default;
        Edge(const int id, const int f, const int t) : id(id), from(f), to(t) {}

        inline int  opposite(const int n) const { return n == from ? to : from; }
        inline bool contains(const int n) const { return n == from || n == to; }
    };

    // Element access

    NodeType&        node(int node_id);
    const NodeType&  node(int node_id) const;
    Span<const int>  neighbors(int node_id) const;
    Span<const Edge> edges() const;

    // Capacity

    size_t num_edges_from_node(int node_id) const;
    bool   node_exists(int node_id) const;

    // Modifiers

    int  insert_node(const NodeType& node);
    void erase_node(int node_id);

    int  insert_edge(int from, int to);
    void erase_edge(int edge_id);
    
    // Dirty flag for caching
    bool is_dirty() const { return dirty_; }
    void mark_dirty() { dirty_ = true; }
    void mark_clean() { dirty_ = false; }

private:
    int current_id_;
    // These contains map to the node id
    IdMap<NodeType>         nodes_;
    IdMap<int>              edges_from_node_;
    IdMap<std::vector<int>> node_neighbors_;

    // This container maps to the edge id
    IdMap<Edge> edges_;
    
    // Dirty flag for optimization
    mutable bool dirty_;
};

template<typename NodeType>
NodeType& Graph<NodeType>::node(const int id)
{
    return const_cast<NodeType&>(static_cast<const Graph*>(this)->node(id));
}

template<typename NodeType>
const NodeType& Graph<NodeType>::node(const int id) const
{
    const auto iter = nodes_.find(id);
    assert(iter != nodes_.end());
    return *iter;
}

template<typename NodeType>
Span<const int> Graph<NodeType>::neighbors(int node_id) const
{
    const auto iter = node_neighbors_.find(node_id);
    assert(iter != node_neighbors_.end());
    return *iter;
}

template<typename NodeType>
Span<const typename Graph<NodeType>::Edge> Graph<NodeType>::edges() const
{
    return edges_.elements();
}

template<typename NodeType>
size_t Graph<NodeType>::num_edges_from_node(const int id) const
{
    auto iter = edges_from_node_.find(id);
    assert(iter != edges_from_node_.end());
    return *iter;
}

template<typename NodeType>
bool Graph<NodeType>::node_exists(const int id) const
{
    return nodes_.contains(id);
}

template<typename NodeType>
int Graph<NodeType>::insert_node(const NodeType& node)
{
    const int id = current_id_++;
    assert(!nodes_.contains(id));
    nodes_.insert(id, node);
    edges_from_node_.insert(id, 0);
    node_neighbors_.insert(id, std::vector<int>());
    dirty_ = true;  // Mark graph as dirty
    return id;
}

template<typename NodeType>
void Graph<NodeType>::erase_node(const int id)
{

    // first, remove any potential dangling edges
    {
        static std::vector<int> edges_to_erase;

        for (const Edge& edge : edges_.elements())
        {
            if (edge.contains(id))
            {
                edges_to_erase.push_back(edge.id);
            }
        }

        for (const int edge_id : edges_to_erase)
        {
            erase_edge(edge_id);
        }

        edges_to_erase.clear();
    }

    nodes_.erase(id);
    edges_from_node_.erase(id);
    node_neighbors_.erase(id);
    dirty_ = true;  // Mark graph as dirty
}

template<typename NodeType>
int Graph<NodeType>::insert_edge(const int from, const int to)
{
    const int id = current_id_++;
    assert(!edges_.contains(id));
    assert(nodes_.contains(from));
    assert(nodes_.contains(to));
    edges_.insert(id, Edge(id, from, to));

    // update neighbor count
    assert(edges_from_node_.contains(from));
    *edges_from_node_.find(from) += 1;
    // update neighbor list
    assert(node_neighbors_.contains(from));
    node_neighbors_.find(from)->push_back(to);
    
    dirty_ = true;  // Mark graph as dirty

    return id;
}

template<typename NodeType>
void Graph<NodeType>::erase_edge(const int edge_id)
{
    // This is a bit lazy, we find the pointer here, but we refind it when we erase the edge based
    // on id key.
    // NOTE: ImNodes가 노드를 삭제하면서 이미 그래프에서 제거된 링크 id를 다시 알려주는 경우가 있어
    // here에 바로 return하여 중복 삭제에 의한 assertion을 방지한다.
    if (!edges_.contains(edge_id))
    {
        return;
    }
    const Edge& edge = *edges_.find(edge_id);

    // update neighbor count
    assert(edges_from_node_.contains(edge.from));
    int& edge_count = *edges_from_node_.find(edge.from);
    assert(edge_count > 0);
    edge_count -= 1;

    // update neighbor list
    {
        assert(node_neighbors_.contains(edge.from));
        auto neighbors = node_neighbors_.find(edge.from);
        auto iter = std::find(neighbors->begin(), neighbors->end(), edge.to);
        assert(iter != neighbors->end());
        neighbors->erase(iter);
    }

    edges_.erase(edge_id);
    dirty_ = true;  // Mark graph as dirty
}

template<typename NodeType, typename Visitor>
void dfs_traverse(const Graph<NodeType>& graph, const int start_node, Visitor visitor)
{
    std::stack<int> stack;

    stack.push(start_node);

    while (!stack.empty())
    {
        const int current_node = stack.top();
        stack.pop();

        visitor(current_node);

        for (const int neighbor : graph.neighbors(current_node))
        {
            stack.push(neighbor);
        }
    }
}
} // namespace example
