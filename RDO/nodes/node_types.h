#pragma once

namespace example
{
namespace nodes
{
enum class NodeType
{
    add,
    multiply,
    output,
    sine,
    time,
    value
};

struct Node
{
    NodeType type;
    float    value;

    explicit Node(const NodeType t) : type(t), value(0.f) {}

    Node(const NodeType t, const float v) : type(t), value(v) {}
};
} // namespace nodes
} // namespace example
