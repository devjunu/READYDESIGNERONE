#pragma once

#include "../graph.h"
#include "node_types.h"
#include <vector>

struct ImVec2;

namespace example
{
namespace nodes
{
struct OutputNode
{
    int r, g, b;
    int id;
};

void RenderOutputNode(const OutputNode& node, Graph<Node>& graph);
void CreateOutputNode(Graph<Node>& graph, std::vector<OutputNode>& nodes, const ImVec2& pos, int& root_node_id);
void DeleteOutputNode(Graph<Node>& graph, std::vector<OutputNode>& nodes, int node_id, int& root_node_id);
} // namespace nodes
} // namespace example
