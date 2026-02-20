#pragma once

#include "../graph.h"
#include "node_types.h"

// ImU32 is defined in imgui.h, but we avoid including it in header
// Use unsigned int as forward declaration
namespace example
{
namespace nodes
{
// Forward declare ImU32 - actual definition is in imgui.h
typedef unsigned int ImU32;

ImU32 EvaluateGraph(Graph<Node>& graph, const int root_node);
void UpdateTime(float time_seconds);
} // namespace nodes
} // namespace example
