#pragma once

#include <set>
#include <unordered_set>
#include <vector>

#include "node.h"

namespace serius {
namespace core {
namespace autograd {

// Performs a topological sort of the computational graph ending at `loss_node`
std::vector<Node *> topological_sort(Node *loss_node);

// Execute the backward pass starting from the given scalar `loss_node`
void backward(Node *loss_node);

// Helper function to recursively perform topological sorting
void topo_sort(Node *node, std::set<Node *> &visited, std::vector<Node *> &order);

// Delete every node reachable from `root` except the parameter nodes listed
// in `params` (owned by the model/module and shared across calls). After a
// forward()-style inference that returns plain Tensors, call this on the
// result node to avoid leaking the whole activation graph. Deleted operands
// are also pruned from kept params' children lists so future traversals
// cannot follow dangling pointers.
void free_graph(Node *root, const std::unordered_set<Node *> &params);

}  // namespace autograd
}  // namespace core
}  // namespace serius
