#pragma once

#include <set>
#include <vector>

#include "node.h"


namespace serius {
namespace autograd {

// Performs a topological sort of the computational graph ending at `loss_node`
std::vector<Node *> topological_sort(Node *loss_node);

// Execute the backward pass starting from the given scalar `loss_node`
void backward(Node *loss_node);

// Helper function to recursively perform topological sorting
void topo_sort(Node *node, std::set<Node *> &visited, std::vector<Node *> &order);

}  // namespace autograd
}  // namespace serius
