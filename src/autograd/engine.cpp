#include "engine.h"

#include <algorithm>

namespace serius {
namespace core {
namespace autograd {

using core::Tensor;

void topo_sort(Node *node, std::set<Node *> &visited, std::vector<Node *> &order)
{
    if (visited.count(node)) return;
    visited.insert(node);
    for (auto *child : node->children) {
        if (child) {
            topo_sort(child, visited, order);
        }
    }
    order.push_back(node);
}

std::vector<Node *> topological_sort(Node *loss_node)
{
    std::vector<Node *> order;
    std::set<Node *> visited;
    topo_sort(loss_node, visited, order);
    return order;
}

void backward(Node *loss_node)
{
    if (loss_node->data.size() != 1) {
        throw std::runtime_error("backward() can only be called on a scalar loss node.");
    }

    // Initialize gradient of the loss node to 1.0 (dL/dL = 1)
    if (loss_node->grad.size() == 0) {
        loss_node->grad = Tensor(loss_node->data.shape());
    }
    loss_node->grad.ones();

    // Get nodes in topological order
    std::vector<Node *> order = topological_sort(loss_node);

    // Call backward functions in REVERSE topological order
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node *node = *it;
        if (node->backward_fn) {
            node->backward_fn();
        }
    }
}

}  // namespace autograd
}  // namespace core
}  // namespace serius
