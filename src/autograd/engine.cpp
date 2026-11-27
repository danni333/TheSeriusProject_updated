#include "engine.h"

#include <algorithm>
#include <unordered_set>

namespace serius {
namespace core {
namespace autograd {

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

void free_graph(Node *root, const std::unordered_set<Node *> &params)
{
    if (!root) {
        return;
    }
    const auto order = topological_sort(root);
    std::unordered_set<Node *> deleted;
    deleted.reserve(order.size());
    // Delete in REVERSE topological order (root first, operands last).
    // Several ops store non-owning views (reshape/transpose/permute/slice/
    // contiguous) of an operand's buffer; tearing down consumers before
    // producers guarantees no node is destroyed while a still-live node
    // aliases its storage.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node *n = *it;
        if (n && params.find(n) == params.end()) {
            deleted.insert(n);
            delete n;
        }
    }
    for (Node *p : params) {
        if (!p) {
            continue;
        }
        p->children.erase(std::remove_if(p->children.begin(), p->children.end(),
                                         [&](Node *c) { return deleted.count(c) > 0; }),
                          p->children.end());
    }
}

}  // namespace autograd
}  // namespace core
}  // namespace serius
