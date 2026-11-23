#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace core {
namespace autograd {

struct Node
{
    core::Tensor data;
    core::Tensor grad;
    std::vector<Node *> children;
    std::function<void()> backward_fn;
    /// Optional forward replay: recomputes this node's output from operand
    /// tensors (same order as `children`). Set by ops that support compiled
    /// execution plans (see optim::GraphCompiler). May be null.
    std::function<core::Tensor(const std::vector<const core::Tensor *> &inputs)> forward_fn;
    bool requires_grad;
    std::string name;  // Optional, for debugging

    // Constructors
    explicit Node(core::Tensor &&t, bool requires_grad = true);
    explicit Node(const core::Tensor &t, bool requires_grad = true);

    // Initialize the gradient tensor with zeros (same shape as data)
    void zero_grad();

    // Disable copy/move to ensure stable pointers in the computational graph
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
};

}  // namespace autograd
}  // namespace core
}  // namespace serius
