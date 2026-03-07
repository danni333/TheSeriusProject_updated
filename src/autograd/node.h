#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace autograd {

struct Node
{
    Tensor data;
    Tensor grad;
    std::vector<Node *> children;
    std::function<void()> backward_fn;
    bool requires_grad;
    std::string name;  // Optional, for debugging

    // Constructors
    explicit Node(Tensor &&t, bool requires_grad = true);
    explicit Node(const Tensor &t, bool requires_grad = true);

    // Initialize the gradient tensor with zeros (same shape as data)
    void zero_grad();

    // Disable copy/move to ensure stable pointers in the computational graph
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
};

}  // namespace autograd
}  // namespace serius
