#include "node.h"

namespace serius {
namespace autograd {

Node::Node(Tensor &&t, bool requires_grad)
    : data(std::move(t)), grad(std::vector<size_t>{}), requires_grad(requires_grad)
{
    if (requires_grad) {
        grad = Tensor(data.shape());
        // Constructor already zero-initializes the memory.
    }
}

Node::Node(const Tensor &t, bool requires_grad) : data(t), grad(std::vector<size_t>{}), requires_grad(requires_grad)
{
    if (requires_grad) {
        grad = Tensor(data.shape());
    }
}

void Node::zero_grad()
{
    if (requires_grad && grad.size() > 0) {
        grad.zeros();
    }
}

}  // namespace autograd
}  // namespace serius
