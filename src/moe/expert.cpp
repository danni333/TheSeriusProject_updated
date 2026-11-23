#include "expert.h"

#include <stdexcept>
#include <unordered_set>

#include "../autograd/engine.h"
#include "../autograd/node.h"

namespace serius {
namespace model {

namespace {

// Deletes every node of the temporary autograd graph produced by FFN::forward,
// except the FFN parameters (owned by the FFN) and the caller-owned input node.
void cleanup_ffn_graph(const FFN &ffn, core::autograd::Node *out, core::autograd::Node *input)
{
    auto params = ffn.parameters();
    std::unordered_set<core::autograd::Node *> keep(params.begin(), params.end());
    keep.insert(input);
    for (auto *n : core::autograd::topological_sort(out)) {
        if (n != nullptr && !keep.count(n)) {
            delete n;
        }
    }
}

}  // namespace

Expert::Expert(size_t d_model, size_t d_ff, unsigned seed)
    : d_model_(d_model), d_ff_(d_ff), ffn_(std::make_unique<FFN>(d_model, d_ff, seed))
{
}

Expert::~Expert() = default;

core::Tensor Expert::forward(const core::Tensor &input) const
{
    if (input.ndim() == 1) {
        if (input.shape()[0] != d_model_) {
            throw std::invalid_argument("Expert expects a 1D token with shape [d_model]");
        }

        core::Tensor row = input.reshape({1, d_model_});
        core::autograd::Node in(row, false);
        core::autograd::Node *out = ffn_->forward(&in);
        core::Tensor result = out->data.reshape({d_model_}).copy();
        cleanup_ffn_graph(*ffn_, out, &in);
        return result;
    }

    if (input.ndim() == 2 && input.shape()[1] == d_model_) {
        core::autograd::Node in(input, false);
        core::autograd::Node *out = ffn_->forward(&in);
        core::Tensor result = out->data.copy();
        cleanup_ffn_graph(*ffn_, out, &in);
        return result;
    }

    throw std::invalid_argument("Expert expects a 1D token or 2D batch with shape [*, d_model]");
}

core::autograd::Node *Expert::forward(core::autograd::Node *input) const
{
    if (input == nullptr || input->data.ndim() != 2 || input->data.shape()[1] != d_model_) {
        throw std::invalid_argument("Expert expects a 2D node with shape [batch, d_model]");
    }
    return ffn_->forward(input);
}

std::vector<core::autograd::Node *> Expert::parameters() const
{
    return ffn_->parameters();
}

}  // namespace model
}  // namespace serius
