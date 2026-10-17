#include "expert.h"

#include <stdexcept>

#include "../autograd/node.h"

namespace serius {
namespace model {

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
        delete out;
        return result;
    }

    if (input.ndim() == 2 && input.shape()[1] == d_model_) {
        core::autograd::Node in(input, false);
        core::autograd::Node *out = ffn_->forward(&in);
        core::Tensor result = out->data.copy();
        delete out;
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
