#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace model {

struct RouterOutput
{
    std::vector<size_t> expert_indices;
    std::vector<float> expert_weights;
    std::vector<float> scores;
};

class Router
{
   public:
    Router(size_t d_model, size_t num_experts, size_t top_k, unsigned seed = 42);

    RouterOutput route(const core::Tensor &input) const;
    core::autograd::Node *forward(core::autograd::Node *input) const;
    core::autograd::Node *selected_weights(core::autograd::Node *input,
                                           const std::vector<size_t> &expert_indices) const;
    std::vector<core::autograd::Node *> parameters() const;

    size_t d_model() const { return d_model_; }
    size_t num_experts() const { return num_experts_; }
    size_t top_k() const { return top_k_; }
    const core::Tensor &gate_weight() const { return gate_weight_->data; }
    const core::Tensor &gate_bias() const { return gate_bias_->data; }

   private:
    size_t d_model_;
    size_t num_experts_;
    size_t top_k_;
    std::unique_ptr<core::autograd::Node> gate_weight_;
    std::unique_ptr<core::autograd::Node> gate_bias_;
};

RouterOutput route(const core::Tensor &x, const core::Tensor &gate_weight, size_t top_k);
RouterOutput route(const core::Tensor &x, const core::Tensor &gate_weight,
                   const core::Tensor &gate_bias, size_t top_k);
RouterOutput route(const core::Tensor &x, size_t top_k, const core::Tensor &gate_weight);

}  // namespace model
}  // namespace serius
