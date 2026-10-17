#include "router.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace serius {
namespace model {

namespace {

std::vector<float> softmax(const std::vector<float> &scores)
{
    if (scores.empty()) {
        return {};
    }

    float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> exps(scores.size());
    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        exps[i] = std::exp(scores[i] - max_score);
        sum += exps[i];
    }

    std::vector<float> result(scores.size());
    if (sum == 0.0f) {
        result.assign(scores.size(), 1.0f / static_cast<float>(scores.size()));
        return result;
    }

    for (size_t i = 0; i < scores.size(); ++i) {
        result[i] = exps[i] / sum;
    }
    return result;
}

}  // namespace

Router::Router(size_t d_model, size_t num_experts, size_t top_k, unsigned seed)
    : d_model_(d_model), num_experts_(num_experts), top_k_(top_k),
      gate_weight_(nullptr), gate_bias_(nullptr)
{
    if (d_model_ == 0 || num_experts_ == 0 || top_k_ == 0 || top_k_ > num_experts_) {
        throw std::invalid_argument("Router requires d_model > 0 and 0 < top_k <= num_experts");
    }

    core::Tensor gate_weight_data({d_model_, num_experts_});
    core::Tensor gate_bias_data({num_experts_});
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < num_experts_; ++j) {
            gate_weight_data.at(i, j) = dist(rng);
        }
    }
    gate_bias_data.fill(0.0f);
    gate_weight_ = std::make_unique<core::autograd::Node>(std::move(gate_weight_data));
    gate_bias_ = std::make_unique<core::autograd::Node>(std::move(gate_bias_data));
}

RouterOutput Router::route(const core::Tensor &input) const
{
    if (input.ndim() != 1) {
        throw std::invalid_argument("Router::route expects a 1D token vector");
    }
    if (input.shape().size() != 1 || input.shape()[0] != d_model_) {
        throw std::invalid_argument("Router input dimension does not match d_model");
    }

    return ::serius::model::route(input, gate_weight_->data, gate_bias_->data, top_k_);
}

core::autograd::Node *Router::forward(core::autograd::Node *input) const
{
    if (input == nullptr) {
        throw std::invalid_argument("Router::forward received a null input");
    }

    const auto decision = ::serius::model::route(input->data, gate_weight_->data, gate_bias_->data, top_k_);
    core::Tensor out(input->data.shape());
    out.fill(0.0f);

    for (size_t k = 0; k < decision.expert_indices.size(); ++k) {
        const auto expert_id = decision.expert_indices[k];
        const auto weight = decision.expert_weights[k];

        for (size_t i = 0; i < input->data.size(); ++i) {
            out[i] += weight * input->data[i] * gate_weight_->data.at(i % d_model_, expert_id);
        }
    }

    return new core::autograd::Node(std::move(out), false);
}

core::autograd::Node *Router::selected_weights(core::autograd::Node *input,
                                               const std::vector<size_t> &expert_indices) const
{
    if (input == nullptr) {
        throw std::invalid_argument("Router::selected_weights received a null input");
    }
    if (input->data.size() != d_model_ ||
        (input->data.ndim() != 1 && !(input->data.ndim() == 2 && input->data.shape()[0] == 1))) {
        throw std::invalid_argument("Router::selected_weights expects one token with d_model features");
    }

    std::vector<float> selected_scores;
    selected_scores.reserve(expert_indices.size());
    for (size_t expert : expert_indices) {
        if (expert >= num_experts_) {
            throw std::invalid_argument("Router selected expert index is out of range");
        }
        float score = gate_bias_->data[expert];
        for (size_t dim = 0; dim < d_model_; ++dim) {
            score += input->data[dim] * gate_weight_->data.at(dim, expert);
        }
        selected_scores.push_back(score);
    }
    const std::vector<float> probabilities = softmax(selected_scores);

    core::Tensor weights({probabilities.size()});
    for (size_t i = 0; i < probabilities.size(); ++i) {
        weights[i] = probabilities[i];
    }
    auto *out = new core::autograd::Node(std::move(weights));
    out->children = {input, gate_weight_.get(), gate_bias_.get()};
    out->backward_fn = [this, input, out, expert_indices, probabilities]() {
        float weighted_grad = 0.0f;
        for (size_t i = 0; i < probabilities.size(); ++i) {
            weighted_grad += out->grad[i] * probabilities[i];
        }

        for (size_t i = 0; i < probabilities.size(); ++i) {
            const size_t expert = expert_indices[i];
            const float score_grad = probabilities[i] * (out->grad[i] - weighted_grad);
            gate_bias_->grad[expert] += score_grad;
            for (size_t dim = 0; dim < d_model_; ++dim) {
                gate_weight_->grad.at(dim, expert) += input->data[dim] * score_grad;
                if (input->requires_grad) {
                    input->grad[dim] += gate_weight_->data.at(dim, expert) * score_grad;
                }
            }
        }
    };
    return out;
}

std::vector<core::autograd::Node *> Router::parameters() const
{
    return {gate_weight_.get(), gate_bias_.get()};
}

RouterOutput route(const core::Tensor &x, const core::Tensor &gate_weight, size_t top_k)
{
    if (gate_weight.ndim() != 2) {
        throw std::invalid_argument("gate_weight shape mismatch in route()");
    }
    return route(x, gate_weight, core::Tensor({gate_weight.shape()[1]}), top_k);
}

RouterOutput route(const core::Tensor &x, const core::Tensor &gate_weight,
                   const core::Tensor &gate_bias, size_t top_k)
{
    if (x.ndim() != 1) {
        throw std::invalid_argument("route() expects a flattened token (1D tensor)");
    }
    if (gate_weight.ndim() != 2 || gate_weight.shape()[0] != x.shape()[0]) {
        throw std::invalid_argument("gate_weight shape mismatch in route()");
    }

    size_t num_experts = gate_weight.shape()[1];
    if (gate_bias.ndim() != 1 || gate_bias.shape()[0] != num_experts) {
        throw std::invalid_argument("gate_bias shape mismatch in route()");
    }
    if (num_experts == 0) {
        return { {}, {}, {} };
    }

    const size_t effective_top_k = std::min(top_k == 0 ? num_experts : top_k, num_experts);
    std::vector<float> scores(num_experts, 0.0f);
    for (size_t expert = 0; expert < num_experts; ++expert) {
        float value = 0.0f;
        for (size_t idx = 0; idx < x.shape()[0]; ++idx) {
            value += x.item(idx) * gate_weight.at(idx, expert);
        }
        scores[expert] = value + gate_bias[expert];
    }

    std::vector<size_t> order(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&scores](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    RouterOutput out;
    out.scores.reserve(effective_top_k);
    out.expert_indices.reserve(effective_top_k);
    out.expert_weights.reserve(effective_top_k);

    for (size_t k = 0; k < effective_top_k; ++k) {
        const size_t expert_id = order[k];
        out.expert_indices.push_back(expert_id);
        out.scores.push_back(scores[expert_id]);
    }

    const auto weights = softmax(out.scores);
    for (size_t k = 0; k < weights.size(); ++k) {
        out.expert_weights.push_back(weights[k]);
    }

    return out;
}

RouterOutput route(const core::Tensor &x, size_t top_k, const core::Tensor &gate_weight)
{
    return ::serius::model::route(x, gate_weight, top_k);
}

}  // namespace model
}  // namespace serius
