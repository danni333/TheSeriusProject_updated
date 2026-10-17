#include "moe_layer.h"

#include <stdexcept>

namespace serius {
namespace model {

MoELayer::MoELayer(size_t d_model, size_t num_experts, size_t top_k, size_t shared_experts, unsigned seed,
                   size_t expert_d_ff)
    : router_(d_model, num_experts, top_k, seed), shared_experts_(shared_experts)
{
    if (num_experts == 0 || top_k == 0 || top_k > num_experts) {
        throw std::invalid_argument("MoELayer requires 0 < top_k <= num_experts");
    }
    const size_t d_ff = expert_d_ff > 0 ? expert_d_ff : d_model * 4;

    shared_expert_layers_.reserve(shared_experts_);
    for (size_t i = 0; i < shared_experts_; ++i) {
        shared_expert_layers_.emplace_back(
            std::make_unique<Expert>(d_model, d_ff, seed + static_cast<unsigned>(i * 17)));
    }

    experts_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.emplace_back(std::make_unique<Expert>(
            d_model, d_ff, seed + 1000U + static_cast<unsigned>(i * 17)));
    }
}

MoELayer::~MoELayer() = default;

std::vector<core::autograd::Node *> MoELayer::parameters() const
{
    std::vector<core::autograd::Node *> params = router_.parameters();
    for (const auto &expert : shared_expert_layers_) {
        const auto expert_params = expert->parameters();
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    for (const auto &expert : experts_) {
        const auto expert_params = expert->parameters();
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    return params;
}

core::autograd::Node *MoELayer::forward(core::autograd::Node *input) const
{
    if (input == nullptr) {
        throw std::invalid_argument("MoELayer::forward called with nullptr");
    }

    const auto &x = input->data;
    if (x.ndim() == 0 || x.shape().back() != router_.d_model()) {
        throw std::invalid_argument("MoELayer input's last dimension must match d_model");
    }

    const size_t d_model = router_.d_model();
    const size_t token_count = x.size() / d_model;
    core::Tensor out_data(x.shape());
    out_data.fill(0.0f);

    struct TokenOutputs
    {
        std::vector<core::autograd::Node *> shared;
        std::vector<core::autograd::Node *> routed;
        core::autograd::Node *route_weights;
    };
    std::vector<TokenOutputs> token_outputs;
    token_outputs.reserve(token_count);
    std::vector<core::autograd::Node *> children;

    for (size_t token_idx = 0; token_idx < token_count; ++token_idx) {
        core::Tensor token_data({1, d_model});
        core::Tensor route_token({d_model});
        for (size_t col = 0; col < d_model; ++col) {
            const float value = x[token_idx * d_model + col];
            token_data.at(0, col) = value;
            route_token[col] = value;
        }

        auto *token_node = new core::autograd::Node(std::move(token_data), input->requires_grad);
        token_node->children = {input};
        token_node->backward_fn = [input, token_node, token_idx, d_model]() {
            if (!input->requires_grad) {
                return;
            }
            for (size_t col = 0; col < d_model; ++col) {
                input->grad[token_idx * d_model + col] += token_node->grad[col];
            }
        };

        TokenOutputs token_result;
        token_result.shared.reserve(shared_expert_layers_.size());
        for (const auto &expert : shared_expert_layers_) {
            auto *expert_out = expert->forward(token_node);
            token_result.shared.push_back(expert_out);
            children.push_back(expert_out);
        }

        const auto decision = router_.route(route_token);
        token_result.route_weights = router_.selected_weights(token_node, decision.expert_indices);
        children.push_back(token_result.route_weights);
        token_result.routed.reserve(decision.expert_indices.size());
        for (size_t k = 0; k < decision.expert_indices.size(); ++k) {
            const auto expert_id = decision.expert_indices[k];
            auto *expert_out = experts_[expert_id]->forward(token_node);
            token_result.routed.push_back(expert_out);
            children.push_back(expert_out);
            for (size_t col = 0; col < d_model; ++col) {
                out_data[token_idx * d_model + col] +=
                    decision.expert_weights[k] * expert_out->data[col];
            }
        }
        for (const auto *expert_out : token_result.shared) {
            for (size_t col = 0; col < d_model; ++col) {
                out_data[token_idx * d_model + col] += expert_out->data[col];
            }
        }
        token_outputs.push_back(std::move(token_result));
    }

    auto *out = new core::autograd::Node(std::move(out_data));
    out->children = std::move(children);
    out->backward_fn = [out, token_outputs, d_model, token_count]() {
        for (size_t token_idx = 0; token_idx < token_count; ++token_idx) {
            const auto &token_result = token_outputs[token_idx];
            for (auto *expert_out : token_result.shared) {
                for (size_t col = 0; col < d_model; ++col) {
                    expert_out->grad[col] += out->grad[token_idx * d_model + col];
                }
            }

            for (size_t k = 0; k < token_result.routed.size(); ++k) {
                auto *expert_out = token_result.routed[k];
                float gate_grad = 0.0f;
                const float gate_weight = token_result.route_weights->data[k];
                for (size_t col = 0; col < d_model; ++col) {
                    const float output_grad = out->grad[token_idx * d_model + col];
                    expert_out->grad[col] += gate_weight * output_grad;
                    gate_grad += output_grad * expert_out->data[col];
                }
                token_result.route_weights->grad[k] += gate_grad;
            }
        }
    };
    return out;
}

}  // namespace model
}  // namespace serius
