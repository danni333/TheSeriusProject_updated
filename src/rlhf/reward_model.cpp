#include "reward_model.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "../autograd/ops.h"

namespace serius {
namespace rlhf {

using core::autograd::ops::matmul;
using core::autograd::ops::rmsnorm;

namespace {

/// Run the transformer up to (and including) the final RMSNorm to obtain the
/// last hidden states of shape (seq_len, hidden_size).
///
/// LLaMAModel::forward() only exposes logits, so the reward model re-runs the
/// embedding + block stack here to read the hidden states required by the
/// scalar reward head.
Node *forward_hidden(const model::LLaMAModel &transformer, const Tensor &tokens)
{
    const size_t seq_len = tokens.shape()[0];
    const size_t hidden_size = transformer.config().hidden_size;
    const size_t vocab_size = transformer.config().vocab_size;

    // Embedding lookup: gather embed_rows[token_id] for each position.
    Node *embed_weight = transformer.embed_tokens();
    Tensor hidden_data({seq_len, hidden_size});
    for (size_t i = 0; i < seq_len; ++i) {
        size_t token_id = static_cast<size_t>(tokens.item(i));
        if (token_id >= vocab_size) {
            token_id = 0;
        }
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden_data.at(i, j) = embed_weight->data.at(token_id, j);
        }
    }

    Node *hidden = new Node(std::move(hidden_data));
    hidden->requires_grad = false;

    // Transformer blocks
    for (const auto &layer : transformer.layers()) {
        hidden = layer->forward(hidden, 0);
    }

    // Final RMSNorm -> last hidden state
    hidden = rmsnorm(hidden, transformer.norm_weight(), transformer.config().rms_norm_eps);
    return hidden;
}

}  // namespace

struct RewardModel::Impl
{
    model::LLamaConfig config;
    model::LLaMAModel transformer;
    Tensor reward_head;  // (hidden_size, 1) projection
    Tensor reward_bias;  // scalar bias

    explicit Impl(const model::LLamaConfig &cfg) : config(cfg), transformer(cfg)
    {
        reward_head = Tensor({config.hidden_size, 1});
        reward_head.rand();
        reward_bias = Tensor({1});
        reward_bias.fill(0.0f);
    }
};

RewardModel::RewardModel(const model::LLamaConfig &config) : impl_(new Impl(config)) {}

RewardModel::RewardModel() : RewardModel(model::LLamaConfig{}) {}

RewardModel::~RewardModel() = default;

float RewardModel::score(const Tensor &tokens) const
{
    // tokens: (seq_len,) -> hidden states (seq_len, hidden_size)
    Node *hidden = forward_hidden(impl_->transformer, tokens);

    const size_t seq_len = tokens.shape()[0];
    const size_t hidden_size = impl_->config.hidden_size;

    // Take the last token's hidden state and project to a scalar reward:
    //   reward = last_hidden @ reward_head + reward_bias
    float reward = 0.0f;
    const float *w = impl_->reward_head.data();
    for (size_t d = 0; d < hidden_size; ++d) {
        reward += hidden->data.at(seq_len - 1, d) * w[d * 1 + 0];
    }
    reward += impl_->reward_bias[0];

    delete hidden;
    return reward;
}

std::vector<float> RewardModel::compute_logprobs(const Tensor &tokens) const
{
    // Hidden states after the final RMSNorm: (seq_len, hidden_size)
    Node *hidden = forward_hidden(impl_->transformer, tokens);

    const size_t seq_len = tokens.shape()[0];
    const size_t vocab_size = impl_->config.vocab_size;

    // logits = hidden @ lm_head  -> (seq_len, vocab_size)
    Node *logits_node = matmul(hidden, impl_->transformer.lm_head());
    const Tensor &logits = logits_node->data;

    std::vector<float> log_probs(seq_len, 0.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        float max_logit = logits.item(t, 0);
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits.item(t, v) > max_logit) {
                max_logit = logits.item(t, v);
            }
        }

        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(logits.item(t, v) - max_logit);
        }
        float log_sum = std::log(sum_exp) + max_logit;

        int target_token = static_cast<int>(tokens.item(t));
        if (target_token >= 0 && static_cast<size_t>(target_token) < vocab_size) {
            log_probs[t] = logits.item(t, static_cast<size_t>(target_token)) - log_sum;
        } else {
            log_probs[t] = -std::log(static_cast<float>(vocab_size));
        }
    }

    delete hidden;
    delete logits_node;
    return log_probs;
}

RewardModel::GenerationResult RewardModel::generate_with_logprobs(const Tensor &prompt_tokens,
                                                                  int max_new_tokens) const
{
    GenerationResult result;
    result.tokens = prompt_tokens.copy();
    result.log_probs.clear();

    Tensor current = prompt_tokens.copy();
    for (int step = 0; step < max_new_tokens; ++step) {
        auto log_probs = compute_logprobs(current);

        size_t seq_len = current.shape()[0];
        float best_log_prob = log_probs[seq_len - 1];
        int best_token = static_cast<int>(current.item(seq_len - 1));

        // Simplified generation: repeat the last token.
        int next_token = best_token;

        std::vector<size_t> new_shape = {seq_len + 1};
        Tensor new_tokens(new_shape);
        for (size_t i = 0; i < seq_len; ++i) {
            new_tokens[i] = current.item(i);
        }
        new_tokens[seq_len] = static_cast<float>(next_token);

        result.log_probs.push_back(best_log_prob);
        current = new_tokens;
    }

    result.tokens = current;
    return result;
}

float reward_preference_loss(float score_win, float score_lose)
{
    // L = -log(sigmoid(score_win - score_lose))
    // Numerically stable: log(1 + exp(-x)) for x >= 0, -x + log(1 + exp(x)) for x < 0
    float diff = score_win - score_lose;
    if (diff >= 0) {
        return std::log1p(std::exp(-diff));
    } else {
        return -diff + std::log1p(std::exp(diff));
    }
}

}  // namespace rlhf
}  // namespace serius
