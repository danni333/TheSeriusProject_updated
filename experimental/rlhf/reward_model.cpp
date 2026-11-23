#include "reward_model.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_set>

#include "autograd/engine.h"
#include "autograd/ops.h"
#include "infer/llama_inference.h"

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

/// Delete every node reachable from \p root except the model's parameter
/// nodes (owned by the transformer via unique_ptr and shared across calls).
/// forward_hidden() builds a fresh autograd graph per call; without this the
/// whole activation graph leaks on every score()/compute_logprobs() call.
void free_graph(Node *root, const std::unordered_set<Node *> &params)
{
    if (!root) return;
    for (Node *n : core::autograd::topological_sort(root)) {
        if (n && params.find(n) == params.end()) {
            delete n;
        }
    }
}

/// Collect the parameter Node pointers owned by the transformer so free_graph
/// can tell them apart from per-call temporaries.
std::unordered_set<Node *> collect_params(const model::LLaMAModel &m)
{
    std::unordered_set<Node *> params;
    auto add = [&params](Node *n) {
        if (n) params.insert(n);
    };
    add(m.embed_tokens());
    add(m.norm_weight());
    add(m.lm_head());
    for (const auto &layer : m.layers()) {
        add(layer->input_layernorm_weight.get());
        add(layer->q_proj.get());
        add(layer->k_proj.get());
        add(layer->v_proj.get());
        add(layer->q_bias.get());
        add(layer->k_bias.get());
        add(layer->v_bias.get());
        add(layer->o_proj.get());
        add(layer->post_attention_layernorm_weight.get());
        add(layer->gate_proj.get());
        add(layer->up_proj.get());
        add(layer->down_proj.get());
    }
    return params;
}

}  // namespace

struct RewardModel::Impl
{
    model::LLamaConfig config;
    model::LLaMAModel transformer;
    Tensor reward_head;  // (hidden_size, 1) projection
    Tensor reward_bias;  // scalar bias
    std::unordered_set<Node *> params;  // transformer params, excluded from free_graph

    explicit Impl(const model::LLamaConfig &cfg) : config(cfg), transformer(cfg)
    {
        reward_head = Tensor({config.hidden_size, 1});
        reward_head.rand();
        reward_bias = Tensor({1});
        reward_bias.fill(0.0f);
        params = collect_params(transformer);
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

    free_graph(hidden, impl_->params);
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

    // logits_node's graph includes hidden and all block intermediates.
    free_graph(logits_node, impl_->params);
    return log_probs;
}

RewardModel::GenerationResult RewardModel::generate_with_logprobs(const Tensor &prompt_tokens,
                                                                  int max_new_tokens) const
{
    GenerationResult result;
    result.log_probs.clear();

    const size_t prompt_len = prompt_tokens.shape()[0];
    const size_t vocab_size = impl_->config.vocab_size;

    std::vector<int> tokens;
    tokens.reserve(prompt_len + static_cast<size_t>(std::max(0, max_new_tokens)));
    for (size_t i = 0; i < prompt_len; ++i) {
        tokens.push_back(static_cast<int>(prompt_tokens.item(i)));
    }

    if (max_new_tokens <= 0 || vocab_size == 0) {
        Tensor out({tokens.size()});
        for (size_t i = 0; i < tokens.size(); ++i) { out[i] = static_cast<float>(tokens[i]); }
        result.tokens = out;
        return result;
    }

    // Real autoregressive decode: prefill the prompt, then run the standard
    // single-token path with a KV cache. Greedy argmax keeps generation
    // deterministic; the recorded log_prob is log_softmax(logits)[chosen].
    infer::KVCache cache;
    Tensor prompt_tensor({prompt_len});
    for (size_t i = 0; i < prompt_len; ++i) { prompt_tensor[i] = static_cast<float>(tokens[i]); }

    Tensor logits = infer::llama_forward_inference(impl_->transformer, prompt_tensor, &cache);
    const size_t log_rows = logits.shape().empty() ? 0 : logits.shape()[0];
    const float *last_row = logits.data() + (log_rows > 0 ? (log_rows - 1) * vocab_size : 0);

    for (int step = 0; step < max_new_tokens; ++step) {
        // argmax over the last-position logits
        int next_token = 0;
        float best = last_row[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (last_row[v] > best) { best = last_row[v]; next_token = static_cast<int>(v); }
        }

        // log_softmax of the chosen logit (numerically stable)
        float max_logit = last_row[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (last_row[v] > max_logit) { max_logit = last_row[v]; }
        }
        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(last_row[v] - max_logit);
        }
        result.log_probs.push_back((best - max_logit) - std::log(sum_exp));

        tokens.push_back(next_token);

        Tensor next_logits = infer::llama_forward_single_token(
            impl_->transformer, next_token, prompt_len + static_cast<size_t>(step), &cache);
        const size_t nrows = next_logits.shape().empty() ? 0 : next_logits.shape()[0];
        last_row = next_logits.data() + (nrows > 0 ? (nrows - 1) * vocab_size : 0);
        // next_logits must outlive this iteration: copy into a stable buffer.
        static thread_local std::vector<float> logits_copy;
        logits_copy.assign(last_row, last_row + vocab_size);
        last_row = logits_copy.data();
    }

    Tensor out({tokens.size()});
    for (size_t i = 0; i < tokens.size(); ++i) { out[i] = static_cast<float>(tokens[i]); }
    result.tokens = out;
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
