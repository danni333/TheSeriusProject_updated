// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#include "inference_capture.h"

#include <cstring>

#include "infer/llama_inference.h"
#include "transformer/llama_model.h"

namespace serius {
namespace interpret {

namespace {

// The hook API is process-global C callbacks, so routing goes through a
// single "active capture" pointer. capture() is not reentrant: concurrent
// inference with hooks installed would interleave records.
InferenceCapture *g_active_capture = nullptr;

}  // namespace

// Defined at namespace scope (not anonymous) so they match the friend
// declarations inside InferenceCapture and can call the private record_*.
void capture_layer_hidden(int layer, const float *hidden, size_t seq_len, size_t hidden_dim)
{
    if (g_active_capture != nullptr) {
        g_active_capture->record_hidden(layer, hidden, seq_len, hidden_dim);
    }
}

void capture_attention_row(int layer, int head, size_t query_idx,
                           const float *probs, size_t npos)
{
    (void)query_idx;  // rows are stored in arrival order per (layer, head)
    if (g_active_capture != nullptr) {
        g_active_capture->record_attention(layer, head, query_idx, probs, npos);
    }
}

InferenceCapture::~InferenceCapture()
{
    // Never leave hooks pointing at a destroyed object.
    if (g_active_capture == this) {
        g_active_capture = nullptr;
        infer::clear_activation_hooks();
    }
}

Tensor InferenceCapture::run(const model::LLaMAModel &model, const Tensor &token_ids)
{
    clear();

    // Save/restore any pre-existing hooks so callers can nest safely.
    const infer::ActivationHooks prev = infer::activation_hooks();
    infer::ActivationHooks hooks{};
    hooks.layer_hidden = &capture_layer_hidden;
    hooks.attention_row = &capture_attention_row;

    g_active_capture = this;
    infer::set_activation_hooks(hooks);

    Tensor logits;
    try {
        infer::KVCache cache;
        logits = infer::llama_forward_inference(model, token_ids, &cache);
    } catch (...) {
        g_active_capture = nullptr;
        infer::set_activation_hooks(prev);
        throw;
    }

    g_active_capture = nullptr;
    infer::set_activation_hooks(prev);
    return logits;
}

void InferenceCapture::ensure_layer(int layer)
{
    if (layer < 0) { return; }
    if (hidden_.size() <= static_cast<size_t>(layer)) {
        hidden_.resize(static_cast<size_t>(layer) + 1);
        attn_.resize(static_cast<size_t>(layer) + 1);
    }
}

void InferenceCapture::record_hidden(int layer, const float *data,
                                     size_t seq_len, size_t hidden_dim)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_layer(layer);
    Tensor t({seq_len, hidden_dim});
    std::memcpy(t.data(), data, seq_len * hidden_dim * sizeof(float));
    hidden_[layer].push_back(std::move(t));
}

void InferenceCapture::record_attention(int layer, int head, size_t query_idx,
                                        const float *probs, size_t npos)
{
    (void)query_idx;
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_layer(layer);
    if (attn_[layer].size() <= static_cast<size_t>(head)) {
        attn_[layer].resize(static_cast<size_t>(head) + 1);
    }
    Tensor t({npos});
    std::memcpy(t.data(), probs, npos * sizeof(float));
    attn_[layer][head].push_back(std::move(t));
}

size_t InferenceCapture::num_layers() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return hidden_.size();
}

const std::vector<Tensor> &InferenceCapture::hidden_calls(int layer) const
{
    static const std::vector<Tensor> kEmpty;
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer < 0 || hidden_.size() <= static_cast<size_t>(layer)) { return kEmpty; }
    return hidden_[layer];
}

Tensor InferenceCapture::hidden_last_position(int layer) const
{
    const auto &calls = hidden_calls(layer);
    if (calls.empty()) { return Tensor(); }
    const Tensor &last = calls.back();   // {seq_len, hidden_dim}
    const size_t hidden_dim = last.shape()[1];
    const size_t seq_len = last.shape()[0];
    Tensor row({1, hidden_dim});
    std::memcpy(row.data(), last.data() + (seq_len - 1) * hidden_dim,
                hidden_dim * sizeof(float));
    return row;
}

const std::vector<Tensor> &InferenceCapture::attention_rows(int layer, int head) const
{
    static const std::vector<Tensor> kEmpty;
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer < 0 || attn_.size() <= static_cast<size_t>(layer)) { return kEmpty; }
    if (head < 0 || attn_[layer].size() <= static_cast<size_t>(head)) { return kEmpty; }
    return attn_[layer][head];
}

size_t InferenceCapture::num_attention_heads(int layer) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer < 0 || attn_.size() <= static_cast<size_t>(layer)) { return 0; }
    return attn_[layer].size();
}

void InferenceCapture::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    hidden_.clear();
    attn_.clear();
}

Tensor logit_lens_head(const model::LLaMAModel &model)
{
    if (const Tensor *lmt = model.lm_head_t()) {
        return *lmt;
    }
    const auto *head = model.lm_head();
    if (head == nullptr || head->data.size() == 0) { return Tensor(); }
    // Stored {hidden, vocab} row-major -> produce {vocab, hidden}.
    const size_t hidden_dim = head->data.shape()[0];
    const size_t vocab = head->data.shape()[1];
    Tensor out({vocab, hidden_dim});
    const float *src = head->data.data();
    float *dst = out.data();
    for (size_t v = 0; v < vocab; ++v) {
        for (size_t d = 0; d < hidden_dim; ++d) {
            dst[v * hidden_dim + d] = src[d * vocab + v];
        }
    }
    return out;
}

}  // namespace interpret
}  // namespace serius
