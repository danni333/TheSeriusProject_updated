#pragma once

#include <string>
#include <vector>

#include "../autograd/node.h"
#include "positional.h"

namespace serius {
namespace model {

/// LLaMA/HuggingFace config (config.json from HF).
struct LLamaConfig
{
    size_t vocab_size = 32000;
    size_t hidden_size = 2048;
    size_t num_attention_heads = 32;
    size_t num_key_value_heads = 32;  // GQA if < num_attention_heads
    size_t num_hidden_layers = 22;
    size_t intermediate_size = 5632;
    size_t max_position_embeddings = 2048;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

/// One LLaMA decoder layer: RMSNorm, Q/K/V/O attention, RMSNorm, MLP (gate/up/down + SiLU).
class LLaMABlock
{
   public:
    LLaMABlock(const LLamaConfig &config);
    ~LLaMABlock();

    core::autograd::Node *forward(core::autograd::Node *input, size_t pos_offset = 0) const;

    size_t hidden_size() const { return hidden_size_; }
    size_t n_heads() const { return n_heads_; }
    size_t n_kv_heads() const { return n_kv_heads_; }
    size_t head_dim() const { return head_dim_; }

    core::autograd::Node *input_layernorm_weight;
    core::autograd::Node *q_proj;
    core::autograd::Node *k_proj;
    core::autograd::Node *v_proj;
    core::autograd::Node *o_proj;
    core::autograd::Node *post_attention_layernorm_weight;
    core::autograd::Node *gate_proj;
    core::autograd::Node *up_proj;
    core::autograd::Node *down_proj;

   private:
    size_t hidden_size_;
    size_t n_heads_;
    size_t n_kv_heads_;
    size_t head_dim_;
    float rms_norm_eps_;
    float rope_theta_;
};

/// Full LLaMA model compatible with HuggingFace LLaMA/LLaMA2/TinyLlama.
class LLaMAModel
{
   public:
    explicit LLaMAModel(const LLamaConfig &config);
    ~LLaMAModel();

    /// Forward: token_ids (shape [seq_len]) -> logits [seq_len, vocab_size].
    core::autograd::Node *forward(core::autograd::Node *token_ids) const;

    const LLamaConfig &config() const { return config_; }
    core::autograd::Node *embed_tokens() { return embed_tokens_; }
    core::autograd::Node *norm_weight() { return norm_weight_; }
    core::autograd::Node *lm_head() { return lm_head_; }
    core::autograd::Node *embed_tokens() const { return embed_tokens_; }
    core::autograd::Node *norm_weight() const { return norm_weight_; }
    core::autograd::Node *lm_head() const { return lm_head_; }
    const std::vector<LLaMABlock *> &layers() const { return layers_; }

   private:
    LLamaConfig config_;
    core::autograd::Node *embed_tokens_ = nullptr;
    std::vector<LLaMABlock *> layers_;
    core::autograd::Node *norm_weight_ = nullptr;
    core::autograd::Node *lm_head_ = nullptr;
};

}  // namespace model
}  // namespace serius
