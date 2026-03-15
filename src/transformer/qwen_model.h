#pragma once

#include <string>
#include <vector>

#include "../autograd/node.h"
#include "positional.h"

namespace serius {
namespace model {

/// Qwen/Qwen2 config structure mirroring config.json.
struct QwenConfig
{
    size_t vocab_size = 151936;
    size_t hidden_size = 2048;
    size_t num_attention_heads = 16;
    size_t num_key_value_heads = 16;
    size_t num_hidden_layers = 24;
    size_t intermediate_size = 5632;
    size_t max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

/// One Qwen decoder layer.
class QwenBlock
{
   public:
    QwenBlock(const QwenConfig &config);
    ~QwenBlock();

    core::autograd::Node *forward(core::autograd::Node *input, size_t pos_offset = 0) const;

    core::autograd::Node *input_layernorm_weight;
    
    // Projections
    core::autograd::Node *q_proj;
    core::autograd::Node *q_proj_bias;
    
    core::autograd::Node *k_proj;
    core::autograd::Node *k_proj_bias;
    
    core::autograd::Node *v_proj;
    core::autograd::Node *v_proj_bias;
    
    core::autograd::Node *o_proj;
    
    core::autograd::Node *post_attention_layernorm_weight;
    
    // MLP
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

/// Full Qwen model compatible with Qwen1.5/Qwen2 architecture.
class QwenModel
{
   public:
    explicit QwenModel(const QwenConfig &config);
    ~QwenModel();

    /// Forward: token_ids (shape [seq_len]) -> logits [seq_len, vocab_size].
    core::autograd::Node *forward(core::autograd::Node *token_ids) const;

    const QwenConfig &config() const { return config_; }

   private:
    QwenConfig config_;
    core::autograd::Node *embed_tokens_ = nullptr;
    std::vector<QwenBlock *> layers_;
    core::autograd::Node *norm_weight_ = nullptr;
    core::autograd::Node *lm_head_ = nullptr;
};

}  // namespace model
}  // namespace serius
