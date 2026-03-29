#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../autograd/node.h"
#include "positional.h"
#include "../distributed/context.h"

namespace serius {
namespace model {

constexpr size_t DEFAULT_VOCAB_SIZE = 32000;
constexpr size_t DEFAULT_HIDDEN_SIZE = 2048;
constexpr size_t DEFAULT_NUM_HEADS = 32;
constexpr size_t DEFAULT_NUM_LAYERS = 22;
constexpr size_t DEFAULT_INTERMEDIATE_SIZE = 5632;
constexpr size_t DEFAULT_MAX_POS = 2048;
constexpr float EPSILON_DEFAULT = 1e-5F;
constexpr float ROPE_THETA_DEFAULT = 10000.0F;

/// LLaMA/HuggingFace config (config.json from HF).
struct LLamaConfig
{
    size_t vocab_size = DEFAULT_VOCAB_SIZE;
    size_t hidden_size = DEFAULT_HIDDEN_SIZE;
    size_t num_attention_heads = DEFAULT_NUM_HEADS;
    size_t num_key_value_heads = DEFAULT_NUM_HEADS;  // GQA if < num_attention_heads
    size_t num_hidden_layers = DEFAULT_NUM_LAYERS;
    size_t intermediate_size = DEFAULT_INTERMEDIATE_SIZE;
    size_t max_position_embeddings = DEFAULT_MAX_POS;
    float rms_norm_eps = EPSILON_DEFAULT;
    float rope_theta = ROPE_THETA_DEFAULT;
};

/// One LLaMA decoder layer: RMSNorm, Q/K/V/O attention, RMSNorm, MLP (gate/up/down + SiLU).
class LLaMABlock
{
   public:
    LLaMABlock(const LLamaConfig &config, 
               const distributed::DistributedTopology &topology = distributed::DistributedTopology::single_node());
    ~LLaMABlock();

    core::autograd::Node *forward(core::autograd::Node *input, size_t pos_offset = 0) const;

    void set_device(hal::DeviceType device) const;

    size_t hidden_size() const { return hidden_size_; }
    size_t n_heads() const { return n_heads_; }
    size_t n_kv_heads() const { return n_kv_heads_; }
    size_t head_dim() const { return head_dim_; }
    const distributed::DistributedTopology& topology() const { return topology_; }

    std::unique_ptr<core::autograd::Node> input_layernorm_weight;
    std::unique_ptr<core::autograd::Node> q_proj;
    std::unique_ptr<core::autograd::Node> k_proj;
    std::unique_ptr<core::autograd::Node> v_proj;
    std::unique_ptr<core::autograd::Node> o_proj;
    std::unique_ptr<core::autograd::Node> post_attention_layernorm_weight;
    std::unique_ptr<core::autograd::Node> gate_proj;
    std::unique_ptr<core::autograd::Node> up_proj;
    std::unique_ptr<core::autograd::Node> down_proj;

   private:
    size_t hidden_size_;
    size_t n_heads_;
    size_t n_kv_heads_;
    size_t head_dim_;
    float rms_norm_eps_;
    float rope_theta_;
    distributed::DistributedTopology topology_;
};

/// Full LLaMA model compatible with HuggingFace LLaMA/LLaMA2/TinyLlama.
class LLaMAModel
{
   public:
    explicit LLaMAModel(const LLamaConfig &config, 
                        const distributed::DistributedTopology &topology = distributed::DistributedTopology::single_node());
    ~LLaMAModel();

    /// Forward: token_ids (shape [seq_len]) -> logits [seq_len, vocab_size].
    core::autograd::Node *forward(core::autograd::Node *token_ids) const;

    void set_device(hal::DeviceType device) const;

    const LLamaConfig &config() const { return config_; }
    const distributed::DistributedTopology& topology() const { return topology_; }
    core::autograd::Node *embed_tokens() { return embed_tokens_.get(); }
    core::autograd::Node *norm_weight() { return norm_weight_.get(); }
    core::autograd::Node *lm_head() { return lm_head_.get(); }
    core::autograd::Node *embed_tokens() const { return embed_tokens_.get(); }
    core::autograd::Node *norm_weight() const { return norm_weight_.get(); }
    core::autograd::Node *lm_head() const { return lm_head_.get(); }
    const std::vector<std::unique_ptr<LLaMABlock>> &layers() const { return layers_; }

    core::autograd::Node *embed_tokens_raw() const { return embed_tokens_.get(); }

   private:
    LLamaConfig config_;
    distributed::DistributedTopology topology_;
    std::unique_ptr<core::autograd::Node> embed_tokens_;
    std::vector<std::unique_ptr<LLaMABlock>> layers_;
    std::unique_ptr<core::autograd::Node> norm_weight_;
    std::unique_ptr<core::autograd::Node> lm_head_;
};

}  // namespace model
}  // namespace serius
