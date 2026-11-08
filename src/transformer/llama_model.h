#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../autograd/node.h"
#include "positional.h"
#include "../distributed/context.h"
#include "../quant/quant_weight.h"

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

    /// Optional packed-quantized copies of the projection weights, populated
    /// by the GGUF loader. Used by the M==1 decode path (matvec_q); the FP32
    /// Node params above remain the source of truth for training/prefill.
    quant::QuantWeight q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
    /// Fused q|k|v and gate|up row-concatenated weights (same type/in/row_bytes)
    /// for fewer, larger GEMV calls in the decode path.
    quant::QuantWeight qkv_w, gate_up_w;

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

    /// Transposed lm_head [vocab, hidden] for fast row-major GEMV inference.
    /// Empty until populated by the weight loader; inference falls back to
    /// the regular matmul path when unset.
    core::Tensor &lm_head_t() { return lm_head_t_; }
    const core::Tensor *lm_head_t() const { return lm_head_t_.size() > 0 ? &lm_head_t_ : nullptr; }

    /// Optional packed-quantized lm_head [vocab rows x hidden cols] for the
    /// fast decode GEMV. Populated by the GGUF loader when available.
    quant::QuantWeight &lm_head_q() { return lm_head_q_; }
    const quant::QuantWeight *lm_head_q() const { return lm_head_q_.valid() ? &lm_head_q_ : nullptr; }

   private:
    LLamaConfig config_;
    distributed::DistributedTopology topology_;
    std::unique_ptr<core::autograd::Node> embed_tokens_;
    std::vector<std::unique_ptr<LLaMABlock>> layers_;
    std::unique_ptr<core::autograd::Node> norm_weight_;
    std::unique_ptr<core::autograd::Node> lm_head_;
    core::Tensor lm_head_t_;
    quant::QuantWeight lm_head_q_;
};

}  // namespace model
}  // namespace serius
