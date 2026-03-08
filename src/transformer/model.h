#pragma once

#include <vector>

#include "../autograd/node.h"
#include "block.h"
#include "layernorm.h"
#include "positional.h"

namespace serius {
namespace transformer {

struct ModelConfig
{
    static constexpr size_t DEFAULT_VOCAB_SIZE = 32000;
    static constexpr size_t DEFAULT_D_MODEL = 512;
    static constexpr size_t DEFAULT_N_HEADS = 8;
    static constexpr size_t DEFAULT_N_LAYERS = 6;
    static constexpr size_t DEFAULT_D_FF = 2048;
    static constexpr size_t DEFAULT_MAX_SEQ_LEN = 512;

    size_t vocab_size = DEFAULT_VOCAB_SIZE;
    size_t d_model = DEFAULT_D_MODEL;
    size_t n_heads = DEFAULT_N_HEADS;
    size_t n_layers = DEFAULT_N_LAYERS;
    size_t d_ff = DEFAULT_D_FF;
    size_t max_seq_len = DEFAULT_MAX_SEQ_LEN;
};

class TransformerModel
{
   public:
    explicit TransformerModel(const ModelConfig &config);
    ~TransformerModel();

    /// Forward pass
    /// Input: token indices as flat array of size seq_len (stored in a Node of shape {seq_len})
    /// Returns: logits of shape (seq_len, vocab_size) as autograd::Node*
    autograd::Node *forward(autograd::Node *token_ids, std::vector<KVCache *> *caches = nullptr) const;

    /// Retrieve all trainable parameters
    std::vector<autograd::Node *> parameters() const;

    // Parameters
    autograd::Node *embedding;  // (vocab_size, d_model)
    std::vector<TransformerBlock *> blocks;
    LayerNorm *final_norm;
    autograd::Node *output_proj;  // (d_model, vocab_size)

   private:
    ModelConfig config_;

    struct EmbeddingDims
    {
        bool is_batched;
        size_t batch_size;
        size_t seq_len;
    };

    autograd::Node *embedding_lookup(autograd::Node *token_ids) const;
    void do_copy_embedding(const Tensor &ids, const EmbeddingDims &dims, Tensor &hidden_data) const;
    void do_embedding_backward(autograd::Node *token_ids, const EmbeddingDims &dims, autograd::Node *hidden) const;
};

}  // namespace transformer
}  // namespace serius
