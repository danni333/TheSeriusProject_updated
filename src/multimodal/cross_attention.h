#pragma once

#include <cstddef>

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace multimodal {

using core::Tensor;
using core::autograd::Node;

/// Bidirectional multi-head self-attention (ViT / audio encoder).
class EncoderSelfAttention
{
   public:
    EncoderSelfAttention(size_t d_model, size_t n_heads, unsigned seed = 42);
    ~EncoderSelfAttention();

    EncoderSelfAttention(const EncoderSelfAttention &) = delete;
    EncoderSelfAttention &operator=(const EncoderSelfAttention &) = delete;

    /// input: (seq, d_model) -> (seq, d_model)
    Node *forward(Node *input) const;

   private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_head_;
    Node *w_q_;
    Node *w_k_;
    Node *w_v_;
    Node *w_o_;
};

/// Cross-attention: text queries attend to visual/audio keys and values.
class CrossAttention
{
   public:
    CrossAttention(size_t d_model, size_t n_heads, unsigned seed = 42);
    ~CrossAttention();

    CrossAttention(const CrossAttention &) = delete;
    CrossAttention &operator=(const CrossAttention &) = delete;

    /// text_hidden: (seq_text, d_model), context: (seq_ctx, d_model)
    Node *forward(Node *text_hidden, Node *context) const;

   private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_head_;
    Node *w_q_;
    Node *w_k_;
    Node *w_v_;
    Node *w_o_;
};

}  // namespace multimodal
}  // namespace serius
