#pragma once

#include "node.h"

namespace serius {
namespace autograd {
namespace ops {

Node *matmul(Node *input_a, Node *input_b);
Node *add(Node *input_a, Node *input_b);
Node *relu(Node *input_node);
Node *gelu(Node *input_node);
Node *softmax(Node *input_node, int dimension = -1);
Node *cross_entropy(Node *input_logits, Node *input_targets);

// New Ops for Modular Attention
Node *reshape(Node *input_node, const std::vector<size_t> &new_shape);
Node *transpose(Node *input_node, size_t dim0 = static_cast<size_t>(-2), size_t dim1 = static_cast<size_t>(-1));
Node *masked_fill(Node *input_node, const Tensor &mask_tensor, float fill_value);
Node *mul_scalar(Node *input_node, float scalar_value);
Node *contiguous(Node *input_node);
Node *layernorm(Node *input, Node *gamma, Node *beta, float eps = 1e-5f);
Node *fused_add_layernorm(Node *input, Node *residual, Node *gamma, Node *beta, float eps = 1e-5f);
Node *fused_masked_softmax(Node *input, const Tensor &mask, float scale = 1.0f);

struct RoPEResult
{
    Node *queries;
    Node *keys;
};

constexpr float DEFAULT_ROPE_BASE = 10000.0F;
RoPEResult rope(Node *query_node, Node *key_node, size_t d_head, float rope_base = DEFAULT_ROPE_BASE,
                size_t pos_offset = 0);

}  // namespace ops
}  // namespace autograd
}  // namespace serius
