#pragma once

#include <memory>
#include <vector>

#include "node.h"

namespace serius {
namespace core {
namespace autograd {
namespace ops {

Node *matmul(Node *input_a, Node *input_b);
Node *add(Node *input_a, Node *input_b);
Node *relu(Node *input_node);
Node *gelu(Node *input_node);
Node *silu(Node *input_node);  // SiLU(x) = x * sigmoid(x), used by LLaMA MLP
Node *softmax(Node *input_node, int dimension = -1);
Node *cross_entropy(Node *input_logits, Node *input_targets);
Node *masked_cross_entropy(Node *input_logits, Node *input_targets, const core::Tensor &loss_mask);

// New Ops for Modular Attention
Node *reshape(Node *input_node, const std::vector<size_t> &new_shape);
Node *transpose(Node *input_node, size_t dim0 = static_cast<size_t>(-2), size_t dim1 = static_cast<size_t>(-1));
Node *masked_fill(Node *input_node, const core::Tensor &mask_tensor, float fill_value);
Node *mul_scalar(Node *input_node, float scalar_value);
Node *contiguous(Node *input_node);
const float DEFAULT_EPS = 1e-5F;
Node *layernorm(Node *input, Node *gamma, Node *beta, float eps = DEFAULT_EPS);
Node *rmsnorm(Node *input, Node *gamma, float eps = DEFAULT_EPS);
Node *fused_add_layernorm(Node *input, Node *residual, Node *gamma, Node *beta, float eps = DEFAULT_EPS);
Node *fused_masked_softmax(Node *input, const core::Tensor &mask, float scale = 1.0F);
Node *fused_bias_gelu(Node *input, Node *bias);
Node *flash_attention(Node *query, Node *key, Node *value, float scale, bool causal);

struct RoPEResult
{
    Node *queries;
    Node *keys;
};

constexpr float DEFAULT_ROPE_BASE = 10000.0F;
RoPEResult rope(Node *query_node, Node *key_node, size_t d_head, float rope_base = DEFAULT_ROPE_BASE,
                size_t pos_offset = 0, size_t training_ctx = 0, size_t target_ctx = 0);

namespace detail {

/// True while replay_forward() is re-executing an op for a compiled plan.
/// Ops use this to skip building backward_fn closures — replayed nodes are
/// temporary and never participate in backward, and some closures capture
/// full-size tensors (e.g. cross_entropy's probabilities) that would be
/// needlessly deep-copied on every execute().
inline thread_local int replay_depth = 0;
inline bool in_replay()
{
    return replay_depth > 0;
}
struct ReplayGuard
{
    ReplayGuard() { ++replay_depth; }
    ~ReplayGuard() { --replay_depth; }
};

/// Returns a non-owning host view of `t` when possible (Float32, contiguous,
/// non-empty), otherwise a deep copy. Used by replay_forward to avoid
/// duplicating operand data during plan execution.
inline core::Tensor replay_tensor_ref(const core::Tensor &t)
{
    if (t.dtype() == hal::DType::Float32 && t.is_contiguous() && t.size() > 0) {
        return core::Tensor(const_cast<float *>(t.data()), t.shape());
    }
    return t.copy();
}

/// Rebuilds temporary operand Nodes around `inputs`, invokes `fn` on them,
/// and returns the produced tensor. `fn` receives the operand Nodes in the
/// same order as `inputs` and must return a heap-allocated Node (which is
/// consumed and deleted here). Backs Node::forward_fn replays.
template <typename F>
core::Tensor replay_forward(F &&fn, const std::vector<const core::Tensor *> &inputs)
{
    std::vector<std::unique_ptr<Node>> owned;
    std::vector<Node *> args;
    owned.reserve(inputs.size());
    args.reserve(inputs.size());
    for (const core::Tensor *t : inputs) {
        owned.push_back(std::make_unique<Node>(replay_tensor_ref(*t), false));
        args.push_back(owned.back().get());
    }
    Node *result;
    {
        ReplayGuard guard;
        result = fn(args);
    }
    core::Tensor out = std::move(result->data);
    delete result;
    return out;
}

}  // namespace detail

}  // namespace ops
}  // namespace autograd
}  // namespace core
}  // namespace serius
