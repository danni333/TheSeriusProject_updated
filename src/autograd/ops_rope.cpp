#define _USE_MATH_DEFINES
#include "ops.h"

// Autograd ops — split from ops.cpp. Declarations live in ops.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "../blas/blas.h"
#include "../attention/flash_attn.h"
#ifdef SERIUS_USE_CUDA
#include "../hal/fused_kernels.h"
#endif
namespace serius {
namespace core {
namespace autograd {
namespace ops {
namespace {

float resolve_rope_base(float rope_base, size_t d_head, size_t training_ctx, size_t target_ctx)
{
    if (training_ctx == 0 || target_ctx <= training_ctx) {
        return rope_base;
    }
    if (d_head <= 2) {
        return rope_base;
    }
    const float scale_ratio = static_cast<float>(target_ctx) / static_cast<float>(training_ctx);
    return rope_base * std::pow(scale_ratio, static_cast<float>(d_head) / static_cast<float>(d_head - 2));
}

}  // namespace

void apply_rope_to_data(Tensor &rot_data, size_t batch_size, size_t seq_len, size_t d_head, size_t d_model,
                        bool is_batched, float rope_base, size_t pos_offset, size_t training_ctx,
                        size_t target_ctx)
{
    const float effective_base = resolve_rope_base(rope_base, d_head, training_ctx, target_ctx);
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            size_t absolute_pos = p_idx + pos_offset;
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(effective_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(absolute_pos) * freq;
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);

                    float x0_v =
                        is_batched ? rot_data.at(b_idx, p_idx, h_idx + i_idx) : rot_data.at(p_idx, h_idx + i_idx);
                    float x1_v = is_batched ? rot_data.at(b_idx, p_idx, h_idx + i_idx + 1)
                                            : rot_data.at(p_idx, h_idx + i_idx + 1);

                    if (is_batched) {
                        rot_data.at(b_idx, p_idx, h_idx + i_idx) = (x0_v * cos_val) - (x1_v * sin_val);
                        rot_data.at(b_idx, p_idx, h_idx + i_idx + 1) = (x0_v * sin_val) + (x1_v * cos_val);
                    } else {
                        rot_data.at(p_idx, h_idx + i_idx) = (x0_v * cos_val) - (x1_v * sin_val);
                        rot_data.at(p_idx, h_idx + i_idx + 1) = (x0_v * sin_val) + (x1_v * cos_val);
                    }
                }
            }
        }
    }
}

void apply_rope_backward_impl(Node *input_node, Node *out, size_t batch_size, size_t seq_len, size_t d_head,
                              size_t d_model, bool is_batched, float rope_base, size_t pos_offset,
                              size_t training_ctx, size_t target_ctx)
{
    if (!input_node->requires_grad) {
        return;
    }
    const float effective_base = resolve_rope_base(rope_base, d_head, training_ctx, target_ctx);
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            size_t absolute_pos = p_idx + pos_offset;
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(effective_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(absolute_pos) * freq;
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);

                    float g0_v =
                        is_batched ? out->grad.at(b_idx, p_idx, h_idx + i_idx) : out->grad.at(p_idx, h_idx + i_idx);
                    float g1_v = is_batched ? out->grad.at(b_idx, p_idx, h_idx + i_idx + 1)
                                            : out->grad.at(p_idx, h_idx + i_idx + 1);

                    if (is_batched) {
                        input_node->grad.at(b_idx, p_idx, h_idx + i_idx) += (g0_v * cos_val) + (g1_v * sin_val);
                        input_node->grad.at(b_idx, p_idx, h_idx + i_idx + 1) += (-(g0_v * sin_val)) + (g1_v * cos_val);
                    } else {
                        input_node->grad.at(p_idx, h_idx + i_idx) += (g0_v * cos_val) + (g1_v * sin_val);
                        input_node->grad.at(p_idx, h_idx + i_idx + 1) += (-(g0_v * sin_val)) + (g1_v * cos_val);
                    }
                }
            }
        }
    }
}

Node *apply_rope_impl(Node *input_node, size_t batch_size, size_t seq_len, size_t d_head, bool is_batched,
                      bool is_query, float rope_base, size_t pos_offset, size_t training_ctx, size_t target_ctx)
{
    Tensor rot_data = input_node->data.copy();
    size_t d_model = is_batched ? input_node->data.shape()[2] : input_node->data.shape()[1];

    apply_rope_to_data(rot_data, batch_size, seq_len, d_head, d_model, is_batched, rope_base, pos_offset,
                       training_ctx, target_ctx);

    Node *out = new Node(std::move(rot_data));
    out->name = is_query ? "rope_q" : "rope_k";
    out->children = {input_node};
    out->forward_fn = [batch_size, seq_len, d_head, d_model, is_batched, is_query, rope_base, pos_offset,
                       training_ctx, target_ctx](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward(
            [batch_size, seq_len, d_head, d_model, is_batched, is_query, rope_base, pos_offset, training_ctx,
             target_ctx](const std::vector<Node *> &a) {
                return apply_rope_impl(a[0], batch_size, seq_len, d_head, is_batched, is_query, rope_base,
                                       pos_offset, training_ctx, target_ctx);
            },
            in);
    };

    out->backward_fn = [input_node, out, batch_size, seq_len, d_head, d_model, is_batched, rope_base, pos_offset,
                        training_ctx, target_ctx]() {
        apply_rope_backward_impl(input_node, out, batch_size, seq_len, d_head, d_model, is_batched, rope_base,
                                 pos_offset, training_ctx, target_ctx);
    };
    return out;
}

RoPEResult rope(Node *query_node, Node *key_node, size_t d_head, float rope_base, size_t pos_offset,
                size_t training_ctx, size_t target_ctx)
{
    bool is_batched = (query_node->data.ndim() == 3);
    size_t batch_size = is_batched ? query_node->data.shape()[0] : 1;
    size_t seq_len = is_batched ? query_node->data.shape()[1] : query_node->data.shape()[0];

    Node *out_q = apply_rope_impl(query_node, batch_size, seq_len, d_head, is_batched, true, rope_base, pos_offset,
                                 training_ctx, target_ctx);
    Node *out_k = apply_rope_impl(key_node, batch_size, seq_len, d_head, is_batched, false, rope_base, pos_offset,
                                 training_ctx, target_ctx);

    return {out_q, out_k};
}

}  // namespace ops
}  // namespace autograd
}  // namespace core
}  // namespace serius

