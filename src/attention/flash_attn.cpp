#include "flash_attn.h"

#include "../autograd/ops.h"

#ifdef SERIUS_USE_CUDA
namespace serius {
namespace hal {
void flash_attention_v2_fwd_cuda(float *output, float *lse, const float *query, const float *key, const float *value,
                                 int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
                                 float softmax_scale, bool is_causal);

void flash_attention_v2_bwd_cuda(float *dq, float *dk, float *dv, const float *q, const float *k, const float *v,
                                 const float *o, const float *do_grad, const float *lse, int batch_size, int num_heads,
                                 int seq_len_q, int seq_len_k, int head_dim, float softmax_scale, bool is_causal);
}  // namespace hal
}  // namespace serius
#endif

namespace serius {
namespace model {
namespace attention {

std::pair<core::Tensor, core::Tensor> flash_attention_forward_internal(
    const core::Tensor &query, const core::Tensor &key, const core::Tensor &value,
    [[maybe_unused]] const FlashAttentionConfig &config)
{
    if (query.is_gpu() != key.is_gpu() || query.is_gpu() != value.is_gpu()) {
        throw std::runtime_error("Flash attention: Q, K, V must be on the same device");
    }

    const auto &q_shape = query.shape();
    [[maybe_unused]] const auto &k_shape = key.shape();

    int batch_size = static_cast<int>(q_shape[0]);
    int num_heads = static_cast<int>(q_shape[1]);
    int seq_len_q = static_cast<int>(q_shape[2]);

    core::Tensor out(q_shape, query.dtype());
    core::Tensor lse({(size_t)batch_size, (size_t)num_heads, (size_t)seq_len_q}, query.dtype());

    if (query.is_gpu()) {
        out.syncToDevice();
        lse.syncToDevice();
    }

#ifdef SERIUS_USE_CUDA
    if (query.is_gpu()) {
        int seq_len_k = static_cast<int>(k_shape[2]);
        int head_dim = static_cast<int>(q_shape[3]);
        serius::hal::flash_attention_v2_fwd_cuda(out.data_gpu(), lse.data_gpu(), query.data_gpu(), key.data_gpu(),
                                                 value.data_gpu(), batch_size, num_heads, seq_len_q, seq_len_k,
                                                 head_dim, config.softmax_scale, config.is_causal);

        out.mark_dirty(core::Tensor::DataLocation::GPU);
        lse.mark_dirty(core::Tensor::DataLocation::GPU);
        return {std::move(out), std::move(lse)};
    }
#endif

    throw std::runtime_error("Flash attention only supported on CUDA.");
}

core::Tensor flash_attention_forward(const core::Tensor &query, const core::Tensor &key, const core::Tensor &value,
                                     const FlashAttentionConfig &config)
{
    auto result = flash_attention_forward_internal(query, key, value, config);
    return std::move(result.first);
}

core::autograd::Node *flash_attention(core::autograd::Node *query, core::autograd::Node *key,
                                      core::autograd::Node *value, const FlashAttentionConfig &config)
{
    // Execute forward pass
    auto [out_tensor, lse_tensor] = flash_attention_forward_internal(query->data, key->data, value->data, config);

    const auto &q_shape = query->data.shape();
    const auto &k_shape = key->data.shape();
    int batch_size = static_cast<int>(q_shape[0]);
    int num_heads = static_cast<int>(q_shape[1]);
    int seq_len_q = static_cast<int>(q_shape[2]);
    int seq_len_k = static_cast<int>(k_shape[2]);
    int head_dim = static_cast<int>(q_shape[3]);

    // Wrap in autograd Node
    core::autograd::Node *out_node = new core::autograd::Node(std::move(out_tensor));
    out_node->name = "flash_attention";
    out_node->children = {query, key, value};

    out_node->backward_fn = [query, key, value, out_node, lse = std::move(lse_tensor), config, batch_size, num_heads,
                             seq_len_q, seq_len_k, head_dim]() mutable {
        if (!query->requires_grad && !key->requires_grad && !value->requires_grad) {
            return;
        }

        // Initialize gradients if not already present
        if (query->requires_grad && query->grad.size() == 0) query->grad = core::Tensor::zeros(query->data.shape());
        if (key->requires_grad && key->grad.size() == 0) key->grad = core::Tensor::zeros(key->data.shape());
        if (value->requires_grad && value->grad.size() == 0) value->grad = core::Tensor::zeros(value->data.shape());

        if (query->requires_grad) query->grad.syncToDevice();
        if (key->requires_grad) key->grad.syncToDevice();
        if (value->requires_grad) value->grad.syncToDevice();

#ifdef SERIUS_USE_CUDA
        if (query->data.is_gpu()) {
            serius::hal::flash_attention_v2_bwd_cuda(
                query->grad.data_gpu(), key->grad.data_gpu(), value->grad.data_gpu(), query->data.data_gpu(),
                key->data.data_gpu(), value->data.data_gpu(), out_node->data.data_gpu(), out_node->grad.data_gpu(),
                lse.data_gpu(), batch_size, num_heads, seq_len_q, seq_len_k, head_dim, config.softmax_scale,
                config.is_causal);

            if (query->requires_grad) query->grad.mark_dirty(core::Tensor::DataLocation::GPU);
            if (key->requires_grad) key->grad.mark_dirty(core::Tensor::DataLocation::GPU);
            if (value->requires_grad) value->grad.mark_dirty(core::Tensor::DataLocation::GPU);
            return;
        }
#endif
        throw std::runtime_error("Flash attention backward not implemented on CPU.");
    };

    return out_node;
}

}  // namespace attention
}  // namespace model
}  // namespace serius
