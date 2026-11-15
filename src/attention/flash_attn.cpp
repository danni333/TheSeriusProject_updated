#include "flash_attn.h"

#include "../autograd/ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef SERIUS_USE_CUDA
namespace serius {
namespace hal {
void flash_attention_v2_fwd_cuda(float *output, float *lse, const float *query, const float *key, const float *value,
                                 int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
                                 float softmax_scale, bool is_causal, int block_m, int block_n);

void flash_attention_v2_bwd_cuda(float *dq, float *dk, float *dv, const float *q, const float *k, const float *v,
                                 const float *o, const float *do_grad, const float *lse, int batch_size, int num_heads,
                                 int seq_len_q, int seq_len_k, int head_dim, float softmax_scale, bool is_causal);
}  // namespace hal
}  // namespace serius
#endif

namespace serius {
namespace model {
namespace attention {

namespace {
struct FlashAttentionBlocks
{
    int query;
    int key_value;
};

FlashAttentionBlocks resolve_flash_attention_blocks(int head_dim, const FlashAttentionConfig &config)
{
    const FlashAttentionBlocks defaults =
        head_dim == 32 ? FlashAttentionBlocks{16, 16}
                       : (head_dim == 64 ? FlashAttentionBlocks{128, 32} : FlashAttentionBlocks{64, 16});
    const int block_m = config.block_m == 0 ? defaults.query : config.block_m;
    const int block_n = config.block_n == 0 ? defaults.key_value : config.block_n;
    const auto is_supported_block = [](int block) { return block == 16 || block == 32 || block == 64 || block == 128; };
    if (!is_supported_block(block_m) || !is_supported_block(block_n)) {
        throw std::invalid_argument("Flash attention: block sizes must be 16, 32, 64, or 128");
    }
    const size_t shared_memory_bytes =
        (static_cast<size_t>(block_m) + 2 * static_cast<size_t>(block_n)) * static_cast<size_t>(head_dim) *
        sizeof(float);
    if (shared_memory_bytes > 48 * 1024) {
        throw std::invalid_argument("Flash attention: requested blocks exceed the 48 KiB shared-memory limit");
    }
    return {block_m, block_n};
}

void validate_flash_attention_inputs(const core::Tensor &query, const core::Tensor &key, const core::Tensor &value,
                                     const FlashAttentionConfig &config)
{
    if (query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4) {
        throw std::invalid_argument("Flash attention: Q, K, and V must have shape (batch, heads, sequence, head_dim)");
    }

    const auto &q_shape = query.shape();
    const auto &k_shape = key.shape();
    const auto &v_shape = value.shape();
    if (q_shape[0] != k_shape[0] || q_shape[1] != k_shape[1] || k_shape != v_shape ||
        q_shape[3] != k_shape[3]) {
        throw std::invalid_argument("Flash attention: incompatible Q, K, and V shapes");
    }
    if (q_shape[0] == 0 || q_shape[1] == 0 || q_shape[2] == 0 || k_shape[2] == 0 || q_shape[3] == 0) {
        throw std::invalid_argument("Flash attention: dimensions must be non-zero");
    }
    if (q_shape[3] != 32 && q_shape[3] != 64 && q_shape[3] != 128) {
        throw std::invalid_argument("Flash attention: head dimension must be 32, 64, or 128");
    }
    if (query.dtype() != serius::hal::DType::Float32 || key.dtype() != serius::hal::DType::Float32 ||
        value.dtype() != serius::hal::DType::Float32) {
        throw std::invalid_argument("Flash attention: only Float32 tensors are supported");
    }
    if (!query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous()) {
        throw std::invalid_argument("Flash attention: Q, K, and V must be contiguous");
    }
    if (query.is_gpu() != key.is_gpu() || query.is_gpu() != value.is_gpu()) {
        throw std::invalid_argument("Flash attention: Q, K, and V must be on the same device");
    }
    if (!std::isfinite(config.softmax_scale)) throw std::invalid_argument("Flash attention: invalid softmax scale");
    (void)resolve_flash_attention_blocks(static_cast<int>(q_shape[3]), config);

    const size_t int_max = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t dim : q_shape) {
        if (dim > int_max) throw std::invalid_argument("Flash attention: dimensions exceed CUDA kernel limits");
    }
    for (size_t dim : k_shape) {
        if (dim > int_max) throw std::invalid_argument("Flash attention: dimensions exceed CUDA kernel limits");
    }
}
// ---------------------------------------------------------------------------
// CPU implementation: blocked/online-softmax forward and a backward that
// recomputes P = exp(S - LSE) row-by-row, never materializing the S^2 matrix.
// ---------------------------------------------------------------------------
namespace cpu {

// Per (b,h,q_pos): scores over key positions, honoring the causal rule
// k_pos > q_pos => masked (same convention as the CUDA kernel and tests).
inline bool key_allowed(bool is_causal, int q_pos, int k_pos)
{
    return !is_causal || k_pos <= q_pos;
}

void forward(const float *q, const float *k, const float *v, float *out, float *lse,
             int batch, int heads, int seq_q, int seq_k, int head_dim, float scale,
             bool is_causal)
{
    const size_t q_stride = static_cast<size_t>(seq_q) * head_dim;
    const size_t kv_stride = static_cast<size_t>(seq_k) * head_dim;
    std::vector<float> scores(static_cast<size_t>(seq_k));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(b) * heads + h);
            const float *qb = q + base * q_stride;
            const float *kb = k + base * kv_stride;
            const float *vb = v + base * kv_stride;
            float *ob = out + base * q_stride;
            float *lb = lse + base * static_cast<size_t>(seq_q);

            for (int i = 0; i < seq_q; ++i) {
                const float *qi = qb + static_cast<size_t>(i) * head_dim;
                // Pass 1: scores + max (online-softmax bookkeeping via two passes
                // is exact and simpler; scores buffer is O(seq_k) per row).
                float max_score = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < seq_k; ++j) {
                    if (!key_allowed(is_causal, i, j)) {
                        scores[j] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const float *kj = kb + static_cast<size_t>(j) * head_dim;
                    float s = 0.0F;
                    for (int d = 0; d < head_dim; ++d) s += qi[d] * kj[d];
                    s *= scale;
                    scores[j] = s;
                    max_score = std::max(max_score, s);
                }
                // Pass 2: exp sum -> lse, weighted V accumulation.
                double sum_exp = 0.0;
                for (int j = 0; j < seq_k; ++j) {
                    if (std::isfinite(scores[j])) sum_exp += std::exp(static_cast<double>(scores[j] - max_score));
                }
                lb[i] = max_score + static_cast<float>(std::log(sum_exp));
                float *oi = ob + static_cast<size_t>(i) * head_dim;
                for (int d = 0; d < head_dim; ++d) oi[d] = 0.0F;
                for (int j = 0; j < seq_k; ++j) {
                    if (!std::isfinite(scores[j])) continue;
                    const float p = static_cast<float>(std::exp(static_cast<double>(scores[j] - max_score)) / sum_exp);
                    const float *vj = vb + static_cast<size_t>(j) * head_dim;
                    for (int d = 0; d < head_dim; ++d) oi[d] += p * vj[d];
                }
            }
        }
    }
}

void backward(const float *q, const float *k, const float *v, const float *out, const float *dout,
              const float *lse, float *dq, float *dk, float *dv, int batch, int heads, int seq_q,
              int seq_k, int head_dim, float scale, bool is_causal)
{
    const size_t q_stride = static_cast<size_t>(seq_q) * head_dim;
    const size_t kv_stride = static_cast<size_t>(seq_k) * head_dim;
    std::vector<float> probs(static_cast<size_t>(seq_k));
    std::vector<float> dp(static_cast<size_t>(seq_k));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(b) * heads + h);
            const float *qb = q + base * q_stride;
            const float *kb = k + base * kv_stride;
            const float *vb = v + base * kv_stride;
            const float *ob = out + base * q_stride;
            const float *dob = dout + base * q_stride;
            const float *lb = lse + base * static_cast<size_t>(seq_q);
            float *dqb = dq ? dq + base * q_stride : nullptr;
            float *dkb = dk ? dk + base * kv_stride : nullptr;
            float *dvb = dv ? dv + base * kv_stride : nullptr;
            if (dkb) for (size_t i = 0; i < kv_stride; ++i) dkb[i] = 0.0F;
            if (dvb) for (size_t i = 0; i < kv_stride; ++i) dvb[i] = 0.0F;

            for (int i = 0; i < seq_q; ++i) {
                const float *qi = qb + static_cast<size_t>(i) * head_dim;
                const float *oi = ob + static_cast<size_t>(i) * head_dim;
                const float *doi = dob + static_cast<size_t>(i) * head_dim;
                float *dqi = dqb ? dqb + static_cast<size_t>(i) * head_dim : nullptr;
                if (dqi) for (int d = 0; d < head_dim; ++d) dqi[d] = 0.0F;

                // Recompute row probabilities: P_ij = exp(S_ij - LSE_i) for
                // unmasked j, 0 otherwise (masked entries contribute nothing).
                for (int j = 0; j < seq_k; ++j) {
                    if (!key_allowed(is_causal, i, j)) {
                        probs[j] = 0.0F;
                        continue;
                    }
                    const float *kj = kb + static_cast<size_t>(j) * head_dim;
                    float s = 0.0F;
                    for (int d = 0; d < head_dim; ++d) s += qi[d] * kj[d];
                    probs[j] = std::exp(s * scale - lb[i]);
                }
                // delta_i = <dO_i, O_i>
                double delta = 0.0;
                for (int d = 0; d < head_dim; ++d) delta += static_cast<double>(doi[d]) * oi[d];
                // dp_j = <dO_i, V_j>
                for (int j = 0; j < seq_k; ++j) {
                    if (probs[j] == 0.0F) { dp[j] = 0.0F; continue; }
                    const float *vj = vb + static_cast<size_t>(j) * head_dim;
                    double acc = 0.0;
                    for (int d = 0; d < head_dim; ++d) acc += static_cast<double>(doi[d]) * vj[d];
                    dp[j] = static_cast<float>(acc);
                }
                // Accumulate dQ_i, dK_j, dV_j.
                for (int j = 0; j < seq_k; ++j) {
                    if (probs[j] == 0.0F) continue;
                    const float ds = probs[j] * (dp[j] - static_cast<float>(delta));
                    const float *kj = kb + static_cast<size_t>(j) * head_dim;
                    float *dkj = dkb ? dkb + static_cast<size_t>(j) * head_dim : nullptr;
                    float *dvj = dvb ? dvb + static_cast<size_t>(j) * head_dim : nullptr;
                    const float p_dout = probs[j];
                    for (int d = 0; d < head_dim; ++d) {
                        if (dqi) dqi[d] += scale * ds * kj[d];
                        if (dkj) dkj[d] += scale * ds * qi[d];
                        if (dvj) dvj[d] += p_dout * doi[d];
                    }
                }
            }
        }
    }
}

}  // namespace cpu
}  // namespace

std::pair<core::Tensor, core::Tensor> flash_attention_forward_internal(
    const core::Tensor &query, const core::Tensor &key, const core::Tensor &value, const FlashAttentionConfig &config)
{
    validate_flash_attention_inputs(query, key, value, config);

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
        const FlashAttentionBlocks blocks = resolve_flash_attention_blocks(static_cast<int>(q_shape[3]), config);
        int seq_len_k = static_cast<int>(k_shape[2]);
        int head_dim = static_cast<int>(q_shape[3]);
        serius::hal::flash_attention_v2_fwd_cuda(out.data_gpu(), lse.data_gpu(), query.data_gpu(), key.data_gpu(),
                                                 value.data_gpu(), batch_size, num_heads, seq_len_q, seq_len_k,
                                                 head_dim, config.softmax_scale, config.is_causal, blocks.query,
                                                 blocks.key_value);

        out.mark_dirty(core::Tensor::DataLocation::GPU);
        lse.mark_dirty(core::Tensor::DataLocation::GPU);
        return {std::move(out), std::move(lse)};
    }
#endif

    // CPU path: online-softmax forward, O(seq_k) scratch per query row.
    cpu::forward(query.data(), key.data(), value.data(), out.data(), lse.data(), batch_size,
                 num_heads, seq_len_q, static_cast<int>(k_shape[2]), static_cast<int>(q_shape[3]),
                 config.softmax_scale, config.is_causal);
    return {std::move(out), std::move(lse)};
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
        // CPU path: recompute P row-by-row from Q,K,V,LSE (no S^2 storage).
        cpu::backward(query->data.data(), key->data.data(), value->data.data(),
                      out_node->data.data(), out_node->grad.data(), lse.data(),
                      query->requires_grad ? query->grad.data() : nullptr,
                      key->requires_grad ? key->grad.data() : nullptr,
                      value->requires_grad ? value->grad.data() : nullptr, batch_size, num_heads,
                      seq_len_q, seq_len_k, head_dim, config.softmax_scale, config.is_causal);
        return;
    };

    return out_node;
}

}  // namespace attention
}  // namespace model
}  // namespace serius
