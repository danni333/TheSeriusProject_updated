#include "llama_inference.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <immintrin.h>

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                             \
    do {                                                                                             \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" \
                      << cudaGetErrorString(err) << "\"" << std::endl;                               \
            throw std::runtime_error("CUDA error");                                                  \
        }                                                                                            \
    } while (0)
#endif
#endif

#include "../blas/blas.h"
#include "../distributed/pipeline.h"
#ifdef SERIUS_USE_CUDA
#include "../hal/fused_kernels.h"
#endif

namespace serius {
namespace infer {

// Constants to eliminate magic numbers (clang-tidy)
constexpr size_t DEFAULT_MAX_SEQ_LEN = 4096;
constexpr float ROPE_THETA_DEFAULT = 10000.0F;
constexpr float EPSILON_DEFAULT = 1e-5F;
constexpr size_t AVX_FLOAT_STRIDE = 8;
constexpr float ATTENTION_MASK_VAL = -1e9F;

using core::Tensor;

[[maybe_unused]] static size_t get_dtype_size(hal::DType dtype) {
    switch (dtype) {
        case hal::DType::Float32:
            return 4;
        case hal::DType::Float16:
        case hal::DType::BFloat16:
            return 2;
        case hal::DType::Int8:
            return 1;
        default:
            return 4;
    }
}

// --- AVX2 Helpers ---

inline float _mm256_reduce_add_ps(__m256 input_vec) {
    __m128 high_vec = _mm256_extractf128_ps(input_vec, 1);
    __m128 low_vec = _mm256_castps256_ps128(input_vec);
    __m128 sum_vec = _mm_add_ps(high_vec, low_vec);
    sum_vec = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
    sum_vec = _mm_add_ss(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, 1));
    return _mm_cvtss_f32(sum_vec);
}

inline float vec_dot_avx2(const float *vec_a, const float *vec_b, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t vec_idx = 0;
    constexpr size_t AVX_STEP = 8;
    for (; vec_idx + AVX_STEP <= n; vec_idx += AVX_STEP) {
        vsum = _mm256_fmadd_ps(_mm256_loadu_ps(vec_a + vec_idx), _mm256_loadu_ps(vec_b + vec_idx), vsum);
    }
    float sum = _mm256_reduce_add_ps(vsum);
    for (; vec_idx < n; ++vec_idx) {
        sum += vec_a[vec_idx] * vec_b[vec_idx];
    }
    return sum;
}

struct ScaleAccumulator {
    float *data;
    const float *increment;
    float scale;
    size_t count;
};

static void vec_add_scale_avx2(const ScaleAccumulator &args)
{
    const __m256 v_scale = _mm256_set1_ps(args.scale);
    size_t vec_idx = 0;
    constexpr size_t AVX_STEP = 8;
    for (; vec_idx + AVX_STEP <= args.count; vec_idx += AVX_STEP) {
        __m256 v_out = _mm256_loadu_ps(args.data + vec_idx);
        const __m256 v_src = _mm256_loadu_ps(args.increment + vec_idx);
        v_out = _mm256_fmadd_ps(v_src, v_scale, v_out);
        _mm256_storeu_ps(args.data + vec_idx, v_out);
    }
    for (; vec_idx < args.count; ++vec_idx) {
        args.data[vec_idx] += args.increment[vec_idx] * args.scale;
    }
}

struct RMSNormArgs {
    const float *inputs;
    const float *weights;
    float *outputs;
    size_t rows;
    size_t dim;
    float eps;
};

static void rms_norm(const RMSNormArgs &args) {
    for (size_t row_idx = 0; row_idx < args.rows; ++row_idx) {
        const float *in_row = args.inputs + (row_idx * args.dim);
        float *out_row = args.outputs + (row_idx * args.dim);
        float sum_sq = 0.0F;
        size_t float_idx = 0;
        __m256 v_sum = _mm256_setzero_ps();
        for (; float_idx + AVX_FLOAT_STRIDE <= args.dim; float_idx += AVX_FLOAT_STRIDE) {
            const __m256 vec_v = _mm256_loadu_ps(in_row + float_idx);
            v_sum = _mm256_fmadd_ps(vec_v, vec_v, v_sum);
        }
        sum_sq = _mm256_reduce_add_ps(v_sum);
        for (; float_idx < args.dim; ++float_idx) {
            sum_sq += in_row[float_idx] * in_row[float_idx];
        }
        const float rms = 1.0F / std::sqrt((sum_sq / static_cast<float>(args.dim)) + args.eps);
        const __m256 v_rms = _mm256_set1_ps(rms);
        for (float_idx = 0; float_idx + AVX_FLOAT_STRIDE <= args.dim; float_idx += AVX_FLOAT_STRIDE) {
            const __m256 vec_x = _mm256_loadu_ps(in_row + float_idx);
            const __m256 vec_w = _mm256_loadu_ps(args.weights + float_idx);
            _mm256_storeu_ps(out_row + float_idx, _mm256_mul_ps(_mm256_mul_ps(vec_x, v_rms), vec_w));
        }
        for (; float_idx < args.dim; ++float_idx) {
            out_row[float_idx] = (in_row[float_idx] * rms) * args.weights[float_idx];
        }
    }
}

struct RoPEArgs {
    float *data;
    size_t seq_len;
    size_t model_dim;
    size_t head_dim;
    float rope_base;
    size_t pos_offset;
};

static void apply_rope_inplace(const RoPEArgs &args) {
    for (size_t pos = 0; pos < args.seq_len; ++pos) {
        const float absolute_pos = static_cast<float>(pos + args.pos_offset);
        float *row_ptr = args.data + (pos * args.model_dim);
        for (size_t head_offset = 0; head_offset < args.model_dim; head_offset += args.head_dim) {
            for (size_t feature_idx = 0; feature_idx < args.head_dim; feature_idx += 2) {
                const float frequency = 1.0F / std::pow(args.rope_base, static_cast<float>(feature_idx) / static_cast<float>(args.head_dim));
                const float theta = absolute_pos * frequency;
                const float cos_value = std::cos(theta);
                const float sin_value = std::sin(theta);
                const float val_x0 = row_ptr[head_offset + feature_idx];
                const float val_x1 = row_ptr[head_offset + feature_idx + 1];
                row_ptr[head_offset + feature_idx]     = (val_x0 * cos_value) - (val_x1 * sin_value);
                row_ptr[head_offset + feature_idx + 1] = (val_x0 * sin_value) + (val_x1 * cos_value);
            }
        }
    }
}

static void silu_inplace(float *data_ptr, size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        const float val = data_ptr[idx];
        data_ptr[idx] = val / (1.0F + std::exp(-val));
    }
}

struct SoftmaxRowsArgs {
    float *data;
    size_t rows;
    size_t cols;
};

static void softmax_rows_inplace(const SoftmaxRowsArgs &args) {
    for (size_t row_idx = 0; row_idx < args.rows; ++row_idx) {
        float *row = args.data + (row_idx * args.cols);
        float max_val = row[0];
        for (size_t col_idx = 1; col_idx < args.cols; ++col_idx) {
            max_val = std::max(max_val, row[col_idx]);
        }
        float sum_val = 0.0F;
        for (size_t col_idx = 0; col_idx < args.cols; ++col_idx) {
            row[col_idx] = std::exp(row[col_idx] - max_val);
            sum_val += row[col_idx];
        }
        const float inv_sum = 1.0F / sum_val;
        for (size_t col_idx = 0; col_idx < args.cols; ++col_idx) {
            row[col_idx] *= inv_sum;
        }
    }
}

// --- Internal Helper Functions for CPU Inference ---

/**
 * @brief Computes Q, K, and V projections and applies RoPE.
 */
static void compute_qkv_rope(
    const model::LLaMABlock *block, const Tensor &normed_tensor,
    Tensor &query_buf, Tensor &key_buf, Tensor &value_buf,
    size_t seq_len, size_t pos_offset, const model::LLamaConfig &cfg) {
    
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t local_hidden_dim = query_buf.shape()[1];
    const size_t local_kv_dim = key_buf.shape()[1];

    core::blas::matmul(normed_tensor, block->q_proj->data, query_buf);
    core::blas::matmul(normed_tensor, block->k_proj->data, key_buf);
    core::blas::matmul(normed_tensor, block->v_proj->data, value_buf);

    apply_rope_inplace({query_buf.data(), seq_len, local_hidden_dim, head_dim, cfg.rope_theta, pos_offset});
    apply_rope_inplace({key_buf.data(), seq_len, local_kv_dim, head_dim, cfg.rope_theta, pos_offset});
}

/**
 * @brief Updates the KV cache with new key and value tensors.
 */
static void update_kv_cache(KVCacheLayer *cache_layer, const Tensor &key_buf, const Tensor &value_buf, size_t seq_len) {
    if (cache_layer == nullptr) {
        return;
    }
    const size_t local_kv_dim = key_buf.shape()[1];
    const size_t start_pos = cache_layer->current_len;
    for (size_t i = 0; i < seq_len; ++i) {
        std::memcpy(cache_layer->k.data() + ((start_pos + i) * local_kv_dim), key_buf.data() + (i * local_kv_dim), local_kv_dim * sizeof(float));
        std::memcpy(cache_layer->v.data() + ((start_pos + i) * local_kv_dim), value_buf.data() + (i * local_kv_dim), local_kv_dim * sizeof(float));
    }
    cache_layer->current_len += seq_len;
}

struct HeadAttentionArgs {
    int head_idx;
    size_t seq_len;
    size_t pos_offset;
    size_t total_len;
    size_t local_hidden_dim;
    size_t local_kv_dim;
    size_t head_dim;
    size_t num_rep;
    const Tensor &query_buf;
    const Tensor &key_buf;
    const Tensor &value_buf;
    KVCacheLayer *cache_layer;
    float attention_scale;
    float *attention_out_ptr;
};

/**
 * @brief Computes attention scores and output for a single head.
 */
static void compute_head_attention(const HeadAttentionArgs &args) {
    const size_t kv_head_idx = static_cast<size_t>(args.head_idx) / args.num_rep;
    for (size_t qi = 0; qi < args.seq_len; ++qi) {
        const size_t abs_qi = args.pos_offset + qi;
        std::vector<float> scores(args.total_len);
        const float *q_ptr = args.query_buf.data() + (qi * args.local_hidden_dim) + (static_cast<size_t>(args.head_idx) * args.head_dim);

        for (size_t kj = 0; kj <= abs_qi; ++kj) {
            const float *k_ptr = (args.cache_layer != nullptr) 
                ? (args.cache_layer->k.data() + (kj * args.local_kv_dim) + (kv_head_idx * args.head_dim))
                : (args.key_buf.data() + (kj * args.local_kv_dim) + (kv_head_idx * args.head_dim));
            scores[kj] = vec_dot_avx2(q_ptr, k_ptr, args.head_dim) * args.attention_scale;
        }
        for (size_t kj = abs_qi + 1; kj < args.total_len; ++kj) {
            scores[kj] = ATTENTION_MASK_VAL;
        }

        softmax_rows_inplace({scores.data(), 1, args.total_len});

        float *out_ptr = args.attention_out_ptr + (qi * args.local_hidden_dim) + (static_cast<size_t>(args.head_idx) * args.head_dim);
        std::memset(out_ptr, 0, args.head_dim * sizeof(float));
        for (size_t kj = 0; kj <= abs_qi; ++kj) {
            const float *v_ptr = (args.cache_layer != nullptr)
                ? (args.cache_layer->v.data() + (kj * args.local_kv_dim) + (kv_head_idx * args.head_dim))
                : (args.value_buf.data() + (kj * args.local_kv_dim) + (kv_head_idx * args.head_dim));
            vec_add_scale_avx2({out_ptr, v_ptr, scores[kj], args.head_dim});
        }
    }
}

static void initialize_kv_cache(KVCache *cache, const model::LLaMAModel &model, size_t local_kv_dim) {
    if (cache != nullptr && (cache->empty() || cache->layers.size() != model.layers().size())) {
        const size_t num_layers = model.layers().size();
        cache->layers.resize(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            cache->layers[i].k = Tensor({DEFAULT_MAX_SEQ_LEN, local_kv_dim});
            cache->layers[i].v = Tensor({DEFAULT_MAX_SEQ_LEN, local_kv_dim});
            cache->layers[i].current_len = 0;
        }
    }
}

static void preprocess_embeddings(const model::LLaMAModel &model, const Tensor &token_ids, Tensor &hidden_states) {
    if (model.embed_tokens() != nullptr) {
        const size_t seq_len = token_ids.size();
        const size_t hidden_dim = model.config().hidden_size;
        const float *embed_ptr = model.embed_tokens()->data.data();
        for (size_t i = 0; i < seq_len; ++i) {
            size_t tid = static_cast<size_t>(token_ids[i]);
            if (tid >= model.config().vocab_size) {
                tid = 0;
            }
            std::copy(embed_ptr + (tid * hidden_dim), embed_ptr + ((tid + 1) * hidden_dim), hidden_states.data() + (i * hidden_dim));
        }
    }
}

static void llama_attention_layer_cpu(
    const model::LLaMABlock *block,
    float *hidden_states,
    float *normed_states,
    size_t seq_len,
    size_t pos_offset,
    const model::LLamaConfig &cfg,
    const distributed::DistributedTopology &topology,
    KVCacheLayer *cache_layer) {

    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t tp_size = static_cast<size_t>(topology.tp_size);
    const size_t num_local_heads = cfg.num_attention_heads / tp_size;
    const size_t num_local_kv_heads = cfg.num_key_value_heads / tp_size;
    const size_t local_hidden_dim = num_local_heads * head_dim;
    const size_t local_kv_dim = num_local_kv_heads * head_dim;

    rms_norm({hidden_states, block->input_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim, cfg.rms_norm_eps});

    Tensor query_buf({seq_len, local_hidden_dim});
    Tensor key_buf({seq_len, local_kv_dim});
    Tensor value_buf({seq_len, local_kv_dim});
    Tensor attention_out({seq_len, local_hidden_dim});
    Tensor attention_proj({seq_len, hidden_dim});

    Tensor normed_tensor({seq_len, hidden_dim});
    std::memcpy(normed_tensor.data(), normed_states, seq_len * hidden_dim * sizeof(float));

    compute_qkv_rope(block, normed_tensor, query_buf, key_buf, value_buf, seq_len, pos_offset, cfg);
    update_kv_cache(cache_layer, key_buf, value_buf, seq_len);

    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const size_t num_rep = num_local_heads / num_local_kv_heads;
    const size_t total_len = (cache_layer != nullptr) ? cache_layer->current_len : seq_len;

    #pragma omp parallel for
    for (int h_idx = 0; h_idx < static_cast<int>(num_local_heads); ++h_idx) {
        compute_head_attention({
            h_idx, seq_len, pos_offset, total_len,
            local_hidden_dim, local_kv_dim, head_dim, num_rep,
            query_buf, key_buf, value_buf, cache_layer, attention_scale, attention_out.data()
        });
    }

    core::blas::matmul(attention_out, block->o_proj->data, attention_proj);
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(attention_proj.data(), attention_proj.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += attention_proj[i];
    }
}

static void llama_mlp_layer_cpu(
    const model::LLaMABlock *block,
    float *hidden_states,
    float *normed_states,
    size_t seq_len,
    const model::LLamaConfig &cfg,
    const distributed::DistributedTopology &topology) {

    const size_t hidden_dim = cfg.hidden_size;
    const size_t tp_size = static_cast<size_t>(topology.tp_size);
    const size_t intermediate_dim = cfg.intermediate_size / tp_size;
    const float rms_norm_eps = cfg.rms_norm_eps;

    rms_norm({hidden_states, block->post_attention_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim, rms_norm_eps});

    Tensor normed_tensor({seq_len, hidden_dim});
    std::memcpy(normed_tensor.data(), normed_states, seq_len * hidden_dim * sizeof(float));

    Tensor gate_buf({seq_len, intermediate_dim});
    Tensor up_buf({seq_len, intermediate_dim});
    Tensor down_proj_out({seq_len, hidden_dim});

    core::blas::matmul(normed_tensor, block->gate_proj->data, gate_buf);
    core::blas::matmul(normed_tensor, block->up_proj->data, up_buf);

    silu_inplace(gate_buf.data(), gate_buf.size());
    for (size_t i = 0; i < gate_buf.size(); ++i) {
        gate_buf[i] *= up_buf[i];
    }

    core::blas::matmul(gate_buf, block->down_proj->data, down_proj_out);
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(down_proj_out.data(), down_proj_out.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += down_proj_out[i];
    }
}

// --- Forward Declarations of CUDA Path ---
#ifdef SERIUS_USE_CUDA
Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache);
Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache);
#endif

// --- Public Interface ---

Tensor llama_forward_inference(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
#ifdef SERIUS_USE_CUDA
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        return llama_forward_cuda(model, token_ids, cache);
    }
#endif

    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size();
    const size_t tp_size = static_cast<size_t>(model.topology().tp_size);
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t num_local_kv_heads = cfg.num_key_value_heads / tp_size;
    const size_t local_kv_dim = num_local_kv_heads * head_dim;
    const float rms_norm_eps = cfg.rms_norm_eps;

    // 1. Initialize KV-Cache if needed
    initialize_kv_cache(cache, model, local_kv_dim);

    Tensor hidden_states({seq_len, hidden_dim});

    // 2. Define the Workhorse Function for this stage
    auto run_transformer_layers = [&](const float *input_act, float *output_act, size_t count) {
        (void)count;
        if (input_act != hidden_states.data()) {
            std::memcpy(hidden_states.data(), input_act, seq_len * hidden_dim * sizeof(float));
        }

        Tensor normed_states_tensor({seq_len, hidden_dim});
        const auto &layer_blocks = model.layers();
        for (size_t layer_idx = 0; layer_idx < layer_blocks.size(); ++layer_idx) {
            KVCacheLayer *cache_layer = (cache != nullptr) ? &cache->layers[layer_idx] : nullptr;
            llama_attention_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(), seq_len, 0, cfg, model.topology(), cache_layer);
            llama_mlp_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(), seq_len, cfg, model.topology());
        }
        std::memcpy(output_act, hidden_states.data(), seq_len * hidden_dim * sizeof(float));
    };

    // 3. Pipeline Orchestration
    if (model.topology().is_pipeline_parallel()) {
        serius::distributed::PipelineConfig p_cfg;
        p_cfg.num_stages = model.topology().pp_size;
        p_cfg.num_micro_batches = 1;
        p_cfg.hidden_size = seq_len * hidden_dim;
        
        serius::distributed::PipelineExecutor executor(model.topology().pp_comm, p_cfg);

        if (model.topology().pp_rank == 0) {
            preprocess_embeddings(model, token_ids, hidden_states);
            executor.execute_forward(run_transformer_layers, hidden_states.data(), nullptr);
        } else if (model.topology().pp_rank == p_cfg.num_stages - 1) {
            executor.execute_forward(run_transformer_layers, nullptr, hidden_states.data());
        } else {
            executor.execute_forward(run_transformer_layers, nullptr, nullptr);
        }

        if (model.topology().pp_rank < model.topology().pp_size - 1) {
            return Tensor({0}); // Placeholder for intermediate ranks
        }
    } else {
        // Single-rank path
        preprocess_embeddings(model, token_ids, hidden_states);
        run_transformer_layers(hidden_states.data(), hidden_states.data(), hidden_states.size());
    }

    // 4. Final Head (Last rank or single rank)
    Tensor final_normed({seq_len, hidden_dim});
    rms_norm({hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), seq_len, hidden_dim, rms_norm_eps});
    Tensor logits({seq_len, cfg.vocab_size});
    core::blas::matmul(final_normed, model.lm_head()->data, logits);
    return logits;
}

Tensor llama_forward_single_token(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache) {
#ifdef SERIUS_USE_CUDA
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        return llama_forward_single_token_cuda(model, token_id, pos, cache);
    }
#endif

    const auto &cfg = model.config();
    const size_t tp_size = static_cast<size_t>(model.topology().tp_size);
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t num_local_kv_heads = cfg.num_key_value_heads / tp_size;
    const size_t local_kv_dim = num_local_kv_heads * head_dim;
    const float rms_norm_eps = cfg.rms_norm_eps;

    // 1. Initialize KV-Cache if needed
    initialize_kv_cache(cache, model, local_kv_dim);

    Tensor hidden_states({1, hidden_dim});

    // 2. Define the Workhorse Function for this stage
    auto run_transformer_layers = [&](const float *input_act, float *output_act, size_t count) {
        (void)count;
        if (input_act != hidden_states.data()) {
            std::memcpy(hidden_states.data(), input_act, hidden_dim * sizeof(float));
        }

        Tensor normed_states_tensor({1, hidden_dim});
        const auto &layer_blocks = model.layers();
        for (size_t layer_idx = 0; layer_idx < layer_blocks.size(); ++layer_idx) {
            KVCacheLayer *cache_layer = (cache != nullptr) ? &cache->layers[layer_idx] : nullptr;
            llama_attention_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(), 1, pos, cfg, model.topology(), cache_layer);
            llama_mlp_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(), 1, cfg, model.topology());
        }
        std::memcpy(output_act, hidden_states.data(), hidden_dim * sizeof(float));
    };

    // 3. Pipeline Orchestration
    if (model.topology().is_pipeline_parallel()) {
        serius::distributed::PipelineConfig p_cfg;
        p_cfg.num_stages = model.topology().pp_size;
        p_cfg.num_micro_batches = 1;
        p_cfg.hidden_size = hidden_dim;
        
        serius::distributed::PipelineExecutor executor(model.topology().pp_comm, p_cfg);

        if (model.topology().pp_rank == 0) {
            Tensor t_ids({1});
            t_ids[0] = static_cast<float>(token_id);
            preprocess_embeddings(model, t_ids, hidden_states);
            executor.execute_forward(run_transformer_layers, hidden_states.data(), nullptr);
        } else if (model.topology().pp_rank == p_cfg.num_stages - 1) {
            executor.execute_forward(run_transformer_layers, nullptr, hidden_states.data());
        } else {
            executor.execute_forward(run_transformer_layers, nullptr, nullptr);
        }

        if (model.topology().pp_rank < model.topology().pp_size - 1) {
            return Tensor({0});
        }
    } else {
        if (model.topology().pp_rank == 0) {
            Tensor t_ids({1});
            t_ids[0] = static_cast<float>(token_id);
            preprocess_embeddings(model, t_ids, hidden_states);
        }
        run_transformer_layers(hidden_states.data(), hidden_states.data(), hidden_states.size());
    }

    // 4. Final Head
    Tensor final_normed({1, hidden_dim});
    rms_norm({hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, rms_norm_eps});
    Tensor logits({1, cfg.vocab_size});
    core::blas::matmul(final_normed, model.lm_head()->data, logits);
    return logits;
}

// --- CUDA Implementations ---
#ifdef SERIUS_USE_CUDA
Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size(), hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();

    Tensor hidden_states_gpu({seq_len, hidden}, dtype); hidden_states_gpu.set_device(hal::DeviceType::CUDA);
    std::vector<int> ids(seq_len); for(size_t i=0; i<seq_len; ++i) ids[i] = (int)token_ids[i];
    int *ids_gpu = nullptr; 
    CHECK_CUDA(cudaMalloc(&ids_gpu, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(ids_gpu, ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    hal::gather(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), ids_gpu, seq_len, hidden, dtype);
    hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU); 
    CHECK_CUDA(cudaFree(ids_gpu));

    Tensor normed_buf({seq_len, hidden}, dtype); normed_buf.set_device(hal::DeviceType::CUDA);
    Tensor query_buf({seq_len, hidden}, dtype); query_buf.set_device(hal::DeviceType::CUDA);
    Tensor key_buf({seq_len, kv_dim}, dtype); key_buf.set_device(hal::DeviceType::CUDA);
    Tensor value_buf({seq_len, kv_dim}, dtype); value_buf.set_device(hal::DeviceType::CUDA);
    Tensor attention_out({seq_len, hidden}, dtype); attention_out.set_device(hal::DeviceType::CUDA);
    Tensor mlp_gate({seq_len, inter}, dtype); mlp_gate.set_device(hal::DeviceType::CUDA);
    Tensor mlp_up({seq_len, inter}, dtype); mlp_up.set_device(hal::DeviceType::CUDA);
    Tensor mlp_down({seq_len, hidden}, dtype); mlp_down.set_device(hal::DeviceType::CUDA);

    const auto &layers = model.layers();
    if (cache && cache->empty()) {
        cache->layers.resize(layers.size());
        for (auto &cl : cache->layers) {
            cl.k = Tensor({4096, kv_dim}, dtype);
            cl.v = Tensor({4096, kv_dim}, dtype);
            cl.k.set_device(hal::DeviceType::CUDA);
            cl.v.set_device(hal::DeviceType::CUDA);
            cl.current_len = 0;
        }
    }

    // Prepare causal mask on GPU
    float *mask_gpu = nullptr;
    if (seq_len > 1) {
        std::vector<float> mask(seq_len * seq_len, 0.0F);
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = i + 1; j < seq_len; ++j) {
                mask[i * seq_len + j] = ATTENTION_MASK_VAL;
            }
        }
        CHECK_CUDA(cudaMalloc(&mask_gpu, seq_len * seq_len * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mask_gpu, mask.data(), seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->input_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->q_proj->data, query_buf);
        core::blas::matmul(normed_buf, blk->k_proj->data, key_buf);
        core::blas::matmul(normed_buf, blk->v_proj->data, value_buf);
        hal::rope(query_buf.raw_data_gpu(), head_dim, seq_len, n_heads, rope_base, 0, dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, seq_len, n_kv_heads, rope_base, 0, dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU); key_buf.mark_dirty(Tensor::DataLocation::GPU);

        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            size_t bytes = seq_len * kv_dim * get_dtype_size(dtype);
            CHECK_CUDA(cudaMemcpy((char*)cl.k.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype), 
                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy((char*)cl.v.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype), 
                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice));
            cl.current_len += seq_len;
        }

        Tensor scores({n_heads, seq_len, seq_len}, dtype); scores.set_device(hal::DeviceType::CUDA);
        Tensor q_heads = query_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1).contiguous();
        Tensor k_heads_raw = key_buf.reshape({seq_len, n_kv_heads, head_dim}).transpose(0, 1).contiguous();
        Tensor v_heads_raw = value_buf.reshape({seq_len, n_kv_heads, head_dim}).transpose(0, 1).contiguous();
        
        Tensor k_heads = k_heads_raw, v_heads = v_heads_raw;
        if (n_kv_heads < n_heads) {
            k_heads = Tensor({n_heads, seq_len, head_dim}, dtype); k_heads.set_device(hal::DeviceType::CUDA);
            v_heads = Tensor({n_heads, seq_len, head_dim}, dtype); v_heads.set_device(hal::DeviceType::CUDA);
            hal::repeat_kv(k_heads.raw_data_gpu(), k_heads_raw.raw_data_gpu(), seq_len, n_heads, n_kv_heads, head_dim, dtype);
            hal::repeat_kv(v_heads.raw_data_gpu(), v_heads_raw.raw_data_gpu(), seq_len, n_heads, n_kv_heads, head_dim, dtype);
        }
        
        float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        Tensor q_heads_c = q_heads.contiguous();
        Tensor k_transposed = k_heads.transpose(1, 2).contiguous();

        core::blas::batched_matmul(q_heads_c, k_transposed, scores);
        hal::fused_masked_softmax(scores.raw_data_gpu(), mask_gpu, n_heads, seq_len, seq_len, dtype, scale);

        Tensor context_heads({n_heads, seq_len, head_dim}, dtype); context_heads.set_device(hal::DeviceType::CUDA);
        core::blas::batched_matmul(scores, v_heads, context_heads);
        
        attention_out = context_heads.transpose(0, 1).contiguous().reshape({seq_len, hidden});
        attention_out.set_device(hal::DeviceType::CUDA);
        core::blas::matmul(attention_out, blk->o_proj->data, mlp_down);
        hal::add_inplace(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->gate_proj->data, mlp_gate);
        core::blas::matmul(normed_buf, blk->up_proj->data, mlp_up);
        hal::swiglu(mlp_gate.raw_data_gpu(), mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), seq_len * inter, dtype);
        mlp_gate.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);
        hal::add_inplace(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
    }
    if (mask_gpu) CHECK_CUDA(cudaFree(mask_gpu));
    hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), model.norm_weight()->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    Tensor logits({seq_len, cfg.vocab_size}, dtype); logits.set_device(hal::DeviceType::CUDA);
    core::blas::matmul(normed_buf, model.lm_head()->data, logits);
    logits.syncToHost();
    return logits;
}

Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache) {
    const auto &cfg = model.config();
    const size_t n_layers = cfg.num_hidden_layers, hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    printf("[DEBUG] Starting forward for token %d at pos %zu\n", token_id, pos); fflush(stdout);

    Tensor hidden_states_gpu({1, hidden}, dtype); hidden_states_gpu.set_device(hal::DeviceType::CUDA);
    int tid = token_id; 
    int *tid_gpu = nullptr; 
    CHECK_CUDA(cudaMalloc(&tid_gpu, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(tid_gpu, &tid, sizeof(int), cudaMemcpyHostToDevice));
    hal::gather(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), tid_gpu, 1, hidden, dtype);
    hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU); 
    
    // Crucial: synchronize before freeing tid_gpu as gather is asynchronous
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaFree(tid_gpu));
    printf("[DEBUG] Embeddings gathered\n"); fflush(stdout);

    Tensor normed_buf({1, hidden}, dtype); normed_buf.set_device(hal::DeviceType::CUDA);
    Tensor query_buf({1, hidden}, dtype); query_buf.set_device(hal::DeviceType::CUDA);
    Tensor key_buf({1, kv_dim}, dtype); key_buf.set_device(hal::DeviceType::CUDA);
    Tensor value_buf({1, kv_dim}, dtype); value_buf.set_device(hal::DeviceType::CUDA);
    Tensor attention_out({1, hidden}, dtype); attention_out.set_device(hal::DeviceType::CUDA);
    Tensor mlp_gate({1, inter}, dtype); mlp_gate.set_device(hal::DeviceType::CUDA);
    Tensor mlp_up({1, inter}, dtype); mlp_up.set_device(hal::DeviceType::CUDA);
    Tensor mlp_down({1, hidden}, dtype); mlp_down.set_device(hal::DeviceType::CUDA);

    const auto &layers = model.layers();
    if (cache && (cache->empty() || cache->layers.size() != layers.size())) {
        printf("[DEBUG] Initializing cache for %zu layers\n", layers.size()); fflush(stdout);
        size_t max_seq_len = 4096;
        cache->layers.resize(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            cache->layers[i].k = Tensor({max_seq_len, kv_dim}, dtype);
            cache->layers[i].v = Tensor({max_seq_len, kv_dim}, dtype);
            cache->layers[i].k.set_device(hal::DeviceType::CUDA);
            cache->layers[i].v.set_device(hal::DeviceType::CUDA);
            cache->layers[i].current_len = 0;
        }
    }
    printf("[DEBUG] Starting layer loop\n"); fflush(stdout);

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->q_proj->data, query_buf);
        core::blas::matmul(normed_buf, blk->k_proj->data, key_buf);
        core::blas::matmul(normed_buf, blk->v_proj->data, value_buf);
        hal::rope(query_buf.raw_data_gpu(), head_dim, 1, n_heads, rope_base, static_cast<int>(pos), dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, 1, n_kv_heads, rope_base, static_cast<int>(pos), dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU); key_buf.mark_dirty(Tensor::DataLocation::GPU);

        size_t effective_len = pos + 1;
        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            size_t bytes = kv_dim * get_dtype_size(dtype);
            CHECK_CUDA(cudaMemcpy((char*)cl.k.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype), 
                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy((char*)cl.v.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype), 
                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice));
            cl.k.mark_dirty(Tensor::DataLocation::GPU);
            cl.v.mark_dirty(Tensor::DataLocation::GPU);
            cl.current_len = effective_len;
        }

        Tensor q_hs = query_buf.reshape({1, n_heads, head_dim}).transpose(0, 1).contiguous();
        q_hs.set_device(hal::DeviceType::CUDA);
        
        Tensor k_full({0}, dtype), v_full({0}, dtype);
        if (cache) {
            k_full = cache->layers[layer_idx].k.slice(0, 0, effective_len);
            v_full = cache->layers[layer_idx].v.slice(0, 0, effective_len);
        } else {
            k_full = key_buf;
            v_full = value_buf;
        }
        
        Tensor k_hs({0}, dtype), v_hs({0}, dtype);
        if (n_kv_heads < n_heads) {
            // repeat_kv kernel expects (batch_seq, heads, head_dim)
            Tensor k_in = k_full.reshape({effective_len, n_kv_heads, head_dim}).contiguous();
            Tensor v_in = v_full.reshape({effective_len, n_kv_heads, head_dim}).contiguous();
            
            Tensor k_rep({effective_len, n_heads, head_dim}, dtype);
            Tensor v_rep({effective_len, n_heads, head_dim}, dtype);
            k_rep.set_device(hal::DeviceType::CUDA); v_rep.set_device(hal::DeviceType::CUDA);
            
            hal::repeat_kv(k_rep.raw_data_gpu(), k_in.raw_data_gpu(), effective_len, n_heads, n_kv_heads, head_dim, dtype);
            hal::repeat_kv(v_rep.raw_data_gpu(), v_in.raw_data_gpu(), effective_len, n_heads, n_kv_heads, head_dim, dtype);
            
            // Transpose to (n_heads, batch_seq, head_dim) for batched_matmul
            k_hs = k_rep.transpose(0, 1).contiguous();
            v_hs = v_rep.transpose(0, 1).contiguous();
            k_hs.set_device(hal::DeviceType::CUDA); v_hs.set_device(hal::DeviceType::CUDA);
        } else {
            k_hs = k_full.reshape({effective_len, n_heads, head_dim}).transpose(0, 1).contiguous();
            v_hs = v_full.reshape({effective_len, n_heads, head_dim}).transpose(0, 1).contiguous();
            k_hs.set_device(hal::DeviceType::CUDA); v_hs.set_device(hal::DeviceType::CUDA);
        }

        Tensor scores({n_heads, 1, effective_len}, dtype); scores.set_device(hal::DeviceType::CUDA);
        core::blas::batched_matmul(q_hs, k_hs.transpose(1, 2), scores);
        
        float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        hal::fused_masked_softmax(scores.raw_data_gpu(), nullptr, n_heads, 1, effective_len, dtype, scale);
        scores.mark_dirty(Tensor::DataLocation::GPU);

        Tensor context_heads({n_heads, 1, head_dim}, dtype); context_heads.set_device(hal::DeviceType::CUDA);
        core::blas::batched_matmul(scores, v_hs, context_heads);
        attention_out = context_heads.transpose(0, 1).contiguous().reshape({1, hidden});
        attention_out.set_device(hal::DeviceType::CUDA);

        core::blas::matmul(attention_out, blk->o_proj->data, mlp_down);
        hal::add_inplace(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), hidden, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->gate_proj->data, mlp_gate);
        core::blas::matmul(normed_buf, blk->up_proj->data, mlp_up);
        hal::swiglu(mlp_gate.raw_data_gpu(), mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), inter, dtype);
        mlp_gate.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);
        hal::add_inplace(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), hidden, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
    }
    printf("[DEBUG] Final norm\n"); fflush(stdout);
    hal::rmsnorm(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    Tensor logits({1, cfg.vocab_size}, dtype); logits.set_device(hal::DeviceType::CUDA);
    core::blas::matmul(normed_buf, model.lm_head()->data, logits);
    printf("[DEBUG] Syncing logits to host\n"); fflush(stdout);
    logits.syncToHost();
    printf("[DEBUG] Done\n"); fflush(stdout);
    return logits;
}
#endif

} // namespace infer
} // namespace serius
