#include "llama_inference.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "activation_hook.h"

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
#include "../common/serius_check.h"
#include "../distributed/pipeline.h"
#ifdef SERIUS_USE_CUDA
#include "../blas/matmul_cuda.h"
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

// Batched prefill (seq_len > 1 in one forward) is enabled by default:
// BatchedVsSequentialPrefill / BatchedVsSequentialRealModel verify KV-cache
// and last-position logit parity vs the sequential path (max diff ~3e-3 on a
// real Q4_K_M GGUF, same argmax). Set SERIUS_PREFILL_BATCH=0 to force the
// slower per-token prefill for debugging.
inline bool prefill_batch_enabled()
{
    static const bool enabled = [] {
#if defined(_MSC_VER)
        char buf[8];
        size_t len = 0;
        if (getenv_s(&len, buf, sizeof(buf), "SERIUS_PREFILL_BATCH") != 0 || len == 0) {
            return true;
        }
        return !(len > 0 && buf[0] == '0');
#else
        const char *v = std::getenv("SERIUS_PREFILL_BATCH");
        return v == nullptr || v[0] != '0';
#endif
    }();
    return enabled;
}

// The CPU batched-prefill kernels are pathologically slow on quantized GGUF
// models (dense FP32 weights, naive batched attention): a ~13-token prompt can
// take minutes vs ~1s per-token. The measured win of batched prefill is on the
// CUDA path only, so the CPU path stays per-token unless explicitly opted in
// with SERIUS_PREFILL_BATCH_CPU=1. SERIUS_PREFILL_BATCH=0 still disables both.
inline bool cpu_prefill_batch_enabled()
{
    static const bool enabled = [] {
#if defined(_MSC_VER)
        char buf[8];
        size_t len = 0;
        if (getenv_s(&len, buf, sizeof(buf), "SERIUS_PREFILL_BATCH_CPU") != 0 ||
            len == 0) {
            return false;
        }
        return buf[0] == '1';
#else
        const char *v = std::getenv("SERIUS_PREFILL_BATCH_CPU");
        return v != nullptr && v[0] == '1';
#endif
    }();
    return enabled && prefill_batch_enabled();
}

using core::Tensor;

// --- Temporary decode profiling (SERIUS_PROF=1) ---
namespace prof {
inline bool enabled()
{
    static const bool e = [] {
#if defined(_MSC_VER)
        size_t n = 0;
        return getenv_s(&n, nullptr, 0, "SERIUS_PROF") == 0 && n > 0;
#else
        return std::getenv("SERIUS_PROF") != nullptr;
#endif
    }();
    return e;
}
struct Acc
{
    double attn_pre = 0, attn_hd = 0, attn_post = 0, mlp = 0, mlp_gu = 0, mlp_rest = 0, head = 0;
    double g_qkv = 0, g_o = 0, g_gu = 0, g_down = 0, g_lm = 0;
    long calls = 0;
};
inline Acc &acc()
{
    static Acc a;
    return a;
}
inline double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void report()
{
    if (!enabled()) {
        return;
    }
    Acc &a = acc();
    ++a.calls;
    if (a.calls % 300 == 0) {
        std::fprintf(stderr,
                     "[prof] attn_pre=%.3f attn_hd=%.3f attn_post=%.3f mlp=%.3f (gu=%.3f rest=%.3f) head=%.3f ms/tok\n"
                     "       gemv: qkv=%.3f o=%.3f gateup=%.3f down=%.3f lm=%.3f ms/tok\n",
                     a.attn_pre * 1e3, a.attn_hd * 1e3, a.attn_post * 1e3, a.mlp * 1e3, a.mlp_gu * 1e3,
                     a.mlp_rest * 1e3, a.head * 1e3, a.g_qkv * 1e3, a.g_o * 1e3, a.g_gu * 1e3, a.g_down * 1e3,
                     a.g_lm * 1e3);
        a.attn_pre = a.attn_hd = a.attn_post = a.mlp = a.mlp_gu = a.mlp_rest = a.head = 0.0;
        a.g_qkv = a.g_o = a.g_gu = a.g_down = a.g_lm = 0.0;
    }
}
}  // namespace prof

// Flush-to-zero / denormals-are-zero on the caller thread AND all OpenMP
// workers. Quantized dequant + exp approximations produce subnormal floats;
// without DAZ/FTZ every FMA on them traps to microcode (~100x slower).
static void enable_ftz_daz_once()
{
    static std::once_flag flag;
    std::call_once(flag, [] {
        const unsigned bits = _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON;
        _mm_setcsr(_mm_getcsr() | bits);
#if defined(_OPENMP)
#pragma omp parallel
        {
            _mm_setcsr(_mm_getcsr() | bits);
        }
#endif
    });
}

// --- AVX2 Helpers ---

inline float _mm256_reduce_add_ps(__m256 input_vec)
{
    __m128 high_vec = _mm256_extractf128_ps(input_vec, 1);
    __m128 low_vec = _mm256_castps256_ps128(input_vec);
    __m128 sum_vec = _mm_add_ps(high_vec, low_vec);
    sum_vec = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
    sum_vec = _mm_add_ss(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, 1));
    return _mm_cvtss_f32(sum_vec);
}

inline float vec_dot_avx2(const float *vec_a, const float *vec_b, size_t n)
{
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

struct ScaleAccumulator
{
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

struct RMSNormArgs
{
    const float *inputs;
    const float *weights;
    float *outputs;
    size_t rows;
    size_t dim;
    float eps;
};

static void rms_norm(const RMSNormArgs &args)
{
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

struct RoPEArgs
{
    float *data;
    size_t seq_len;
    size_t model_dim;
    size_t head_dim;
    float rope_base;
    size_t pos_offset;
};

static void apply_rope_inplace(const RoPEArgs &args)
{
    const size_t half_dim = args.head_dim / 2;

    // Decode fast path (seq_len == 1): inv_freq is position-independent, so
    // cache it; cos/sin for the single position go on the stack. All heads
    // and both Q/K share the same tables.
    if (args.seq_len == 1) {
        struct InvFreqCache
        {
            size_t head_dim = 0;
            float rope_base = 0.0F;
            std::vector<float> inv_freq;
        };
        static thread_local InvFreqCache cache;
        if (cache.head_dim != args.head_dim || cache.rope_base != args.rope_base) {
            cache.inv_freq.resize(half_dim);
            for (size_t i = 0; i < half_dim; ++i) {
                cache.inv_freq[i] =
                    1.0F / std::pow(args.rope_base, static_cast<float>(2 * i) / static_cast<float>(args.head_dim));
            }
            cache.head_dim = args.head_dim;
            cache.rope_base = args.rope_base;
        }
        const float abs_pos = static_cast<float>(args.pos_offset);
        float cos_t[128];
        float sin_t[128];
        SERIUS_CHECK(half_dim <= 128, "head_dim too large for rope fast path");
        for (size_t i = 0; i < half_dim; ++i) {
            const float theta = abs_pos * cache.inv_freq[i];
            cos_t[i] = std::cos(theta);
            sin_t[i] = std::sin(theta);
        }
        float *row_ptr = args.data;
        for (size_t head_offset = 0; head_offset < args.model_dim; head_offset += args.head_dim) {
            float *head_ptr = row_ptr + head_offset;
            for (size_t i = 0; i < half_dim; ++i) {
                const float val_x0 = head_ptr[i];
                const float val_x1 = head_ptr[i + half_dim];
                head_ptr[i] = (val_x0 * cos_t[i]) - (val_x1 * sin_t[i]);
                head_ptr[i + half_dim] = (val_x0 * sin_t[i]) + (val_x1 * cos_t[i]);
            }
        }
        return;
    }

    // Precompute cos/sin for all positions × all freq pairs once per call.
    // Use stack buffers for single-token decoding (zero heap allocation overhead).
    constexpr size_t ROPE_STACK_CAP = 256;
    float stack_cos[ROPE_STACK_CAP];
    float stack_sin[ROPE_STACK_CAP];
    std::vector<float> heap_cos;
    std::vector<float> heap_sin;
    float *cos_table = stack_cos;
    float *sin_table = stack_sin;
    const size_t total_elements = args.seq_len * half_dim;
    if (total_elements > ROPE_STACK_CAP) {
        heap_cos.resize(total_elements);
        heap_sin.resize(total_elements);
        cos_table = heap_cos.data();
        sin_table = heap_sin.data();
    }

    for (size_t pos = 0; pos < args.seq_len; ++pos) {
        const float abs_pos = static_cast<float>(pos + args.pos_offset);
        for (size_t i = 0; i < half_dim; ++i) {
            const float freq =
                1.0F / std::pow(args.rope_base, static_cast<float>(2 * i) / static_cast<float>(args.head_dim));
            const float theta = abs_pos * freq;
            cos_table[pos * half_dim + i] = std::cos(theta);
            sin_table[pos * half_dim + i] = std::sin(theta);
        }
    }

    for (size_t pos = 0; pos < args.seq_len; ++pos) {
        float *row_ptr = args.data + (pos * args.model_dim);
        const float *cos_row = cos_table + pos * half_dim;
        const float *sin_row = sin_table + pos * half_dim;
        for (size_t head_offset = 0; head_offset < args.model_dim; head_offset += args.head_dim) {
            float *head_ptr = row_ptr + head_offset;
            for (size_t i = 0; i < half_dim; ++i) {
                const float val_x0 = head_ptr[i];
                const float val_x1 = head_ptr[i + half_dim];
                head_ptr[i] = (val_x0 * cos_row[i]) - (val_x1 * sin_row[i]);
                head_ptr[i + half_dim] = (val_x0 * sin_row[i]) + (val_x1 * cos_row[i]);
            }
        }
    }
}

// Vectorized exp: exp(x) = poly(r) * 2^n with n = round(x*log2e),
// r = x - n*ln2 in [-0.347, 0.347]. Degree-7 Taylor -> rel err ~1e-7.
inline __m256 exp256_ps(__m256 x)
{
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0F));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0F));
    const __m256 fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341F));
    const __m256 n = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453F), x);
    __m256 p = _mm256_set1_ps(1.0F / 5040.0F);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 720.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 120.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 24.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 6.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(p, pow2n);
}

struct SoftmaxRowsArgs
{
    float *data;
    size_t rows;
    size_t cols;
};

static void softmax_rows_inplace(const SoftmaxRowsArgs &args)
{
    for (size_t row_idx = 0; row_idx < args.rows; ++row_idx) {
        float *row = args.data + (row_idx * args.cols);
        float max_val = row[0];
        size_t col_idx = 1;
#if defined(__AVX2__) || defined(_MSC_VER)
        {
            __m256 vmax = _mm256_set1_ps(row[0]);
            for (; col_idx + AVX_FLOAT_STRIDE <= args.cols; col_idx += AVX_FLOAT_STRIDE) {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + col_idx));
            }
            alignas(32) float mbuf[AVX_FLOAT_STRIDE];
            _mm256_store_ps(mbuf, vmax);
            for (int i = 0; i < AVX_FLOAT_STRIDE; ++i) {
                max_val = std::max(max_val, mbuf[i]);
            }
        }
#endif
        for (; col_idx < args.cols; ++col_idx) {
            max_val = std::max(max_val, row[col_idx]);
        }
        const __m256 v_max = _mm256_set1_ps(max_val);
        float sum_val = 0.0F;
        col_idx = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
        {
            __m256 vsum = _mm256_setzero_ps();
            for (; col_idx + AVX_FLOAT_STRIDE <= args.cols; col_idx += AVX_FLOAT_STRIDE) {
                const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(row + col_idx), v_max));
                _mm256_storeu_ps(row + col_idx, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            sum_val = _mm256_reduce_add_ps(vsum);
        }
#endif
        for (; col_idx < args.cols; ++col_idx) {
            row[col_idx] = std::exp(row[col_idx] - max_val);
            sum_val += row[col_idx];
        }
        const float inv_sum = 1.0F / sum_val;
        const __m256 v_invs = _mm256_set1_ps(inv_sum);
        col_idx = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
        for (; col_idx + AVX_FLOAT_STRIDE <= args.cols; col_idx += AVX_FLOAT_STRIDE) {
            _mm256_storeu_ps(row + col_idx, _mm256_mul_ps(_mm256_loadu_ps(row + col_idx), v_invs));
        }
#endif
        for (; col_idx < args.cols; ++col_idx) {
            row[col_idx] *= inv_sum;
        }
    }
}

// --- Internal Helper Functions for CPU Inference ---

/**
 * @brief Computes Q, K, and V projections and applies RoPE.
 */
static void compute_qkv_rope(const model::LLaMABlock *block, const Tensor &normed_tensor, Tensor &query_buf,
                             Tensor &key_buf, Tensor &value_buf, size_t seq_len, size_t pos_offset,
                             const model::LLamaConfig &cfg)
{
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t local_hidden_dim = query_buf.shape()[1];
    const size_t local_kv_dim = key_buf.shape()[1];

    // Fast path: packed-quantized weights with on-the-fly dequant GEMV (M==1 only)
    const size_t qkv_total = local_hidden_dim + 2 * local_kv_dim;
    if (seq_len == 1 && block->qkv_w.valid() && block->qkv_w.in == hidden_dim && block->qkv_w.out == qkv_total) {
        thread_local std::vector<float> qkv_tmp;
        qkv_tmp.resize(qkv_total);
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->qkv_w, normed_tensor.data(), qkv_tmp.data());
        if (p) {
            prof::acc().g_qkv += prof::now() - t0;
        }
        std::memcpy(query_buf.data(), qkv_tmp.data(), local_hidden_dim * sizeof(float));
        std::memcpy(key_buf.data(), qkv_tmp.data() + local_hidden_dim, local_kv_dim * sizeof(float));
        std::memcpy(value_buf.data(), qkv_tmp.data() + local_hidden_dim + local_kv_dim, local_kv_dim * sizeof(float));
    } else {
        core::blas::matmul(normed_tensor, block->q_proj->data, query_buf);
        core::blas::matmul(normed_tensor, block->k_proj->data, key_buf);
        core::blas::matmul(normed_tensor, block->v_proj->data, value_buf);
    }

    // Q/K/V projection biases (Qwen2/Qwen2.5 have large ones; zero elsewhere).
    auto add_qkv_bias = [](float *buf, size_t rows, size_t cols, const core::Tensor &bias) {
        if (bias.size() != cols) {
            return;
        }
        const float *b = bias.data();
        for (size_t r = 0; r < rows; ++r) {
            float *row = buf + (r * cols);
            for (size_t c = 0; c < cols; ++c) {
                row[c] += b[c];
            }
        }
    };
    if (block->q_bias) {
        add_qkv_bias(query_buf.data(), seq_len, local_hidden_dim, block->q_bias->data);
    }
    if (block->k_bias) {
        add_qkv_bias(key_buf.data(), seq_len, local_kv_dim, block->k_bias->data);
    }
    if (block->v_bias) {
        add_qkv_bias(value_buf.data(), seq_len, local_kv_dim, block->v_bias->data);
    }

    apply_rope_inplace({query_buf.data(), seq_len, local_hidden_dim, head_dim, cfg.rope_theta, pos_offset});
    apply_rope_inplace({key_buf.data(), seq_len, local_kv_dim, head_dim, cfg.rope_theta, pos_offset});
}

/**
 * @brief Updates the KV cache with new key and value tensors.
 */
// FP16 helpers for the KV cache (round-to-nearest via F16C where available).
static inline __m128i f32x8_to_f16x8(__m256 v)
{
#if defined(__F16C__) || defined(_MSC_VER)
    return _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
#else
    alignas(32) float f[8];
    _mm256_store_ps(f, v);
    __m128i h = _mm_setzero_si128();
    for (int i = 0; i < 8; ++i) {
        const float x = f[i];
        uint32_t b;
        std::memcpy(&b, &x, 4);
        const uint32_t sign = (b >> 16) & 0x8000u;
        int e = static_cast<int>((b >> 23) & 0xFF) - 127 + 15;
        uint32_t m = b & 0x7FFFFFu;
        uint16_t r;
        if (e <= 0) {
            r = static_cast<uint16_t>(sign);
        } else if (e >= 31) {
            r = static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            r = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m >> 13));
        }
        h = _mm_insert_epi16(h, static_cast<int>(r), i);
    }
    return h;
#endif
}

static inline __m256 f16x8_to_f32x8(__m128i h)
{
#if defined(__F16C__) || defined(_MSC_VER)
    return _mm256_cvtph_ps(h);
#else
    alignas(16) uint16_t u[8];
    _mm_store_si128(reinterpret_cast<__m128i *>(u), h);
    alignas(32) float f[8];
    for (int i = 0; i < 8; ++i) {
        const uint32_t b = static_cast<uint32_t>(u[i]);
        const uint32_t sign = (b & 0x8000u) << 16;
        const int e = static_cast<int>((b >> 10) & 0x1F);
        const uint32_t m = b & 0x3FFu;
        uint32_t r;
        if (e == 0) {
            if (m == 0) {
                r = sign;
            } else {
                int exp = -1;
                uint32_t mm = m;
                while ((mm & 0x400u) == 0) {
                    mm <<= 1;
                    --exp;
                }
                mm &= 0x3FFu;
                r = sign | static_cast<uint32_t>(127 - 15 + 1 + exp << 23) | (mm << 13);
            }
        } else if (e == 31) {
            r = sign | 0x7F800000u | (m << 13);
        } else {
            r = sign | static_cast<uint32_t>(e - 15 + 127) << 23 | (m << 13);
        }
        std::memcpy(&f[i], &r, 4);
    }
    return _mm256_load_ps(f);
#endif
}

static inline uint16_t f32_to_f16_scalar(float x)
{
    uint32_t b;
    std::memcpy(&b, &x, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int e = static_cast<int>((b >> 23) & 0xFF) - 127 + 15;
    const uint32_t m = b & 0x7FFFFFu;
    if (e <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (e >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m >> 13));
}

static inline float f16_to_f32_scalar(uint16_t h)
{
    const uint32_t b = static_cast<uint32_t>(h);
    const uint32_t sign = (b & 0x8000u) << 16;
    const int e = static_cast<int>((b >> 10) & 0x1F);
    const uint32_t m = b & 0x3FFu;
    uint32_t r;
    if (e == 0) {
        if (m == 0) {
            r = sign;
        } else {
            int exp = -1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0) {
                mm <<= 1;
                --exp;
            }
            mm &= 0x3FFu;
            r = sign | (static_cast<uint32_t>(127 - 14 + exp) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        r = sign | 0x7F800000u | (m << 13);
    } else {
        r = sign | (static_cast<uint32_t>(e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &r, 4);
    return f;
}

// Dot product q(fp32) . kv(fp16 or fp32) at element offset `off`.
static float kv_dot_f32(const float *q, const uint16_t *p16, const float *p32, size_t off, size_t dim)
{
    __m256 acc = _mm256_setzero_ps();
    size_t d = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
    for (; d + 8 <= dim; d += 8) {
        const __m256 kv = p16 ? f16x8_to_f32x8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p16 + off + d)))
                              : _mm256_loadu_ps(p32 + off + d);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), kv, acc);
    }
#endif
    float s = _mm256_reduce_add_ps(acc);
    for (; d < dim; ++d) {
        const float kv = p16 ? f16_to_f32_scalar(p16[off + d]) : p32[off + d];
        s += q[d] * kv;
    }
    return s;
}

/// Grow a KV cache layer's buffers so they hold at least `needed` positions,
/// preserving existing contents. Called on demand from update_kv_cache so
/// prompts longer than the initial DEFAULT_MAX_SEQ_LEN capacity don't write
/// out of bounds (previously a heap overflow that crashed the server).
static void ensure_kv_capacity(KVCacheLayer *cache_layer, size_t needed, size_t local_kv_dim)
{
    const size_t cap16 = cache_layer->k16.empty() ? 0 : cache_layer->k16.size() / local_kv_dim;
    const size_t cap32 = cache_layer->k.shape().size() >= 2 ? cache_layer->k.shape()[0] : 0;
    const size_t cap = std::max(cap16, cap32);
    if (needed <= cap) {
        return;
    }
    size_t new_cap = cap == 0 ? needed : cap;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    std::vector<uint16_t> nk(new_cap * local_kv_dim);
    std::vector<uint16_t> nv(new_cap * local_kv_dim);
    if (!cache_layer->k16.empty()) {
        std::memcpy(nk.data(), cache_layer->k16.data(), cache_layer->k16.size() * sizeof(uint16_t));
        std::memcpy(nv.data(), cache_layer->v16.data(), cache_layer->v16.size() * sizeof(uint16_t));
    }
    cache_layer->k16.swap(nk);
    cache_layer->v16.swap(nv);

    Tensor new_k({new_cap, local_kv_dim});
    Tensor new_v({new_cap, local_kv_dim});
    if (cache_layer->k.size() > 0) {
        const size_t old_k = std::min(cache_layer->k.size(), new_cap * local_kv_dim);
        std::memcpy(new_k.data(), cache_layer->k.data(), old_k * sizeof(float));
    }
    if (cache_layer->v.size() > 0) {
        const size_t old_v = std::min(cache_layer->v.size(), new_cap * local_kv_dim);
        std::memcpy(new_v.data(), cache_layer->v.data(), old_v * sizeof(float));
    }
    cache_layer->k = std::move(new_k);
    cache_layer->v = std::move(new_v);
}

static void update_kv_cache(KVCacheLayer *cache_layer, const Tensor &key_buf, const Tensor &value_buf, size_t seq_len)
{
    if (cache_layer == nullptr) {
        return;
    }
    const size_t local_kv_dim = key_buf.shape()[1];
    const size_t start_pos = cache_layer->current_len;
    ensure_kv_capacity(cache_layer, start_pos + seq_len, local_kv_dim);
    // FP16 store (halves attention read bandwidth); fp32 copy kept for the
    // CUDA path and any non-fp16 consumer.
    for (size_t i = 0; i < seq_len; ++i) {
        const float *kp = key_buf.data() + (i * local_kv_dim);
        const float *vp = value_buf.data() + (i * local_kv_dim);
        uint16_t *k16 = cache_layer->k16.data() + ((start_pos + i) * local_kv_dim);
        uint16_t *v16 = cache_layer->v16.data() + ((start_pos + i) * local_kv_dim);
        std::memcpy(cache_layer->k.data() + ((start_pos + i) * local_kv_dim), kp, local_kv_dim * sizeof(float));
        std::memcpy(cache_layer->v.data() + ((start_pos + i) * local_kv_dim), vp, local_kv_dim * sizeof(float));
        size_t d = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
        for (; d + 8 <= local_kv_dim; d += 8) {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(k16 + d), f32x8_to_f16x8(_mm256_loadu_ps(kp + d)));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(v16 + d), f32x8_to_f16x8(_mm256_loadu_ps(vp + d)));
        }
#endif
        for (; d < local_kv_dim; ++d) {
            k16[d] = f32_to_f16_scalar(kp[d]);
            v16[d] = f32_to_f16_scalar(vp[d]);
        }
    }
    cache_layer->current_len += seq_len;
}

struct HeadAttentionArgs
{
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
    int layer_idx = -1;  // -1 disables the attention_row observability hook
};

/**
 * @brief Computes attention scores and output for a single head.
 */
static void compute_head_attention(const HeadAttentionArgs &args)
{
    const ActivationHooks &ahooks = activation_hooks();
    const bool want_attn_hook = (ahooks.attention_row != nullptr && args.layer_idx >= 0);
    const size_t kv_head_idx = static_cast<size_t>(args.head_idx) / args.num_rep;
    constexpr size_t STACK_SCORES_CAP = 1024;
    float stack_scores[STACK_SCORES_CAP];
    std::vector<float> heap_scores;

    for (size_t qi = 0; qi < args.seq_len; ++qi) {
        float *scores = stack_scores;
        if (args.total_len > STACK_SCORES_CAP) {
            heap_scores.resize(args.total_len);
            scores = heap_scores.data();
        }

        const float *q_ptr =
            args.query_buf.data() + (qi * args.local_hidden_dim) + (static_cast<size_t>(args.head_idx) * args.head_dim);
        // With a KV cache, positions are absolute (pos_offset + qi). Without
        // one, key_buf/value_buf only hold the seq_len rows being processed —
        // attending to pos_offset + qi + 1 positions would read out of bounds.
        const size_t npos = (args.cache_layer != nullptr) ? (args.pos_offset + qi + 1) : (qi + 1);
        const size_t koff = kv_head_idx * args.head_dim;
        // FP16 KV cache when available (halves attention DRAM traffic).
        // SERIUS_KV_FP32=1 forces the fp32 path for max precision.
        static const bool no_kv_fp16 = [] {
#if defined(_MSC_VER)
            char buf[8];
            size_t len = 0;
            return getenv_s(&len, buf, sizeof(buf), "SERIUS_KV_FP32") == 0 && len > 0;
#else
            return std::getenv("SERIUS_KV_FP32") != nullptr;
#endif
        }();
        const bool kv16 = !no_kv_fp16 && (args.cache_layer != nullptr) && !args.cache_layer->k16.empty();
        const uint16_t *k16base = kv16 ? args.cache_layer->k16.data() : nullptr;
        const uint16_t *v16base = kv16 ? args.cache_layer->v16.data() : nullptr;
        const float *kbase = (!kv16 && args.cache_layer != nullptr) ? args.cache_layer->k.data() : args.key_buf.data();
        const float *vbase32 =
            (!kv16 && args.cache_layer != nullptr) ? args.cache_layer->v.data() : args.value_buf.data();
        auto load_kv = [](const uint16_t *p16, const float *p32, size_t off) -> __m256 {
            return p16 ? f16x8_to_f32x8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p16 + off)))
                       : _mm256_loadu_ps(p32 + off);
        };

        const bool simd_ok = (args.head_dim % AVX_FLOAT_STRIDE) == 0 && args.head_dim <= 128;
        if (simd_ok) {
            // Scores: 4 positions per iteration, q broadcast across k streams.
            size_t kj = 0;
            for (; kj + 4 <= npos; kj += 4) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (size_t d = 0; d < args.head_dim; d += AVX_FLOAT_STRIDE) {
                    const __m256 qv = _mm256_loadu_ps(q_ptr + d);
                    const size_t base = (kj * args.local_kv_dim) + koff + d;
                    a0 = _mm256_fmadd_ps(qv, load_kv(k16base, kbase, base + 0 * args.local_kv_dim), a0);
                    a1 = _mm256_fmadd_ps(qv, load_kv(k16base, kbase, base + 1 * args.local_kv_dim), a1);
                    a2 = _mm256_fmadd_ps(qv, load_kv(k16base, kbase, base + 2 * args.local_kv_dim), a2);
                    a3 = _mm256_fmadd_ps(qv, load_kv(k16base, kbase, base + 3 * args.local_kv_dim), a3);
                }
                scores[kj + 0] = _mm256_reduce_add_ps(a0) * args.attention_scale;
                scores[kj + 1] = _mm256_reduce_add_ps(a1) * args.attention_scale;
                scores[kj + 2] = _mm256_reduce_add_ps(a2) * args.attention_scale;
                scores[kj + 3] = _mm256_reduce_add_ps(a3) * args.attention_scale;
            }
            for (; kj < npos; ++kj) {
                scores[kj] = kv_dot_f32(q_ptr, k16base, kbase, (kj * args.local_kv_dim) + koff, args.head_dim) *
                             args.attention_scale;
            }
        } else {
            for (size_t kj = 0; kj < npos; ++kj) {
                scores[kj] = kv_dot_f32(q_ptr, k16base, kbase, (kj * args.local_kv_dim) + koff, args.head_dim) *
                             args.attention_scale;
            }
        }
        for (size_t kj = npos; kj < args.total_len; ++kj) {
            scores[kj] = ATTENTION_MASK_VAL;
        }

        softmax_rows_inplace({scores, 1, args.total_len});

        // Observability hook (interpret): attention distribution of this
        // query row. `scores` is reused across qi — consumers must copy.
        if (want_attn_hook) {
            ahooks.attention_row(args.layer_idx, args.head_idx, qi, scores, npos);
        }

        float *out_ptr = args.attention_out_ptr + (qi * args.local_hidden_dim) +
                         (static_cast<size_t>(args.head_idx) * args.head_dim);
        if (simd_ok) {
            // Weighted-V sum with the output kept in registers across all positions.
            constexpr size_t MAX_VVEC = 16;  // head_dim <= 128
            __m256 vacc[MAX_VVEC];
            const size_t nvec = args.head_dim / AVX_FLOAT_STRIDE;
            for (size_t d = 0; d < nvec; ++d) {
                vacc[d] = _mm256_setzero_ps();
            }
            const size_t voff = kv_head_idx * args.head_dim;
            for (size_t kj = 0; kj < npos; ++kj) {
                const __m256 s = _mm256_set1_ps(scores[kj]);
                const size_t vo = kj * args.local_kv_dim + voff;
                for (size_t d = 0; d < nvec; ++d) {
                    vacc[d] = _mm256_fmadd_ps(load_kv(v16base, vbase32, vo + d * AVX_FLOAT_STRIDE), s, vacc[d]);
                }
            }
            for (size_t d = 0; d < nvec; ++d) {
                _mm256_storeu_ps(out_ptr + d * AVX_FLOAT_STRIDE, vacc[d]);
            }
        } else {
            std::memset(out_ptr, 0, args.head_dim * sizeof(float));
            std::vector<float> vrow;
            if (kv16) {
                vrow.resize(args.head_dim);
            }
            for (size_t kj = 0; kj < npos; ++kj) {
                const size_t vo = kj * args.local_kv_dim + (kv_head_idx * args.head_dim);
                const float *v_ptr;
                if (kv16) {
                    const uint16_t *s = v16base + vo;
                    for (size_t d = 0; d < args.head_dim; ++d) {
                        vrow[d] = f16_to_f32_scalar(s[d]);
                    }
                    v_ptr = vrow.data();
                } else {
                    v_ptr = (args.cache_layer != nullptr) ? (args.cache_layer->v.data() + vo)
                                                          : (args.value_buf.data() + vo);
                }
                vec_add_scale_avx2({out_ptr, v_ptr, scores[kj], args.head_dim});
            }
        }
    }
}

static void initialize_kv_cache(KVCache *cache, const model::LLaMAModel &model, size_t local_kv_dim)
{
    if (cache != nullptr && (cache->empty() || cache->layers.size() != model.layers().size())) {
        const size_t num_layers = model.layers().size();
        cache->layers.resize(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            cache->layers[i].k = Tensor({DEFAULT_MAX_SEQ_LEN, local_kv_dim});
            cache->layers[i].v = Tensor({DEFAULT_MAX_SEQ_LEN, local_kv_dim});
            cache->layers[i].k16.assign(DEFAULT_MAX_SEQ_LEN * local_kv_dim, 0);
            cache->layers[i].v16.assign(DEFAULT_MAX_SEQ_LEN * local_kv_dim, 0);
            cache->layers[i].current_len = 0;
        }
    }
}

static void preprocess_embeddings(const model::LLaMAModel &model, const Tensor &token_ids, Tensor &hidden_states)
{
    if (model.embed_tokens() != nullptr) {
        const size_t seq_len = token_ids.size();
        const size_t hidden_dim = model.config().hidden_size;
        const float *embed_ptr = model.embed_tokens()->data.data();
        for (size_t i = 0; i < seq_len; ++i) {
            size_t tid = static_cast<size_t>(token_ids[i]);
            if (tid >= model.config().vocab_size) {
                tid = 0;
            }
            std::copy(embed_ptr + (tid * hidden_dim), embed_ptr + ((tid + 1) * hidden_dim),
                      hidden_states.data() + (i * hidden_dim));
        }
    }
}

static void llama_attention_layer_cpu(const model::LLaMABlock *block, float *hidden_states, float *normed_states,
                                      size_t seq_len, size_t pos_offset, const model::LLamaConfig &cfg,
                                      const distributed::DistributedTopology &topology, KVCacheLayer *cache_layer,
                                      int layer_idx = -1)
{
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t tp_size = static_cast<size_t>(topology.tp_size);
    const size_t num_local_heads = cfg.num_attention_heads / tp_size;
    const size_t num_local_kv_heads = cfg.num_key_value_heads / tp_size;
    const size_t local_hidden_dim = num_local_heads * head_dim;
    const size_t local_kv_dim = num_local_kv_heads * head_dim;

    const bool prof = prof::enabled();
    double t_pre0 = prof ? prof::now() : 0.0;
    rms_norm({hidden_states, block->input_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim,
              cfg.rms_norm_eps});

    // Decode/prefill scratch buffers: reused across calls (avoids ~8 malloc+memset per layer).
    thread_local std::vector<float> qb, kb, vb, ab, pb;
    qb.resize(seq_len * local_hidden_dim);
    kb.resize(seq_len * local_kv_dim);
    vb.resize(seq_len * local_kv_dim);
    ab.resize(seq_len * local_hidden_dim);
    pb.resize(seq_len * hidden_dim);
    Tensor query_buf(qb.data(), {seq_len, local_hidden_dim});
    Tensor key_buf(kb.data(), {seq_len, local_kv_dim});
    Tensor value_buf(vb.data(), {seq_len, local_kv_dim});
    Tensor attention_out(ab.data(), {seq_len, local_hidden_dim});
    Tensor attention_proj(pb.data(), {seq_len, hidden_dim});

    // Zero-copy non-owning view on already computed normed_states
    Tensor normed_tensor(normed_states, {seq_len, hidden_dim});

    compute_qkv_rope(block, normed_tensor, query_buf, key_buf, value_buf, seq_len, pos_offset, cfg);
    update_kv_cache(cache_layer, key_buf, value_buf, seq_len);
    if (prof) {
        prof::acc().attn_pre += prof::now() - t_pre0;
    }

    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const size_t num_rep = num_local_heads / num_local_kv_heads;
    const size_t total_len = (cache_layer != nullptr) ? cache_layer->current_len : seq_len;

    double t_hd0 = prof ? prof::now() : 0.0;
#pragma omp parallel for schedule(static) if (num_local_heads >= 2)
    for (int h_idx = 0; h_idx < static_cast<int>(num_local_heads); ++h_idx) {
        compute_head_attention({h_idx, seq_len, pos_offset, total_len, local_hidden_dim, local_kv_dim, head_dim,
                                num_rep, query_buf, key_buf, value_buf, cache_layer, attention_scale,
                                attention_out.data(), layer_idx});
    }
    if (prof) {
        prof::acc().attn_hd += prof::now() - t_hd0;
    }

    double t_post0 = prof ? prof::now() : 0.0;
    if (seq_len == 1 && block->o_w.valid() && block->o_w.in == local_hidden_dim && block->o_w.out == hidden_dim) {
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->o_w, attention_out.data(), attention_proj.data());
        if (p) {
            prof::acc().g_o += prof::now() - t0;
        }
    } else {
        core::blas::matmul(attention_out, block->o_proj->data, attention_proj);
    }
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(attention_proj.data(), attention_proj.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += attention_proj[i];
    }
    if (prof) {
        prof::acc().attn_post += prof::now() - t_post0;
    }
}

static void llama_mlp_layer_cpu(const model::LLaMABlock *block, float *hidden_states, float *normed_states,
                                size_t seq_len, const model::LLamaConfig &cfg,
                                const distributed::DistributedTopology &topology)
{
    const size_t hidden_dim = cfg.hidden_size;
    const size_t tp_size = static_cast<size_t>(topology.tp_size);
    const size_t intermediate_dim = cfg.intermediate_size / tp_size;
    const float rms_norm_eps = cfg.rms_norm_eps;
    const bool prof = prof::enabled();
    const double t_mlp0 = prof ? prof::now() : 0.0;

    rms_norm({hidden_states, block->post_attention_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim,
              rms_norm_eps});
    const double t_rms1 = prof ? prof::now() : 0.0;

    // Zero-copy non-owning view on already computed normed_states
    Tensor normed_tensor(normed_states, {seq_len, hidden_dim});

    thread_local std::vector<float> gate_v, up_v, down_v;
    gate_v.resize(seq_len * intermediate_dim);
    up_v.resize(seq_len * intermediate_dim);
    down_v.resize(seq_len * hidden_dim);
    Tensor gate_buf(gate_v.data(), {seq_len, intermediate_dim});
    Tensor up_buf(up_v.data(), {seq_len, intermediate_dim});
    Tensor down_proj_out(down_v.data(), {seq_len, hidden_dim});

    if (seq_len == 1 && block->gate_up_w.valid() && block->gate_up_w.in == hidden_dim &&
        block->gate_up_w.out == 2 * intermediate_dim) {
        thread_local std::vector<float> gu_tmp;
        gu_tmp.resize(2 * intermediate_dim);
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->gate_up_w, normed_tensor.data(), gu_tmp.data());
        if (p) {
            prof::acc().g_gu += prof::now() - t0;
        }
        std::memcpy(gate_buf.data(), gu_tmp.data(), intermediate_dim * sizeof(float));
        std::memcpy(up_buf.data(), gu_tmp.data() + intermediate_dim, intermediate_dim * sizeof(float));
    } else {
        core::blas::matmul(normed_tensor, block->gate_proj->data, gate_buf);
        core::blas::matmul(normed_tensor, block->up_proj->data, up_buf);
    }
    if (prof) {
        prof::acc().mlp_gu += prof::now() - t_rms1;
    }

    // Fused SiLU activation & gate multiply in a single memory pass
    float *g_ptr = gate_buf.data();
    const float *u_ptr = up_buf.data();
    const size_t total_mlp_elem = seq_len * intermediate_dim;
    size_t si = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
    {
        const __m256 v_one = _mm256_set1_ps(1.0F);
        const __m256 v_sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        for (; si + 8 <= total_mlp_elem; si += 8) {
            const __m256 x = _mm256_loadu_ps(g_ptr + si);
            const __m256 u = _mm256_loadu_ps(u_ptr + si);
            const __m256 e = exp256_ps(_mm256_xor_ps(x, v_sign));  // exp(-x)
            const __m256 sig = _mm256_div_ps(v_one, _mm256_add_ps(v_one, e));
            _mm256_storeu_ps(g_ptr + si, _mm256_mul_ps(_mm256_mul_ps(x, sig), u));
        }
    }
#endif
    for (; si < total_mlp_elem; ++si) {
        const float val = g_ptr[si];
        const float silu_val = val / (1.0F + std::exp(-val));
        g_ptr[si] = silu_val * u_ptr[si];
    }

    const bool use_quant_down = (seq_len == 1) && block->down_w.valid() && block->down_w.in == intermediate_dim &&
                                block->down_w.out == hidden_dim;
    const double t_sd0 = prof ? prof::now() : 0.0;
    if (use_quant_down) {
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->down_w, gate_buf.data(), down_proj_out.data());
        if (p) {
            prof::acc().g_down += prof::now() - t0;
        }
    } else {
        core::blas::matmul(gate_buf, block->down_proj->data, down_proj_out);
    }
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(down_proj_out.data(), down_proj_out.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += down_proj_out[i];
    }
    if (prof) {
        prof::acc().mlp_rest += prof::now() - t_sd0;
    }
    if (prof) {
        prof::acc().mlp += prof::now() - t_mlp0;
    }
}

// --- Forward Declarations of CUDA Path ---
#ifdef SERIUS_USE_CUDA
Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache);
Tensor llama_forward_cuda_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache);
Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache);
Tensor llama_forward_single_token_cuda_prefix(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache,
                                              size_t n_gpu_layers, float *hidden_out, size_t hidden_size);
Tensor llama_forward_single_token_cuda_graph(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache);
#endif

// --- Public Interface ---

/**
 * @brief Computes logits = final_normed @ lm_head^T.
 * Uses the transposed [vocab, hidden] lm_head_t with a row-major GEMV when
 * available (sequential memory access, much faster for large vocab); falls
 * back to the generic matmul otherwise.
 */
static void compute_logits(const model::LLaMAModel &model, const Tensor &final_normed, Tensor &logits, size_t seq_len,
                           const model::LLamaConfig &cfg)
{
    if (seq_len == 1) {
        if (const quant::QuantWeight *lq = model.lm_head_q()) {
            const bool p = prof::enabled();
            const double t0 = p ? prof::now() : 0.0;
            core::blas::matvec_q(*lq, final_normed.data(), logits.data());
            if (p) {
                prof::acc().g_lm += prof::now() - t0;
            }
            return;
        }
    }
    if (const core::Tensor *lmt = model.lm_head_t()) {
        const size_t vocab = cfg.vocab_size;
        const size_t hidden = cfg.hidden_size;
        for (size_t s = 0; s < seq_len; ++s) {
            core::blas::matvec_t(lmt->data(), final_normed.data() + (s * hidden), logits.data() + (s * vocab), vocab,
                                 hidden);
        }
    } else {
        core::blas::matmul(final_normed, model.lm_head()->data, logits);
    }
}

Tensor llama_forward_inference(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache)
{
    enable_ftz_daz_once();
#ifdef SERIUS_USE_CUDA
    if (prefill_batch_enabled() && partial_gpu_layers(model) == 0 && model.embed_tokens() &&
        model.embed_tokens()->data.is_gpu()) {
        return llama_forward_cuda(model, token_ids, cache);
    }
#endif

    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size();

    // Fast prefill: when the model carries packed quantized weights, run the
    // prompt through the single-token decode path. Only worthwhile for tiny
    // prompts — the batched prefill below is far faster per token. Note the
    // batched path uses dense FP32 weights while decode uses lossy-requantized
    // packed weights, so outputs can differ slightly (~1e-3 in logits).
    const bool has_quant = !model.layers().empty() && model.layers()[0]->qkv_w.valid();
    // Partial offload always takes the per-token path: batched prefill would
    // write every layer's KV on CPU while decode reads the GPU prefix KV.
    if (!cpu_prefill_batch_enabled() || partial_gpu_layers(model) > 0 || (has_quant && seq_len <= 8)) {
        core::Tensor last_logits({0});
        for (size_t i = 0; i < seq_len; ++i) {
            const int tid = static_cast<int>(token_ids[i]);
            last_logits = llama_forward_single_token(model, tid, i, cache);
        }
        return last_logits;
    }

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
        const ActivationHooks &hooks = activation_hooks();
        for (size_t layer_idx = 0; layer_idx < layer_blocks.size(); ++layer_idx) {
            KVCacheLayer *cache_layer = (cache != nullptr) ? &cache->layers[layer_idx] : nullptr;
            llama_attention_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(),
                                      seq_len, 0, cfg, model.topology(), cache_layer, static_cast<int>(layer_idx));
            llama_mlp_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(),
                                seq_len, cfg, model.topology());
            if (hooks.layer_hidden != nullptr) {
                hooks.layer_hidden(static_cast<int>(layer_idx), hidden_states.data(), seq_len, hidden_dim);
            }
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
            // Intermediate PP ranks only forward hidden states through the
            // pipeline; logits exist solely on the last rank. Return an empty
            // rank-1 tensor ({0}) — same contract as the per-token path —
            // callers must not read logits on non-last ranks.
            return Tensor({0});
        }
    } else {
        // Single-rank path
        preprocess_embeddings(model, token_ids, hidden_states);
        run_transformer_layers(hidden_states.data(), hidden_states.data(), hidden_states.size());
    }

    // 4. Final Head (Last rank or single rank)
    Tensor final_normed({seq_len, hidden_dim});
    rms_norm({hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), seq_len, hidden_dim,
              rms_norm_eps});
    Tensor logits({seq_len, cfg.vocab_size});
    compute_logits(model, final_normed, logits, seq_len, cfg);
    return logits;
}

Tensor llama_forward_inference_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache)
{
    enable_ftz_daz_once();
    if (!prefill_batch_enabled()) {
        // Batched prefill disabled (SERIUS_PREFILL_BATCH=0) — decode one
        // token at a time instead. llama_forward_single_token dispatches to
        // the CUDA graph path for GPU-resident models, so this stays
        // reasonably fast.
        Tensor last_logits({0});
        for (size_t i = 0; i < token_ids.size(); ++i) {
            last_logits = llama_forward_single_token(model, static_cast<int>(token_ids[i]), i, cache);
        }
        return last_logits;
    }
#ifdef SERIUS_USE_CUDA
    if (partial_gpu_layers(model) == 0 && model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        return llama_forward_cuda_last(model, token_ids, cache);
    }
#endif
    Tensor logits = llama_forward_inference(model, token_ids, cache);
    if (logits.size() == 0) {
        // Non-last PP rank: no logits exist on this rank.
        return logits;
    }
    const size_t vocab = logits.shape().back();
    const size_t rows = logits.shape()[0];
    Tensor out({1, vocab});
    const float *src = logits.data() + (rows - 1) * vocab;
    std::copy(src, src + vocab, out.data());
    return out;
}

/// CPU suffix for partial GPU offload: continues the residual stream at
/// `layer_begin` through the final norm + lm_head. `hidden` is a {1,hidden}
/// FP32 buffer updated in place; KV for these layers lives in `cache` as CPU
/// tensors grown on demand by update_kv_cache.
/// Only used by the CUDA partial-offload path below.
#ifdef SERIUS_USE_CUDA
static Tensor llama_forward_suffix_cpu(const model::LLaMAModel &model, float *hidden, size_t pos, KVCache *cache,
                                       size_t layer_begin)
{
    const auto &cfg = model.config();
    const size_t hidden_dim = cfg.hidden_size;

    Tensor normed({1, hidden_dim});
    const auto &layer_blocks = model.layers();
    for (size_t i = layer_begin; i < layer_blocks.size(); ++i) {
        KVCacheLayer *cl = (cache != nullptr && i < cache->layers.size()) ? &cache->layers[i] : nullptr;
        llama_attention_layer_cpu(layer_blocks[i].get(), hidden, normed.data(), 1, pos, cfg, model.topology(), cl,
                                  static_cast<int>(i));
        llama_mlp_layer_cpu(layer_blocks[i].get(), hidden, normed.data(), 1, cfg, model.topology());
    }
    Tensor final_normed({1, hidden_dim});
    rms_norm({hidden, model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, cfg.rms_norm_eps});
    Tensor logits({1, cfg.vocab_size});
    compute_logits(model, final_normed, logits, 1, cfg);
    return logits;
}
#endif  // SERIUS_USE_CUDA

/// One decode/prefill-token step under partial offload: GPU prefix (layers
/// [0, n_gpu)) produces the residual stream, D2H once, CPU suffix finishes.
/// Every token goes through this path so each layer's KV cache is written by
/// the device that owns it — batched prefill is deliberately bypassed.
/// CUDA-only: partial offload requires the GPU prefix path.
#ifdef SERIUS_USE_CUDA
static Tensor llama_forward_single_token_partial(const model::LLaMAModel &model, int token_id, size_t pos,
                                                 KVCache *cache, size_t n_gpu)
{
    const size_t hidden_dim = model.config().hidden_size;
    std::vector<float> hbuf(hidden_dim, 0.0f);
    // Returns an empty tensor by design: it emits the hidden state via hbuf.
    // CUDA errors throw — there is no silent wrong-result path.
    llama_forward_single_token_cuda_prefix(model, token_id, pos, cache, n_gpu, hbuf.data(), hidden_dim);
    return llama_forward_suffix_cpu(model, hbuf.data(), pos, cache, n_gpu);
}
#endif  // SERIUS_USE_CUDA

Tensor llama_forward_single_token(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache)
{
    enable_ftz_daz_once();
#ifdef SERIUS_USE_CUDA
    const size_t ngpu = partial_gpu_layers(model);
    if (ngpu > 0) {
        return llama_forward_single_token_partial(model, token_id, pos, cache, ngpu);
    }
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        Tensor graph_logits = llama_forward_single_token_cuda_graph(model, token_id, pos, cache);
        if (graph_logits.size() > 0) {
            return graph_logits;
        }
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
        const ActivationHooks &hooks = activation_hooks();
        for (size_t layer_idx = 0; layer_idx < layer_blocks.size(); ++layer_idx) {
            KVCacheLayer *cache_layer = (cache != nullptr) ? &cache->layers[layer_idx] : nullptr;
            llama_attention_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(),
                                      1, pos, cfg, model.topology(), cache_layer, static_cast<int>(layer_idx));
            llama_mlp_layer_cpu(layer_blocks[layer_idx].get(), hidden_states.data(), normed_states_tensor.data(), 1,
                                cfg, model.topology());
            if (hooks.layer_hidden != nullptr) {
                hooks.layer_hidden(static_cast<int>(layer_idx), hidden_states.data(), 1, hidden_dim);
            }
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

        // 4. Final Head
        const bool prof_h = prof::enabled();
        const double t_head0 = prof_h ? prof::now() : 0.0;
        Tensor final_normed({1, hidden_dim});
        rms_norm(
            {hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, rms_norm_eps});
        Tensor logits({1, cfg.vocab_size});
        compute_logits(model, final_normed, logits, 1, cfg);
        if (prof_h) {
            prof::acc().head += prof::now() - t_head0;
            prof::report();
        }
        return logits;
    }

    // 4. Final Head
    Tensor final_normed({1, hidden_dim});
    rms_norm(
        {hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, rms_norm_eps});
    Tensor logits({1, cfg.vocab_size});
    compute_logits(model, final_normed, logits, 1, cfg);
    if (prof::enabled()) {
        prof::report();
    }
    return logits;
}

}  // namespace infer
}  // namespace serius
