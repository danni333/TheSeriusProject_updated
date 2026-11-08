#include "llama_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
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

using core::Tensor;

// --- Temporary decode profiling (SERIUS_PROF=1) ---
namespace prof {
inline bool enabled() {
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
struct Acc {
    double attn_pre = 0, attn_hd = 0, attn_post = 0, mlp = 0, mlp_gu = 0, mlp_rest = 0, head = 0;
    double g_qkv = 0, g_o = 0, g_gu = 0, g_down = 0, g_lm = 0;
    long calls = 0;
};
inline Acc &acc() { static Acc a; return a; }
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void report() {
    if (!enabled()) { return; }
    Acc &a = acc();
    ++a.calls;
    if (a.calls % 300 == 0) {
        std::fprintf(stderr,
            "[prof] attn_pre=%.3f attn_hd=%.3f attn_post=%.3f mlp=%.3f (gu=%.3f rest=%.3f) head=%.3f ms/tok\n"
            "       gemv: qkv=%.3f o=%.3f gateup=%.3f down=%.3f lm=%.3f ms/tok\n",
            a.attn_pre * 1e3, a.attn_hd * 1e3, a.attn_post * 1e3, a.mlp * 1e3,
            a.mlp_gu * 1e3, a.mlp_rest * 1e3, a.head * 1e3,
            a.g_qkv * 1e3, a.g_o * 1e3, a.g_gu * 1e3, a.g_down * 1e3, a.g_lm * 1e3);
        a.attn_pre = a.attn_hd = a.attn_post = a.mlp = a.mlp_gu = a.mlp_rest = a.head = 0.0;
        a.g_qkv = a.g_o = a.g_gu = a.g_down = a.g_lm = 0.0;
    }
}
}  // namespace prof

// Flush-to-zero / denormals-are-zero on the caller thread AND all OpenMP
// workers. Quantized dequant + exp approximations produce subnormal floats;
// without DAZ/FTZ every FMA on them traps to microcode (~100x slower).
static void enable_ftz_daz_once() {
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
    const size_t half_dim = args.head_dim / 2;

    // Decode fast path (seq_len == 1): inv_freq is position-independent, so
    // cache it; cos/sin for the single position go on the stack. All heads
    // and both Q/K share the same tables.
    if (args.seq_len == 1) {
        struct InvFreqCache {
            size_t head_dim = 0;
            float rope_base = 0.0F;
            std::vector<float> inv_freq;
        };
        static thread_local InvFreqCache cache;
        if (cache.head_dim != args.head_dim || cache.rope_base != args.rope_base) {
            cache.inv_freq.resize(half_dim);
            for (size_t i = 0; i < half_dim; ++i) {
                cache.inv_freq[i] = 1.0F / std::pow(args.rope_base,
                    static_cast<float>(2 * i) / static_cast<float>(args.head_dim));
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
                head_ptr[i]            = (val_x0 * cos_t[i]) - (val_x1 * sin_t[i]);
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
            const float freq = 1.0F / std::pow(args.rope_base,
                static_cast<float>(2 * i) / static_cast<float>(args.head_dim));
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
                head_ptr[i]            = (val_x0 * cos_row[i]) - (val_x1 * sin_row[i]);
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
    const __m256 pow2n = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(p, pow2n);
}

[[maybe_unused]] static void silu_inplace(float *data_ptr, size_t count) {
    // AVX2 path: process 8 floats at a time.
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // We approximate exp via a fast polynomial for the AVX lanes and use the
    // scalar std::exp only for the tail (at most 7 elements).
    size_t idx = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
    const __m256 ones  = _mm256_set1_ps(1.0F);
    const __m256 zeros = _mm256_setzero_ps();
    // Fast sigmoid approximation: 1/(1+exp(-x)) ≈ clamp(x/8+0.5, 0, 1)
    // Good enough for SiLU's smooth region; for high accuracy the scalar path
    // is used at the tail anyway.
    // Actually use the exact scalar formula but vectorised via libm:
    // We do x * (x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))) — same thing.
    // Faster trick: replace 1/(1+exp(-x)) with the tanh identity:
    //   sigmoid(x) = (tanh(x/2) + 1) / 2
    // which is not available as an AVX intrinsic either; keep scalar for
    // correctness but vectorise the multiply back.
    // Best available: compute in scalar, store back vectorised.
    // For now do the full 8-wide loop with scalar exp per element:
    for (; idx + 8 <= count; idx += 8) {
        // Load 8 values
        __m256 v = _mm256_loadu_ps(data_ptr + idx);
        // Compute SiLU element-wise (scalar exp, fastest portable approach)
        float buf[8];
        _mm256_storeu_ps(buf, v);
        for (int k = 0; k < 8; ++k) {
            buf[k] = buf[k] / (1.0F + std::exp(-buf[k]));
        }
        _mm256_storeu_ps(data_ptr + idx, _mm256_loadu_ps(buf));
    }
    (void)ones; (void)zeros;
#endif
    // Scalar tail
    for (; idx < count; ++idx) {
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
        size_t col_idx = 1;
#if defined(__AVX2__) || defined(_MSC_VER)
        {
            __m256 vmax = _mm256_set1_ps(row[0]);
            for (; col_idx + AVX_FLOAT_STRIDE <= args.cols; col_idx += AVX_FLOAT_STRIDE) {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + col_idx));
            }
            alignas(32) float mbuf[AVX_FLOAT_STRIDE];
            _mm256_store_ps(mbuf, vmax);
            for (int i = 0; i < AVX_FLOAT_STRIDE; ++i) { max_val = std::max(max_val, mbuf[i]); }
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
static void compute_qkv_rope(
    const model::LLaMABlock *block, const Tensor &normed_tensor,
    Tensor &query_buf, Tensor &key_buf, Tensor &value_buf,
    size_t seq_len, size_t pos_offset, const model::LLamaConfig &cfg) {
    
    const size_t hidden_dim = cfg.hidden_size;
    const size_t head_dim = hidden_dim / cfg.num_attention_heads;
    const size_t local_hidden_dim = query_buf.shape()[1];
    const size_t local_kv_dim = key_buf.shape()[1];

    // Fast path: packed-quantized weights with on-the-fly dequant GEMV (M==1 only)
    const size_t qkv_total = local_hidden_dim + 2 * local_kv_dim;
    if (seq_len == 1 && block->qkv_w.valid() && block->qkv_w.in == hidden_dim &&
        block->qkv_w.out == qkv_total) {
        thread_local std::vector<float> qkv_tmp;
        qkv_tmp.resize(qkv_total);
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->qkv_w, normed_tensor.data(), qkv_tmp.data());
        if (p) { prof::acc().g_qkv += prof::now() - t0; }
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
        if (bias.size() != cols) { return; }
        const float *b = bias.data();
        for (size_t r = 0; r < rows; ++r) {
            float *row = buf + (r * cols);
            for (size_t c = 0; c < cols; ++c) { row[c] += b[c]; }
        }
    };
    if (block->q_bias) { add_qkv_bias(query_buf.data(), seq_len, local_hidden_dim, block->q_bias->data); }
    if (block->k_bias) { add_qkv_bias(key_buf.data(), seq_len, local_kv_dim, block->k_bias->data); }
    if (block->v_bias) { add_qkv_bias(value_buf.data(), seq_len, local_kv_dim, block->v_bias->data); }

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
        if (e <= 0) { r = static_cast<uint16_t>(sign); }
        else if (e >= 31) { r = static_cast<uint16_t>(sign | 0x7C00u); }
        else { r = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m >> 13)); }
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
            if (m == 0) { r = sign; }
            else {
                int exp = -1;
                uint32_t mm = m;
                while ((mm & 0x400u) == 0) { mm <<= 1; --exp; }
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
    if (e <= 0) { return static_cast<uint16_t>(sign); }
    if (e >= 31) { return static_cast<uint16_t>(sign | 0x7C00u); }
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
        if (m == 0) { r = sign; }
        else {
            int exp = -1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0) { mm <<= 1; --exp; }
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
static float kv_dot_f32(const float *q, const uint16_t *p16, const float *p32,
                        size_t off, size_t dim)
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

static void update_kv_cache(KVCacheLayer *cache_layer, const Tensor &key_buf, const Tensor &value_buf, size_t seq_len) {
    if (cache_layer == nullptr) {
        return;
    }
    const size_t local_kv_dim = key_buf.shape()[1];
    const size_t start_pos = cache_layer->current_len;
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
    constexpr size_t STACK_SCORES_CAP = 1024;
    float stack_scores[STACK_SCORES_CAP];
    std::vector<float> heap_scores;

    for (size_t qi = 0; qi < args.seq_len; ++qi) {
        const size_t abs_qi = args.pos_offset + qi;
        float *scores = stack_scores;
        if (args.total_len > STACK_SCORES_CAP) {
            heap_scores.resize(args.total_len);
            scores = heap_scores.data();
        }

        const float *q_ptr = args.query_buf.data() + (qi * args.local_hidden_dim) + (static_cast<size_t>(args.head_idx) * args.head_dim);
        const size_t npos = abs_qi + 1;
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
        const float *vbase32 = (!kv16 && args.cache_layer != nullptr) ? args.cache_layer->v.data() : args.value_buf.data();
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
                scores[kj] = kv_dot_f32(q_ptr, k16base, kbase,
                    (kj * args.local_kv_dim) + koff, args.head_dim) * args.attention_scale;
            }
        } else {
            for (size_t kj = 0; kj < npos; ++kj) {
                scores[kj] = kv_dot_f32(q_ptr, k16base, kbase,
                    (kj * args.local_kv_dim) + koff, args.head_dim) * args.attention_scale;
            }
        }
        for (size_t kj = npos; kj < args.total_len; ++kj) {
            scores[kj] = ATTENTION_MASK_VAL;
        }

        softmax_rows_inplace({scores, 1, args.total_len});

        float *out_ptr = args.attention_out_ptr + (qi * args.local_hidden_dim) + (static_cast<size_t>(args.head_idx) * args.head_dim);
        if (simd_ok) {
            // Weighted-V sum with the output kept in registers across all positions.
            constexpr size_t MAX_VVEC = 16;  // head_dim <= 128
            __m256 vacc[MAX_VVEC];
            const size_t nvec = args.head_dim / AVX_FLOAT_STRIDE;
            for (size_t d = 0; d < nvec; ++d) { vacc[d] = _mm256_setzero_ps(); }
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
            if (kv16) { vrow.resize(args.head_dim); }
            for (size_t kj = 0; kj <= abs_qi; ++kj) {
                const size_t vo = kj * args.local_kv_dim + (kv_head_idx * args.head_dim);
                const float *v_ptr;
                if (kv16) {
                    const uint16_t *s = v16base + vo;
                    for (size_t d = 0; d < args.head_dim; ++d) { vrow[d] = f16_to_f32_scalar(s[d]); }
                    v_ptr = vrow.data();
                } else {
                    v_ptr = (args.cache_layer != nullptr)
                        ? (args.cache_layer->v.data() + vo)
                        : (args.value_buf.data() + vo);
                }
                vec_add_scale_avx2({out_ptr, v_ptr, scores[kj], args.head_dim});
            }
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
            cache->layers[i].k16.assign(DEFAULT_MAX_SEQ_LEN * local_kv_dim, 0);
            cache->layers[i].v16.assign(DEFAULT_MAX_SEQ_LEN * local_kv_dim, 0);
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

    const bool prof = prof::enabled();
    double t_pre0 = prof ? prof::now() : 0.0;
    rms_norm({hidden_states, block->input_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim, cfg.rms_norm_eps});

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
    if (prof) { prof::acc().attn_pre += prof::now() - t_pre0; }

    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const size_t num_rep = num_local_heads / num_local_kv_heads;
    const size_t total_len = (cache_layer != nullptr) ? cache_layer->current_len : seq_len;

    double t_hd0 = prof ? prof::now() : 0.0;
    #pragma omp parallel for schedule(static) if (num_local_heads >= 2)
    for (int h_idx = 0; h_idx < static_cast<int>(num_local_heads); ++h_idx) {
        compute_head_attention({
            h_idx, seq_len, pos_offset, total_len,
            local_hidden_dim, local_kv_dim, head_dim, num_rep,
            query_buf, key_buf, value_buf, cache_layer, attention_scale, attention_out.data()
        });
    }
    if (prof) { prof::acc().attn_hd += prof::now() - t_hd0; }

    double t_post0 = prof ? prof::now() : 0.0;
    if (seq_len == 1 && block->o_w.valid() && block->o_w.in == local_hidden_dim && block->o_w.out == hidden_dim) {
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->o_w, attention_out.data(), attention_proj.data());
        if (p) { prof::acc().g_o += prof::now() - t0; }
    } else {
        core::blas::matmul(attention_out, block->o_proj->data, attention_proj);
    }
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(attention_proj.data(), attention_proj.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += attention_proj[i];
    }
    if (prof) { prof::acc().attn_post += prof::now() - t_post0; }
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
    const bool prof = prof::enabled();
    const double t_mlp0 = prof ? prof::now() : 0.0;

    rms_norm({hidden_states, block->post_attention_layernorm_weight->data.data(), normed_states, seq_len, hidden_dim, rms_norm_eps});
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
        if (p) { prof::acc().g_gu += prof::now() - t0; }
        std::memcpy(gate_buf.data(), gu_tmp.data(), intermediate_dim * sizeof(float));
        std::memcpy(up_buf.data(), gu_tmp.data() + intermediate_dim, intermediate_dim * sizeof(float));
    } else {
        core::blas::matmul(normed_tensor, block->gate_proj->data, gate_buf);
        core::blas::matmul(normed_tensor, block->up_proj->data, up_buf);
    }
    if (prof) { prof::acc().mlp_gu += prof::now() - t_rms1; }

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

    const bool use_quant_down = (seq_len == 1) && block->down_w.valid() &&
                                block->down_w.in == intermediate_dim && block->down_w.out == hidden_dim;
    const double t_sd0 = prof ? prof::now() : 0.0;
    if (use_quant_down) {
        const bool p = prof::enabled();
        const double t0 = p ? prof::now() : 0.0;
        core::blas::matvec_q(block->down_w, gate_buf.data(), down_proj_out.data());
        if (p) { prof::acc().g_down += prof::now() - t0; }
    } else {
        core::blas::matmul(gate_buf, block->down_proj->data, down_proj_out);
    }
    if (topology.is_tensor_parallel()) {
        topology.tp_comm->all_reduce(down_proj_out.data(), down_proj_out.size(), distributed::ReduceOp::Sum);
    }

    for (size_t i = 0; i < seq_len * hidden_dim; ++i) {
        hidden_states[i] += down_proj_out[i];
    }
    if (prof) { prof::acc().mlp_rest += prof::now() - t_sd0; }
    if (prof) { prof::acc().mlp += prof::now() - t_mlp0; }
}

// --- Forward Declarations of CUDA Path ---
#ifdef SERIUS_USE_CUDA
Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache);
Tensor llama_forward_cuda_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache);
Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache);
bool llama_forward_single_token_cuda_graph(const model::LLaMAModel &model, int token_id, size_t pos,
                                           KVCache *cache, Tensor &out_logits);
#endif

// --- Public Interface ---

/**
 * @brief Computes logits = final_normed @ lm_head^T.
 * Uses the transposed [vocab, hidden] lm_head_t with a row-major GEMV when
 * available (sequential memory access, much faster for large vocab); falls
 * back to the generic matmul otherwise.
 */
static void compute_logits(const model::LLaMAModel &model, const Tensor &final_normed, Tensor &logits,
                           size_t seq_len, const model::LLamaConfig &cfg)
{
    if (seq_len == 1) {
        if (const quant::QuantWeight *lq = model.lm_head_q()) {
            const bool p = prof::enabled();
            const double t0 = p ? prof::now() : 0.0;
            core::blas::matvec_q(*lq, final_normed.data(), logits.data());
            if (p) { prof::acc().g_lm += prof::now() - t0; }
            return;
        }
    }
    if (const core::Tensor *lmt = model.lm_head_t()) {
        const size_t vocab  = cfg.vocab_size;
        const size_t hidden = cfg.hidden_size;
        for (size_t s = 0; s < seq_len; ++s) {
            core::blas::matvec_t(lmt->data(), final_normed.data() + (s * hidden),
                                 logits.data() + (s * vocab), vocab, hidden);
        }
    } else {
        core::blas::matmul(final_normed, model.lm_head()->data, logits);
    }
}

Tensor llama_forward_inference(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
    enable_ftz_daz_once();
#ifdef SERIUS_USE_CUDA
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        return llama_forward_cuda(model, token_ids, cache);
    }
#endif

    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size();

    // Fast prefill: when the model carries packed quantized weights, run the
    // prompt through the single-token decode path. Only worthwhile for tiny
    // prompts — the batched cuBLAS prefill below is far faster per token.
    const bool has_quant = !model.layers().empty() && model.layers()[0]->qkv_w.valid();
    if (has_quant && seq_len <= 8) {
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
    compute_logits(model, final_normed, logits, seq_len, cfg);
    return logits;
}

Tensor llama_forward_inference_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
    enable_ftz_daz_once();
#ifdef SERIUS_USE_CUDA
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        return llama_forward_cuda_last(model, token_ids, cache);
    }
#endif
    Tensor logits = llama_forward_inference(model, token_ids, cache);
    const size_t vocab = logits.shape().back();
    const size_t rows = logits.shape()[0];
    Tensor out({1, vocab});
    const float *src = logits.data() + (rows - 1) * vocab;
    std::copy(src, src + vocab, out.data());
    return out;
}

Tensor llama_forward_single_token(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache) {
    enable_ftz_daz_once();
#ifdef SERIUS_USE_CUDA
    if (model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        Tensor graph_logits;
        if (llama_forward_single_token_cuda_graph(model, token_id, pos, cache, graph_logits)) {
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

        // 4. Final Head
        const bool prof_h = prof::enabled();
        const double t_head0 = prof_h ? prof::now() : 0.0;
        Tensor final_normed({1, hidden_dim});
        rms_norm({hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, rms_norm_eps});
        Tensor logits({1, cfg.vocab_size});
        compute_logits(model, final_normed, logits, 1, cfg);
        if (prof_h) { prof::acc().head += prof::now() - t_head0; prof::report(); }
        return logits;
    }

    // 4. Final Head
    Tensor final_normed({1, hidden_dim});
    rms_norm({hidden_states.data(), model.norm_weight()->data.data(), final_normed.data(), 1, hidden_dim, rms_norm_eps});
    Tensor logits({1, cfg.vocab_size});
    compute_logits(model, final_normed, logits, 1, cfg);
    if (prof::enabled()) { prof::report(); }
    return logits;
}

// --- CUDA Implementations ---
#ifdef SERIUS_USE_CUDA
// last_only: compute logits for the final position only ({1, vocab} instead of
// {seq_len, vocab}) — chat/server callers only ever read the last row, and the
// full logits tensor is seq_len*vocab*4 bytes of host memset + D2H copy.
static Tensor llama_forward_cuda_impl(const model::LLaMAModel &model, const Tensor &token_ids,
                                      KVCache *cache, bool last_only) {
    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size(), hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    const hal::DType stream_dtype = hal::DType::Float32;  // FP32 residual stream

    const bool prof_p = [] {
        size_t n = 0;
        return getenv_s(&n, nullptr, 0, "SERIUS_PROF") == 0 && n > 0;
    }();
    auto pt_now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double pt0 = prof_p ? pt_now() : 0.0;

    Tensor hidden_states_gpu = Tensor::gpu_only({seq_len, hidden}, stream_dtype);
    std::vector<int> ids(seq_len); for(size_t i=0; i<seq_len; ++i) ids[i] = (int)token_ids[i];
    int *ids_gpu = nullptr; 
    CHECK_CUDA(cudaMalloc(&ids_gpu, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(ids_gpu, ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    hal::gather_mixed(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), ids_gpu, seq_len, hidden, dtype, stream_dtype);
    hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU); 
    CHECK_CUDA(cudaFree(ids_gpu));
    const double pt_embed = prof_p ? pt_now() : 0.0;

    Tensor normed_buf = Tensor::gpu_only({seq_len, hidden}, dtype);
    Tensor query_buf = Tensor::gpu_only({seq_len, hidden}, dtype);
    Tensor key_buf = Tensor::gpu_only({seq_len, kv_dim}, dtype);
    Tensor value_buf = Tensor::gpu_only({seq_len, kv_dim}, dtype);
    Tensor attention_out = Tensor::gpu_only({seq_len, hidden}, dtype);
    Tensor mlp_gate = Tensor::gpu_only({seq_len, inter}, dtype);
    Tensor mlp_up = Tensor::gpu_only({seq_len, inter}, dtype);
    Tensor mlp_down = Tensor::gpu_only({seq_len, hidden}, dtype);

    // Persistent attention scratch: allocated once instead of ~10 tensors per
    // layer (each Tensor ctor memsets its host buffer — hundreds of MB of
    // wasted page faults per prefill otherwise).
    Tensor scores = Tensor::gpu_only({n_heads, seq_len, seq_len}, dtype);
    Tensor q_hs_buf = Tensor::gpu_only({n_heads, seq_len, head_dim}, dtype);
    Tensor v_hs_buf = Tensor::gpu_only({n_heads, seq_len, head_dim}, dtype);
    Tensor kT_buf = Tensor::gpu_only({n_heads, head_dim, seq_len}, dtype);
    Tensor ctx_hs_buf = Tensor::gpu_only({n_heads, seq_len, head_dim}, dtype);
    Tensor v_rep_buf = Tensor::gpu_only({seq_len, n_heads, head_dim}, dtype);
    Tensor k_rep_buf = Tensor::gpu_only({seq_len, n_heads, head_dim}, dtype);
    // attention_out ({seq, hidden}) has the same contiguous layout as
    // {seq, n_heads, head_dim} — copy_layout writes straight into it.

    const double pt_loop0 = prof_p ? pt_now() : 0.0;
    const auto &layers = model.layers();
    if (cache && cache->empty()) {
        cache->layers.resize(layers.size());
        // One arena allocation for every layer's K and V: 112 individual
        // cudaMallocs cost ~200 ms of driver time; a single slab + non-owning
        // views costs one.
        const size_t slab_elems = 4096 * kv_dim;
        cache->gpu_arena = Tensor::gpu_only({layers.size() * 2 * slab_elems}, dtype);
        char *base = static_cast<char *>(cache->gpu_arena.raw_data_gpu());
        const size_t slab_bytes = slab_elems * get_dtype_size(dtype);
        for (size_t i = 0; i < layers.size(); ++i) {
            auto &cl = cache->layers[i];
            cl.k = Tensor::gpu_view(base + i * 2 * slab_bytes, {4096, kv_dim}, dtype);
            cl.v = Tensor::gpu_view(base + i * 2 * slab_bytes + slab_bytes, {4096, kv_dim}, dtype);
            cl.current_len = 0;
        }
    }
    const double pt_cache = prof_p ? pt_now() : 0.0;

    const double pt_mask = prof_p ? pt_now() : 0.0;

    double acc_proj = 0, acc_attn = 0, acc_mlp = 0;
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const double seg0 = prof_p ? pt_now() : 0.0;
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->input_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->q_proj->data, query_buf);
        core::blas::matmul(normed_buf, blk->k_proj->data, key_buf);
        core::blas::matmul(normed_buf, blk->v_proj->data, value_buf);
        if (blk->q_bias) { hal::add_row_bias(query_buf.raw_data_gpu(), blk->q_bias->data.raw_data_gpu(), seq_len, hidden, dtype); }
        if (blk->k_bias) { hal::add_row_bias(key_buf.raw_data_gpu(), blk->k_bias->data.raw_data_gpu(), seq_len, kv_dim, dtype); }
        if (blk->v_bias) { hal::add_row_bias(value_buf.raw_data_gpu(), blk->v_bias->data.raw_data_gpu(), seq_len, kv_dim, dtype); }
        hal::rope(query_buf.raw_data_gpu(), head_dim, seq_len, n_heads, rope_base, 0, dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, seq_len, n_kv_heads, rope_base, 0, dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU); key_buf.mark_dirty(Tensor::DataLocation::GPU);
        // Pre-scale Q by 1/sqrt(head_dim): keeps attention scores inside the
        // reduced-precision range (Qwen logits can exceed FP16 max otherwise).
        const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        if (dtype != hal::DType::Float32) {
            hal::scale_inplace(query_buf.raw_data_gpu(), seq_len * hidden, q_scale, dtype);
        }
        const double seg1 = prof_p ? pt_now() : 0.0;

        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            size_t bytes = seq_len * kv_dim * get_dtype_size(dtype);
            CHECK_CUDA(cudaMemcpyAsync((char*)cl.k.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype), 
                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync((char*)cl.v.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype), 
                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            cl.current_len += seq_len;
        }

        // Attention with persistent scratch: transpose+contiguous chains are
        // replaced by copy_layout into preallocated buffers (no per-layer
        // Tensor churn / host memset).
        {
            // Q: {seq, n_heads, hd} -> {n_heads, seq, hd} in q_hs_buf
            Tensor qv = query_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
            hal::copy_layout(q_hs_buf.raw_data_gpu(), qv.raw_data_gpu(), qv.shape().data(),
                             qv.strides().data(), qv.ndim(), dtype);

            // K/V: expand GQA into {seq, n_heads, hd} rep buffers, then copy
            // to the {n_heads, seq, hd} (v) and {n_heads, hd, seq} (k) layouts
            // the batched GEMMs expect.
            if (n_kv_heads < n_heads) {
                hal::repeat_kv(k_rep_buf.raw_data_gpu(), key_buf.raw_data_gpu(), seq_len,
                               n_heads, n_kv_heads, head_dim, dtype);
                hal::repeat_kv(v_rep_buf.raw_data_gpu(), value_buf.raw_data_gpu(), seq_len,
                               n_heads, n_kv_heads, head_dim, dtype);
                Tensor kt = k_rep_buf.transpose(0, 1);   // {nh, seq, hd}
                Tensor vt = v_rep_buf.transpose(0, 1);
                hal::copy_layout(v_hs_buf.raw_data_gpu(), vt.raw_data_gpu(), vt.shape().data(),
                                 vt.strides().data(), vt.ndim(), dtype);
                Tensor ktt = kt.transpose(1, 2);         // {nh, hd, seq}
                hal::copy_layout(kT_buf.raw_data_gpu(), ktt.raw_data_gpu(), ktt.shape().data(),
                                 ktt.strides().data(), ktt.ndim(), dtype);
            } else {
                Tensor kt = key_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
                Tensor vt = value_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
                hal::copy_layout(v_hs_buf.raw_data_gpu(), vt.raw_data_gpu(), vt.shape().data(),
                                 vt.strides().data(), vt.ndim(), dtype);
                Tensor ktt = kt.transpose(1, 2);
                hal::copy_layout(kT_buf.raw_data_gpu(), ktt.raw_data_gpu(), ktt.shape().data(),
                                 ktt.strides().data(), ktt.ndim(), dtype);
            }
        }

        float scale = (dtype == hal::DType::Float32) ? (1.0F / std::sqrt(static_cast<float>(head_dim))) : 1.0F;
        {
            Tensor q_t = Tensor::gpu_view(q_hs_buf.raw_data_gpu(), {n_heads, seq_len, head_dim}, dtype);
            Tensor kT_t = Tensor::gpu_view(kT_buf.raw_data_gpu(), {n_heads, head_dim, seq_len}, dtype);
            core::blas::batched_matmul(q_t, kT_t, scores);
        }
        // mask=nullptr → the kernel applies causal masking (i > q_idx) itself.
        hal::fused_masked_softmax(scores.raw_data_gpu(), nullptr, n_heads, seq_len, seq_len, dtype, scale);

        {
            Tensor v_t = Tensor::gpu_view(v_hs_buf.raw_data_gpu(), {n_heads, seq_len, head_dim}, dtype);
            Tensor ctx_t = Tensor::gpu_view(ctx_hs_buf.raw_data_gpu(), {n_heads, seq_len, head_dim}, dtype);
            core::blas::batched_matmul(scores, v_t, ctx_t);
            // context {nh, seq, hd} -> attention_out {seq, hidden}
            Tensor cv = ctx_hs_buf.transpose(0, 1);
            hal::copy_layout(attention_out.raw_data_gpu(), cv.raw_data_gpu(), cv.shape().data(),
                             cv.strides().data(), cv.ndim(), dtype);
        }
        attention_out.mark_dirty(Tensor::DataLocation::GPU);
        const double seg2 = prof_p ? pt_now() : 0.0;
        core::blas::matmul(attention_out, blk->o_proj->data, mlp_down);
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden, stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(normed_buf, blk->gate_proj->data, mlp_gate);
        core::blas::matmul(normed_buf, blk->up_proj->data, mlp_up);
        hal::swiglu(mlp_gate.raw_data_gpu(), mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), seq_len * inter, dtype);
        mlp_gate.mark_dirty(Tensor::DataLocation::GPU);
        core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden, stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        if (prof_p) {
            const double seg3 = pt_now();
            acc_proj += seg1 - seg0; acc_attn += seg2 - seg1; acc_mlp += seg3 - seg2;
        }
    }
    if (prof_p) std::fprintf(stderr, "[prefillprof] proj=%.1fms attn=%.1fms mlp=%.1fms cache=%.1fms mask=%.1fms\n",
                             acc_proj * 1e3, acc_attn * 1e3, acc_mlp * 1e3,
                             (pt_cache - pt_loop0) * 1e3, (pt_mask - pt_cache) * 1e3);
    const double pt_loop1 = prof_p ? pt_now() : 0.0;
    hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), model.norm_weight()->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);

    if (prof_p) {
        std::fprintf(stderr, "[prefillprof] embed=%.1fms setup=%.1fms layers=%.1fms\n",
                     (pt_embed - pt0) * 1e3, (pt_loop0 - pt_embed) * 1e3, (pt_loop1 - pt_loop0) * 1e3);
    }
    const double pt_head0 = prof_p ? pt_now() : 0.0;

    // When only the last row's logits are needed, view just that row of the
    // {seq_len, hidden} buffer — the logits GEMM cost drops by seq_len× and
    // the {seq_len, vocab} host tensor + D2H copy disappears entirely.
    const size_t log_rows = last_only ? 1 : seq_len;
    const size_t row_off = last_only ? (seq_len - 1) * hidden : 0;
    const size_t esz = get_dtype_size(dtype);
    void *norm_ptr = static_cast<char *>(normed_buf.raw_data_gpu()) + row_off * esz;
    Tensor normed_last = Tensor::gpu_view(norm_ptr, {log_rows, hidden}, dtype);

    if (dtype == hal::DType::Float16) {
        if (const core::Tensor *lhead16 = model.lm_head_fp16_t()) {
        // FP16 weights + FP32 cuBLAS accumulation for the logits GEMV.
        Tensor logits({log_rows, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        core::blas::logits_gemv_mixed_cublas(norm_ptr, lhead16->raw_data_gpu(),
                                             logits.raw_data_gpu(), log_rows, cfg.vocab_size, hidden);
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (prof_p) std::fprintf(stderr, "[prefillprof] head=%.1fms\n", (pt_now() - pt_head0) * 1e3);
        return logits;
        }
    }
    if (const core::Tensor *lhead = model.lm_head_fp32()) {
        // FP32 logits via transposed-lm_head GEMV: lm_head_fp32 is [vocab, hidden].
        Tensor normed_f32({log_rows, hidden}, hal::DType::Float32);
        normed_f32.alloc_gpu();
        hal::convert_dtype(normed_f32.raw_data_gpu(), norm_ptr, log_rows * hidden, dtype,
                           hal::DType::Float32);
        Tensor logits({log_rows, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        hal::logits_gemv_f32(normed_f32.raw_data_gpu(), lhead->raw_data_gpu(), logits.raw_data_gpu(),
                             log_rows, cfg.vocab_size, hidden);
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        return logits;
    }
    Tensor logits({log_rows, cfg.vocab_size}, dtype);
    logits.alloc_gpu();
    core::blas::matmul(normed_last, model.lm_head()->data, logits);
    if (last_only) {
        Tensor out = logits.cast(hal::DType::Float32);
        out.syncToHost();
        return out;
    }
    return logits.cast(hal::DType::Float32);
}

Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
    return llama_forward_cuda_impl(model, token_ids, cache, false);
}

Tensor llama_forward_cuda_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache) {
    return llama_forward_cuda_impl(model, token_ids, cache, true);
}

#ifdef SERIUS_USE_CUDA
// --- CUDA Graph decode path ---
// The single-token decode step is a fixed sequence of ~700 kernels whose only
// per-token variation is the token id and the position. Both are written to
// device memory (tid_gpu / pos_gpu) before each replay, so the whole step can
// be captured once and replayed with cudaGraphLaunch — eliminating all launch
// overhead and inter-kernel gaps.
namespace {

struct GraphDecodeCtx {
    bool ready = false;          // tensors allocated, dims cached
    bool captured = false;       // graph instantiated
    const model::LLaMAModel *model = nullptr;
    const KVCache *cache = nullptr;
    // dims (for staleness check)
    size_t hidden = 0, inter = 0, kv_dim = 0, qkv_total = 0, n_layers = 0, vocab = 0;
    hal::DType dtype = hal::DType::Float16;
    // persistent device tensors (never re-allocated while ctx is valid)
    Tensor hidden_states_gpu, normed_buf, query_buf, key_buf, value_buf;
    Tensor attention_out, mlp_down;
    Tensor qf32_normed, qf32_attn, qf32_gu, qf32_sx, qkv_buf, gu_buf, qacc, logits_dev;
    int *tid_gpu = nullptr;
    int *pos_gpu = nullptr;
    uint64_t cache_inst = 0;    // KVCache::instance_id the graph was captured for
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t cap_stream = nullptr;
};

GraphDecodeCtx &graph_ctx() {
    static GraphDecodeCtx c;
    return c;
}

// (model, cache-instance) pairs whose graph capture already failed — don't
// retry every token; the eager path handles them. Uses instance_id because
// stack addresses are reused across turns.
struct GraphDisabledKey { const void *model; uint64_t cache_inst; };
GraphDisabledKey &graph_disabled() {
    static GraphDisabledKey k{nullptr, 0};
    return k;
}

void graph_ctx_destroy(GraphDecodeCtx &c) {
    if (c.exec) { cudaGraphExecDestroy(c.exec); c.exec = nullptr; }
    if (c.graph) { cudaGraphDestroy(c.graph); c.graph = nullptr; }
    if (c.tid_gpu) { cudaFree(c.tid_gpu); c.tid_gpu = nullptr; }
    if (c.pos_gpu) { cudaFree(c.pos_gpu); c.pos_gpu = nullptr; }
    if (c.cap_stream) { cudaStreamDestroy(c.cap_stream); c.cap_stream = nullptr; }
    c.ready = false;
    c.captured = false;
    c.model = nullptr;
    c.cache = nullptr;
    c.cache_inst = 0;
}

// Issues the full decode step on stream `s`. All pos/token dependence goes
// through ctx.tid_gpu / ctx.pos_gpu on device; no host-side pos math, no
// allocations, no synchronization — capture-safe. Only the quantized FP16
// path is supported (eligibility is checked by the caller).
void issue_decode_graph(GraphDecodeCtx &ctx, const model::LLaMAModel &model,
                        KVCache *cache, cudaStream_t s) {
    const auto &cfg = model.config();
    const auto &layers = model.layers();
    const size_t hidden = ctx.hidden, kv_dim = ctx.kv_dim, inter = ctx.inter;
    const size_t n_heads = cfg.num_attention_heads, n_kv_heads = cfg.num_key_value_heads;
    const size_t head_dim = hidden / n_heads;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    const hal::DType dtype = ctx.dtype;
    const hal::DType stream_dtype = hal::DType::Float32;
    const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    hal::gather_mixed(ctx.hidden_states_gpu.raw_data_gpu(),
                      model.embed_tokens()->data.raw_data_gpu(), ctx.tid_gpu, 1, hidden,
                      dtype, stream_dtype, s);
    ctx.hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);

    float *qacc_ptr = static_cast<float *>(ctx.qacc.raw_data_gpu());
    auto qproj = [&](const quant::QuantWeight &qw, const core::Tensor &x_f16, core::Tensor &out,
                     core::Tensor &x_f32) {
        if (dtype == hal::DType::Float16 && qw.dev_stride != 0) {
            // Fused FP16-in/FP16-out GEMV: skips both convert_dtype launches.
            const void *sx = nullptr;
            if (qw.dev_kind == 3 || qw.dev_kind == 4) {
                hal::block_sums16_f16(x_f16.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(),
                                      (int)(qw.in / 16), s);
                sx = ctx.qf32_sx.raw_data_gpu();
            }
            hal::gemv_quant_vec_f16io(x_f16.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in,
                                      qw.out, (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d,
                                      (int)qw.dev_m, out.raw_data_gpu(), true, s);
            out.mark_dirty(Tensor::DataLocation::GPU);
            return;
        }
        hal::convert_dtype(x_f32.raw_data_gpu(), x_f16.raw_data_gpu(), qw.in, dtype,
                           hal::DType::Float32, s);
        const void *sx = nullptr;
        if (qw.dev_kind == 3 || qw.dev_kind == 4) {
            hal::block_sums16_f32(x_f32.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(),
                                  (int)(qw.in / 16), s);
            sx = ctx.qf32_sx.raw_data_gpu();
        }
        hal::gemv_quant_vec_f32(x_f32.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in,
                                qw.out, (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d,
                                (int)qw.dev_m, qacc_ptr, s);
        hal::convert_dtype(out.raw_data_gpu(), qacc_ptr, qw.out, hal::DType::Float32, dtype, s);
        out.mark_dirty(Tensor::DataLocation::GPU);
    };

    const size_t esz = get_dtype_size(dtype);
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                           blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps,
                           dtype, s);
        ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);

        // Fused QKV GEMV -> q/k/v are contiguous slices of qkv_buf
        // ([q | k | v]); downstream kernels take raw pointers, so the three
        // DtoD slice copies are unnecessary — alias them directly.
        qproj(blk->qkv_w, ctx.normed_buf, ctx.qkv_buf, ctx.qf32_normed);
        char *qptr = static_cast<char *>(ctx.qkv_buf.raw_data_gpu());
        char *kptr = qptr + hidden * esz;
        char *vptr = kptr + kv_dim * esz;

        // Fused bias+RoPE(+scale) for Q/K; V bias is folded into the cache write.
        hal::rope_bias_devpos(qptr, head_dim, 1, n_heads, rope_base,
                              ctx.pos_gpu,
                              blk->q_bias ? blk->q_bias->data.raw_data_gpu() : nullptr,
                              (dtype != hal::DType::Float32) ? q_scale : 1.0F, dtype, s);
        hal::rope_bias_devpos(kptr, head_dim, 1, n_kv_heads, rope_base,
                              ctx.pos_gpu,
                              blk->k_bias ? blk->k_bias->data.raw_data_gpu() : nullptr,
                              1.0F, dtype, s);

        // Write the new K/V row into the cache at the device-side position.
        KVCacheLayer &cl = cache->layers[layer_idx];
        hal::kv_write_devpos(cl.k.raw_data_gpu(), kptr,
                             kv_dim * esz, ctx.pos_gpu, kv_dim * esz, s);
        hal::kv_write_bias_devpos(cl.v.raw_data_gpu(), vptr,
                                  blk->v_bias ? blk->v_bias->data.raw_data_gpu() : nullptr,
                                  kv_dim, ctx.pos_gpu, kv_dim, dtype, s);
        cl.k.mark_dirty(Tensor::DataLocation::GPU);
        cl.v.mark_dirty(Tensor::DataLocation::GPU);

        // Fused GQA decode attention: pos+1 valid rows read from device.
        hal::gqa_decode_attention_f16_devpos(
            qptr, cl.k.raw_data_gpu(), cl.v.raw_data_gpu(),
            ctx.attention_out.raw_data_gpu(), ctx.pos_gpu,
            (int)n_heads, (int)n_kv_heads, (int)head_dim, (int)kv_dim, s);
        ctx.attention_out.mark_dirty(Tensor::DataLocation::GPU);

        qproj(blk->o_w, ctx.attention_out, ctx.mlp_down, ctx.qf32_attn);
        hal::add_inplace_mixed(ctx.hidden_states_gpu.raw_data_gpu(), ctx.mlp_down.raw_data_gpu(),
                               hidden, stream_dtype, dtype, s);
        ctx.hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                           blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, 1,
                           eps, dtype, s);
        ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);

        // Fused gate|up + SwiGLU + down.
        qproj(blk->gate_up_w, ctx.normed_buf, ctx.gu_buf, ctx.qf32_normed);
        hal::swiglu(ctx.gu_buf.raw_data_gpu(), ctx.gu_buf.raw_data_gpu(),
                    (char *)ctx.gu_buf.raw_data_gpu() + inter * esz, inter, dtype, s);
        ctx.gu_buf.mark_dirty(Tensor::DataLocation::GPU);
        qproj(blk->down_w, ctx.gu_buf, ctx.mlp_down, ctx.qf32_gu);
        hal::add_inplace_mixed(ctx.hidden_states_gpu.raw_data_gpu(), ctx.mlp_down.raw_data_gpu(),
                               hidden, stream_dtype, dtype, s);
        ctx.hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
    }

    // Final norm + quantized lm_head straight into ctx.logits_dev (FP32).
    hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                       model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype, s);
    ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    const quant::QuantWeight *lq = model.lm_head_q();
    // FP16 x in, FP32 logits out — still skips the input convert_dtype.
    const void *sx = nullptr;
    if (lq->dev_kind == 3 || lq->dev_kind == 4) {
        hal::block_sums16_f16(ctx.normed_buf.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(),
                              (int)(lq->in / 16), s);
        sx = ctx.qf32_sx.raw_data_gpu();
    }
    hal::gemv_quant_vec_f16io(ctx.normed_buf.raw_data_gpu(), sx, lq->gpu_data, lq->dev_kind,
                              lq->in, lq->out, (int)lq->dev_stride, (int)lq->dev_qs,
                              (int)lq->dev_d, (int)lq->dev_m,
                              ctx.logits_dev.raw_data_gpu(), false, s);
    ctx.logits_dev.mark_dirty(Tensor::DataLocation::GPU);
}

}  // namespace

bool llama_forward_single_token_cuda_graph(const model::LLaMAModel &model, int token_id,
                                           size_t pos, KVCache *cache, Tensor &out_logits) {
    if (cache == nullptr || pos + 1 > 4096) return false;
    {
        size_t n = 0;
        if (getenv_s(&n, nullptr, 0, "SERIUS_NO_GRAPH") == 0 && n > 0) return false;
    }
    {
        const GraphDisabledKey &dk = graph_disabled();
        if (dk.model == &model && dk.cache_inst == cache->instance_id) return false;
    }
    const auto &cfg = model.config();
    const size_t hidden = cfg.hidden_size, inter = cfg.intermediate_size;
    const size_t kv_dim = cfg.num_key_value_heads * (hidden / cfg.num_attention_heads);
    const size_t qkv_total = hidden + 2 * kv_dim;
    const size_t n_layers = cfg.num_hidden_layers;
    hal::DType dtype = model.embed_tokens()->data.dtype();

    // Eligibility: FP16 activations, every projection packed on GPU, quantized
    // lm_head — anything else falls back to the eager path.
    if (dtype != hal::DType::Float16 || model.embed_tokens()->data.raw_data_gpu() == nullptr ||
        model.lm_head_q() == nullptr || model.lm_head_q()->gpu_data == nullptr ||
        model.lm_head_q()->dev_stride == 0 || model.lm_head_q()->out != cfg.vocab_size) {
        return false;
    }
    for (const auto &blk : model.layers()) {
        const auto *b = blk.get();
        if (!b->qkv_w.gpu_data || !b->gate_up_w.gpu_data || !b->o_w.gpu_data || !b->down_w.gpu_data ||
            b->qkv_w.dev_stride == 0 || b->gate_up_w.dev_stride == 0 || b->o_w.dev_stride == 0 ||
            b->down_w.dev_stride == 0) {
            return false;
        }
    }

    GraphDecodeCtx &ctx = graph_ctx();
    const bool stale = !ctx.ready || ctx.model != &model || ctx.cache != cache ||
                       ctx.cache_inst != cache->instance_id ||
                       ctx.hidden != hidden || ctx.inter != inter || ctx.kv_dim != kv_dim ||
                       ctx.n_layers != n_layers || ctx.vocab != cfg.vocab_size || ctx.dtype != dtype;
    if (stale) {
        graph_ctx_destroy(ctx);
        ctx.model = &model;
        ctx.cache = cache;
        ctx.cache_inst = cache->instance_id;
        ctx.hidden = hidden; ctx.inter = inter; ctx.kv_dim = kv_dim;
        ctx.qkv_total = qkv_total; ctx.n_layers = n_layers; ctx.vocab = cfg.vocab_size;
        ctx.dtype = dtype;

        ctx.hidden_states_gpu = Tensor({1, hidden}, hal::DType::Float32); ctx.hidden_states_gpu.alloc_gpu();
        ctx.normed_buf = Tensor({1, hidden}, dtype); ctx.normed_buf.alloc_gpu();
        ctx.query_buf = Tensor({1, hidden}, dtype); ctx.query_buf.alloc_gpu();
        ctx.key_buf = Tensor({1, kv_dim}, dtype); ctx.key_buf.alloc_gpu();
        ctx.value_buf = Tensor({1, kv_dim}, dtype); ctx.value_buf.alloc_gpu();
        ctx.attention_out = Tensor({1, hidden}, dtype); ctx.attention_out.alloc_gpu();
        ctx.mlp_down = Tensor({1, hidden}, dtype); ctx.mlp_down.alloc_gpu();
        ctx.qf32_normed = Tensor({hidden}, hal::DType::Float32); ctx.qf32_normed.alloc_gpu();
        ctx.qf32_attn = Tensor({hidden}, hal::DType::Float32); ctx.qf32_attn.alloc_gpu();
        ctx.qf32_gu = Tensor({inter}, hal::DType::Float32); ctx.qf32_gu.alloc_gpu();
        ctx.qf32_sx = Tensor({(std::max(hidden, inter) + 15) / 16 + 16}, hal::DType::Float32);
        ctx.qf32_sx.alloc_gpu();
        ctx.qkv_buf = Tensor({1, qkv_total}, dtype); ctx.qkv_buf.alloc_gpu();
        ctx.gu_buf = Tensor({1, 2 * inter}, dtype); ctx.gu_buf.alloc_gpu();
        const size_t qacc_cap = std::max({cfg.vocab_size, qkv_total, 2 * inter, hidden, inter});
        ctx.qacc = Tensor({qacc_cap}, hal::DType::Float32); ctx.qacc.alloc_gpu();
        ctx.logits_dev = Tensor({1, cfg.vocab_size}, hal::DType::Float32);
        ctx.logits_dev.alloc_gpu();

        CHECK_CUDA(cudaMalloc(&ctx.tid_gpu, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&ctx.pos_gpu, sizeof(int)));
        CHECK_CUDA(cudaStreamCreateWithFlags(&ctx.cap_stream, cudaStreamNonBlocking));
        ctx.ready = true;
    }

    // Device inputs updated before every launch (outside the graph).
    const int pos_i = static_cast<int>(pos);
    CHECK_CUDA(cudaMemcpyAsync(ctx.tid_gpu, &token_id, sizeof(int), cudaMemcpyHostToDevice, 0));
    CHECK_CUDA(cudaMemcpyAsync(ctx.pos_gpu, &pos_i, sizeof(int), cudaMemcpyHostToDevice, 0));

    if (!ctx.captured) {
        // Capture on a side stream; any failure falls back to the eager path.
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaError_t cap_err = cudaSuccess;
        try {
            CHECK_CUDA(cudaStreamBeginCapture(ctx.cap_stream, cudaStreamCaptureModeGlobal));
            issue_decode_graph(ctx, model, cache, ctx.cap_stream);
            cap_err = cudaStreamEndCapture(ctx.cap_stream, &ctx.graph);
        } catch (...) {
            cudaStreamEndCapture(ctx.cap_stream, &ctx.graph);
            cudaGetLastError();
            graph_ctx_destroy(ctx);
            graph_disabled() = {&model, cache->instance_id};
            return false;
        }
        if (cap_err != cudaSuccess || ctx.graph == nullptr ||
            cudaGraphInstantiateWithFlags(&ctx.exec, ctx.graph, 0ULL) != cudaSuccess) {
            cudaGetLastError();
            graph_ctx_destroy(ctx);
            graph_disabled() = {&model, cache->instance_id};
            return false;
        }
        ctx.captured = true;
    }

    CHECK_CUDA(cudaGraphLaunch(ctx.exec, 0));
    for (size_t i = 0; i < cache->layers.size(); ++i) {
        cache->layers[i].current_len = pos + 1;
    }
    CHECK_CUDA(cudaStreamSynchronize(0));

    out_logits = Tensor({1, cfg.vocab_size}, hal::DType::Float32);
    out_logits.alloc_gpu();
    // Host copy for the CPU fallback sampler; device copy so the GPU top-k
    // preselection (raw_data_gpu) keeps working on the returned tensor.
    CHECK_CUDA(cudaMemcpy(out_logits.data(), ctx.logits_dev.raw_data_gpu(),
                          cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_logits.raw_data_gpu(), ctx.logits_dev.raw_data_gpu(),
                          cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
    out_logits.mark_dirty(Tensor::DataLocation::GPU);
    return true;
}
#endif

Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache) {
    static const bool gpu_time = [] {
        size_t len = 0;
        return getenv_s(&len, nullptr, 0, "SERIUS_GPU_TIME") == 0 && len > 0;
    }();
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (gpu_time) { cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventRecord(ev0); }
    const auto &cfg = model.config();
    const size_t n_layers = cfg.num_hidden_layers, hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    const hal::DType stream_dtype = hal::DType::Float32;

    Tensor hidden_states_gpu({1, hidden}, stream_dtype); hidden_states_gpu.alloc_gpu();
    // Persistent device buffer for the token id: avoids a cudaMalloc +
    // cudaDeviceSynchronize + cudaFree round-trip on every decode step.
    static int *tid_gpu = [] {
        int *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, sizeof(int)));
        return p;
    }();
    int tid = token_id;
    CHECK_CUDA(cudaMemcpyAsync(tid_gpu, &tid, sizeof(int), cudaMemcpyHostToDevice, 0));
    hal::gather_mixed(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), tid_gpu, 1, hidden, dtype, stream_dtype);
    hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);

    Tensor normed_buf({1, hidden}, dtype); normed_buf.alloc_gpu();
    Tensor query_buf({1, hidden}, dtype); query_buf.alloc_gpu();
    Tensor key_buf({1, kv_dim}, dtype); key_buf.alloc_gpu();
    Tensor value_buf({1, kv_dim}, dtype); value_buf.alloc_gpu();
    Tensor attention_out({1, hidden}, dtype); attention_out.alloc_gpu();
    Tensor mlp_gate({1, inter}, dtype); mlp_gate.alloc_gpu();
    Tensor mlp_up({1, inter}, dtype); mlp_up.alloc_gpu();
    Tensor mlp_down({1, hidden}, dtype); mlp_down.alloc_gpu();
    // FP32 scratch inputs for the quantized decode GEMVs (converted once per
    // projection source instead of once per weight row).
    Tensor qf32_normed({hidden}, hal::DType::Float32); qf32_normed.alloc_gpu();
    Tensor qf32_attn({hidden}, hal::DType::Float32); qf32_attn.alloc_gpu();
    Tensor qf32_gu({inter}, hal::DType::Float32); qf32_gu.alloc_gpu();
    // Per-16-element sums of the FP32 GEMV inputs (Q4_0/Q4_1 affine fold).
    // Largest `in` is max(hidden, inter); in/16 floats suffice.
    Tensor qf32_sx({(std::max(hidden, inter) + 15) / 16 + 16}, hal::DType::Float32);
    qf32_sx.alloc_gpu();
    // Reused per-call scratch for the quantized decode path: avoids ~200
    // device allocations per generated token (they were previously created
    // inside the layer loop and inside every qproj call).
    const size_t qkv_total = hidden + 2 * kv_dim;
    Tensor qkv_buf({1, qkv_total}, dtype); qkv_buf.alloc_gpu();
    Tensor gu_buf({1, 2 * inter}, dtype); gu_buf.alloc_gpu();
    const size_t qacc_cap = std::max({cfg.vocab_size, qkv_total, 2 * inter, hidden, inter});
    Tensor qacc({qacc_cap}, hal::DType::Float32);
    qacc.alloc_gpu();

    const auto &layers = model.layers();
    if (cache && (cache->empty() || cache->layers.size() != layers.size())) {
        size_t max_seq_len = 4096;
        cache->layers.resize(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            cache->layers[i].k = Tensor({max_seq_len, kv_dim}, dtype);
            cache->layers[i].v = Tensor({max_seq_len, kv_dim}, dtype);
            cache->layers[i].k.alloc_gpu();
            cache->layers[i].v.alloc_gpu();
            cache->layers[i].current_len = 0;
        }
    }

    // Decode attends to the full KV history: pass an all-zero mask so the
    // kernel's nullptr fallback (which applies causal masking) is not used.
    // Allocated once: the mask is all zeros and never changes.
    static float *decode_mask_gpu = [] {
        float *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, 4096 * sizeof(float)));
        CHECK_CUDA(cudaMemset(p, 0, 4096 * sizeof(float)));
        return p;
    }();

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        // M==1 projections as custom streaming GEMVs (one kernel each, weight
        // read exactly once) — much faster than cuBLAS heuristics for GEMV.
        auto proj = [&](const core::Tensor &x, const core::Tensor &w, core::Tensor &out, size_t K, size_t N) {
            if (dtype == hal::DType::Float16 && w.is_contiguous() && w.raw_data_gpu() != nullptr) {
                // Split-K FP16 GEMV with FP32 accumulation.
                Tensor acc({N}, hal::DType::Float32); acc.alloc_gpu();
                CHECK_CUDA(cudaMemsetAsync(acc.raw_data_gpu(), 0, N * sizeof(float), 0));
                hal::gemv_kn_sk_f16(x.raw_data_gpu(), w.raw_data_gpu(), acc.raw_data_gpu(), K, N);
                hal::convert_dtype(out.raw_data_gpu(), acc.raw_data_gpu(), N, hal::DType::Float32, dtype);
                out.mark_dirty(Tensor::DataLocation::GPU);
            } else {
                core::blas::matmul(x, w, out);
            }
        };
        // Packed GGUF weights (Q4_0/Q4_1/Q8_0) resident on the GPU: dequantize
        // on the fly inside the GEMV — ~4x less DRAM traffic than FP16.
        const bool use_quant = (dtype == hal::DType::Float16) && blk->qkv_w.gpu_data &&
            blk->gate_up_w.gpu_data && blk->o_w.gpu_data && blk->down_w.gpu_data;
        // Quantized GEMV consumes FP32 x (converted once per projection input
        // into the provided scratch buffer) and writes FP32 y directly.
        auto qproj = [&](const quant::QuantWeight &qw, const core::Tensor &x_f16, core::Tensor &out,
                         core::Tensor &x_f32) {
            if (qw.dev_stride != 0 && dtype == hal::DType::Float16) {
                // Fused FP16-in/FP16-out GEMV: skips both convert_dtype launches.
                const void *sx = nullptr;
                if (qw.dev_kind == 3 || qw.dev_kind == 4) {
                    hal::block_sums16_f16(x_f16.raw_data_gpu(), qf32_sx.raw_data_gpu(),
                                          (int)(qw.in / 16));
                    sx = qf32_sx.raw_data_gpu();
                }
                hal::gemv_quant_vec_f16io(x_f16.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind,
                                          qw.in, qw.out, (int)qw.dev_stride, (int)qw.dev_qs,
                                          (int)qw.dev_d, (int)qw.dev_m, out.raw_data_gpu(),
                                          true);
                out.mark_dirty(Tensor::DataLocation::GPU);
                return;
            }
            hal::convert_dtype(x_f32.raw_data_gpu(), x_f16.raw_data_gpu(), qw.in, dtype,
                               hal::DType::Float32);
            float *acc_ptr = static_cast<float *>(qacc.raw_data_gpu());
            if (qw.dev_stride != 0) {
                const void *sx = nullptr;
                if (qw.dev_kind == 3 || qw.dev_kind == 4) {
                    hal::block_sums16_f32(x_f32.raw_data_gpu(), qf32_sx.raw_data_gpu(),
                                          (int)(qw.in / 16));
                    sx = qf32_sx.raw_data_gpu();
                }
                hal::gemv_quant_vec_f32(x_f32.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in,
                                        qw.out, (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d,
                                        (int)qw.dev_m, acc_ptr);
            } else {
                hal::gemv_quant_f32(x_f32.raw_data_gpu(), qw.gpu_data, (int)qw.type, qw.in, qw.out,
                                    acc_ptr);
            }
            hal::convert_dtype(out.raw_data_gpu(), acc_ptr, qw.out, hal::DType::Float32, dtype);
            out.mark_dirty(Tensor::DataLocation::GPU);
        };
        if (use_quant) {
            // Fused QKV: rows are [q | k | v], slice into the RoPE inputs.
            const size_t esz = get_dtype_size(dtype);
            qproj(blk->qkv_w, normed_buf, qkv_buf, qf32_normed);
            CHECK_CUDA(cudaMemcpyAsync(query_buf.raw_data_gpu(), qkv_buf.raw_data_gpu(),
                                       hidden * esz, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync(key_buf.raw_data_gpu(),
                                       (char *)qkv_buf.raw_data_gpu() + hidden * esz,
                                       kv_dim * esz, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync(value_buf.raw_data_gpu(),
                                       (char *)qkv_buf.raw_data_gpu() + (hidden + kv_dim) * esz,
                                       kv_dim * esz, cudaMemcpyDeviceToDevice, 0));
            query_buf.mark_dirty(Tensor::DataLocation::GPU);
            key_buf.mark_dirty(Tensor::DataLocation::GPU);
            value_buf.mark_dirty(Tensor::DataLocation::GPU);
        } else {
            proj(normed_buf, blk->q_proj->data, query_buf, hidden, hidden);
            proj(normed_buf, blk->k_proj->data, key_buf, hidden, kv_dim);
            proj(normed_buf, blk->v_proj->data, value_buf, hidden, kv_dim);
        }
        if (blk->q_bias) { hal::add_row_bias(query_buf.raw_data_gpu(), blk->q_bias->data.raw_data_gpu(), 1, hidden, dtype); }
        if (blk->k_bias) { hal::add_row_bias(key_buf.raw_data_gpu(), blk->k_bias->data.raw_data_gpu(), 1, kv_dim, dtype); }
        if (blk->v_bias) { hal::add_row_bias(value_buf.raw_data_gpu(), blk->v_bias->data.raw_data_gpu(), 1, kv_dim, dtype); }
        hal::rope(query_buf.raw_data_gpu(), head_dim, 1, n_heads, rope_base, static_cast<int>(pos), dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, 1, n_kv_heads, rope_base, static_cast<int>(pos), dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU); key_buf.mark_dirty(Tensor::DataLocation::GPU);
        const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        if (dtype != hal::DType::Float32) {
            hal::scale_inplace(query_buf.raw_data_gpu(), hidden, q_scale, dtype);
        }

        size_t effective_len = pos + 1;
        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            size_t bytes = kv_dim * get_dtype_size(dtype);
            CHECK_CUDA(cudaMemcpyAsync((char*)cl.k.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype), 
                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync((char*)cl.v.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype), 
                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            cl.k.mark_dirty(Tensor::DataLocation::GPU);
            cl.v.mark_dirty(Tensor::DataLocation::GPU);
            cl.current_len = effective_len;
        }

        if (dtype == hal::DType::Float16 && cache && effective_len <= 4096) {
            // Fused GQA decode attention: single kernel reads the KV cache
            // directly (query already RoPE'd and pre-scaled above).
            hal::gqa_decode_attention_f16(
                query_buf.raw_data_gpu(),
                cache->layers[layer_idx].k.raw_data_gpu(),
                cache->layers[layer_idx].v.raw_data_gpu(),
                attention_out.raw_data_gpu(),
                (int)effective_len, (int)n_heads, (int)n_kv_heads, (int)head_dim,
                (int)kv_dim);
            attention_out.mark_dirty(Tensor::DataLocation::GPU);
        } else {
        Tensor q_hs = query_buf.reshape({1, n_heads, head_dim}).transpose(0, 1).contiguous();
        q_hs.alloc_gpu();

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
            k_rep.alloc_gpu(); v_rep.alloc_gpu();

            hal::repeat_kv(k_rep.raw_data_gpu(), k_in.raw_data_gpu(), effective_len, n_heads, n_kv_heads, head_dim, dtype);
            hal::repeat_kv(v_rep.raw_data_gpu(), v_in.raw_data_gpu(), effective_len, n_heads, n_kv_heads, head_dim, dtype);

            // Transpose to (n_heads, batch_seq, head_dim) for batched_matmul
            k_hs = k_rep.transpose(0, 1).contiguous();
            v_hs = v_rep.transpose(0, 1).contiguous();
            k_hs.alloc_gpu(); v_hs.alloc_gpu();
        } else {
            k_hs = k_full.reshape({effective_len, n_heads, head_dim}).transpose(0, 1).contiguous();
            v_hs = v_full.reshape({effective_len, n_heads, head_dim}).transpose(0, 1).contiguous();
            k_hs.alloc_gpu(); v_hs.alloc_gpu();
        }

        Tensor scores({n_heads, 1, effective_len}, dtype); scores.alloc_gpu();
        core::blas::batched_matmul(q_hs, k_hs.transpose(1, 2), scores);

        float scale = (dtype == hal::DType::Float32) ? (1.0F / std::sqrt(static_cast<float>(head_dim))) : 1.0F;
        hal::fused_masked_softmax(scores.raw_data_gpu(), decode_mask_gpu, n_heads, 1, effective_len, dtype, scale);
        scores.mark_dirty(Tensor::DataLocation::GPU);

        Tensor context_heads({n_heads, 1, head_dim}, dtype); context_heads.alloc_gpu();
        core::blas::batched_matmul(scores, v_hs, context_heads);
        attention_out = context_heads.transpose(0, 1).contiguous().reshape({1, hidden});
        attention_out.alloc_gpu();
        }

        if (use_quant) {
            qproj(blk->o_w, attention_out, mlp_down, qf32_attn);
        } else {
            proj(attention_out, blk->o_proj->data, mlp_down, hidden, hidden);
        }
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), hidden, stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        if (use_quant) {
            // Fused gate|up: swiglu in place on the first half, down reads it.
            const size_t esz = get_dtype_size(dtype);
            qproj(blk->gate_up_w, normed_buf, gu_buf, qf32_normed);
            hal::swiglu(gu_buf.raw_data_gpu(), gu_buf.raw_data_gpu(),
                        (char *)gu_buf.raw_data_gpu() + inter * esz, inter, dtype);
            gu_buf.mark_dirty(Tensor::DataLocation::GPU);
            qproj(blk->down_w, gu_buf, mlp_down, qf32_gu);
        } else {
            proj(normed_buf, blk->gate_proj->data, mlp_gate, hidden, inter);
            proj(normed_buf, blk->up_proj->data, mlp_up, hidden, inter);
            hal::swiglu(mlp_gate.raw_data_gpu(), mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), inter, dtype);
            mlp_gate.mark_dirty(Tensor::DataLocation::GPU);
            proj(mlp_gate, blk->down_proj->data, mlp_down, inter, hidden);
        }
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), hidden, stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
    }
    hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(), model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    // Quantized lm_head (GGUF): single dequant GEMV straight into FP32 logits.
    if (const quant::QuantWeight *lq = model.lm_head_q();
        lq != nullptr && lq->gpu_data != nullptr && lq->out == cfg.vocab_size &&
        dtype == hal::DType::Float16) {
        // normed_buf is the final RMSNorm output; reuse the FP32 scratch.
        hal::convert_dtype(qf32_normed.raw_data_gpu(), normed_buf.raw_data_gpu(), hidden, dtype,
                           hal::DType::Float32);
        Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        if (lq->dev_stride != 0) {
            const void *sx = nullptr;
            if (lq->dev_kind == 3 || lq->dev_kind == 4) {
                hal::block_sums16_f32(qf32_normed.raw_data_gpu(), qf32_sx.raw_data_gpu(),
                                      (int)(lq->in / 16));
                sx = qf32_sx.raw_data_gpu();
            }
            hal::gemv_quant_vec_f32(qf32_normed.raw_data_gpu(), sx, lq->gpu_data, lq->dev_kind, lq->in,
                                    lq->out, (int)lq->dev_stride, (int)lq->dev_qs, (int)lq->dev_d,
                                    (int)lq->dev_m, logits.raw_data_gpu());
        } else {
            hal::gemv_quant_f32(qf32_normed.raw_data_gpu(), lq->gpu_data, (int)lq->type, lq->in,
                                lq->out, logits.raw_data_gpu());
        }
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (gpu_time) {
            cudaEventRecord(ev1); cudaEventSynchronize(ev1);
            float ms; cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        }
        return logits;
    }
    if (dtype == hal::DType::Float16) {
        if (const core::Tensor *lhead16 = model.lm_head_fp16_t()) {
        Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        core::blas::logits_gemv_mixed_cublas(normed_buf.raw_data_gpu(), lhead16->raw_data_gpu(),
                                             logits.raw_data_gpu(), 1, cfg.vocab_size, hidden);
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (gpu_time) {
            cudaEventRecord(ev1); cudaEventSynchronize(ev1);
            float ms; cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        }
        return logits;
        }
    }
    if (const core::Tensor *lhead = model.lm_head_fp32()) {
        Tensor normed_f32({1, hidden}, hal::DType::Float32);
        normed_f32.alloc_gpu();
        hal::convert_dtype(normed_f32.raw_data_gpu(), normed_buf.raw_data_gpu(), hidden, dtype,
                           hal::DType::Float32);
        Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        hal::logits_gemv_f32(normed_f32.raw_data_gpu(), lhead->raw_data_gpu(), logits.raw_data_gpu(),
                             1, cfg.vocab_size, hidden);
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (gpu_time) {
            cudaEventRecord(ev1); cudaEventSynchronize(ev1);
            float ms; cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        }
        return logits;
    }
    Tensor logits({1, cfg.vocab_size}, dtype); logits.alloc_gpu();
    core::blas::matmul(normed_buf, model.lm_head()->data, logits);
    auto ret = logits.cast(hal::DType::Float32);
    if (gpu_time) {
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    return ret;
}
#endif

} // namespace infer
} // namespace serius
