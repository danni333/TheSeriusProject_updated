// CUDA inference path for LLaMA — split from llama_inference.cpp.
// Contains the cuBLAS prefill, fused prefill weights, CUDA-graph decode
// step and single-token CUDA forward. Everything is inside
// SERIUS_USE_CUDA guards, matching the original layout.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "llama_inference.h"
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
#include "../distributed/pipeline.h"
#ifdef SERIUS_USE_CUDA
#include "../blas/matmul_cuda.h"
#include "../hal/device.h"
#include "../hal/fused_kernels.h"
#endif
namespace serius {
namespace infer {

using core::Tensor;

// Non-empty env check: getenv_s on MSVC, getenv elsewhere.
[[maybe_unused]] static bool env_nonempty(const char *name)
{
#if defined(_MSC_VER)
    size_t n = 0;
    return getenv_s(&n, nullptr, 0, name) == 0 && n > 0;
#else
    const char *v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
#endif
}

[[maybe_unused]] static size_t get_dtype_size(hal::DType dtype)
{
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

// --- CUDA Implementations ---
#ifdef SERIUS_USE_CUDA
// last_only: compute logits for the final position only ({1, vocab} instead of
// {seq_len, vocab}) — chat/server callers only ever read the last row, and the
// full logits tensor is seq_len*vocab*4 bytes of host memset + D2H copy.

// Initial per-layer GPU KV capacity. Grows on demand via
// ensure_kv_capacity_gpu — sizing to the real context instead of a fixed
// 4096 saves VRAM on short requests and allows >4096-token contexts on the
// eager decode path (the CUDA-graph path stays capped at 4096).
constexpr size_t KV_INIT_CAP = 1024;

/// Grow a GPU KV cache layer to hold `needed` positions, preserving the
/// `current_len` valid rows. Returns true when the buffers were reallocated —
/// callers embedding raw KV pointers (the decode graph) must re-capture.
/// Works for both arena-backed views and standalone tensors: the old storage
/// is simply dropped (arena slabs are owned by KVCache::gpu_arena).
bool ensure_kv_capacity_gpu(KVCacheLayer &cl, size_t needed, size_t kv_dim, hal::DType dtype)
{
    const size_t cap = cl.k.shape().empty() ? 0 : cl.k.shape()[0];
    if (needed <= cap) {
        return false;
    }
    const size_t new_cap = std::max({needed, cap * 2, KV_INIT_CAP});
    Tensor nk = Tensor::gpu_only({new_cap, kv_dim}, dtype);
    Tensor nv = Tensor::gpu_only({new_cap, kv_dim}, dtype);
    const size_t esz = get_dtype_size(dtype);
    const size_t rows = std::min(cl.current_len, cap);
    if (cl.k.raw_data_gpu() != nullptr && cl.v.raw_data_gpu() != nullptr && rows > 0) {
        CHECK_CUDA(
            cudaMemcpyAsync(nk.raw_data_gpu(), cl.k.raw_data_gpu(), rows * kv_dim * esz, cudaMemcpyDeviceToDevice, 0));
        CHECK_CUDA(
            cudaMemcpyAsync(nv.raw_data_gpu(), cl.v.raw_data_gpu(), rows * kv_dim * esz, cudaMemcpyDeviceToDevice, 0));
    }
    cl.k = std::move(nk);
    cl.v = std::move(nv);
    return true;
}

// All-zero attention mask for decode, grown on demand: the eager path needs
// `effective_len` floats and contexts can now exceed the old fixed 4096.
float *decode_mask_for(size_t len)
{
    static float *p = nullptr;
    static size_t cap = 0;
    if (len > cap) {
        if (p != nullptr) {
            cudaFree(p);
        }
        cap = std::max(len, (size_t)4096);
        CHECK_CUDA(cudaMalloc(&p, cap * sizeof(float)));
        CHECK_CUDA(cudaMemset(p, 0, cap * sizeof(float)));
    }
    return p;
}

// Fused dense QKV / gate|up weights for batched prefill: 4 GEMMs per layer
// instead of 7 (better cuBLAS efficiency at M=seq_len). Built once per model
// on GPU; the separate dense q/k/v/gate/up tensors are then released — decode
// uses the quantized weights whenever all four quant tensors are resident
// (same gate as the decode `use_quant` path), so the dense copies are dead.
struct FusedPrefillW
{
    std::vector<core::Tensor> qkv;  // {hidden, hidden + 2*kv_dim} per layer
    std::vector<core::Tensor> gu;   // {hidden, 2*inter} per layer
    std::vector<char> fused;        // per-layer flag
    size_t hidden = 0, kv_dim = 0, inter = 0;
    const void *fingerprint = nullptr;
};

static const FusedPrefillW *fused_prefill_weights(const model::LLaMAModel &model, hal::DType dtype)
{
    static const bool no_quant = env_nonempty("SERIUS_NO_QUANT");
    static const bool no_fuse = env_nonempty("SERIUS_NO_FUSE");
    if (no_quant || no_fuse || dtype != hal::DType::Float16) return nullptr;

    const auto &cfg = model.config();
    const size_t hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t kv_dim = cfg.num_key_value_heads * (hidden / n_heads);
    const size_t inter = cfg.intermediate_size;
    const void *fp = model.embed_tokens()->data.raw_data_gpu();
    if (!fp) return nullptr;

    static std::unordered_map<const model::LLaMAModel *, FusedPrefillW> cache;
    if (cache.size() > 2) cache.clear();  // bound VRAM (~1.7 GB fused per model)
    auto it = cache.find(&model);
    if (it != cache.end() && it->second.fingerprint == fp && it->second.hidden == hidden &&
        it->second.kv_dim == kv_dim && it->second.inter == inter && it->second.qkv.size() == model.layers().size()) {
        return &it->second;
    }
    cache.erase(&model);

    auto ok_shape = [](const core::Tensor &t, size_t r, size_t c) {
        return t.dtype() == hal::DType::Float16 && t.shape().size() == 2 && t.shape()[0] == r && t.shape()[1] == c &&
               t.raw_data_gpu() != nullptr;
    };

    FusedPrefillW w;
    w.hidden = hidden;
    w.kv_dim = kv_dim;
    w.inter = inter;
    w.fingerprint = fp;
    const auto &layers = model.layers();
    w.qkv.resize(layers.size());
    w.gu.resize(layers.size());
    w.fused.assign(layers.size(), 0);

    size_t built = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto *blk = layers[i].get();  // mutable: unique_ptr yields LLaMABlock*
        // Decode must be able to run fully quantized before we drop the dense
        // originals (mirrors the use_quant gate in llama_forward_single_token_cuda).
        const bool quant_resident =
            blk->qkv_w.gpu_data && blk->gate_up_w.gpu_data && blk->o_w.gpu_data && blk->down_w.gpu_data;
        if (!quant_resident) continue;
        if (!ok_shape(blk->q_proj->data, hidden, hidden) || !ok_shape(blk->k_proj->data, hidden, kv_dim) ||
            !ok_shape(blk->v_proj->data, hidden, kv_dim) || !ok_shape(blk->gate_proj->data, hidden, inter) ||
            !ok_shape(blk->up_proj->data, hidden, inter))
            continue;

        auto qkv_t = core::Tensor::gpu_only({hidden, hidden + 2 * kv_dim}, dtype);
        hal::concat_rows_f16(qkv_t.raw_data_gpu(), blk->q_proj->data.raw_data_gpu(), blk->k_proj->data.raw_data_gpu(),
                             blk->v_proj->data.raw_data_gpu(), hidden, hidden, kv_dim, kv_dim, 0);
        auto gu_t = core::Tensor::gpu_only({hidden, 2 * inter}, dtype);
        hal::concat_rows_f16(gu_t.raw_data_gpu(), blk->gate_proj->data.raw_data_gpu(),
                             blk->up_proj->data.raw_data_gpu(), nullptr, hidden, inter, inter, 0, 0);
        w.qkv[i] = std::move(qkv_t);
        w.gu[i] = std::move(gu_t);
        w.fused[i] = 1;
        ++built;
        blk->q_proj->data = core::Tensor();
        blk->k_proj->data = core::Tensor();
        blk->v_proj->data = core::Tensor();
        blk->gate_proj->data = core::Tensor();
        blk->up_proj->data = core::Tensor();
    }
    if (built == 0) {
        cache.erase(&model);
        return nullptr;
    }
    // The dense tensors freed above land in the backend's 1 GiB free-list
    // cache instead of returning to the driver — flush it so fused weights
    // don't stack on top of them and overcommit VRAM.
    if (hal::IBackend *be = hal::get_gpu_backend()) be->flush_cache();
    return &cache.emplace(&model, std::move(w)).first->second;
}

// Batched prefill straight from packed quantized weights: when every layer
// projection is resident on the GPU in the repacked QuantWeight layout, each
// weight is dequantized into a small FP16 scratch and transposed for cuBLAS —
// the dense FP16 copies (which would ~double the weight footprint) are never
// needed. Decode already runs fully quantized, so this also removes the
// prefill/decode weight mismatch for GGUF models.
// Range-limited variant of quant_prefill_ready: partial offload only uploads
// the packed weights of the GPU prefix layers, so readiness must be checked
// per range (suffix layers legitimately stay host-only).
static bool quant_prefill_ready_range(const model::LLaMAModel &model, size_t layer_begin, size_t layer_end)
{
    if (model.layers().empty() || layer_begin >= layer_end) return false;
    for (size_t i = layer_begin; i < layer_end && i < model.layers().size(); ++i) {
        const auto *b = model.layers()[i].get();
        const quant::QuantWeight *ws[] = {&b->qkv_w, &b->gate_up_w, &b->o_w, &b->down_w};
        for (const quant::QuantWeight *qw : ws) {
            if (!qw->valid() || qw->dev_stride == 0 || qw->gpu_data == nullptr) {
                return false;
            }
        }
    }
    return true;
}

// Grow-only scratch for the dequant -> transpose -> GEMM prefill path, sized
// to the largest single fused weight (gate|up). Reused across layers, calls
// and models (a bigger model simply grows it once).
struct QuantPrefillScratch
{
    core::Tensor dq;    // [out, in] FP16 dequant output
    core::Tensor dq_t;  // [in, out] FP16 weight layout cuBLAS expects
    size_t cap = 0;     // element capacity of each buffer
};

static QuantPrefillScratch &quant_prefill_scratch()
{
    static QuantPrefillScratch s;
    return s;
}

// Dequantize qw ([out,in] packed) into scratch, transpose to [in,out], and
// run x[*,in] @ w^T -> out[*,out] through cuBLAS (FP16 in, FP32 compute).
// Dequantized-FP16 weight cache for batched prefill. quant_prefill_gemm
// used to re-dequantize + transpose every packed weight on EVERY prefill;
// on multi-hundred-token prompts that traffic dominates the GEMM itself.
// Entries are keyed by a fingerprint of dims/type/layout plus sampled
// packed bytes, store the transposed [in,out] FP16 matrix, and are
// allocated lazily — on VRAM exhaustion we stop caching and keep the
// per-call path (correctness preserved).
namespace {
struct PrefillDQCache
{
    struct Entry
    {
        core::Tensor t;  // dense [in,out] in `dtype`
    };
    std::unordered_map<uint64_t, Entry> map;
    std::mutex mu;
};

PrefillDQCache &prefill_dq_cache()
{
    static PrefillDQCache c;
    return c;
}

uint64_t qw_cache_key(const quant::QuantWeight &qw, hal::DType dtype)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(qw.type));
    mix(qw.in);
    mix(qw.out);
    mix(qw.dev_stride);
    mix(static_cast<uint64_t>(qw.dev_kind));
    mix(static_cast<uint64_t>(dtype));
    if (qw.bytes.empty()) {
        mix(reinterpret_cast<uintptr_t>(qw.gpu_data));  // no host bytes: pointer disambiguates
    } else {
        const size_t n = qw.bytes.size();
        auto sample = [&](size_t off) {
            if (off + 8 <= n) {
                uint64_t v = 0;
                std::memcpy(&v, qw.bytes.data() + off, 8);
                mix(v);
            }
        };
        sample(0);
        if (n > 16) {
            sample(n / 2 - 4);
            sample(n - 8);
        }
    }
    return h;
}
}  // namespace

static void quant_prefill_gemm(const quant::QuantWeight &qw, const core::Tensor &x, core::Tensor &out, hal::DType dtype,
                               QuantPrefillScratch &sc)
{
    const size_t elems = qw.in * qw.out;
    const uint64_t key = qw_cache_key(qw, dtype);

    {
        auto &cache = prefill_dq_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        auto it = cache.map.find(key);
        if (it != cache.map.end()) {
            core::Tensor w_t = core::Tensor::gpu_view(it->second.t.raw_data_gpu(), {qw.in, qw.out}, dtype);
            core::blas::matmul(x, w_t, out);
            return;
        }
    }

    if (sc.cap < elems) {
        sc.dq = core::Tensor::gpu_only({elems}, hal::DType::Float16);
        sc.dq_t = core::Tensor::gpu_only({elems}, hal::DType::Float16);
        sc.cap = elems;
    }
    hal::dequant_qw_f16(qw.gpu_data, qw.dev_kind, (int)qw.in, (int)qw.out, (int)qw.dev_stride, (int)qw.dev_qs,
                        (int)qw.dev_d, (int)qw.dev_m, sc.dq.raw_data_gpu());
    // Transpose [out,in] -> [in,out] with the layout-copy kernel.
    core::Tensor dq_view = core::Tensor::gpu_view(sc.dq.raw_data_gpu(), {qw.out, qw.in}, dtype);
    core::Tensor dq_t_view = dq_view.transpose(0, 1);  // {in,out}, strides {1,in}
    hal::copy_layout(sc.dq_t.raw_data_gpu(), dq_view.raw_data_gpu(), dq_t_view.shape().data(),
                     dq_t_view.strides().data(), dq_t_view.ndim(), dtype);
    core::Tensor w_t = core::Tensor::gpu_view(sc.dq_t.raw_data_gpu(), {qw.in, qw.out}, dtype);
    core::blas::matmul(x, w_t, out);

    try {
        core::Tensor t = core::Tensor::gpu_only({qw.in, qw.out}, dtype);
        CHECK_CUDA(cudaMemcpy(t.raw_data_gpu(), sc.dq_t.raw_data_gpu(), elems * t.element_size(),
                              cudaMemcpyDeviceToDevice));
        t.mark_dirty(Tensor::DataLocation::GPU);
        auto &cache = prefill_dq_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        cache.map.emplace(key, PrefillDQCache::Entry{std::move(t)});
    } catch (...) {
        // VRAM exhausted or CUDA error: subsequent prefills keep the
        // per-call dequant path for this weight.
    }
}

// Split a fused [q | k | v] GEMM output into the three projection buffers.
static void split_qkv_f16(void *q, void *k, void *v, const void *qkv, size_t seq, size_t qkv_total, size_t hidden,
                          size_t kv_dim)
{
    hal::extract_cols_f16(q, qkv, seq, qkv_total, 0, hidden);
    hal::extract_cols_f16(k, qkv, seq, qkv_total, hidden, kv_dim);
    hal::extract_cols_f16(v, qkv, seq, qkv_total, hidden + kv_dim, kv_dim);
}

// Split a fused [gate | up] GEMM output into the two MLP buffers.
static void split_gu_f16(void *gate, void *up, const void *gu, size_t seq, size_t inter)
{
    hal::extract_cols_f16(gate, gu, seq, 2 * inter, 0, inter);
    hal::extract_cols_f16(up, gu, seq, 2 * inter, inter, inter);
}

// Copies `count` elements of a device buffer of dtype `src_dtype` into the
// host float buffer `dst`. Non-FP32 sources are staged through `stage_f32`
// (grown on demand) so per-layer calls don't reallocate.
static void cuda_dev_to_host_f32(float *dst, const void *src, size_t count, hal::DType src_dtype,
                                 core::Tensor &stage_f32)
{
    if (src_dtype == hal::DType::Float32) {
        CHECK_CUDA(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
        return;
    }
    if (stage_f32.size() < count) {
        stage_f32 = core::Tensor::gpu_only({count}, hal::DType::Float32);
    }
    hal::convert_dtype(stage_f32.raw_data_gpu(), src, count, src_dtype, hal::DType::Float32);
    stage_f32.mark_dirty(Tensor::DataLocation::GPU);
    CHECK_CUDA(cudaMemcpy(dst, stage_f32.raw_data_gpu(), count * sizeof(float), cudaMemcpyDeviceToHost));
}

static Tensor llama_forward_cuda_impl(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache,
                                      bool last_only, size_t layer_begin = 0, size_t layer_end = SIZE_MAX,
                                      [[maybe_unused]] float *hidden_host_out = nullptr,
                                      [[maybe_unused]] size_t hidden_out_elems = 0)
{
    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size(), hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    const hal::DType stream_dtype = hal::DType::Float32;  // FP32 residual stream

    const bool prof_p = env_nonempty("SERIUS_PROF");
    auto pt_now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double pt0 = prof_p ? pt_now() : 0.0;

    Tensor hidden_states_gpu = Tensor::gpu_only({seq_len, hidden}, stream_dtype);
    std::vector<int> ids(seq_len);
    for (size_t i = 0; i < seq_len; ++i) ids[i] = (int)token_ids[i];
    int *ids_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&ids_gpu, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(ids_gpu, ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    hal::gather_mixed(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), ids_gpu, seq_len,
                      hidden, dtype, stream_dtype);
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
    // Fused-GEMM outputs (only used when fused_prefill_weights is available).
    Tensor qkv_buf = Tensor::gpu_only({seq_len, hidden + 2 * kv_dim}, dtype);
    Tensor gu_buf = Tensor::gpu_only({seq_len, 2 * inter}, dtype);
    const FusedPrefillW *fpw = fused_prefill_weights(model, dtype);
    const size_t lay_b = std::min(layer_begin, model.layers().size());
    const size_t lay_e = std::min(layer_end, model.layers().size());
    const bool qpf = (dtype == hal::DType::Float16) && quant_prefill_ready_range(model, lay_b, lay_e);
    QuantPrefillScratch &qps = quant_prefill_scratch();

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
    const bool full_range = (lay_b == 0 && lay_e == layers.size());
    if (cache && cache->empty()) {
        cache->layers.resize(layers.size());
        if (full_range) {
            // One arena allocation for every layer's K and V: 112 individual
            // cudaMallocs cost ~200 ms of driver time; a single slab +
            // non-owning views costs one.
            const size_t slab_elems = KV_INIT_CAP * kv_dim;
            cache->gpu_arena = Tensor::gpu_only({layers.size() * 2 * slab_elems}, dtype);
            char *base = static_cast<char *>(cache->gpu_arena.raw_data_gpu());
            const size_t slab_bytes = slab_elems * get_dtype_size(dtype);
            for (size_t i = 0; i < layers.size(); ++i) {
                auto &cl = cache->layers[i];
                cl.k = Tensor::gpu_view(base + i * 2 * slab_bytes, {KV_INIT_CAP, kv_dim}, dtype);
                cl.v = Tensor::gpu_view(base + i * 2 * slab_bytes + slab_bytes, {KV_INIT_CAP, kv_dim}, dtype);
                cl.current_len = 0;
            }
        } else {
            // Partial offload: only the GPU prefix layers get device KV; the
            // CPU suffix layers keep their (empty) host-side KVCacheLayer so
            // update_kv_cache can fill them later.
            for (size_t i = lay_b; i < lay_e; ++i) {
                auto &cl = cache->layers[i];
                cl.k = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.v = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.k.alloc_gpu();
                cl.v.alloc_gpu();
                cl.current_len = 0;
            }
        }
    }
    // Grow GPU KV layers to fit this prefill's write window
    // (current_len + seq_len); buffers double on each growth. Restricted to
    // the processed range: suffix layers may hold CPU tensors whose
    // raw_data_gpu() is null and must not be touched.
    if (cache) {
        for (size_t i = lay_b; i < lay_e && i < cache->layers.size(); ++i) {
            auto &cl = cache->layers[i];
            if (cl.k.raw_data_gpu() == nullptr || cl.v.raw_data_gpu() == nullptr) {
                cl.k = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.v = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.k.alloc_gpu();
                cl.v.alloc_gpu();
            }
            ensure_kv_capacity_gpu(cl, cl.current_len + seq_len, kv_dim, dtype);
        }
    }
    const double pt_cache = prof_p ? pt_now() : 0.0;

    const double pt_mask = prof_p ? pt_now() : 0.0;

    double acc_proj = 0, acc_attn = 0, acc_mlp = 0;

    // Interpretability hooks (activation_hook.h): opt-in per-layer D2H
    // copies. Fetched once — toggling hooks mid-forward is unsupported.
    const ActivationHooks &ahooks = activation_hooks();
    const bool want_hidden_hook = (ahooks.layer_hidden != nullptr);
    const bool want_attn_hook = (ahooks.attention_row != nullptr);
    std::vector<float> hook_hidden_host;
    std::vector<float> hook_scores_host;
    core::Tensor hook_stage_f32;
    if (want_hidden_hook) {
        hook_hidden_host.resize(seq_len * hidden);
    }
    if (want_attn_hook) {
        hook_scores_host.resize(n_heads * seq_len * seq_len);
        if (dtype != hal::DType::Float32) {
            hook_stage_f32 = core::Tensor::gpu_only({n_heads * seq_len * seq_len}, hal::DType::Float32);
        }
    }

    for (size_t layer_idx = lay_b; layer_idx < lay_e; ++layer_idx) {
        const double seg0 = prof_p ? pt_now() : 0.0;
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                           blk->input_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        if (qpf) {
            // Quantized prefill: dequant qkv_w -> [in,out] -> fused GEMM.
            quant_prefill_gemm(blk->qkv_w, normed_buf, qkv_buf, dtype, qps);
            split_qkv_f16(query_buf.raw_data_gpu(), key_buf.raw_data_gpu(), value_buf.raw_data_gpu(),
                          qkv_buf.raw_data_gpu(), seq_len, hidden + 2 * kv_dim, hidden, kv_dim);
        } else if (fpw && fpw->fused[layer_idx]) {
            // One fused GEMM {seq,in}x{in,hidden+2kv} then split q|k|v columns.
            core::blas::matmul(normed_buf, fpw->qkv[layer_idx], qkv_buf);
            split_qkv_f16(query_buf.raw_data_gpu(), key_buf.raw_data_gpu(), value_buf.raw_data_gpu(),
                          qkv_buf.raw_data_gpu(), seq_len, hidden + 2 * kv_dim, hidden, kv_dim);
        } else {
            core::blas::matmul(normed_buf, blk->q_proj->data, query_buf);
            core::blas::matmul(normed_buf, blk->k_proj->data, key_buf);
            core::blas::matmul(normed_buf, blk->v_proj->data, value_buf);
        }
        if (blk->q_bias) {
            hal::add_row_bias(query_buf.raw_data_gpu(), blk->q_bias->data.raw_data_gpu(), seq_len, hidden, dtype);
        }
        if (blk->k_bias) {
            hal::add_row_bias(key_buf.raw_data_gpu(), blk->k_bias->data.raw_data_gpu(), seq_len, kv_dim, dtype);
        }
        if (blk->v_bias) {
            hal::add_row_bias(value_buf.raw_data_gpu(), blk->v_bias->data.raw_data_gpu(), seq_len, kv_dim, dtype);
        }
        hal::rope(query_buf.raw_data_gpu(), head_dim, seq_len, n_heads, rope_base, 0, dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, seq_len, n_kv_heads, rope_base, 0, dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU);
        key_buf.mark_dirty(Tensor::DataLocation::GPU);
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
            CHECK_CUDA(cudaMemcpyAsync((char *)cl.k.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype),
                                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync((char *)cl.v.raw_data_gpu() + cl.current_len * kv_dim * get_dtype_size(dtype),
                                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            cl.current_len += seq_len;
        }

        // Attention with persistent scratch: transpose+contiguous chains are
        // replaced by copy_layout into preallocated buffers (no per-layer
        // Tensor churn / host memset).
        {
            // Q: {seq, n_heads, hd} -> {n_heads, seq, hd} in q_hs_buf
            Tensor qv = query_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
            hal::copy_layout(q_hs_buf.raw_data_gpu(), qv.raw_data_gpu(), qv.shape().data(), qv.strides().data(),
                             qv.ndim(), dtype);

            // K/V: expand GQA into {seq, n_heads, hd} rep buffers, then copy
            // to the {n_heads, seq, hd} (v) and {n_heads, hd, seq} (k) layouts
            // the batched GEMMs expect.
            if (n_kv_heads < n_heads) {
                hal::repeat_kv(k_rep_buf.raw_data_gpu(), key_buf.raw_data_gpu(), seq_len, n_heads, n_kv_heads, head_dim,
                               dtype);
                hal::repeat_kv(v_rep_buf.raw_data_gpu(), value_buf.raw_data_gpu(), seq_len, n_heads, n_kv_heads,
                               head_dim, dtype);
                Tensor kt = k_rep_buf.transpose(0, 1);  // {nh, seq, hd}
                Tensor vt = v_rep_buf.transpose(0, 1);
                hal::copy_layout(v_hs_buf.raw_data_gpu(), vt.raw_data_gpu(), vt.shape().data(), vt.strides().data(),
                                 vt.ndim(), dtype);
                Tensor ktt = kt.transpose(1, 2);  // {nh, hd, seq}
                hal::copy_layout(kT_buf.raw_data_gpu(), ktt.raw_data_gpu(), ktt.shape().data(), ktt.strides().data(),
                                 ktt.ndim(), dtype);
            } else {
                Tensor kt = key_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
                Tensor vt = value_buf.reshape({seq_len, n_heads, head_dim}).transpose(0, 1);
                hal::copy_layout(v_hs_buf.raw_data_gpu(), vt.raw_data_gpu(), vt.shape().data(), vt.strides().data(),
                                 vt.ndim(), dtype);
                Tensor ktt = kt.transpose(1, 2);
                hal::copy_layout(kT_buf.raw_data_gpu(), ktt.raw_data_gpu(), ktt.shape().data(), ktt.strides().data(),
                                 ktt.ndim(), dtype);
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

        if (want_attn_hook) {
            // scores is {n_heads, seq, seq} row-major; row q attends to
            // positions [0, q] under the causal mask applied above.
            cuda_dev_to_host_f32(hook_scores_host.data(), scores.raw_data_gpu(), hook_scores_host.size(), dtype,
                                 hook_stage_f32);
            for (size_t h = 0; h < n_heads; ++h) {
                for (size_t q = 0; q < seq_len; ++q) {
                    ahooks.attention_row(static_cast<int>(layer_idx), static_cast<int>(h), q,
                                         hook_scores_host.data() + (h * seq_len + q) * seq_len, q + 1);
                }
            }
        }

        {
            Tensor v_t = Tensor::gpu_view(v_hs_buf.raw_data_gpu(), {n_heads, seq_len, head_dim}, dtype);
            Tensor ctx_t = Tensor::gpu_view(ctx_hs_buf.raw_data_gpu(), {n_heads, seq_len, head_dim}, dtype);
            core::blas::batched_matmul(scores, v_t, ctx_t);
            // context {nh, seq, hd} -> attention_out {seq, hidden}
            Tensor cv = ctx_hs_buf.transpose(0, 1);
            hal::copy_layout(attention_out.raw_data_gpu(), cv.raw_data_gpu(), cv.shape().data(), cv.strides().data(),
                             cv.ndim(), dtype);
        }
        attention_out.mark_dirty(Tensor::DataLocation::GPU);
        const double seg2 = prof_p ? pt_now() : 0.0;
        if (qpf) {
            quant_prefill_gemm(blk->o_w, attention_out, mlp_down, dtype, qps);
        } else {
            core::blas::matmul(attention_out, blk->o_proj->data, mlp_down);
        }
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden,
                               stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                           blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        if (qpf) {
            quant_prefill_gemm(blk->gate_up_w, normed_buf, gu_buf, dtype, qps);
            split_gu_f16(mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), gu_buf.raw_data_gpu(), seq_len, inter);
        } else if (fpw && fpw->fused[layer_idx]) {
            core::blas::matmul(normed_buf, fpw->gu[layer_idx], gu_buf);
            split_gu_f16(mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), gu_buf.raw_data_gpu(), seq_len, inter);
        } else {
            core::blas::matmul(normed_buf, blk->gate_proj->data, mlp_gate);
            core::blas::matmul(normed_buf, blk->up_proj->data, mlp_up);
        }
        hal::swiglu(mlp_gate.raw_data_gpu(), mlp_gate.raw_data_gpu(), mlp_up.raw_data_gpu(), seq_len * inter, dtype);
        mlp_gate.mark_dirty(Tensor::DataLocation::GPU);
        if (qpf) {
            quant_prefill_gemm(blk->down_w, mlp_gate, mlp_down, dtype, qps);
        } else {
            core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);
        }
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), seq_len * hidden,
                               stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        if (want_hidden_hook) {
            // hidden_states_gpu is the FP32 residual stream — direct D2H.
            CHECK_CUDA(cudaMemcpy(hook_hidden_host.data(), hidden_states_gpu.raw_data_gpu(),
                                  hook_hidden_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ahooks.layer_hidden(static_cast<int>(layer_idx), hook_hidden_host.data(), seq_len, hidden);
        }
        if (prof_p) {
            const double seg3 = pt_now();
            acc_proj += seg1 - seg0;
            acc_attn += seg2 - seg1;
            acc_mlp += seg3 - seg2;
        }
    }
    if (prof_p)
        std::fprintf(stderr, "[prefillprof] proj=%.1fms attn=%.1fms mlp=%.1fms cache=%.1fms mask=%.1fms\n",
                     acc_proj * 1e3, acc_attn * 1e3, acc_mlp * 1e3, (pt_cache - pt_loop0) * 1e3,
                     (pt_mask - pt_cache) * 1e3);
    const double pt_loop1 = prof_p ? pt_now() : 0.0;

    // Partial offload: hand the FP32 residual stream for every prompt
    // position back to the host — the CPU suffix layers + final norm/lm_head
    // finish the forward there. Synchronous copy: the caller needs the data
    // immediately and this runs once per prompt, not per token.
    if (hidden_host_out != nullptr) {
        CHECK_CUDA(cudaMemcpy(hidden_host_out, hidden_states_gpu.raw_data_gpu(),
                              hidden_out_elems * sizeof(float), cudaMemcpyDeviceToHost));
        return Tensor({0});
    }

    hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                       model.norm_weight()->data.raw_data_gpu(), hidden, seq_len, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);

    if (prof_p) {
        std::fprintf(stderr, "[prefillprof] embed=%.1fms setup=%.1fms layers=%.1fms\n", (pt_embed - pt0) * 1e3,
                     (pt_loop0 - pt_embed) * 1e3, (pt_loop1 - pt_loop0) * 1e3);
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

    // Packed lm_head resident on GPU: dequantize vocab rows in tiles straight
    // into the logits GEMM — lm_head_q rows are already [vocab, hidden], the
    // exact layout logits_gemm_mixed_cublas wants, so no transpose is needed
    // and no persistent vocab×hidden FP16 copy is kept.
    if (const quant::QuantWeight *lq = model.lm_head_q();
        lq != nullptr && lq->gpu_data != nullptr && lq->dev_stride != 0 && lq->out == cfg.vocab_size &&
        lq->in == hidden && (lq->in % 32) == 0 && dtype == hal::DType::Float16) {
        Tensor logits({log_rows, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        constexpr size_t TILE_V = 8192;
        static core::Tensor lm_tile;
        static size_t lm_tile_cap = 0;
        const size_t tile_elems = TILE_V * hidden;
        if (lm_tile_cap < tile_elems) {
            lm_tile = core::Tensor::gpu_only({tile_elems}, hal::DType::Float16);
            lm_tile_cap = tile_elems;
        }
        for (size_t v0 = 0; v0 < cfg.vocab_size; v0 += TILE_V) {
            const size_t nv = std::min(TILE_V, cfg.vocab_size - v0);
            const uint8_t *w_row = static_cast<const uint8_t *>(lq->gpu_data) + v0 * lq->dev_stride;
            hal::dequant_qw_f16(w_row, lq->dev_kind, (int)hidden, (int)nv, (int)lq->dev_stride, (int)lq->dev_qs,
                                (int)lq->dev_d, (int)lq->dev_m, lm_tile.raw_data_gpu());
            // Tiles cover disjoint vocab columns of an uninitialized buffer:
            // beta must be 0 on every tile (nothing to accumulate onto).
            core::blas::logits_gemm_mixed_cublas(norm_ptr, lm_tile.raw_data_gpu(), logits.raw_data_gpu(), log_rows,
                                                 cfg.vocab_size, v0, nv, hidden, 0.0f);
        }
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (prof_p) std::fprintf(stderr, "[prefillprof] head=%.1fms\n", (pt_now() - pt_head0) * 1e3);
        return logits;
    }

    if (dtype == hal::DType::Float16) {
        if (const core::Tensor *lhead16 = model.lm_head_fp16_t()) {
            // FP16 weights + FP32 cuBLAS accumulation for the logits GEMV.
            Tensor logits({log_rows, cfg.vocab_size}, hal::DType::Float32);
            logits.alloc_gpu();
            core::blas::logits_gemv_mixed_cublas(norm_ptr, lhead16->raw_data_gpu(), logits.raw_data_gpu(), log_rows,
                                                 cfg.vocab_size, hidden);
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
        hal::convert_dtype(normed_f32.raw_data_gpu(), norm_ptr, log_rows * hidden, dtype, hal::DType::Float32);
        Tensor logits({log_rows, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        hal::logits_gemv_f32(normed_f32.raw_data_gpu(), lhead->raw_data_gpu(), logits.raw_data_gpu(), log_rows,
                             cfg.vocab_size, hidden);
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

Tensor llama_forward_cuda(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache)
{
    return llama_forward_cuda_impl(model, token_ids, cache, false);
}

Tensor llama_forward_cuda_last(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache)
{
    return llama_forward_cuda_impl(model, token_ids, cache, true);
}

// --- CUDA Graph decode path ---
// The single-token decode step is a fixed sequence of ~700 kernels whose only
// per-token variation is the token id and the position. Both are written to
// device memory (tid_gpu / pos_gpu) before each replay, so the whole step can
// be captured once and replayed with cudaGraphLaunch — eliminating all launch
// overhead and inter-kernel gaps.
namespace {

struct GraphDecodeCtx
{
    bool ready = false;     // tensors allocated, dims cached
    bool captured = false;  // graph instantiated
    const model::LLaMAModel *model = nullptr;
    const KVCache *cache = nullptr;
    // dims (for staleness check)
    size_t hidden = 0, inter = 0, kv_dim = 0, qkv_total = 0, n_layers = 0, vocab = 0;
    hal::DType dtype = hal::DType::Float16;
    // persistent device tensors (never re-allocated while ctx is valid)
    Tensor hidden_states_gpu, normed_buf, query_buf, key_buf, value_buf;
    Tensor attention_out, mlp_down;
    Tensor qf32_normed, qf32_attn, qf32_gu, qf32_sx, qkv_buf, gu_buf, qacc, logits_dev;
    Tensor logits_host;      // persistent D2H target for the logits view
    int *io_gpu = nullptr;   // single 2-int device buffer: [token_id, pos]
    int *tid_gpu = nullptr;  // aliases into io_gpu
    int *pos_gpu = nullptr;
    uint64_t cache_inst = 0;  // KVCache::instance_id the graph was captured for
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t cap_stream = nullptr;
};

GraphDecodeCtx &graph_ctx()
{
    static GraphDecodeCtx c;
    return c;
}

// (model, cache-instance) pairs whose graph capture already failed — don't
// retry every token; the eager path handles them. Uses instance_id because
// stack addresses are reused across turns.
struct GraphDisabledKey
{
    const void *model;
    uint64_t cache_inst;
};
GraphDisabledKey &graph_disabled()
{
    static GraphDisabledKey k{nullptr, 0};
    return k;
}

void graph_ctx_destroy(GraphDecodeCtx &c)
{
    if (c.exec) {
        cudaGraphExecDestroy(c.exec);
        c.exec = nullptr;
    }
    if (c.graph) {
        cudaGraphDestroy(c.graph);
        c.graph = nullptr;
    }
    if (c.io_gpu) {
        cudaFree(c.io_gpu);
        c.io_gpu = nullptr;
    }
    c.tid_gpu = nullptr;
    c.pos_gpu = nullptr;
    if (c.cap_stream) {
        cudaStreamDestroy(c.cap_stream);
        c.cap_stream = nullptr;
    }
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
void issue_decode_graph(GraphDecodeCtx &ctx, const model::LLaMAModel &model, KVCache *cache, cudaStream_t s)
{
    const auto &cfg = model.config();
    const auto &layers = model.layers();
    const size_t hidden = ctx.hidden, kv_dim = ctx.kv_dim, inter = ctx.inter;
    const size_t n_heads = cfg.num_attention_heads, n_kv_heads = cfg.num_key_value_heads;
    const size_t head_dim = hidden / n_heads;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    const hal::DType dtype = ctx.dtype;
    const hal::DType stream_dtype = hal::DType::Float32;
    const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    hal::gather_mixed(ctx.hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), ctx.tid_gpu, 1,
                      hidden, dtype, stream_dtype, s);
    ctx.hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);

    float *qacc_ptr = static_cast<float *>(ctx.qacc.raw_data_gpu());
    auto qproj = [&](const quant::QuantWeight &qw, const core::Tensor &x_f16, core::Tensor &out, core::Tensor &x_f32) {
        if (dtype == hal::DType::Float16 && qw.dev_stride != 0) {
            // Fused FP16-in/FP16-out GEMV: skips both convert_dtype launches.
            const void *sx = nullptr;
            if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
                hal::block_sums16_f16(x_f16.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(), (int)(qw.in / 16), s);
                sx = ctx.qf32_sx.raw_data_gpu();
            }
            hal::gemv_quant_vec_f16io(x_f16.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in, qw.out,
                                      (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d, (int)qw.dev_m,
                                      out.raw_data_gpu(), true, s);
            out.mark_dirty(Tensor::DataLocation::GPU);
            return;
        }
        hal::convert_dtype(x_f32.raw_data_gpu(), x_f16.raw_data_gpu(), qw.in, dtype, hal::DType::Float32, s);
        const void *sx = nullptr;
        if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
            hal::block_sums16_f32(x_f32.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(), (int)(qw.in / 16), s);
            sx = ctx.qf32_sx.raw_data_gpu();
        }
        hal::gemv_quant_vec_f32(x_f32.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in, qw.out, (int)qw.dev_stride,
                                (int)qw.dev_qs, (int)qw.dev_d, (int)qw.dev_m, qacc_ptr, s);
        hal::convert_dtype(out.raw_data_gpu(), qacc_ptr, qw.out, hal::DType::Float32, dtype, s);
        out.mark_dirty(Tensor::DataLocation::GPU);
    };

    const size_t esz = get_dtype_size(dtype);
    // The MLP down-projection delta is kept pending and folded into the next
    // RMSNorm (this layer's input norm, or the final norm after the loop), so
    // each residual add costs zero extra launches.
    const void *pending_delta = nullptr;
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        if (pending_delta != nullptr) {
            hal::rmsnorm_residual_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                                        pending_delta, blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps,
                                        dtype, dtype, s);
        } else {
            hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                               blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype, s);
        }
        ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);

        // Fused QKV GEMV -> q/k/v are contiguous slices of qkv_buf
        // ([q | k | v]); downstream kernels take raw pointers, so the three
        // DtoD slice copies are unnecessary — alias them directly.
        qproj(blk->qkv_w, ctx.normed_buf, ctx.qkv_buf, ctx.qf32_normed);
        char *qptr = static_cast<char *>(ctx.qkv_buf.raw_data_gpu());
        char *kptr = qptr + hidden * esz;
        char *vptr = kptr + kv_dim * esz;

        // Fused bias+RoPE(+scale) for Q and K in a single launch; V bias is
        // folded into the cache write.
        hal::rope_bias_devpos_qk(qptr, kptr, head_dim, n_heads, n_kv_heads, rope_base, ctx.pos_gpu,
                                 blk->q_bias ? blk->q_bias->data.raw_data_gpu() : nullptr,
                                 blk->k_bias ? blk->k_bias->data.raw_data_gpu() : nullptr,
                                 (dtype != hal::DType::Float32) ? q_scale : 1.0F, 1.0F, dtype, s);

        // Fused K+V cache write at the device-side position.
        KVCacheLayer &cl = cache->layers[layer_idx];
        hal::kv_write_kv_devpos(cl.k.raw_data_gpu(), cl.v.raw_data_gpu(), kptr, vptr,
                                blk->v_bias ? blk->v_bias->data.raw_data_gpu() : nullptr, kv_dim, ctx.pos_gpu, kv_dim,
                                dtype, s);
        cl.k.mark_dirty(Tensor::DataLocation::GPU);
        cl.v.mark_dirty(Tensor::DataLocation::GPU);

        // Fused GQA decode attention: pos+1 valid rows read from device.
        hal::gqa_decode_attention_f16_devpos(qptr, cl.k.raw_data_gpu(), cl.v.raw_data_gpu(),
                                             ctx.attention_out.raw_data_gpu(), ctx.pos_gpu, (int)n_heads,
                                             (int)n_kv_heads, (int)head_dim, (int)kv_dim, s);
        ctx.attention_out.mark_dirty(Tensor::DataLocation::GPU);

        qproj(blk->o_w, ctx.attention_out, ctx.mlp_down, ctx.qf32_attn);
        hal::rmsnorm_residual_mixed(
            ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(), ctx.mlp_down.raw_data_gpu(),
            blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype, dtype, s);
        ctx.hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);

        // Fused gate|up + SwiGLU + down.
        qproj(blk->gate_up_w, ctx.normed_buf, ctx.gu_buf, ctx.qf32_normed);
        hal::swiglu(ctx.gu_buf.raw_data_gpu(), ctx.gu_buf.raw_data_gpu(),
                    (char *)ctx.gu_buf.raw_data_gpu() + inter * esz, inter, dtype, s);
        ctx.gu_buf.mark_dirty(Tensor::DataLocation::GPU);
        qproj(blk->down_w, ctx.gu_buf, ctx.mlp_down, ctx.qf32_gu);
        pending_delta = ctx.mlp_down.raw_data_gpu();
    }

    // Final norm + quantized lm_head straight into ctx.logits_dev (FP32).
    if (pending_delta != nullptr) {
        hal::rmsnorm_residual_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(), pending_delta,
                                    model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype, dtype, s);
    } else {
        hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                           model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype, s);
    }
    ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    const quant::QuantWeight *lq = model.lm_head_q();
    // FP16 x in, FP32 logits out — still skips the input convert_dtype.
    const void *sx = nullptr;
    if (lq->dev_kind == 3 || lq->dev_kind == 4 || lq->dev_kind == 6) {
        hal::block_sums16_f16(ctx.normed_buf.raw_data_gpu(), ctx.qf32_sx.raw_data_gpu(), (int)(lq->in / 16), s);
        sx = ctx.qf32_sx.raw_data_gpu();
    }
    hal::gemv_quant_vec_f16io(ctx.normed_buf.raw_data_gpu(), sx, lq->gpu_data, lq->dev_kind, lq->in, lq->out,
                              (int)lq->dev_stride, (int)lq->dev_qs, (int)lq->dev_d, (int)lq->dev_m,
                              ctx.logits_dev.raw_data_gpu(), false, s);
    ctx.logits_dev.mark_dirty(Tensor::DataLocation::GPU);
}

}  // namespace

// Returns a non-owning host+device view over the persistent graph buffers
// (empty Tensor when the graph path is unavailable — caller falls back to the
// eager CUDA path). The view aliases ctx.logits_dev / ctx.logits_host: valid
// until the next decode call or graph_ctx rebuild, so callers must consume it
// immediately (sample, don't store).
Tensor llama_forward_single_token_cuda_graph(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache)
{
    if (cache == nullptr || pos + 1 > 4096) return Tensor();
    // Captured graphs can't emit activation hooks — fall back to the eager
    // path while interpretability hooks are installed.
    if (activation_hooks_enabled()) return Tensor();
    {
        if (env_nonempty("SERIUS_NO_GRAPH")) return Tensor();
    }
    {
        const GraphDisabledKey &dk = graph_disabled();
        if (dk.model == &model && dk.cache_inst == cache->instance_id) return Tensor();
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
        model.lm_head_q() == nullptr || model.lm_head_q()->gpu_data == nullptr || model.lm_head_q()->dev_stride == 0 ||
        model.lm_head_q()->out != cfg.vocab_size) {
        return Tensor();
    }
    for (const auto &blk : model.layers()) {
        const auto *b = blk.get();
        if (!b->qkv_w.gpu_data || !b->gate_up_w.gpu_data || !b->o_w.gpu_data || !b->down_w.gpu_data ||
            b->qkv_w.dev_stride == 0 || b->gate_up_w.dev_stride == 0 || b->o_w.dev_stride == 0 ||
            b->down_w.dev_stride == 0) {
            return Tensor();
        }
    }

    // The graph reads/writes cache->layers[i].k/v directly during capture —
    // the eager path allocates them lazily, the graph path must do the same
    // or issue_decode_graph indexes an empty vector.
    if (cache->empty() || cache->layers.size() != n_layers) {
        cache->layers.resize(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            if (cache->layers[i].k.raw_data_gpu() == nullptr) {
                cache->layers[i].k = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cache->layers[i].v = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cache->layers[i].k.alloc_gpu();
                cache->layers[i].v.alloc_gpu();
            }
            cache->layers[i].current_len = 0;
        }
    }
    // The captured graph embeds the KV tensor pointers: if this step needs
    // more capacity than the buffers hold, grow them and force a re-capture.
    bool kv_grew = false;
    for (auto &cl : cache->layers) {
        kv_grew |= ensure_kv_capacity_gpu(cl, pos + 1, kv_dim, dtype);
    }
    if (kv_grew) {
        graph_ctx_destroy(graph_ctx());
    }

    GraphDecodeCtx &ctx = graph_ctx();
    const bool stale = !ctx.ready || ctx.model != &model || ctx.cache != cache ||
                       ctx.cache_inst != cache->instance_id || ctx.hidden != hidden || ctx.inter != inter ||
                       ctx.kv_dim != kv_dim || ctx.n_layers != n_layers || ctx.vocab != cfg.vocab_size ||
                       ctx.dtype != dtype;
    if (stale) {
        graph_ctx_destroy(ctx);
        ctx.model = &model;
        ctx.cache = cache;
        ctx.cache_inst = cache->instance_id;
        ctx.hidden = hidden;
        ctx.inter = inter;
        ctx.kv_dim = kv_dim;
        ctx.qkv_total = qkv_total;
        ctx.n_layers = n_layers;
        ctx.vocab = cfg.vocab_size;
        ctx.dtype = dtype;

        ctx.hidden_states_gpu = Tensor({1, hidden}, hal::DType::Float32);
        ctx.hidden_states_gpu.alloc_gpu();
        ctx.normed_buf = Tensor({1, hidden}, dtype);
        ctx.normed_buf.alloc_gpu();
        ctx.query_buf = Tensor({1, hidden}, dtype);
        ctx.query_buf.alloc_gpu();
        ctx.key_buf = Tensor({1, kv_dim}, dtype);
        ctx.key_buf.alloc_gpu();
        ctx.value_buf = Tensor({1, kv_dim}, dtype);
        ctx.value_buf.alloc_gpu();
        ctx.attention_out = Tensor({1, hidden}, dtype);
        ctx.attention_out.alloc_gpu();
        ctx.mlp_down = Tensor({1, hidden}, dtype);
        ctx.mlp_down.alloc_gpu();
        ctx.qf32_normed = Tensor({hidden}, hal::DType::Float32);
        ctx.qf32_normed.alloc_gpu();
        ctx.qf32_attn = Tensor({hidden}, hal::DType::Float32);
        ctx.qf32_attn.alloc_gpu();
        ctx.qf32_gu = Tensor({inter}, hal::DType::Float32);
        ctx.qf32_gu.alloc_gpu();
        ctx.qf32_sx = Tensor({(std::max(hidden, inter) + 15) / 16 + 16}, hal::DType::Float32);
        ctx.qf32_sx.alloc_gpu();
        ctx.qkv_buf = Tensor({1, qkv_total}, dtype);
        ctx.qkv_buf.alloc_gpu();
        ctx.gu_buf = Tensor({1, 2 * inter}, dtype);
        ctx.gu_buf.alloc_gpu();
        const size_t qacc_cap = std::max({cfg.vocab_size, qkv_total, 2 * inter, hidden, inter});
        ctx.qacc = Tensor({qacc_cap}, hal::DType::Float32);
        ctx.qacc.alloc_gpu();
        ctx.logits_dev = Tensor({1, cfg.vocab_size}, hal::DType::Float32);
        ctx.logits_dev.alloc_gpu();
        ctx.logits_host = Tensor({1, cfg.vocab_size}, hal::DType::Float32);

        CHECK_CUDA(cudaMalloc(&ctx.io_gpu, 2 * sizeof(int)));
        ctx.tid_gpu = ctx.io_gpu;
        ctx.pos_gpu = ctx.io_gpu + 1;
        CHECK_CUDA(cudaStreamCreateWithFlags(&ctx.cap_stream, cudaStreamNonBlocking));
        ctx.ready = true;
    }

    // Device inputs updated before every launch (outside the graph): one
    // 8-byte H2D copy fills both [token_id, pos].
    const int io_host[2] = {token_id, static_cast<int>(pos)};
    CHECK_CUDA(cudaMemcpyAsync(ctx.io_gpu, io_host, sizeof(io_host), cudaMemcpyHostToDevice, 0));

    if (!ctx.captured) {
        // Capture on a side stream; any failure falls back to the eager path.
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaError_t cap_err = cudaSuccess;
        try {
            CHECK_CUDA(cudaStreamBeginCapture(ctx.cap_stream, cudaStreamCaptureModeGlobal));
            issue_decode_graph(ctx, model, cache, ctx.cap_stream);
            cap_err = cudaStreamEndCapture(ctx.cap_stream, &ctx.graph);
        }
        catch (...) {
            cudaStreamEndCapture(ctx.cap_stream, &ctx.graph);
            cudaGetLastError();
            graph_ctx_destroy(ctx);
            graph_disabled() = {&model, cache->instance_id};
            return Tensor();
        }
        if (cap_err != cudaSuccess || ctx.graph == nullptr ||
            cudaGraphInstantiateWithFlags(&ctx.exec, ctx.graph, 0ULL) != cudaSuccess) {
            cudaGetLastError();
            graph_ctx_destroy(ctx);
            graph_disabled() = {&model, cache->instance_id};
            return Tensor();
        }
        ctx.captured = true;
    }

    CHECK_CUDA(cudaGraphLaunch(ctx.exec, 0));
    for (size_t i = 0; i < cache->layers.size(); ++i) {
        cache->layers[i].current_len = pos + 1;
    }

    // The blocking D2H copy is ordered after the graph on stream 0, so it
    // doubles as the completion wait — no separate stream sync needed.
    CHECK_CUDA(cudaMemcpy(ctx.logits_host.data(), ctx.logits_dev.raw_data_gpu(), cfg.vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return Tensor::host_gpu_view(ctx.logits_host.data(), ctx.logits_dev.raw_data_gpu(), {1, cfg.vocab_size},
                                 hal::DType::Float32);
}

// Persistent decode scratch: the single-token path runs once per generated
// token, so every Tensor({shape}) + alloc_gpu() used to cost a host memset +
// cudaMalloc + cudaFree — ~200 driver calls per token across the layer loop.
// Buffers are kept across calls and only reallocated when the model dims or
// the required sequence capacity change (geometric growth, like the KV cache).
struct DecodeScratch {
    size_t hidden = 0, kv_dim = 0, inter = 0, n_heads = 0, head_dim = 0;
    hal::DType dtype = hal::DType::Float32;
    size_t seq_cap = 0;
    Tensor hidden_states_gpu, normed_buf, query_buf, key_buf, value_buf, attention_out, mlp_gate, mlp_up,
        mlp_down, qf32_normed, qf32_attn, qf32_gu, qf32_sx, qkv_buf, gu_buf, qacc, proj_acc, q_hs_buf,
        scores_buf, v_hs_buf, kT_buf, ctx_hs_buf, k_rep_buf, v_rep_buf;
};

static DecodeScratch &decode_scratch()
{
    static DecodeScratch s;
    return s;
}

// layer_begin/layer_end restrict the transformer loop (partial offload runs
// only the GPU-resident prefix). When hidden_host_out != nullptr the final
// norm + lm_head are skipped and the {1,hidden} FP32 residual stream is
// copied back to the host instead of producing logits.
static Tensor llama_forward_single_token_cuda_impl(const model::LLaMAModel &model, int token_id, size_t pos,
                                                   KVCache *cache, size_t layer_begin, size_t layer_end,
                                                   float *hidden_host_out, size_t hidden_out_size)
{
    static const bool gpu_time = env_nonempty("SERIUS_GPU_TIME");
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (gpu_time) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
    }
    const auto &cfg = model.config();
    const size_t n_layers = cfg.num_hidden_layers, hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    const hal::DType stream_dtype = hal::DType::Float32;

    const size_t qkv_total = hidden + 2 * kv_dim;
    const size_t qacc_cap = std::max({cfg.vocab_size, qkv_total, 2 * inter, hidden, inter});
    DecodeScratch &ds = decode_scratch();
    const size_t eff_need = pos + 1;
    const bool dims_ok = ds.hidden == hidden && ds.kv_dim == kv_dim && ds.inter == inter &&
                         ds.n_heads == n_heads && ds.head_dim == head_dim && ds.dtype == dtype;
    if (!dims_ok || eff_need > ds.seq_cap) {
        ds.hidden = hidden;
        ds.kv_dim = kv_dim;
        ds.inter = inter;
        ds.n_heads = n_heads;
        ds.head_dim = head_dim;
        ds.dtype = dtype;
        ds.seq_cap = dims_ok ? std::max(ds.seq_cap * 2, eff_need) : std::max<size_t>(eff_need, KV_INIT_CAP);
        ds.hidden_states_gpu = Tensor::gpu_only({1, hidden}, stream_dtype);
        ds.normed_buf = Tensor::gpu_only({1, hidden}, dtype);
        ds.query_buf = Tensor::gpu_only({1, hidden}, dtype);
        ds.key_buf = Tensor::gpu_only({1, kv_dim}, dtype);
        ds.value_buf = Tensor::gpu_only({1, kv_dim}, dtype);
        ds.attention_out = Tensor::gpu_only({1, hidden}, dtype);
        ds.mlp_gate = Tensor::gpu_only({1, inter}, dtype);
        ds.mlp_up = Tensor::gpu_only({1, inter}, dtype);
        ds.mlp_down = Tensor::gpu_only({1, hidden}, dtype);
        ds.qf32_normed = Tensor::gpu_only({hidden}, hal::DType::Float32);
        ds.qf32_attn = Tensor::gpu_only({hidden}, hal::DType::Float32);
        ds.qf32_gu = Tensor::gpu_only({inter}, hal::DType::Float32);
        ds.qf32_sx = Tensor::gpu_only({(std::max(hidden, inter) + 15) / 16 + 16}, hal::DType::Float32);
        ds.qkv_buf = Tensor::gpu_only({1, qkv_total}, dtype);
        ds.gu_buf = Tensor::gpu_only({1, 2 * inter}, dtype);
        ds.qacc = Tensor::gpu_only({qacc_cap}, hal::DType::Float32);
        ds.proj_acc = Tensor::gpu_only({std::max({hidden, kv_dim, inter})}, hal::DType::Float32);
        ds.q_hs_buf = Tensor::gpu_only({n_heads, 1, head_dim}, dtype);
        ds.scores_buf = Tensor::gpu_only({n_heads * ds.seq_cap}, dtype);
        ds.v_hs_buf = Tensor::gpu_only({n_heads * ds.seq_cap * head_dim}, dtype);
        ds.kT_buf = Tensor::gpu_only({n_heads * head_dim * ds.seq_cap}, dtype);
        ds.ctx_hs_buf = Tensor::gpu_only({n_heads * head_dim}, dtype);
        ds.k_rep_buf = Tensor::gpu_only({ds.seq_cap * n_heads * head_dim}, dtype);
        ds.v_rep_buf = Tensor::gpu_only({ds.seq_cap * n_heads * head_dim}, dtype);
    }
    auto &hidden_states_gpu = ds.hidden_states_gpu;
    auto &normed_buf = ds.normed_buf;
    auto &query_buf = ds.query_buf;
    auto &key_buf = ds.key_buf;
    auto &value_buf = ds.value_buf;
    auto &attention_out = ds.attention_out;
    auto &mlp_gate = ds.mlp_gate;
    auto &mlp_up = ds.mlp_up;
    auto &mlp_down = ds.mlp_down;
    auto &qf32_normed = ds.qf32_normed;
    auto &qf32_attn = ds.qf32_attn;
    auto &qf32_gu = ds.qf32_gu;
    auto &qf32_sx = ds.qf32_sx;
    auto &qkv_buf = ds.qkv_buf;
    auto &gu_buf = ds.gu_buf;
    auto &qacc = ds.qacc;
    // Decode-state scratch buffers reserved for upcoming fused-decode work;
    // unused in this translation unit's current path.
    [[maybe_unused]] auto &proj_acc = ds.proj_acc;
    [[maybe_unused]] auto &q_hs_buf = ds.q_hs_buf;
    [[maybe_unused]] auto &v_hs_buf = ds.v_hs_buf;
    [[maybe_unused]] auto &kT_buf = ds.kT_buf;
    [[maybe_unused]] auto &ctx_hs_buf = ds.ctx_hs_buf;
    [[maybe_unused]] auto &k_rep_buf = ds.k_rep_buf;
    [[maybe_unused]] auto &v_rep_buf = ds.v_rep_buf;
    [[maybe_unused]] auto &scores_buf = ds.scores_buf;
    // Persistent device buffer for the token id: avoids a cudaMalloc +
    // cudaDeviceSynchronize + cudaFree round-trip on every decode step.
    static int *tid_gpu = [] {
        int *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, sizeof(int)));
        return p;
    }();
    int tid = token_id;
    CHECK_CUDA(cudaMemcpyAsync(tid_gpu, &tid, sizeof(int), cudaMemcpyHostToDevice, 0));
    hal::gather_mixed(hidden_states_gpu.raw_data_gpu(), model.embed_tokens()->data.raw_data_gpu(), tid_gpu, 1, hidden,
                      dtype, stream_dtype);
    hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);

    const auto &layers = model.layers();
    if (cache && (cache->empty() || cache->layers.size() != layers.size())) {
        cache->layers.resize(n_layers);
        for (size_t i = layer_begin; i < layer_end && i < n_layers; ++i) {
            cache->layers[i].k = Tensor({KV_INIT_CAP, kv_dim}, dtype);
            cache->layers[i].v = Tensor({KV_INIT_CAP, kv_dim}, dtype);
            cache->layers[i].k.alloc_gpu();
            cache->layers[i].v.alloc_gpu();
            cache->layers[i].current_len = 0;
        }
    }
    // Partial offload: the cache may already hold CPU tensors for every
    // layer (the suffix uses them); the GPU prefix layers need device KV.
    if (cache) {
        for (size_t i = layer_begin; i < layer_end && i < n_layers; ++i) {
            auto &cl = cache->layers[i];
            if (cl.k.raw_data_gpu() == nullptr || cl.v.raw_data_gpu() == nullptr) {
                cl.k = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.v = Tensor({KV_INIT_CAP, kv_dim}, dtype);
                cl.k.alloc_gpu();
                cl.v.alloc_gpu();
            }
        }
    }

    // Interpretability hooks (activation_hook.h): opt-in D2H copies.
    const ActivationHooks &ahooks = activation_hooks();
    const bool want_hidden_hook = (ahooks.layer_hidden != nullptr);
    const bool want_attn_hook = (ahooks.attention_row != nullptr);
    std::vector<float> hook_hidden_host(want_hidden_hook ? hidden : 0);
    std::vector<float> hook_scores_host;
    core::Tensor hook_stage_f32;
    if (want_attn_hook) {
        hook_scores_host.resize(n_heads * (pos + 1));
        if (dtype != hal::DType::Float32) {
            hook_stage_f32 = core::Tensor::gpu_only({n_heads * (pos + 1)}, hal::DType::Float32);
        }
    }

    for (size_t layer_idx = layer_begin; layer_idx < layer_end && layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                           blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        // M==1 projections as custom streaming GEMVs (one kernel each, weight
        // read exactly once) — much faster than cuBLAS heuristics for GEMV.
        auto proj = [&](const core::Tensor &x, const core::Tensor &w, core::Tensor &out, size_t K, size_t N) {
            if (dtype == hal::DType::Float16 && w.is_contiguous() && w.raw_data_gpu() != nullptr) {
                // Split-K FP16 GEMV with FP32 accumulation (persistent scratch).
                float *accp = static_cast<float *>(proj_acc.raw_data_gpu());
                CHECK_CUDA(cudaMemsetAsync(accp, 0, N * sizeof(float), 0));
                hal::gemv_kn_sk_f16(x.raw_data_gpu(), w.raw_data_gpu(), accp, K, N);
                hal::convert_dtype(out.raw_data_gpu(), accp, N, hal::DType::Float32, dtype);
                out.mark_dirty(Tensor::DataLocation::GPU);
            } else {
                core::blas::matmul(x, w, out);
            }
        };
        // Packed GGUF weights (Q4_0/Q4_1/Q8_0) resident on the GPU: dequantize
        // on the fly inside the GEMV — ~4x less DRAM traffic than FP16.
        // dev_stride != 0 guarantees the repacked layout the vec GEMV reads;
        // gpu_data is only ever set for that layout, but check both so a
        // future raw-layout upload can't silently feed an unsupported kernel.
        const bool use_quant = (dtype == hal::DType::Float16) && blk->qkv_w.gpu_data && blk->qkv_w.dev_stride != 0 &&
                               blk->gate_up_w.gpu_data && blk->gate_up_w.dev_stride != 0 && blk->o_w.gpu_data &&
                               blk->o_w.dev_stride != 0 && blk->down_w.gpu_data && blk->down_w.dev_stride != 0;
        // Quantized GEMV consumes FP32 x (converted once per projection input
        // into the provided scratch buffer) and writes FP32 y directly.
        auto qproj = [&](const quant::QuantWeight &qw, const core::Tensor &x_f16, core::Tensor &out,
                         core::Tensor &x_f32) {
            if (qw.dev_stride != 0 && dtype == hal::DType::Float16) {
                // Fused FP16-in/FP16-out GEMV: skips both convert_dtype launches.
                const void *sx = nullptr;
                if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
                    hal::block_sums16_f16(x_f16.raw_data_gpu(), qf32_sx.raw_data_gpu(), (int)(qw.in / 16));
                    sx = qf32_sx.raw_data_gpu();
                }
                hal::gemv_quant_vec_f16io(x_f16.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in, qw.out,
                                          (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d, (int)qw.dev_m,
                                          out.raw_data_gpu(), true);
                out.mark_dirty(Tensor::DataLocation::GPU);
                return;
            }
            hal::convert_dtype(x_f32.raw_data_gpu(), x_f16.raw_data_gpu(), qw.in, dtype, hal::DType::Float32);
            float *acc_ptr = static_cast<float *>(qacc.raw_data_gpu());
            if (qw.dev_stride != 0) {
                const void *sx = nullptr;
                if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
                    hal::block_sums16_f32(x_f32.raw_data_gpu(), qf32_sx.raw_data_gpu(), (int)(qw.in / 16));
                    sx = qf32_sx.raw_data_gpu();
                }
                hal::gemv_quant_vec_f32(x_f32.raw_data_gpu(), sx, qw.gpu_data, qw.dev_kind, qw.in, qw.out,
                                        (int)qw.dev_stride, (int)qw.dev_qs, (int)qw.dev_d, (int)qw.dev_m, acc_ptr);
            } else {
                hal::gemv_quant_f32(x_f32.raw_data_gpu(), qw.gpu_data, (int)qw.type, qw.in, qw.out, acc_ptr);
            }
            hal::convert_dtype(out.raw_data_gpu(), acc_ptr, qw.out, hal::DType::Float32, dtype);
            out.mark_dirty(Tensor::DataLocation::GPU);
        };
        if (use_quant) {
            // Fused QKV: rows are [q | k | v], slice into the RoPE inputs.
            const size_t esz = get_dtype_size(dtype);
            qproj(blk->qkv_w, normed_buf, qkv_buf, qf32_normed);
            CHECK_CUDA(cudaMemcpyAsync(query_buf.raw_data_gpu(), qkv_buf.raw_data_gpu(), hidden * esz,
                                       cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync(key_buf.raw_data_gpu(), (char *)qkv_buf.raw_data_gpu() + hidden * esz,
                                       kv_dim * esz, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync(value_buf.raw_data_gpu(),
                                       (char *)qkv_buf.raw_data_gpu() + (hidden + kv_dim) * esz, kv_dim * esz,
                                       cudaMemcpyDeviceToDevice, 0));
            query_buf.mark_dirty(Tensor::DataLocation::GPU);
            key_buf.mark_dirty(Tensor::DataLocation::GPU);
            value_buf.mark_dirty(Tensor::DataLocation::GPU);
        } else {
            proj(normed_buf, blk->q_proj->data, query_buf, hidden, hidden);
            proj(normed_buf, blk->k_proj->data, key_buf, hidden, kv_dim);
            proj(normed_buf, blk->v_proj->data, value_buf, hidden, kv_dim);
        }
        if (blk->q_bias) {
            hal::add_row_bias(query_buf.raw_data_gpu(), blk->q_bias->data.raw_data_gpu(), 1, hidden, dtype);
        }
        if (blk->k_bias) {
            hal::add_row_bias(key_buf.raw_data_gpu(), blk->k_bias->data.raw_data_gpu(), 1, kv_dim, dtype);
        }
        if (blk->v_bias) {
            hal::add_row_bias(value_buf.raw_data_gpu(), blk->v_bias->data.raw_data_gpu(), 1, kv_dim, dtype);
        }
        hal::rope(query_buf.raw_data_gpu(), head_dim, 1, n_heads, rope_base, static_cast<int>(pos), dtype);
        hal::rope(key_buf.raw_data_gpu(), head_dim, 1, n_kv_heads, rope_base, static_cast<int>(pos), dtype);
        query_buf.mark_dirty(Tensor::DataLocation::GPU);
        key_buf.mark_dirty(Tensor::DataLocation::GPU);
        const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        if (dtype != hal::DType::Float32) {
            hal::scale_inplace(query_buf.raw_data_gpu(), hidden, q_scale, dtype);
        }

        size_t effective_len = pos + 1;
        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            ensure_kv_capacity_gpu(cl, effective_len, kv_dim, dtype);
            size_t bytes = kv_dim * get_dtype_size(dtype);
            CHECK_CUDA(cudaMemcpyAsync((char *)cl.k.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype),
                                       key_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            CHECK_CUDA(cudaMemcpyAsync((char *)cl.v.raw_data_gpu() + pos * kv_dim * get_dtype_size(dtype),
                                       value_buf.raw_data_gpu(), bytes, cudaMemcpyDeviceToDevice, 0));
            cl.k.mark_dirty(Tensor::DataLocation::GPU);
            cl.v.mark_dirty(Tensor::DataLocation::GPU);
            cl.current_len = effective_len;
        }

        if (dtype == hal::DType::Float16 && cache && effective_len <= 4096) {
            // Fused GQA decode attention: single kernel reads the KV cache
            // directly (query already RoPE'd and pre-scaled above).
            hal::gqa_decode_attention_f16(query_buf.raw_data_gpu(), cache->layers[layer_idx].k.raw_data_gpu(),
                                          cache->layers[layer_idx].v.raw_data_gpu(), attention_out.raw_data_gpu(),
                                          (int)effective_len, (int)n_heads, (int)n_kv_heads, (int)head_dim,
                                          (int)kv_dim);
            attention_out.mark_dirty(Tensor::DataLocation::GPU);
        } else {
            // Persistent scratch + copy_layout: same layout strategy as the
            // prefill path — no per-layer Tensor churn or host memsets.
            Tensor qv = query_buf.reshape({1, n_heads, head_dim}).transpose(0, 1);  // {nh,1,hd}
            hal::copy_layout(q_hs_buf.raw_data_gpu(), qv.raw_data_gpu(), qv.shape().data(), qv.strides().data(),
                             qv.ndim(), dtype);

            Tensor k_full({0}, dtype), v_full({0}, dtype);
            if (cache) {
                k_full = cache->layers[layer_idx].k.slice(0, 0, effective_len);
                v_full = cache->layers[layer_idx].v.slice(0, 0, effective_len);
            } else {
                k_full = key_buf;
                v_full = value_buf;
            }

            // ks/vs: {effective_len, n_heads, head_dim} contiguous views.
            Tensor ks({0}, dtype), vs({0}, dtype);
            if (n_kv_heads < n_heads) {
                // repeat_kv expects (seq, heads, head_dim); k_full is already
                // contiguous in that layout ({eff, kv_dim} == {eff, nkv, hd}).
                hal::repeat_kv(k_rep_buf.raw_data_gpu(), k_full.raw_data_gpu(), effective_len, n_heads, n_kv_heads,
                               head_dim, dtype);
                hal::repeat_kv(v_rep_buf.raw_data_gpu(), v_full.raw_data_gpu(), effective_len, n_heads, n_kv_heads,
                               head_dim, dtype);
                ks = Tensor::gpu_view(k_rep_buf.raw_data_gpu(), {effective_len, n_heads, head_dim}, dtype);
                vs = Tensor::gpu_view(v_rep_buf.raw_data_gpu(), {effective_len, n_heads, head_dim}, dtype);
            } else {
                ks = k_full.reshape({effective_len, n_heads, head_dim});
                vs = v_full.reshape({effective_len, n_heads, head_dim});
            }
            Tensor vt = vs.transpose(0, 1);  // {nh, eff, hd}
            hal::copy_layout(v_hs_buf.raw_data_gpu(), vt.raw_data_gpu(), vt.shape().data(), vt.strides().data(),
                             vt.ndim(), dtype);
            Tensor kt = ks.transpose(0, 1);  // {nh, eff, hd}
            Tensor ktt = kt.transpose(1, 2);  // {nh, hd, eff}
            hal::copy_layout(kT_buf.raw_data_gpu(), ktt.raw_data_gpu(), ktt.shape().data(), ktt.strides().data(),
                             ktt.ndim(), dtype);

            Tensor q_hs = Tensor::gpu_view(q_hs_buf.raw_data_gpu(), {n_heads, 1, head_dim}, dtype);
            Tensor kT_t = Tensor::gpu_view(kT_buf.raw_data_gpu(), {n_heads, head_dim, effective_len}, dtype);
            Tensor scores = Tensor::gpu_view(scores_buf.raw_data_gpu(), {n_heads, 1, effective_len}, dtype);
            core::blas::batched_matmul(q_hs, kT_t, scores);

            float scale = (dtype == hal::DType::Float32) ? (1.0F / std::sqrt(static_cast<float>(head_dim))) : 1.0F;
            hal::fused_masked_softmax(scores.raw_data_gpu(), decode_mask_for(effective_len), n_heads, 1,
                                      effective_len, dtype, scale);
            scores.mark_dirty(Tensor::DataLocation::GPU);

            if (want_attn_hook) {
                // scores is {n_heads, 1, effective_len} — the single query
                // row attends to all effective_len cached positions.
                cuda_dev_to_host_f32(hook_scores_host.data(), scores.raw_data_gpu(),
                                     static_cast<size_t>(n_heads) * effective_len, dtype, hook_stage_f32);
                for (size_t h = 0; h < n_heads; ++h) {
                    ahooks.attention_row(static_cast<int>(layer_idx), static_cast<int>(h), 0,
                                         hook_scores_host.data() + h * effective_len, effective_len);
                }
            }

            Tensor v_t = Tensor::gpu_view(v_hs_buf.raw_data_gpu(), {n_heads, effective_len, head_dim}, dtype);
            Tensor ctx_t = Tensor::gpu_view(ctx_hs_buf.raw_data_gpu(), {n_heads, 1, head_dim}, dtype);
            core::blas::batched_matmul(scores, v_t, ctx_t);
            Tensor cv = ctx_t.transpose(0, 1);  // {1, nh, hd} == {1, hidden}
            hal::copy_layout(attention_out.raw_data_gpu(), cv.raw_data_gpu(), cv.shape().data(), cv.strides().data(),
                             cv.ndim(), dtype);
            attention_out.mark_dirty(Tensor::DataLocation::GPU);
        }

        if (use_quant) {
            qproj(blk->o_w, attention_out, mlp_down, qf32_attn);
        } else {
            proj(attention_out, blk->o_proj->data, mlp_down, hidden, hidden);
        }
        hal::add_inplace_mixed(hidden_states_gpu.raw_data_gpu(), mlp_down.raw_data_gpu(), hidden, stream_dtype, dtype);
        hidden_states_gpu.mark_dirty(Tensor::DataLocation::GPU);
        hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                           blk->post_attention_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps, dtype);
        normed_buf.mark_dirty(Tensor::DataLocation::GPU);
        if (use_quant) {
            // Fused gate|up: swiglu in place on the first half, down reads it.
            const size_t esz = get_dtype_size(dtype);
            qproj(blk->gate_up_w, normed_buf, gu_buf, qf32_normed);
            hal::swiglu(gu_buf.raw_data_gpu(), gu_buf.raw_data_gpu(), (char *)gu_buf.raw_data_gpu() + inter * esz,
                        inter, dtype);
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
        if (want_hidden_hook) {
            CHECK_CUDA(cudaMemcpy(hook_hidden_host.data(), hidden_states_gpu.raw_data_gpu(),
                                  hidden * sizeof(float), cudaMemcpyDeviceToHost));
            ahooks.layer_hidden(static_cast<int>(layer_idx), hook_hidden_host.data(), 1, hidden);
        }
    }

    // Partial-offload prefix: hand the FP32 residual stream back to the host
    // — the CPU suffix runs the remaining layers + final norm + lm_head.
    if (hidden_host_out != nullptr) {
        if (hidden_out_size != hidden) {
            return Tensor();
        }
        CHECK_CUDA(cudaMemcpyAsync(hidden_host_out, hidden_states_gpu.raw_data_gpu(), hidden * sizeof(float),
                                   cudaMemcpyDeviceToHost, 0));
        CHECK_CUDA(cudaStreamSynchronize(0));
        if (gpu_time) {
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms;
            cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] prefix_fwd_gpu=%.2fms layers=[%zu,%zu)\n", ms, layer_begin, layer_end);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
        return Tensor();
    }

    hal::rmsnorm_mixed(normed_buf.raw_data_gpu(), hidden_states_gpu.raw_data_gpu(),
                       model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype);
    normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    // Quantized lm_head (GGUF): single dequant GEMV straight into FP32 logits.
    if (const quant::QuantWeight *lq = model.lm_head_q(); lq != nullptr && lq->gpu_data != nullptr &&
                                                          lq->dev_stride != 0 && lq->out == cfg.vocab_size &&
                                                          dtype == hal::DType::Float16) {
        // normed_buf is the final RMSNorm output; reuse the FP32 scratch.
        hal::convert_dtype(qf32_normed.raw_data_gpu(), normed_buf.raw_data_gpu(), hidden, dtype, hal::DType::Float32);
        Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        if (lq->dev_stride != 0) {
            const void *sx = nullptr;
            if (lq->dev_kind == 3 || lq->dev_kind == 4 || lq->dev_kind == 6) {
                hal::block_sums16_f32(qf32_normed.raw_data_gpu(), qf32_sx.raw_data_gpu(), (int)(lq->in / 16));
                sx = qf32_sx.raw_data_gpu();
            }
            hal::gemv_quant_vec_f32(qf32_normed.raw_data_gpu(), sx, lq->gpu_data, lq->dev_kind, lq->in, lq->out,
                                    (int)lq->dev_stride, (int)lq->dev_qs, (int)lq->dev_d, (int)lq->dev_m,
                                    logits.raw_data_gpu());
        } else {
            hal::gemv_quant_f32(qf32_normed.raw_data_gpu(), lq->gpu_data, (int)lq->type, lq->in, lq->out,
                                logits.raw_data_gpu());
        }
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (gpu_time) {
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms;
            cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
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
                cudaEventRecord(ev1);
                cudaEventSynchronize(ev1);
                float ms;
                cudaEventElapsedTime(&ms, ev0, ev1);
                std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
                cudaEventDestroy(ev0);
                cudaEventDestroy(ev1);
            }
            return logits;
        }
    }
    if (const core::Tensor *lhead = model.lm_head_fp32()) {
        Tensor normed_f32({1, hidden}, hal::DType::Float32);
        normed_f32.alloc_gpu();
        hal::convert_dtype(normed_f32.raw_data_gpu(), normed_buf.raw_data_gpu(), hidden, dtype, hal::DType::Float32);
        Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
        logits.alloc_gpu();
        hal::logits_gemv_f32(normed_f32.raw_data_gpu(), lhead->raw_data_gpu(), logits.raw_data_gpu(), 1, cfg.vocab_size,
                             hidden);
        logits.mark_dirty(Tensor::DataLocation::GPU);
        logits.syncToHost();
        if (gpu_time) {
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms;
            cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
        return logits;
    }
    Tensor logits({1, cfg.vocab_size}, dtype);
    logits.alloc_gpu();
    core::blas::matmul(normed_buf, model.lm_head()->data, logits);
    auto ret = logits.cast(hal::DType::Float32);
    if (gpu_time) {
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        std::fprintf(stderr, "[gputime] decode_fwd_gpu=%.2fms\n", ms);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
    return ret;
}

Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache)
{
    return llama_forward_single_token_cuda_impl(model, token_id, pos, cache, 0, model.layers().size(), nullptr, 0);
}

Tensor llama_forward_single_token_cuda_prefix(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache,
                                              size_t n_gpu_layers, float *hidden_out, size_t hidden_size)
{
    const size_t n = std::min(n_gpu_layers, model.layers().size());
    return llama_forward_single_token_cuda_impl(model, token_id, pos, cache, 0, n, hidden_out, hidden_size);
}

// --- Batched prefill for partial offload ---
// The GPU prefix processes the whole prompt in one batched pass (same GEMM
// path as full offload, restricted to layers [0, n_gpu)), then the FP32
// residual stream for every position is copied to the host once. The CPU
// suffix finishes the remaining layers + lm_head.
//
// Correctness constraint: the batched attention covers only the prompt
// window — there is no cross-call history attention — so this is only valid
// on a fresh KV cache. Callers must check llama_cuda_prefix_batched_fresh()
// first and fall back to the per-token path otherwise.

bool llama_cuda_prefix_batched_ready(const model::LLaMAModel &model, size_t n_gpu_layers)
{
    const size_t n = std::min(n_gpu_layers, model.layers().size());
    const size_t total = model.layers().size();
    return n > 0 && n < total && model.embed_tokens() && model.embed_tokens()->data.is_gpu() &&
           model.embed_tokens()->data.dtype() == hal::DType::Float16 &&
           quant_prefill_ready_range(model, 0, n);
}

bool llama_cuda_prefix_batched_fresh(const KVCache *cache)
{
    if (cache == nullptr) {
        return true;
    }
    for (const auto &cl : cache->layers) {
        if (cl.current_len != 0) {
            return false;
        }
    }
    return true;
}

Tensor llama_forward_cuda_prefix_batched(const model::LLaMAModel &model, const Tensor &token_ids, KVCache *cache,
                                         size_t n_gpu_layers, float *hidden_out, size_t hidden_out_elems)
{
    const size_t n = std::min(n_gpu_layers, model.layers().size());
    return llama_forward_cuda_impl(model, token_ids, cache, false, 0, n, hidden_out, hidden_out_elems);
}

// Logits for the partial-offload CPU suffix: the suffix leaves a normed
// {1,hidden} FP32 row on the host; running the (packed, GPU-resident) lm_head
// there turns a ~50 ms CPU GEMV into a sub-ms tiled dequant + cuBLAS GEMM.
// Returns an empty tensor when no GPU-ready packed lm_head exists so callers
// fall back to compute_logits on CPU.
Tensor llama_logits_gpu_last(const model::LLaMAModel &model, const float *normed_host, size_t hidden_size)
{
    const auto &cfg = model.config();
    const quant::QuantWeight *lq = model.lm_head_q();
    if (lq == nullptr || lq->gpu_data == nullptr || lq->dev_stride == 0 || lq->out != cfg.vocab_size ||
        lq->in != hidden_size || (lq->in % 32) != 0) {
        return Tensor({0});
    }

    core::Tensor x_host({1, hidden_size}, hal::DType::Float32);
    std::memcpy(x_host.data(), normed_host, hidden_size * sizeof(float));
    core::Tensor x = x_host.cast(hal::DType::Float16);
    x.set_device(hal::DeviceType::CUDA);
    const void *norm_ptr = x.raw_data_gpu();

    Tensor logits({1, cfg.vocab_size}, hal::DType::Float32);
    logits.alloc_gpu();
    constexpr size_t TILE_V = 8192;
    static core::Tensor lm_tile;
    static size_t lm_tile_cap = 0;
    const size_t tile_elems = TILE_V * hidden_size;
    if (lm_tile_cap < tile_elems) {
        lm_tile = core::Tensor::gpu_only({tile_elems}, hal::DType::Float16);
        lm_tile_cap = tile_elems;
    }
    for (size_t v0 = 0; v0 < cfg.vocab_size; v0 += TILE_V) {
        const size_t nv = std::min(TILE_V, cfg.vocab_size - v0);
        const uint8_t *w_row = static_cast<const uint8_t *>(lq->gpu_data) + v0 * lq->dev_stride;
        hal::dequant_qw_f16(w_row, lq->dev_kind, (int)hidden_size, (int)nv, (int)lq->dev_stride, (int)lq->dev_qs,
                            (int)lq->dev_d, (int)lq->dev_m, lm_tile.raw_data_gpu());
        core::blas::logits_gemm_mixed_cublas(norm_ptr, lm_tile.raw_data_gpu(), logits.raw_data_gpu(), 1,
                                             cfg.vocab_size, v0, nv, hidden_size, 0.0f);
    }
    logits.mark_dirty(Tensor::DataLocation::GPU);
    logits.syncToHost();
    return logits;
}
#endif

}  // namespace infer
}  // namespace serius
