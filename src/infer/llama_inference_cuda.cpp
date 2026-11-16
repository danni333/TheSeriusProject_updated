// CUDA inference path for LLaMA — split from llama_inference.cpp.
// Contains the cuBLAS prefill, fused prefill weights, CUDA-graph decode
// step and single-token CUDA forward. Everything is inside
// SERIUS_USE_CUDA guards, matching the original layout.
#include "llama_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

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

// Portable non-empty env check (getenv_s is MSVC-only).
static bool env_nonempty(const char *name) {
    const char *v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
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

// --- CUDA Implementations ---
#ifdef SERIUS_USE_CUDA
// last_only: compute logits for the final position only ({1, vocab} instead of
// {seq_len, vocab}) — chat/server callers only ever read the last row, and the
// full logits tensor is seq_len*vocab*4 bytes of host memset + D2H copy.
#ifdef SERIUS_USE_CUDA
// Fused dense QKV / gate|up weights for batched prefill: 4 GEMMs per layer
// instead of 7 (better cuBLAS efficiency at M=seq_len). Built once per model
// on GPU; the separate dense q/k/v/gate/up tensors are then released — decode
// uses the quantized weights whenever all four quant tensors are resident
// (same gate as the decode `use_quant` path), so the dense copies are dead.
struct FusedPrefillW {
    std::vector<core::Tensor> qkv;    // {hidden, hidden + 2*kv_dim} per layer
    std::vector<core::Tensor> gu;     // {hidden, 2*inter} per layer
    std::vector<char> fused;          // per-layer flag
    size_t hidden = 0, kv_dim = 0, inter = 0;
    const void *fingerprint = nullptr;
};

static const FusedPrefillW *fused_prefill_weights(const model::LLaMAModel &model,
                                                  hal::DType dtype) {
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
    if (it != cache.end() && it->second.fingerprint == fp &&
        it->second.hidden == hidden && it->second.kv_dim == kv_dim &&
        it->second.inter == inter && it->second.qkv.size() == model.layers().size()) {
        return &it->second;
    }
    cache.erase(&model);

    auto ok_shape = [](const core::Tensor &t, size_t r, size_t c) {
        return t.dtype() == hal::DType::Float16 && t.shape().size() == 2 &&
               t.shape()[0] == r && t.shape()[1] == c && t.raw_data_gpu() != nullptr;
    };

    FusedPrefillW w;
    w.hidden = hidden; w.kv_dim = kv_dim; w.inter = inter; w.fingerprint = fp;
    const auto &layers = model.layers();
    w.qkv.resize(layers.size()); w.gu.resize(layers.size());
    w.fused.assign(layers.size(), 0);

    size_t built = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto *blk = layers[i].get();  // mutable: unique_ptr yields LLaMABlock*
        // Decode must be able to run fully quantized before we drop the dense
        // originals (mirrors the use_quant gate in llama_forward_single_token_cuda).
        const bool quant_resident = blk->qkv_w.gpu_data && blk->gate_up_w.gpu_data &&
                                    blk->o_w.gpu_data && blk->down_w.gpu_data;
        if (!quant_resident) continue;
        if (!ok_shape(blk->q_proj->data, hidden, hidden) ||
            !ok_shape(blk->k_proj->data, hidden, kv_dim) ||
            !ok_shape(blk->v_proj->data, hidden, kv_dim) ||
            !ok_shape(blk->gate_proj->data, hidden, inter) ||
            !ok_shape(blk->up_proj->data, hidden, inter)) continue;

        auto qkv_t = core::Tensor::gpu_only({hidden, hidden + 2 * kv_dim}, dtype);
        hal::concat_rows_f16(qkv_t.raw_data_gpu(), blk->q_proj->data.raw_data_gpu(),
                             blk->k_proj->data.raw_data_gpu(), blk->v_proj->data.raw_data_gpu(),
                             hidden, hidden, kv_dim, kv_dim, 0);
        auto gu_t = core::Tensor::gpu_only({hidden, 2 * inter}, dtype);
        hal::concat_rows_f16(gu_t.raw_data_gpu(), blk->gate_proj->data.raw_data_gpu(),
                             blk->up_proj->data.raw_data_gpu(), nullptr,
                             hidden, inter, inter, 0, 0);
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
    if (built == 0) { cache.erase(&model); return nullptr; }
    // The dense tensors freed above land in the backend's 1 GiB free-list
    // cache instead of returning to the driver — flush it so fused weights
    // don't stack on top of them and overcommit VRAM.
    if (hal::IBackend *be = hal::get_gpu_backend()) be->flush_cache();
    return &cache.emplace(&model, std::move(w)).first->second;
}
#endif

static Tensor llama_forward_cuda_impl(const model::LLaMAModel &model, const Tensor &token_ids,
                                      KVCache *cache, bool last_only) {
    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size(), hidden = cfg.hidden_size, n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads, head_dim = hidden / n_heads, kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps, rope_base = cfg.rope_theta;
    hal::DType dtype = model.embed_tokens()->data.dtype();
    const hal::DType stream_dtype = hal::DType::Float32;  // FP32 residual stream

    const bool prof_p = env_nonempty("SERIUS_PROF");
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
    // Fused-GEMM outputs (only used when fused_prefill_weights is available).
    Tensor qkv_buf = Tensor::gpu_only({seq_len, hidden + 2 * kv_dim}, dtype);
    Tensor gu_buf = Tensor::gpu_only({seq_len, 2 * inter}, dtype);
    const FusedPrefillW *fpw = fused_prefill_weights(model, dtype);

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
        if (fpw && fpw->fused[layer_idx]) {
            // One fused GEMM {seq,in}x{in,hidden+2kv} then split q|k|v columns.
            core::blas::matmul(normed_buf, fpw->qkv[layer_idx], qkv_buf);
            const size_t qkv_total = hidden + 2 * kv_dim;
            hal::extract_cols_f16(query_buf.raw_data_gpu(), qkv_buf.raw_data_gpu(),
                                  seq_len, qkv_total, 0, hidden);
            hal::extract_cols_f16(key_buf.raw_data_gpu(), qkv_buf.raw_data_gpu(),
                                  seq_len, qkv_total, hidden, kv_dim);
            hal::extract_cols_f16(value_buf.raw_data_gpu(), qkv_buf.raw_data_gpu(),
                                  seq_len, qkv_total, hidden + kv_dim, kv_dim);
        } else {
            core::blas::matmul(normed_buf, blk->q_proj->data, query_buf);
            core::blas::matmul(normed_buf, blk->k_proj->data, key_buf);
            core::blas::matmul(normed_buf, blk->v_proj->data, value_buf);
        }
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
        if (fpw && fpw->fused[layer_idx]) {
            core::blas::matmul(normed_buf, fpw->gu[layer_idx], gu_buf);
            hal::extract_cols_f16(mlp_gate.raw_data_gpu(), gu_buf.raw_data_gpu(),
                                  seq_len, 2 * inter, 0, inter);
            hal::extract_cols_f16(mlp_up.raw_data_gpu(), gu_buf.raw_data_gpu(),
                                  seq_len, 2 * inter, inter, inter);
        } else {
            core::blas::matmul(normed_buf, blk->gate_proj->data, mlp_gate);
            core::blas::matmul(normed_buf, blk->up_proj->data, mlp_up);
        }
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
    Tensor logits_host;       // persistent D2H target for the logits view
    int *io_gpu = nullptr;    // single 2-int device buffer: [token_id, pos]
    int *tid_gpu = nullptr;   // aliases into io_gpu
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
    if (c.io_gpu) { cudaFree(c.io_gpu); c.io_gpu = nullptr; }
    c.tid_gpu = nullptr;
    c.pos_gpu = nullptr;
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
            if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
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
        if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
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
    // The MLP down-projection delta is kept pending and folded into the next
    // RMSNorm (this layer's input norm, or the final norm after the loop), so
    // each residual add costs zero extra launches.
    const void *pending_delta = nullptr;
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx].get();
        if (pending_delta != nullptr) {
            hal::rmsnorm_residual_mixed(ctx.normed_buf.raw_data_gpu(),
                                        ctx.hidden_states_gpu.raw_data_gpu(), pending_delta,
                                        blk->input_layernorm_weight->data.raw_data_gpu(),
                                        hidden, 1, eps, dtype, dtype, s);
        } else {
            hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                               blk->input_layernorm_weight->data.raw_data_gpu(), hidden, 1, eps,
                               dtype, s);
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
        hal::rope_bias_devpos_qk(qptr, kptr, head_dim, n_heads, n_kv_heads, rope_base,
                                 ctx.pos_gpu,
                                 blk->q_bias ? blk->q_bias->data.raw_data_gpu() : nullptr,
                                 blk->k_bias ? blk->k_bias->data.raw_data_gpu() : nullptr,
                                 (dtype != hal::DType::Float32) ? q_scale : 1.0F, 1.0F,
                                 dtype, s);

        // Fused K+V cache write at the device-side position.
        KVCacheLayer &cl = cache->layers[layer_idx];
        hal::kv_write_kv_devpos(cl.k.raw_data_gpu(), cl.v.raw_data_gpu(), kptr, vptr,
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
        hal::rmsnorm_residual_mixed(ctx.normed_buf.raw_data_gpu(),
                                    ctx.hidden_states_gpu.raw_data_gpu(),
                                    ctx.mlp_down.raw_data_gpu(),
                                    blk->post_attention_layernorm_weight->data.raw_data_gpu(),
                                    hidden, 1, eps, dtype, dtype, s);
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
        hal::rmsnorm_residual_mixed(ctx.normed_buf.raw_data_gpu(),
                                    ctx.hidden_states_gpu.raw_data_gpu(), pending_delta,
                                    model.norm_weight()->data.raw_data_gpu(),
                                    hidden, 1, eps, dtype, dtype, s);
    } else {
        hal::rmsnorm_mixed(ctx.normed_buf.raw_data_gpu(), ctx.hidden_states_gpu.raw_data_gpu(),
                           model.norm_weight()->data.raw_data_gpu(), hidden, 1, eps, dtype, s);
    }
    ctx.normed_buf.mark_dirty(Tensor::DataLocation::GPU);
    const quant::QuantWeight *lq = model.lm_head_q();
    // FP16 x in, FP32 logits out — still skips the input convert_dtype.
    const void *sx = nullptr;
    if (lq->dev_kind == 3 || lq->dev_kind == 4 || lq->dev_kind == 6) {
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

// Returns a non-owning host+device view over the persistent graph buffers
// (empty Tensor when the graph path is unavailable — caller falls back to the
// eager CUDA path). The view aliases ctx.logits_dev / ctx.logits_host: valid
// until the next decode call or graph_ctx rebuild, so callers must consume it
// immediately (sample, don't store).
Tensor llama_forward_single_token_cuda_graph(const model::LLaMAModel &model, int token_id,
                                             size_t pos, KVCache *cache) {
    if (cache == nullptr || pos + 1 > 4096) return Tensor();
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
        model.lm_head_q() == nullptr || model.lm_head_q()->gpu_data == nullptr ||
        model.lm_head_q()->dev_stride == 0 || model.lm_head_q()->out != cfg.vocab_size) {
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
    CHECK_CUDA(cudaMemcpyAsync(ctx.io_gpu, io_host, sizeof(io_host),
                               cudaMemcpyHostToDevice, 0));

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
    CHECK_CUDA(cudaMemcpy(ctx.logits_host.data(), ctx.logits_dev.raw_data_gpu(),
                          cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return Tensor::host_gpu_view(ctx.logits_host.data(), ctx.logits_dev.raw_data_gpu(),
                                 {1, cfg.vocab_size}, hal::DType::Float32);
}
#endif

Tensor llama_forward_single_token_cuda(const model::LLaMAModel &model, int token_id, size_t pos, KVCache *cache) {
    static const bool gpu_time = env_nonempty("SERIUS_GPU_TIME");
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
                if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
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
                if (qw.dev_kind == 3 || qw.dev_kind == 4 || qw.dev_kind == 6) {
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
            if (lq->dev_kind == 3 || lq->dev_kind == 4 || lq->dev_kind == 6) {
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

