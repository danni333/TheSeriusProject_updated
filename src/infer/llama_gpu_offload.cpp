#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include "../blas/blas.h"
#include "../quant/quant_weight.h"
#include "llama_inference.h"
#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>

#include "../blas/matmul_cuda.h"
#include "../hal/device.h"
#endif

namespace serius {
namespace infer {

namespace {

/// Bytes currently held by a packed quantized weight (host or device copy).
size_t quant_weight_bytes(const quant::QuantWeight &qw)
{
    if (!qw.valid()) {
        return 0;
    }
    return qw.dev_stride ? qw.dev_bytes.size() : qw.bytes.size();
}

/// Whether a packed weight can live on the GPU: valid, 32-aligned inner dim,
/// and a type the repack/GEMV kernels support.
bool qw_gpu_supported(const quant::QuantWeight &qw)
{
    if (!qw.valid() || (qw.in % 32) != 0) {
        return false;
    }
    switch (qw.type) {
        case quant::QuantType::Q4_0:
        case quant::QuantType::Q4_1:
        case quant::QuantType::Q8_0:
            return true;
        case quant::QuantType::Q4_K:
        case quant::QuantType::Q5_K:
            return (qw.in % 256) == 0;
        default:
            return false;
    }
}

/// True when packed quantized weights may be uploaded to the GPU at all.
bool quant_upload_enabled()
{
    return std::getenv("SERIUS_NO_QUANT") == nullptr;
}

/// True when every layer's fused projections are packed-quantized in a GPU
/// supported layout — the offload can then skip the dense FP16 projection
/// copies entirely (decode is already fully quantized; batched prefill
/// dequantizes on the fly). The quantized CUDA paths require FP16
/// activations, which the offload itself guarantees by casting the
/// embedding (and every moved param) to Float16.
bool model_quant_layers(const model::LLaMAModel &model)
{
    if (!quant_upload_enabled() || model.layers().empty()) {
        return false;
    }
    for (const auto &blk : model.layers()) {
        if (!qw_gpu_supported(blk->qkv_w) || !qw_gpu_supported(blk->gate_up_w) || !qw_gpu_supported(blk->o_w) ||
            !qw_gpu_supported(blk->down_w)) {
            return false;
        }
    }
    return true;
}

}  // namespace

LlamaGpuPlan plan_llama_gpu_offload(const model::LLaMAModel &model)
{
    LlamaGpuPlan plan;
    const auto &cfg = model.config();
    const size_t hidden = cfg.hidden_size;
    const size_t inter = cfg.intermediate_size;
    const size_t kv_dim = cfg.num_key_value_heads * (hidden / cfg.num_attention_heads);
    const size_t vocab = cfg.vocab_size;
    const size_t n_layers = cfg.num_hidden_layers;

    // GPU KV cache starts at 1024 positions/layer and grows on demand
    // (ensure_kv_capacity_gpu) — size the estimate to the initial footprint;
    // long contexts add KV bytes at runtime.
    const size_t kv_bytes = n_layers * 1024 * kv_dim * 2 * 2;  // K+V, fp16

    const bool quant = model_quant_layers(model);
    const auto *lm_head_q = model.lm_head_q();
    // The packed lm_head can replace BOTH dense transposed copies only when
    // it will actually be repacked+uploaded (qw_gpu_supported predicts that)
    // and has the [vocab, hidden] shape the logits kernels expect. FP16
    // activations come for free: the offload casts embed_tokens to Float16.
    const bool quant_lm = quant_upload_enabled() && lm_head_q != nullptr && qw_gpu_supported(*lm_head_q) &&
                          lm_head_q->out == vocab && lm_head_q->in == hidden;

    size_t need = 0;
    if (quant) {
        // Packed-quant mode: embed table, norms and biases stay dense FP16;
        // projections exist on the GPU only as packed QuantWeights.
        need += vocab * hidden * 2;                        // embed_tokens
        need += n_layers * (3 * hidden + 2 * kv_dim) * 2;  // 2 LN + q/k/v bias
        need += hidden * 2;                                // final norm
        for (const auto &blk : model.layers()) {
            need += quant_weight_bytes(blk->qkv_w);
            need += quant_weight_bytes(blk->gate_up_w);
            need += quant_weight_bytes(blk->o_w);
            need += quant_weight_bytes(blk->down_w);
        }
        // Batched-prefill dequant scratch: dq + dq_t, each max(in*out) FP16.
        size_t max_elems = 0;
        for (const auto &blk : model.layers()) {
            const quant::QuantWeight *ws[] = {&blk->qkv_w, &blk->gate_up_w, &blk->o_w, &blk->down_w};
            for (const auto *qw : ws) {
                max_elems = std::max(max_elems, qw->in * qw->out);
            }
        }
        need += max_elems * 2 * 2;
        need += kv_bytes;
    } else {
        // Dense FP16 params uploaded by the offload (embed + per-block weights).
        const size_t per_layer_params = hidden * hidden +      // q_proj
                                        2 * hidden * kv_dim +  // k_proj + v_proj
                                        hidden * hidden +      // o_proj
                                        3 * hidden * inter;    // gate + up + down
        size_t dense_params = vocab * hidden + n_layers * per_layer_params;
        need = dense_params * 2;  // FP16
        need += kv_bytes;

        // Packed weights still upload alongside the dense copies whenever
        // the repack supports them (mixed-quant models keep both paths).
        for (const auto &blk : model.layers()) {
            const quant::QuantWeight *ws[] = {&blk->qkv_w, &blk->gate_up_w, &blk->o_w, &blk->down_w};
            for (const auto *qw : ws) {
                if (qw_gpu_supported(*qw)) {
                    need += quant_weight_bytes(*qw);
                }
            }
        }
        // Batched-prefill dequant scratch is still needed for layers whose
        // packed weights did upload even in dense mode.
        size_t max_elems = 0;
        for (const auto &blk : model.layers()) {
            const quant::QuantWeight *ws[] = {&blk->qkv_w, &blk->gate_up_w, &blk->o_w, &blk->down_w};
            for (const auto *qw : ws) {
                if (qw_gpu_supported(*qw)) {
                    max_elems = std::max(max_elems, qw->in * qw->out);
                }
            }
        }
        need += max_elems * 2 * 2;
    }

    // lm_head: packed + dequant tile scratch when GPU-ready, otherwise the
    // FP16 transposed copy (always needed by the prefill logits GEMV) plus
    // the FP32 copy only when no packed lm_head exists at all.
    if (quant_lm) {
        need += quant_weight_bytes(*lm_head_q);
        need += 8192 * hidden * 2;
    } else {
        need += vocab * hidden * 2;
        if (lm_head_q == nullptr) {
            need += vocab * hidden * 4;
        } else if (quant_upload_enabled() && qw_gpu_supported(*lm_head_q)) {
            need += quant_weight_bytes(*lm_head_q);  // uploaded, logits gates skip it
        }
    }

    // Prefill scratch + decode buffers + fragmentation margin.
    need = need * 6 / 5 + (size_t{256} << 20);
    plan.need_bytes = need;

#ifdef SERIUS_USE_CUDA
    if (std::getenv("SERIUS_NO_GPU") != nullptr) {
        plan.fit = LlamaGpuFit::kNoCuda;
        return plan;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        plan.fit = LlamaGpuFit::kNoCuda;
        return plan;
    }
    plan.free_bytes = free_b;
    // Keep 10% of free VRAM as headroom for other allocations/fragmentation.
    plan.fit = (need <= free_b * 9 / 10) ? LlamaGpuFit::kFits : LlamaGpuFit::kInsufficientVram;
#else
    plan.fit = LlamaGpuFit::kNoCuda;
#endif
    return plan;
}

size_t estimate_llama_params(const model::LLaMAModel &model)
{
    const auto &cfg = model.config();
    const size_t hidden = cfg.hidden_size;
    const size_t inter = cfg.intermediate_size;
    const size_t kv_dim = cfg.num_key_value_heads * (hidden / cfg.num_attention_heads);
    const size_t per_layer = hidden * hidden +      // q_proj
                             2 * hidden * kv_dim +  // k_proj + v_proj
                             hidden * hidden +      // o_proj
                             3 * hidden * inter;    // gate + up + down
    return cfg.vocab_size * hidden + cfg.num_hidden_layers * per_layer;
}

LlamaGpuDecision resolve_llama_gpu(const model::LLaMAModel &model, LlamaGpuMode mode)
{
    LlamaGpuDecision d;
    d.est_params = estimate_llama_params(model);
    if (mode == LlamaGpuMode::kCpu) {
        d.reason = "cpu requested";
        return d;
    }
    const LlamaGpuPlan plan = plan_llama_gpu_offload(model);
    d.fit = plan.fit;
    d.need_bytes = plan.need_bytes;
    d.free_bytes = plan.free_bytes;
    if (mode == LlamaGpuMode::kCuda) {
        // Forced GPU: offload whenever CUDA exists; the caller logs a warning
        // when the estimate says it might not fit.
        d.use_gpu = (plan.fit != LlamaGpuFit::kNoCuda);
        d.reason = (plan.fit == LlamaGpuFit::kInsufficientVram)
                       ? "cuda requested but estimated VRAM need exceeds free memory"
                       : "cuda requested";
        return d;
    }
    // kAuto: offload only when the footprint fits safely.
    d.use_gpu = (plan.fit == LlamaGpuFit::kFits);
    if (plan.fit == LlamaGpuFit::kFits) {
        d.reason = "auto: fits in free VRAM";
    } else if (plan.fit == LlamaGpuFit::kInsufficientVram) {
        d.reason = "auto: insufficient VRAM";
    } else {
        d.reason = "auto: CUDA unavailable";
    }
    return d;
}

bool offload_llama_model_to_gpu(model::LLaMAModel &model)
{
#ifdef SERIUS_USE_CUDA
    if (std::getenv("SERIUS_NO_GPU") != nullptr) {
        return false;
    }

    // Packed GGUF quantized weights (Q4_0/Q4_1/Q8_0/Q4_K) go first: whether
    // the dense FP16 projection copies are needed at all is decided by the
    // residency check below, not by a prediction.
    if (std::getenv("SERIUS_NO_QUANT") == nullptr) {
        hal::IBackend *be = hal::get_gpu_backend();
        auto upload_qw = [be](quant::QuantWeight &qw) {
            if (!qw.valid() || qw.gpu_data != nullptr) {
                return;
            }
            // Repack into the aligned layout the vectorized kernels read.
            // Types/layouts it rejects stay host-only: gpu_data is only ever
            // set for the repacked dev_bytes layout (dev_stride != 0), so a
            // non-null gpu_data always means a usable GPU-resident weight.
            quant::repack_quant_for_gpu(qw);
            if (qw.dev_stride == 0) {
                return;
            }
            qw.gpu_data = be->malloc(qw.dev_bytes.size());
            be->memcpy_h2d(qw.gpu_data, qw.dev_bytes.data(), qw.dev_bytes.size());
        };
        for (auto &blk : model.layers()) {
            upload_qw(blk->qkv_w);
            upload_qw(blk->gate_up_w);
            upload_qw(blk->o_w);
            upload_qw(blk->down_w);
        }
        upload_qw(model.lm_head_q());
    }

    // A projection set is GPU-quantized only when every fused weight was
    // repacked and uploaded — the CUDA-graph decode, the eager decode and
    // the batched prefill all have quantized paths for that case, so the
    // dense FP16 copies (the dominant VRAM cost for GGUF models) can be
    // skipped entirely. The FP32 host tensors remain as the CPU fallback /
    // source of truth. FP16 activations are guaranteed: move_param below
    // casts embed_tokens (and every moved param) to Float16.
    auto qw_ready = [](const quant::QuantWeight &qw) { return qw.gpu_data != nullptr && qw.dev_stride != 0; };
    bool quant_layers = !model.layers().empty();
    for (const auto &blk : model.layers()) {
        if (!qw_ready(blk->qkv_w) || !qw_ready(blk->gate_up_w) || !qw_ready(blk->o_w) || !qw_ready(blk->down_w)) {
            quant_layers = false;
            break;
        }
    }
    const auto &cfg = model.config();
    const auto &lq = model.lm_head_q();
    const bool quant_lm_head = lq.valid() && qw_ready(lq) && lq.out == cfg.vocab_size && lq.in == cfg.hidden_size;

    // Keep an FP32 transposed lm_head [vocab, hidden] on the GPU:
    // the dedicated logits GEMV kernel reads it coalesced, and FP32
    // preserves the fine-grained ranking over the 150k+ vocabulary.
    // The FP32 transposed lm_head (~933 MB VRAM) is only read by the
    // non-quantized logits fallback; skip it when a packed lm_head
    // exists. The FP16 copy is still needed by the prefill logits GEMV —
    // unless the packed lm_head feeds the tiled dequant head instead.
    if (model.lm_head() != nullptr && !quant_lm_head) {
        core::Tensor w_t = model.lm_head()->data.transpose(0, 1).contiguous();
        if (!model.lm_head_q().valid()) {
            model.lm_head_fp32() = w_t;
            model.lm_head_fp32().set_device(hal::DeviceType::CUDA);
        }
        // FP16 copy for the cuBLAS mixed-precision logits GEMV (half the
        // memory traffic of the FP32 path; FP32 accumulation preserved).
        model.lm_head_fp16_t() = w_t.cast(hal::DType::Float16);
        model.lm_head_fp16_t().set_device(hal::DeviceType::CUDA);
    }
    auto move_param = [](core::autograd::Node *p) {
        if (p == nullptr) {
            return;
        }
        p->data = p->data.cast(hal::DType::Float16);
        p->data.set_device(hal::DeviceType::CUDA);
    };
    move_param(model.embed_tokens_raw());
    for (auto &blk : model.layers()) {
        move_param(blk->input_layernorm_weight.get());
        move_param(blk->q_bias.get());
        move_param(blk->k_bias.get());
        move_param(blk->v_bias.get());
        move_param(blk->post_attention_layernorm_weight.get());
        if (!quant_layers) {
            move_param(blk->q_proj.get());
            move_param(blk->k_proj.get());
            move_param(blk->v_proj.get());
            move_param(blk->o_proj.get());
            move_param(blk->gate_proj.get());
            move_param(blk->up_proj.get());
            move_param(blk->down_proj.get());
        }
    }
    move_param(model.norm_weight());

    // cuBLAS warmup: the first GemmEx/GemmStridedBatchedEx per kernel
    // variant lazily loads driver modules (cuLibraryLoadData, ~300 ms
    // total) — pay it here during weight upload instead of inside the
    // first prompt's prefill. Shapes mirror the real prefill GEMMs.
    // Skipped in packed-quant mode: the dense projection tensors are not on
    // the GPU (prefill dequantizes into scratch instead).
    if (!quant_layers) {
        hal::IBackend *gbe = hal::get_gpu_backend();
        const size_t hidden = model.config().hidden_size;
        const size_t inter = model.config().intermediate_size;
        const size_t heads = model.config().num_attention_heads;
        const size_t head_dim = hidden / heads;
        const size_t wm = 256;  // representative M (seq_len)
        const size_t max_k = (std::max)(hidden, inter);
        void *buf = gbe->malloc(wm * max_k * 2 + 4 * 1024 * 1024);
        auto f16_view = [&](size_t m, size_t k) { return core::Tensor::gpu_view(buf, {m, k}, hal::DType::Float16); };
        const auto &l0 = model.layers()[0];
        struct WP
        {
            core::autograd::Node *w;
        } wps[] = {
            {l0->q_proj.get()},    {l0->k_proj.get()},  {l0->v_proj.get()},    {l0->o_proj.get()},
            {l0->gate_proj.get()}, {l0->up_proj.get()}, {l0->down_proj.get()},
        };
        for (auto &wp : wps) {
            if (wp.w == nullptr) continue;
            const auto &ws = wp.w->data.shape();
            const size_t K = ws[0], N = ws.size() > 1 ? ws[1] : ws[0];
            core::Tensor A = f16_view(wm, K);
            core::Tensor C = core::Tensor::gpu_view(buf, {wm, N}, hal::DType::Float16);
            core::blas::matmul(A, wp.w->data, C);
        }
        // Batched attention shapes: {heads, seq, head_dim} variants.
        {
            core::Tensor A = core::Tensor::gpu_view(buf, {heads, wm, head_dim}, hal::DType::Float16);
            core::Tensor B = core::Tensor::gpu_view(buf, {heads, head_dim, wm}, hal::DType::Float16);
            core::Tensor C = core::Tensor::gpu_view(buf, {heads, wm, wm}, hal::DType::Float16);
            core::blas::batched_matmul(A, B, C);
            core::Tensor D = core::Tensor::gpu_view(buf, {heads, wm, head_dim}, hal::DType::Float16);
            core::blas::batched_matmul(C, A, D);
        }
        // Logits GEMV (M=1 path used by llama_forward_inference_last).
        if (model.lm_head_fp16_t().data_gpu() != nullptr) {
            core::Tensor x = f16_view(1, hidden);
            core::Tensor out = core::Tensor::gpu_view(buf, {1, model.config().vocab_size}, hal::DType::Float32);
            core::blas::logits_gemv_mixed_cublas(x.raw_data_gpu(), model.lm_head_fp16_t().raw_data_gpu(),
                                                 out.raw_data_gpu(), 1, model.config().vocab_size, hidden);
        }
        cudaDeviceSynchronize();
        gbe->free(buf);
    }

    // Real-path prefill warmup: pays the one-time costs the shape
    // warmup above can't — first-touch page faults on the prefill
    // scratch arena, custom kernel module loads, and cuBLASLt
    // heuristic selection for the actual projection shapes.
    {
        KVCache warm_cache;
        const size_t wn = 128;
        core::Tensor warm_in({wn});
        for (size_t i = 0; i < wn; ++i) {
            warm_in[i] = static_cast<float>(1000 + (i % 1000));
        }
        (void)llama_forward_inference_last(model, warm_in, &warm_cache);
    }

    return true;
#else
    (void)model;
    return false;
#endif
}

namespace {

/// Models with a partial GPU offload: leading N layers resident on device.
/// Keyed by model address — the registry lives as long as the process and
/// model objects are unique_ptr-stable for the CLI/server lifetime.
std::unordered_map<const model::LLaMAModel *, size_t> &partial_layers_registry()
{
    static std::unordered_map<const model::LLaMAModel *, size_t> reg;
    return reg;
}

}  // namespace

size_t partial_gpu_layers(const model::LLaMAModel &model)
{
    const auto &reg = partial_layers_registry();
    const auto it = reg.find(&model);
    return it == reg.end() ? 0 : it->second;
}

bool offload_llama_model_partial(model::LLaMAModel &model, size_t n_layers)
{
#ifdef SERIUS_USE_CUDA
    hal::IBackend *be = hal::get_gpu_backend();
    if (be == nullptr || !be->is_available()) {
        return false;
    }
    const size_t total = model.layers().size();
    if (n_layers == 0 || n_layers >= total) {
        return false;  // 0 or "all" is handled by the full-offload path
    }

    // Packed-quant prefix (v2): upload the fused QuantWeights for the GPU
    // layers — ~0.55 B/param vs ~4 B/param FP32, so many more layers fit and
    // the prefix runs the same fast GEMV kernels as full offload. Layers
    // whose packed weights can't be repacked/uploaded fall back to dense
    // FP16 copies (still much faster than the old FP32 path).
    auto upload_qw = [be](quant::QuantWeight &qw) {
        if (!qw.valid() || qw.gpu_data != nullptr) {
            return;
        }
        quant::repack_quant_for_gpu(qw);
        if (qw.dev_stride == 0) {
            return;
        }
        qw.gpu_data = be->malloc(qw.dev_bytes.size());
        be->memcpy_h2d(qw.gpu_data, qw.dev_bytes.data(), qw.dev_bytes.size());
    };
    auto qw_ready = [](const quant::QuantWeight &qw) { return qw.gpu_data != nullptr && qw.dev_stride != 0; };
    // FP16 param copies for the GPU prefix. cast() replaces the tensor, so
    // the host side becomes FP16 too — fine: prefix params are never read
    // by the CPU suffix (which only touches layers [n_layers, total)).
    auto move_param_f16 = [](core::autograd::Node *p) {
        if (p != nullptr) {
            p->data = p->data.cast(hal::DType::Float16);
            p->data.set_device(hal::DeviceType::CUDA);
        }
    };

    move_param_f16(model.embed_tokens_raw());
    size_t quant_layers = 0, dense_layers = 0;
    for (size_t i = 0; i < n_layers; ++i) {
        auto &blk = model.layers()[i];
        move_param_f16(blk->input_layernorm_weight.get());
        move_param_f16(blk->q_bias.get());
        move_param_f16(blk->k_bias.get());
        move_param_f16(blk->v_bias.get());
        move_param_f16(blk->post_attention_layernorm_weight.get());
        upload_qw(blk->qkv_w);
        upload_qw(blk->gate_up_w);
        upload_qw(blk->o_w);
        upload_qw(blk->down_w);
        if (qw_ready(blk->qkv_w) && qw_ready(blk->gate_up_w) && qw_ready(blk->o_w) && qw_ready(blk->down_w)) {
            ++quant_layers;
        } else {
            ++dense_layers;
            move_param_f16(blk->q_proj.get());
            move_param_f16(blk->k_proj.get());
            move_param_f16(blk->v_proj.get());
            move_param_f16(blk->o_proj.get());
            move_param_f16(blk->gate_proj.get());
            move_param_f16(blk->up_proj.get());
            move_param_f16(blk->down_proj.get());
        }
    }
    // norm_weight stays CPU-only (the suffix computes it there). The packed
    // lm_head IS uploaded when possible: the logits GEMV is ~20-25% of CPU
    // suffix time on large-vocab models and runs in <1 ms on GPU. VRAM is
    // not planned for the partial path — a failed upload just keeps the CPU
    // logits path (gpu_data stays null).
    try {
        upload_qw(model.lm_head_q());
    } catch (...) {
    }

    partial_layers_registry()[&model] = n_layers;
    std::fprintf(stderr,
                 "[offload] partial: %zu/%zu layers on GPU (%zu packed-quant, %zu dense-FP16; CPU suffix keeps FP32 "
                 "host weights)\n",
                 n_layers, total, quant_layers, dense_layers);
    return true;
#else
    (void)model;
    (void)n_layers;
    return false;
#endif
}

}  // namespace infer
}  // namespace serius
