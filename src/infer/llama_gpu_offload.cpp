#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "llama_inference.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "../blas/blas.h"
#include "../quant/quant_weight.h"
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

    // Dense FP16 params uploaded by the offload (embed + per-block weights).
    const size_t per_layer_params =
        hidden * hidden +            // q_proj
        2 * hidden * kv_dim +        // k_proj + v_proj
        hidden * hidden +            // o_proj
        3 * hidden * inter;          // gate + up + down
    size_t dense_params = vocab * hidden + n_layers * per_layer_params;
    size_t need = dense_params * 2;  // FP16

    // lm_head copies: FP16 transposed always; FP32 only without packed lm_head.
    need += vocab * hidden * 2;
    const auto *lm_head_q = model.lm_head_q();
    if (lm_head_q == nullptr) {
        need += vocab * hidden * 4;
    }

    // Packed quantized weights kept alongside the dense FP16 copies.
    for (const auto &blk : model.layers()) {
        need += quant_weight_bytes(blk->qkv_w);
        need += quant_weight_bytes(blk->gate_up_w);
        need += quant_weight_bytes(blk->o_w);
        need += quant_weight_bytes(blk->down_w);
    }
    if (lm_head_q != nullptr) {
        need += quant_weight_bytes(*lm_head_q);
    }

    // KV cache on GPU: K+V per layer, FP16, sized on max context (capped —
    // prompts rarely reach the full training context).
    const size_t max_seq = cfg.max_position_embeddings > 0
        ? std::min<size_t>(cfg.max_position_embeddings, 16384)
        : 4096;
    need += n_layers * max_seq * kv_dim * 2 * 2;  // K+V, fp16

    // cuBLAS/prefill scratch + decode-graph buffers + fragmentation margin.
    need = need * 6 / 5 + (size_t{512} << 20);
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
    plan.fit = (need <= free_b * 9 / 10) ? LlamaGpuFit::kFits
                                        : LlamaGpuFit::kInsufficientVram;
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
    const size_t per_layer =
        hidden * hidden +                 // q_proj
        2 * hidden * kv_dim +             // k_proj + v_proj
        hidden * hidden +                 // o_proj
        3 * hidden * inter;               // gate + up + down
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

    // Keep an FP32 transposed lm_head [vocab, hidden] on the GPU:
    // the dedicated logits GEMV kernel reads it coalesced, and FP32
    // preserves the fine-grained ranking over the 150k+ vocabulary.
    // The FP32 transposed lm_head (~933 MB VRAM) is only read by the
    // non-quantized logits fallback; skip it when a packed lm_head
    // exists. The FP16 copy is still needed by the prefill logits GEMV.
    const bool need_fp32_lm_head = !model.lm_head_q().valid();
    if (model.lm_head() != nullptr) {
        core::Tensor w_t = model.lm_head()->data.transpose(0, 1).contiguous();
        if (need_fp32_lm_head) {
            model.lm_head_fp32() = w_t;
            model.lm_head_fp32().set_device(hal::DeviceType::CUDA);
        }
        // FP16 copy for the cuBLAS mixed-precision logits GEMV (half the
        // memory traffic of the FP32 path; FP32 accumulation preserved).
        model.lm_head_fp16_t() = w_t.cast(hal::DType::Float16);
        model.lm_head_fp16_t().set_device(hal::DeviceType::CUDA);
    }
    auto move_param = [](core::autograd::Node *p) {
        if (p == nullptr) { return; }
        p->data = p->data.cast(hal::DType::Float16);
        p->data.set_device(hal::DeviceType::CUDA);
    };
    move_param(model.embed_tokens_raw());
    for (auto &blk : model.layers()) {
        move_param(blk->input_layernorm_weight.get());
        move_param(blk->q_proj.get());
        move_param(blk->k_proj.get());
        move_param(blk->v_proj.get());
        move_param(blk->q_bias.get());
        move_param(blk->k_bias.get());
        move_param(blk->v_bias.get());
        move_param(blk->o_proj.get());
        move_param(blk->post_attention_layernorm_weight.get());
        move_param(blk->gate_proj.get());
        move_param(blk->up_proj.get());
        move_param(blk->down_proj.get());
    }
    move_param(model.norm_weight());
    // Upload packed GGUF quantized weights (Q4_0/Q4_1/Q8_0) for the
    // CUDA decode GEMV kernels; no-op for non-quantized models.
    if (std::getenv("SERIUS_NO_QUANT") == nullptr) {
        hal::IBackend *be = hal::get_gpu_backend();
        auto upload_qw = [be](quant::QuantWeight &qw) {
            if (!qw.valid() || qw.gpu_data != nullptr) { return; }
            // Repack into the aligned layout the vectorized kernel reads.
            quant::repack_quant_for_gpu(qw);
            const std::vector<uint8_t> &src = qw.dev_stride ? qw.dev_bytes : qw.bytes;
            qw.gpu_data = be->malloc(src.size());
            be->memcpy_h2d(qw.gpu_data, src.data(), src.size());
        };
        for (auto &blk : model.layers()) {
            upload_qw(blk->qkv_w);
            upload_qw(blk->gate_up_w);
            upload_qw(blk->o_w);
            upload_qw(blk->down_w);
        }
        upload_qw(model.lm_head_q());
    }

    // cuBLAS warmup: the first GemmEx/GemmStridedBatchedEx per kernel
    // variant lazily loads driver modules (cuLibraryLoadData, ~300 ms
    // total) — pay it here during weight upload instead of inside the
    // first prompt's prefill. Shapes mirror the real prefill GEMMs.
    {
        hal::IBackend *gbe = hal::get_gpu_backend();
        const size_t hidden = model.config().hidden_size;
        const size_t inter = model.config().intermediate_size;
        const size_t heads = model.config().num_attention_heads;
        const size_t head_dim = hidden / heads;
        const size_t wm = 256;  // representative M (seq_len)
        const size_t max_k = (std::max)(hidden, inter);
        void *buf = gbe->malloc(wm * max_k * 2 + 4 * 1024 * 1024);
        auto f16_view = [&](size_t m, size_t k) {
            return core::Tensor::gpu_view(buf, {m, k}, hal::DType::Float16);
        };
        const auto &l0 = model.layers()[0];
        struct WP { core::autograd::Node *w; } wps[] = {
            {l0->q_proj.get()}, {l0->k_proj.get()}, {l0->v_proj.get()},
            {l0->o_proj.get()}, {l0->gate_proj.get()}, {l0->up_proj.get()},
            {l0->down_proj.get()},
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

}  // namespace infer
}  // namespace serius
