#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

#include "infer/llama_inference.h"
#include "infer/activation_hook.h"
#if defined(SERIUS_EXPERIMENTAL_INTERPRET)
#include "interpret/inference_capture.h"
#endif
#include "io/hf_llama_loader.h"
#include "io/gguf_loader.h"
#include "transformer/llama_model.h"
#include "tensor/tensor.h"
#include "blas/blas.h"
#include "hal/device.h"

using namespace serius;
using core::Tensor;
using namespace serius::model;
using namespace serius::infer;

class LLaMAInferenceCUDATest : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef SERIUS_USE_CUDA
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }

    LLamaConfig get_test_config() {
        LLamaConfig cfg;
        cfg.vocab_size = 128;
        cfg.hidden_size = 64;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;  // GQA: exercises kv-head mapping in batched path
        cfg.num_hidden_layers = 2;
        cfg.intermediate_size = 128;
        cfg.rms_norm_eps = 1e-5f;
        return cfg;
    }

    void randomize_model(LLaMAModel& model) {
        auto randomize_node = [](core::autograd::Node* node) {
            for (size_t i = 0; i < node->data.size(); ++i) {
                node->data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.2f - 0.1f;
            }
        };

        randomize_node(model.embed_tokens());
        randomize_node(model.norm_weight());
        randomize_node(model.lm_head());

        for (const auto &layer : model.layers()) {
            randomize_node(layer->input_layernorm_weight.get());
            randomize_node(layer->q_proj.get());
            randomize_node(layer->k_proj.get());
            randomize_node(layer->v_proj.get());
            randomize_node(layer->o_proj.get());
            randomize_node(layer->post_attention_layernorm_weight.get());
            randomize_node(layer->gate_proj.get());
            randomize_node(layer->up_proj.get());
            randomize_node(layer->down_proj.get());
        }
    }
};

// llama_forward_inference returns [seq_len, vocab] on the batched-prefill
// paths but only the last position [1, vocab] on the per-token fallback
// (CPU batched prefill is opt-in via SERIUS_PREFILL_BATCH_CPU=1). Reduce any
// of those layouts to the last-position logits row for comparison.
static std::vector<float> last_position_logits(const Tensor &t)
{
    const auto &shape = t.shape();
    if (shape.empty() || t.size() == 0) { return {}; }
    const size_t vocab = shape.back();
    const size_t rows = shape.size() > 1 ? shape[shape.size() - 2] : 1;
    std::vector<float> out(vocab);
    const size_t base = (rows - 1) * vocab;
    for (size_t v = 0; v < vocab; ++v) { out[v] = t[base + v]; }
    return out;
}

TEST_F(LLaMAInferenceCUDATest, PrefillParity) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);

    size_t seq_len = 8;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(i % cfg.vocab_size);

    // CPU Forward
    model.set_device(hal::DeviceType::CPU);
    Tensor cpu_logits = llama_forward_inference(model, token_ids, nullptr);

    // GPU Forward
    model.set_device(hal::DeviceType::CUDA);
    Tensor gpu_logits = llama_forward_inference(model, token_ids, nullptr);

    // Cross-device parity on last-position logits: CPU may take the per-token
    // fallback ([1, vocab]) while GPU runs batched prefill ([seq, vocab]).
    // FP32 accumulation order differs between GPU parallel reductions and CPU
    // sequential FMA chains across multiple layers.
    // Tolerance of 0.15 covers expected numerical drift for 2-layer FP32 inference.
    const auto cpu_last = last_position_logits(cpu_logits);
    const auto gpu_last = last_position_logits(gpu_logits);
    ASSERT_EQ(cpu_last.size(), gpu_last.size());
    for (size_t i = 0; i < cpu_last.size(); ++i) {
        EXPECT_NEAR(cpu_last[i], gpu_last[i], 0.15f);
    }
}

TEST_F(LLaMAInferenceCUDATest, PrefillThenDecodeParity) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);
    size_t seq_len = 6;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(i % cfg.vocab_size);

    model.set_device(hal::DeviceType::CPU);
    serius::infer::KVCache cpu_cache;
    llama_forward_inference(model, token_ids, &cpu_cache);
    std::vector<Tensor> cpu_logits;
    for (int step = 0; step < 4; ++step) {
        cpu_logits.push_back(llama_forward_single_token(model, 7 + step, seq_len + step, &cpu_cache));
    }

    model.set_device(hal::DeviceType::CUDA);
    // Reset caches: GPU cache tensors were allocated by CPU path on host;
    // force fresh GPU cache by clearing.
    serius::infer::KVCache gpu_cache;
    llama_forward_inference(model, token_ids, &gpu_cache);
    for (int step = 0; step < 4; ++step) {
        Tensor lg = llama_forward_single_token(model, 7 + step, seq_len + step, &gpu_cache);
        ASSERT_EQ(lg.size(), cpu_logits[step].size());
        double max_diff = 0.0;
        for (size_t i = 0; i < lg.size(); ++i) {
            max_diff = std::max(max_diff, static_cast<double>(std::fabs(lg[i] - cpu_logits[step][i])));
        }
        std::cerr << "[decode step " << step << "] max_diff=" << max_diff << "\n";
        EXPECT_LT(max_diff, 0.5);
    }
}

TEST_F(LLaMAInferenceCUDATest, SingleTokenWithKVCacheParity) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);

    int token_id = 42;
    size_t pos = 5;
    
    // CPU with Cache
    model.set_device(hal::DeviceType::CPU);
    serius::infer::KVCache cpu_cache;
    // Cache is auto-initialized by llama_forward_single_token if empty, 
    // but the CPU/GPU parity requires consistent initialization.
    Tensor cpu_logits = llama_forward_single_token(model, token_id, pos, &cpu_cache);

    // GPU with Cache
    model.set_device(hal::DeviceType::CUDA);
    serius::infer::KVCache gpu_cache;
    Tensor gpu_logits = llama_forward_single_token(model, token_id, pos, &gpu_cache);

    // Compare logits — same tolerance rationale as PrefillParity
    for (size_t i = 0; i < cpu_logits.size(); ++i) {
        EXPECT_NEAR(cpu_logits[i], gpu_logits[i], 0.15f);
    }
    
    // Compare caches
    for (size_t l = 0; l < cfg.num_hidden_layers; ++l) {
        serius::infer::KVCacheLayer &cl = cpu_cache.layers[l];
        serius::infer::KVCacheLayer &gl = gpu_cache.layers[l];
        gl.k.syncToHost();
        gl.v.syncToHost();
        
        // Only check up to pos+1
        size_t head_dim = cfg.hidden_size / cfg.num_attention_heads;
        size_t kv_dim = cfg.num_key_value_heads * head_dim;
        for (size_t i = 0; i < (pos + 1) * kv_dim; ++i) {
            EXPECT_NEAR(cl.k[i], gl.k[i], 0.15f);
            EXPECT_NEAR(cl.v[i], gl.v[i], 0.15f);
        }
    }
}

TEST_F(LLaMAInferenceCUDATest, GqaBf16PrefillParity) {
    LLamaConfig cfg = get_test_config();
    cfg.hidden_size = 512;
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 2;  // GQA
    cfg.intermediate_size = 1024;
    cfg.vocab_size = 256;
    LLaMAModel model(cfg);
    randomize_model(model);

    // Give the QKV biases non-zero values (Qwen-style).
    auto set_bias = [](core::autograd::Node *n, size_t size) {
        if (n == nullptr) return;
        n->data = core::Tensor({size});
        for (size_t i = 0; i < size; ++i) n->data[i] = 0.01f * static_cast<float>(i % 7);
    };
    for (const auto &layer : model.layers()) {
        set_bias(layer->q_bias.get(), cfg.hidden_size);
        set_bias(layer->k_bias.get(), cfg.num_key_value_heads * (cfg.hidden_size / cfg.num_attention_heads));
        set_bias(layer->v_bias.get(), cfg.num_key_value_heads * (cfg.hidden_size / cfg.num_attention_heads));
    }

    size_t seq_len = 8;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(i % cfg.vocab_size);

    model.set_device(hal::DeviceType::CPU);
    Tensor cpu_logits = llama_forward_inference(model, token_ids, nullptr);

    // Emulate the chat path: cast all params to BF16 and move to GPU.
    auto cast_param = [](core::autograd::Node *p) {
        if (p == nullptr) return;
        p->data = p->data.cast(hal::DType::BFloat16);
        p->data.set_device(hal::DeviceType::CUDA);
    };
    cast_param(model.embed_tokens());
    for (const auto &layer : model.layers()) {
        cast_param(layer->input_layernorm_weight.get());
        cast_param(layer->q_proj.get());
        cast_param(layer->k_proj.get());
        cast_param(layer->v_proj.get());
        cast_param(layer->q_bias.get());
        cast_param(layer->k_bias.get());
        cast_param(layer->v_bias.get());
        cast_param(layer->o_proj.get());
        cast_param(layer->post_attention_layernorm_weight.get());
        cast_param(layer->gate_proj.get());
        cast_param(layer->up_proj.get());
        cast_param(layer->down_proj.get());
    }
    cast_param(model.norm_weight());
    cast_param(model.lm_head());

    Tensor gpu_logits = llama_forward_inference(model, token_ids, nullptr);

    const auto cpu_last = last_position_logits(cpu_logits);
    const auto gpu_last = last_position_logits(gpu_logits);
    ASSERT_EQ(cpu_last.size(), gpu_last.size());
    double max_diff = 0.0;
    size_t cpu_am = 0, gpu_am = 0;
    float cpu_mx = -1e30f, gpu_mx = -1e30f;
    for (size_t i = 0; i < cpu_last.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(cpu_last[i] - gpu_last[i])));
        if (cpu_last[i] > cpu_mx) { cpu_mx = cpu_last[i]; cpu_am = i; }
        if (gpu_last[i] > gpu_mx) { gpu_mx = gpu_last[i]; gpu_am = i; }
    }
    std::cerr << "[bf16-gqa] max_diff=" << max_diff << " cpu_argmax=" << cpu_am
              << " gpu_argmax=" << gpu_am << " cpu_max=" << cpu_mx << " gpu_max=" << gpu_mx << "\n";
    // BF16 should preserve ranking reasonably; allow drift but not collapse.
    EXPECT_LT(max_diff, 2.0);
}

TEST_F(LLaMAInferenceCUDATest, GqaFp16PrefillParity) {
    LLamaConfig cfg = get_test_config();
    cfg.hidden_size = 512;
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 1024;
    cfg.vocab_size = 256;
    LLaMAModel model(cfg);
    randomize_model(model);

    size_t seq_len = 8;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(i % cfg.vocab_size);

    model.set_device(hal::DeviceType::CPU);
    Tensor cpu_logits = llama_forward_inference(model, token_ids, nullptr);

    auto cast_param = [](core::autograd::Node *p) {
        if (p == nullptr) return;
        p->data = p->data.cast(hal::DType::Float16);
        p->data.set_device(hal::DeviceType::CUDA);
    };
    cast_param(model.embed_tokens());
    for (const auto &layer : model.layers()) {
        cast_param(layer->input_layernorm_weight.get());
        cast_param(layer->q_proj.get());
        cast_param(layer->k_proj.get());
        cast_param(layer->v_proj.get());
        cast_param(layer->o_proj.get());
        cast_param(layer->post_attention_layernorm_weight.get());
        cast_param(layer->gate_proj.get());
        cast_param(layer->up_proj.get());
        cast_param(layer->down_proj.get());
    }
    cast_param(model.norm_weight());
    cast_param(model.lm_head());

    Tensor gpu_logits = llama_forward_inference(model, token_ids, nullptr);

    const auto cpu_last = last_position_logits(cpu_logits);
    const auto gpu_last = last_position_logits(gpu_logits);
    ASSERT_EQ(cpu_last.size(), gpu_last.size());
    size_t nan_count = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < gpu_last.size(); ++i) {
        if (!std::isfinite(gpu_last[i])) { ++nan_count; continue; }
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(cpu_last[i] - gpu_last[i])));
    }
    std::cerr << "[fp16-gqa] nan=" << nan_count << "/" << gpu_last.size()
              << " max_diff=" << max_diff << "\n";
    EXPECT_EQ(nan_count, 0u);
}

// Diagnostic: does the batched prefill (seq_len > 1 in one forward) produce
// the same KV cache and last-position logits as the sequential
// single-token path? Run with SERIUS_PREFILL_BATCH=1 so
// llama_forward_inference takes the batched branch for dense models.
TEST_F(LLaMAInferenceCUDATest, BatchedVsSequentialPrefill) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);

    const size_t seq_len = 12;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(3 + i);

    model.set_device(hal::DeviceType::CPU);

    // Sequential reference: one token at a time.
    serius::infer::KVCache seq_cache;
    Tensor seq_last({0});
    for (size_t i = 0; i < seq_len; ++i) {
        seq_last = llama_forward_single_token(model, static_cast<int>(token_ids[i]), i, &seq_cache);
    }

    // Batched path (requires SERIUS_PREFILL_BATCH=1 for dense models).
    serius::infer::KVCache batch_cache;
    Tensor batch_logits = llama_forward_inference(model, token_ids, &batch_cache);
    ASSERT_EQ(batch_logits.shape()[0], seq_len);
    const size_t vocab = batch_logits.shape()[1];

    // Compare K cache layer 0, position 5 (mid-sequence).
    const size_t kv_dim = cfg.num_key_value_heads * (cfg.hidden_size / cfg.num_attention_heads);
    double k_max_diff = 0.0;
    for (size_t d = 0; d < kv_dim; ++d) {
        k_max_diff = std::max(k_max_diff,
            static_cast<double>(std::fabs(seq_cache.layers[0].k[5 * kv_dim + d] -
                                          batch_cache.layers[0].k[5 * kv_dim + d])));
    }
    std::cerr << "[batched-vs-seq] K[0][5] max_diff=" << k_max_diff << "\n";

    // Compare last-row logits.
    double logit_max_diff = 0.0;
    for (size_t v = 0; v < vocab; ++v) {
        logit_max_diff = std::max(logit_max_diff,
            static_cast<double>(std::fabs(seq_last[v] - batch_logits[(seq_len - 1) * vocab + v])));
    }
    std::cerr << "[batched-vs-seq] last-logits max_diff=" << logit_max_diff << "\n";
    EXPECT_LT(logit_max_diff, 0.5);
}

// Diagnostic on a REAL GGUF: does batched prefill match sequential decode?
// Requires SERIUS_PREFILL_BATCH=1 and the model file on disk.
TEST(LLaMARealGGUFTest, BatchedVsSequentialRealModel) {
    const char *env_path = nullptr;
    size_t env_len = 0;
    getenv_s(&env_len, nullptr, 0, "SERIUS_TEST_GGUF");
    std::string env_buf;
    if (env_len > 0) {
        env_buf.resize(env_len);
        getenv_s(&env_len, env_buf.data(), env_buf.size(), "SERIUS_TEST_GGUF");
        env_path = env_buf.c_str();
    }
    const std::string gguf = env_path && env_path[0]
        ? env_path
        : "../models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if (!std::filesystem::exists(gguf)) { GTEST_SKIP() << "model file missing"; }

    const auto mc = serius::io::gguf_to_model_config(gguf);
    ASSERT_GT(mc.vocab_size, 0u);
    model::LLamaConfig cfg;
    cfg.vocab_size = mc.vocab_size;
    cfg.hidden_size = mc.d_model;
    cfg.num_hidden_layers = mc.n_layers;
    cfg.num_attention_heads = mc.n_heads;
    cfg.num_key_value_heads = mc.n_kv_heads;
    cfg.intermediate_size = mc.d_ff;
    cfg.max_position_embeddings = mc.max_seq_len;
    cfg.rope_theta = mc.rope_theta;
    cfg.rms_norm_eps = mc.rms_norm_eps;

    model::LLaMAModel model(cfg);
    ASSERT_TRUE(serius::io::load_llama_weights_from_gguf_file(gguf, &model));
    model.set_device(hal::DeviceType::CPU);

    // Strip packed-quantized weights so the sequential decode path falls back
    // to the same dense FP32 weights the batched path uses. Packed weights are
    // lossy requants (e.g. Q6_K->Q4_1) and would introduce an expected,
    // unrelated precision gap; this test validates batched math parity.
    for (auto &blk : model.layers()) {
        blk->q_w = {}; blk->k_w = {}; blk->v_w = {}; blk->o_w = {};
        blk->gate_w = {}; blk->up_w = {}; blk->down_w = {};
        blk->qkv_w = {}; blk->gate_up_w = {};
    }
    model.lm_head_q() = {};

    const auto &blk0 = model.layers()[0];
    std::cerr << "[real-gguf] qkv_w=" << blk0->qkv_w.valid() << " o_w=" << blk0->o_w.valid()
              << " gate_up_w=" << blk0->gate_up_w.valid() << " down_w=" << blk0->down_w.valid()
              << " lm_head_t_size=" << model.lm_head_t().size() << "\n";

    const size_t seq_len = 16;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) token_ids[i] = static_cast<float>(1000 + i);

    KVCache seq_cache;
    Tensor seq_last({0});
    for (size_t i = 0; i < seq_len; ++i) {
        seq_last = llama_forward_single_token(model, static_cast<int>(token_ids[i]), i, &seq_cache);
    }

    KVCache batch_cache;
    Tensor batch_logits = llama_forward_inference(model, token_ids, &batch_cache);
    ASSERT_EQ(batch_logits.shape()[0], seq_len);
    const size_t vocab = batch_logits.shape()[1];

    // Manual reference for layer-0 K at pos 0 (rope is identity there):
    // K = rms_norm(embed[tok]) @ k_proj + k_bias.
    {
        const size_t H = cfg.hidden_size;
        const size_t kvd = cfg.num_key_value_heads * (H / cfg.num_attention_heads);
        const float *emb = model.embed_tokens()->data.data();
        const float *x = emb + 1000 * H;  // token_ids[0] = 1000
        std::vector<float> normed(H), kref(kvd, 0.0f);
        double ss = 0.0;
        for (size_t i = 0; i < H; ++i) ss += double(x[i]) * x[i];
        const float rms = 1.0f / std::sqrt(float(ss / H) + cfg.rms_norm_eps);
        const float *nw = model.layers()[0]->input_layernorm_weight->data.data();
        for (size_t i = 0; i < H; ++i) normed[i] = x[i] * rms * nw[i];
        Tensor xn(normed.data(), {1, H});
        Tensor kn(kref.data(), {1, kvd});
        core::blas::matmul(xn, model.layers()[0]->k_proj->data, kn);
        if (model.layers()[0]->k_bias) {
            const float *b = model.layers()[0]->k_bias->data.data();
            for (size_t i = 0; i < kvd; ++i) kref[i] += b[i];
        }

        double ds = 0, db = 0;
        for (size_t d = 0; d < kvd; ++d) {
            ds = std::max(ds, double(std::fabs(kref[d] - seq_cache.layers[0].k[d])));
            db = std::max(db, double(std::fabs(kref[d] - batch_cache.layers[0].k[d])));
        }
        std::cerr << "[real-gguf] K0 vs manual: seq_diff=" << ds << " batch_diff=" << db << "\n";
    }

    // Bisect: compare K cache layer 0 across all positions.
    const size_t kv_dim = cfg.num_key_value_heads * (cfg.hidden_size / cfg.num_attention_heads);
    double k_max = 0.0, v_max = 0.0;
    for (size_t p = 0; p < seq_len; ++p) {
        double kp = 0.0;
        for (size_t d = 0; d < kv_dim; ++d) {
            kp = std::max(kp, static_cast<double>(std::fabs(
                seq_cache.layers[0].k[p * kv_dim + d] - batch_cache.layers[0].k[p * kv_dim + d])));
            v_max = std::max(v_max, static_cast<double>(std::fabs(
                seq_cache.layers[0].v[p * kv_dim + d] - batch_cache.layers[0].v[p * kv_dim + d])));
        }
        k_max = std::max(k_max, kp);
    }
    std::cerr << "[real-gguf] K max_diff=" << k_max << " V max_diff=" << v_max << "\n";
    // Same check on the LAST layer — divergence there means it accumulates
    // through depth (MLP dense-vs-quant or attention), not just layer 0.
    {
        const size_t last = seq_cache.layers.size() - 1;
        double lk = 0.0;
        for (size_t p = 0; p < seq_len; ++p)
            for (size_t d = 0; d < kv_dim; ++d)
                lk = std::max(lk, double(std::fabs(seq_cache.layers[last].k[p * kv_dim + d] -
                                                   batch_cache.layers[last].k[p * kv_dim + d])));
        std::cerr << "[real-gguf] last-layer K max_diff=" << lk << "\n";
    }

    double max_diff = 0.0;
    size_t seq_arg = 0, batch_arg = 0;
    for (size_t v = 0; v < vocab; ++v) {
        const float a = seq_last[v], b = batch_logits[(seq_len - 1) * vocab + v];
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a - b)));
        if (a > seq_last[seq_arg]) seq_arg = v;
        if (b > batch_logits[(seq_len - 1) * vocab + batch_arg]) batch_arg = v;
    }
    std::cerr << "[real-gguf] max_diff=" << max_diff
              << " seq_argmax=" << seq_arg << " batch_argmax=" << batch_arg << "\n";
    EXPECT_LT(max_diff, 0.5);
}

#if defined(SERIUS_EXPERIMENTAL_INTERPRET)
TEST_F(LLaMAInferenceCUDATest, CudaPrefillCaptureHooks) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);
    model.set_device(hal::DeviceType::CUDA);

    const size_t seq_len = 4;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        token_ids[i] = static_cast<float>(i % cfg.vocab_size);
    }

    interpret::InferenceCapture capture;
    Tensor logits = capture.run(model, token_ids);
    EXPECT_GT(logits.size(), 0);
    EXPECT_EQ(capture.num_layers(), cfg.num_hidden_layers);
    EXPECT_EQ(capture.num_attention_heads(0), cfg.num_attention_heads);

    // Per (layer, head): one attention row per query position; row q has
    // q+1 valid entries summing to ~1 under the causal mask.
    const auto &rows = capture.attention_rows(0, 0);
    ASSERT_EQ(rows.size(), seq_len);
    for (size_t q = 0; q < seq_len; ++q) {
        EXPECT_EQ(rows[q].size(), q + 1);
        double sum = 0.0;
        for (size_t k = 0; k < rows[q].size(); ++k) {
            sum += rows[q][k];
        }
        EXPECT_NEAR(sum, 1.0, 1e-4);
    }
    EXPECT_EQ(capture.hidden_last_position(0).size(), cfg.hidden_size);
}
#endif

namespace {
int g_hook_hidden_calls = 0;
int g_hook_attn_calls = 0;
}  // namespace

TEST_F(LLaMAInferenceCUDATest, CudaDecodeHooksFire) {
    LLamaConfig cfg = get_test_config();
    LLaMAModel model(cfg);
    randomize_model(model);
    model.set_device(hal::DeviceType::CUDA);

    const size_t seq_len = 3;
    Tensor token_ids({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        token_ids[i] = static_cast<float>(i % cfg.vocab_size);
    }

    KVCache cache;
    llama_forward_inference(model, token_ids, &cache);  // prefill, hooks off

    g_hook_hidden_calls = 0;
    g_hook_attn_calls = 0;
    ActivationHooks hooks{};
    hooks.layer_hidden = [](int, const float *, size_t, size_t) { ++g_hook_hidden_calls; };
    hooks.attention_row = [](int, int, size_t, const float *, size_t) { ++g_hook_attn_calls; };
    set_activation_hooks(hooks);

    Tensor next = llama_forward_single_token(model, 5, seq_len, &cache);
    clear_activation_hooks();

    EXPECT_GT(next.size(), 0);
    EXPECT_EQ(g_hook_hidden_calls, static_cast<int>(cfg.num_hidden_layers));
    // FP32 model takes the non-fused attention branch: one row per head.
    EXPECT_EQ(g_hook_attn_calls,
              static_cast<int>(cfg.num_hidden_layers * cfg.num_attention_heads));
}

int main(int argc, char **argv) {
    // These tests exercise the batched prefill path; it is opt-in at runtime.
#ifdef _WIN32
    _putenv("SERIUS_PREFILL_BATCH=1");
#else
    setenv("SERIUS_PREFILL_BATCH", "1", 1);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
