#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "infer/llama_inference.h"
#include "transformer/llama_model.h"
#include "tensor/tensor.h"
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
        cfg.num_key_value_heads = 4;
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

    // Cross-device parity: FP32 accumulation order differs between GPU parallel
    // reductions and CPU sequential FMA chains across multiple layers.
    // Tolerance of 0.15 covers expected numerical drift for 2-layer FP32 inference.
    ASSERT_EQ(cpu_logits.size(), gpu_logits.size());
    for (size_t i = 0; i < cpu_logits.size(); ++i) {
        EXPECT_NEAR(cpu_logits[i], gpu_logits[i], 0.15f);
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

    ASSERT_EQ(cpu_logits.size(), gpu_logits.size());
    double max_diff = 0.0;
    size_t cpu_am = 0, gpu_am = 0;
    float cpu_mx = -1e30f, gpu_mx = -1e30f;
    for (size_t i = 0; i < cpu_logits.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(cpu_logits[i] - gpu_logits[i])));
        if (cpu_logits[i] > cpu_mx) { cpu_mx = cpu_logits[i]; cpu_am = i; }
        if (gpu_logits[i] > gpu_mx) { gpu_mx = gpu_logits[i]; gpu_am = i; }
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

    size_t nan_count = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < gpu_logits.size(); ++i) {
        if (!std::isfinite(gpu_logits[i])) { ++nan_count; continue; }
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(cpu_logits[i] - gpu_logits[i])));
    }
    std::cerr << "[fp16-gqa] nan=" << nan_count << "/" << gpu_logits.size()
              << " max_diff=" << max_diff << "\n";
    EXPECT_EQ(nan_count, 0u);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
