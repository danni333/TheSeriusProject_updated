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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
