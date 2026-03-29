#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

#include "distributed/comm.h"
#include "distributed/local_comm.h"
#include "infer/llama_inference.h"
#include "tensor/tensor.h"
#include "transformer/llama_model.h"

namespace serius {
namespace test {

using namespace distributed;
using namespace model;
using namespace core;
using namespace infer;

void populate_random(Tensor &t) {
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = (rand() % 1000) / 1000.0f - 0.5f;
    }
}

// Helper to manually copy a subset of a full weight matrix into a TP-sliced weight matrix
void copy_sliced_weight(const Tensor &full, Tensor &local, bool is_column_parallel, int tp_rank, int tp_size) {
    size_t rows = is_column_parallel ? full.shape()[0] : full.shape()[0] / tp_size;
    size_t cols = is_column_parallel ? full.shape()[1] / tp_size : full.shape()[1];
    
    size_t r_start = is_column_parallel ? 0 : (tp_rank * rows);
    size_t c_start = is_column_parallel ? (tp_rank * cols) : 0;
    
    size_t full_cols = full.shape()[1];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            local.data()[r * cols + c] = full.data()[(r_start + r) * full_cols + (c_start + c)];
        }
    }
}

TEST(DistributedLLaMA, TensorParallelInferenceEquivalence) {
    LLamaConfig config;
    config.hidden_size = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 128;
    config.num_hidden_layers = 1;
    config.vocab_size = 100;
    config.max_position_embeddings = 64;
    
    int tp_size = 2;
    auto comms = create_local_group(tp_size);
    
    // Create reference model
    LLaMAModel ref_model(config, DistributedTopology::single_node());
    
    // Populate reference model with random weights
    populate_random(ref_model.embed_tokens()->data);
    populate_random(ref_model.norm_weight()->data);
    populate_random(ref_model.lm_head()->data);
    for (const auto &layer : ref_model.layers()) {
        populate_random(layer->input_layernorm_weight->data);
        populate_random(layer->post_attention_layernorm_weight->data);
        populate_random(layer->q_proj->data);
        populate_random(layer->k_proj->data);
        populate_random(layer->v_proj->data);
        populate_random(layer->o_proj->data);
        populate_random(layer->gate_proj->data);
        populate_random(layer->up_proj->data);
        populate_random(layer->down_proj->data);
    }
    
    auto run_rank = [&](int rank) {
        DistributedTopology topology;
        topology.tp_comm = comms[rank].get();
        topology.pp_comm = nullptr; // PP not tested here
        topology.tp_rank = rank;
        topology.tp_size = tp_size;
        topology.pp_rank = 0;
        topology.pp_size = 1;
        
        LLaMAModel local_model(config, topology);
        
        // Copy non-sliced weights
        std::copy(ref_model.embed_tokens()->data.data(), ref_model.embed_tokens()->data.data() + ref_model.embed_tokens()->data.size(), local_model.embed_tokens()->data.data());
        std::copy(ref_model.norm_weight()->data.data(), ref_model.norm_weight()->data.data() + ref_model.norm_weight()->data.size(), local_model.norm_weight()->data.data());
        std::copy(ref_model.lm_head()->data.data(), ref_model.lm_head()->data.data() + ref_model.lm_head()->data.size(), local_model.lm_head()->data.data());
        
        for (size_t i = 0; i < config.num_hidden_layers; ++i) {
            auto *ref_layer = ref_model.layers()[i].get();
            auto *local_layer = local_model.layers()[i].get();
            
            std::copy(ref_layer->input_layernorm_weight->data.data(), ref_layer->input_layernorm_weight->data.data() + ref_layer->input_layernorm_weight->data.size(), local_layer->input_layernorm_weight->data.data());
            std::copy(ref_layer->post_attention_layernorm_weight->data.data(), ref_layer->post_attention_layernorm_weight->data.data() + ref_layer->post_attention_layernorm_weight->data.size(), local_layer->post_attention_layernorm_weight->data.data());
            
            // Column Parallel
            copy_sliced_weight(ref_layer->q_proj->data, local_layer->q_proj->data, true, rank, tp_size);
            copy_sliced_weight(ref_layer->k_proj->data, local_layer->k_proj->data, true, rank, tp_size);
            copy_sliced_weight(ref_layer->v_proj->data, local_layer->v_proj->data, true, rank, tp_size);
            copy_sliced_weight(ref_layer->gate_proj->data, local_layer->gate_proj->data, true, rank, tp_size);
            copy_sliced_weight(ref_layer->up_proj->data, local_layer->up_proj->data, true, rank, tp_size);
            
            // Row Parallel
            copy_sliced_weight(ref_layer->o_proj->data, local_layer->o_proj->data, false, rank, tp_size);
            copy_sliced_weight(ref_layer->down_proj->data, local_layer->down_proj->data, false, rank, tp_size);
        }
        
        // Run Inference
        Tensor tokens_tensor({3});
        tokens_tensor.data()[0] = 10;
        tokens_tensor.data()[1] = 20;
        tokens_tensor.data()[2] = 30;
        Tensor local_logits = llama_forward_inference(local_model, tokens_tensor);
        return local_logits;
    };
    
    // Run all ranks
    std::vector<std::future<Tensor>> futures;
    for (int r = 0; r < tp_size; ++r) {
        futures.push_back(std::async(std::launch::async, run_rank, r));
    }
    
    // Run refernece
    Tensor tokens_tensor({3});
    tokens_tensor.data()[0] = 10;
    tokens_tensor.data()[1] = 20;
    tokens_tensor.data()[2] = 30;
    Tensor ref_logits = llama_forward_inference(ref_model, tokens_tensor);
    
    // Compare
    for (int r = 0; r < tp_size; ++r) {
        Tensor local_logits = futures[r].get();
        
        ASSERT_EQ(ref_logits.size(), local_logits.size());
        for (size_t i = 0; i < ref_logits.size(); ++i) {
            EXPECT_NEAR(ref_logits.data()[i], local_logits.data()[i], 1e-4) << "Mismatch at rank " << r << " index " << i;
        }
    }
}

} // namespace test
} // namespace serius
