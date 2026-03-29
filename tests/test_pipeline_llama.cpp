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

TEST(DistributedLLaMA, PipelineParallelInferenceEquivalence) {
    LLamaConfig config;
    config.hidden_size = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 128;
    config.num_hidden_layers = 4; // 4 layers total for 2 PP ranks (2 layers each)
    config.vocab_size = 100;
    config.max_position_embeddings = 64;
    
    int pp_size = 2;
    auto comms = create_local_group(pp_size);
    
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
        topology.tp_comm = nullptr;
        topology.pp_comm = comms[rank].get();
        topology.tp_rank = 0;
        topology.tp_size = 1;
        topology.pp_rank = rank;
        topology.pp_size = pp_size;
        
        LLaMAModel local_model(config, topology);
        
        // Copy relevant parts
        if (rank == 0) {
            std::copy(ref_model.embed_tokens()->data.data(), ref_model.embed_tokens()->data.data() + ref_model.embed_tokens()->data.size(), local_model.embed_tokens()->data.data());
        }
        if (rank == pp_size - 1) {
            std::copy(ref_model.norm_weight()->data.data(), ref_model.norm_weight()->data.data() + ref_model.norm_weight()->data.size(), local_model.norm_weight()->data.data());
            std::copy(ref_model.lm_head()->data.data(), ref_model.lm_head()->data.data() + ref_model.lm_head()->data.size(), local_model.lm_head()->data.data());
        }
        
        size_t layers_per_stage = config.num_hidden_layers / pp_size;
        size_t start_layer = rank * layers_per_stage;
        size_t end_layer = start_layer + layers_per_stage;
        if (rank == pp_size - 1) end_layer = config.num_hidden_layers;

        for (size_t i = start_layer; i < end_layer; ++i) {
            size_t local_idx = i - start_layer;
            auto *ref_layer = ref_model.layers()[i].get();
            auto *local_layer = local_model.layers()[local_idx].get();
            
            std::copy(ref_layer->input_layernorm_weight->data.data(), ref_layer->input_layernorm_weight->data.data() + ref_layer->input_layernorm_weight->data.size(), local_layer->input_layernorm_weight->data.data());
            std::copy(ref_layer->post_attention_layernorm_weight->data.data(), ref_layer->post_attention_layernorm_weight->data.data() + ref_layer->post_attention_layernorm_weight->data.size(), local_layer->post_attention_layernorm_weight->data.data());
            std::copy(ref_layer->q_proj->data.data(), ref_layer->q_proj->data.data() + ref_layer->q_proj->data.size(), local_layer->q_proj->data.data());
            std::copy(ref_layer->k_proj->data.data(), ref_layer->k_proj->data.data() + ref_layer->k_proj->data.size(), local_layer->k_proj->data.data());
            std::copy(ref_layer->v_proj->data.data(), ref_layer->v_proj->data.data() + ref_layer->v_proj->data.size(), local_layer->v_proj->data.data());
            std::copy(ref_layer->o_proj->data.data(), ref_layer->o_proj->data.data() + ref_layer->o_proj->data.size(), local_layer->o_proj->data.data());
            std::copy(ref_layer->gate_proj->data.data(), ref_layer->gate_proj->data.data() + ref_layer->gate_proj->data.size(), local_layer->gate_proj->data.data());
            std::copy(ref_layer->up_proj->data.data(), ref_layer->up_proj->data.data() + ref_layer->up_proj->data.size(), local_layer->up_proj->data.data());
            std::copy(ref_layer->down_proj->data.data(), ref_layer->down_proj->data.data() + ref_layer->down_proj->data.size(), local_layer->down_proj->data.data());
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
    std::future<Tensor> rank0 = std::async(std::launch::async, run_rank, 0);
    std::future<Tensor> rank1 = std::async(std::launch::async, run_rank, 1);
    
    // Output from rank0 is a dummy empty tensor. Rank 1 returns the actual logits.
    Tensor dummy_logits = rank0.get();
    Tensor local_logits = rank1.get();

    // Run refernece
    Tensor tokens_tensor({3});
    tokens_tensor.data()[0] = 10;
    tokens_tensor.data()[1] = 20;
    tokens_tensor.data()[2] = 30;
    Tensor ref_logits = llama_forward_inference(ref_model, tokens_tensor);
    
    // Compare
    ASSERT_EQ(dummy_logits.shape().size(), 1);
    ASSERT_EQ(dummy_logits.size(), 0);

    ASSERT_EQ(ref_logits.size(), local_logits.size());
    for (size_t i = 0; i < ref_logits.size(); ++i) {
        EXPECT_NEAR(ref_logits.data()[i], local_logits.data()[i], 1e-4) << "Mismatch at index " << i << " (expected " << ref_logits.data()[i] << " != " << local_logits.data()[i] << ")";
    }
}

} // namespace test
} // namespace serius
