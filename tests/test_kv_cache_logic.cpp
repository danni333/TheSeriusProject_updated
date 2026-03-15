#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "transformer/model.h"
#include "transformer/paged_kv_cache.h"


namespace serius {
namespace test {

namespace autograd = core::autograd;
using core::Tensor;
using model::BlockManager;
using model::ModelConfig;
using model::PagedKVCache;
using model::TransformerModel;

TEST(KVCacheTest, IncrementalInferenceConsistency)
{
    ModelConfig config;
    config.vocab_size = 100;
    config.d_model = 32;
    config.n_heads = 4;
    config.n_layers = 2;
    config.d_ff = 128;
    config.max_seq_len = 16;

    TransformerModel model(config);

    // 1. Full sequence forward
    Tensor tokens({1, 4});  // batch=1, seq=4
    for (size_t i = 0; i < 4; ++i) tokens.at(0, i) = static_cast<float>(i + 1);

    autograd::Node input_node(tokens, false);
    autograd::Node *full_logits_node = model.forward(&input_node);
    Tensor full_logits = full_logits_node->data;

    // 2. Incremental forward with Paged KV Cache
    auto block_manager = std::make_shared<BlockManager>(128, 8, config.n_heads, config.d_model / config.n_heads);
    std::vector<std::unique_ptr<PagedKVCache>> caches;
    std::vector<PagedKVCache *> cache_ptrs;
    for (size_t i = 0; i < config.n_layers; ++i) {
        caches.push_back(std::make_unique<PagedKVCache>(block_manager));
        cache_ptrs.push_back(caches.back().get());
    }

    // Process tokens one by one
    for (size_t t = 0; t < 4; ++t) {
        Tensor step_token({1, 1});
        step_token.at(0, 0) = tokens.at(0, t);

        autograd::Node step_input(step_token, false);
        autograd::Node *step_logits_node = model.forward(&step_input, &cache_ptrs);
        Tensor step_logits = step_logits_node->data;

        // Compare step_logits with the corresponding row in full_logits
        // full_logits shape: (1, 4, 100)
        // step_logits shape: (1, 1, 100)
        for (size_t v = 0; v < 100; ++v) {
            float expected = full_logits.at(0, t, v);
            float actual = step_logits.at(0, 0, v);
            if (v % 10 == 0 && std::abs(expected - actual) > 1e-4F) {  // Check every 10th to avoid too much output
                std::cout << "Mismatch at Step=" << t << ", VocabIndex=" << v << ", Expected=" << expected
                          << ", Actual=" << actual << std::endl;
            }
            EXPECT_NEAR(expected, actual, 1e-4F) << "Mismatch at step " << t << " vocab dim " << v;
        }
    }
}

}  // namespace test
}  // namespace serius
