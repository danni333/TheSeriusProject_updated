#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "autograd/node.h"
#include "transformer/model.h"
#include "transformer/paged_kv_cache.h"

namespace serius {
namespace test {

using core::Tensor;
using core::autograd::Node;
using model::BlockManager;
using model::ModelConfig;
using model::PagedKVCache;
using model::TransformerModel;

TEST(TransformerOptimTest, NumericalConsistency)
{
    fprintf(stderr, "[Test] NumericalConsistency Start\n");
    fflush(stderr);
    ModelConfig config;
    config.n_layers = 1;
    config.d_model = 64;
    config.n_heads = 4;
    config.d_ff = 128;
    config.vocab_size = 100;

    TransformerModel model(config);

    // Create random batched input (B=2, S=8, D=64)
    size_t batch_size = 2;
    size_t seq_len = 8;
    Tensor ids_t({batch_size, seq_len});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            ids_t.at(b, s) = (float)((b * seq_len + s) % config.vocab_size);
        }
    }
    Node *token_ids = new Node(std::move(ids_t));

    // Forward pass
    // TransformerModel::forward calls forward_optimized internally now if implemented
    Node *logits = model.forward(token_ids);
    EXPECT_EQ(logits->data.shape().size(), 3);
    EXPECT_EQ(logits->data.shape()[0], batch_size);
    EXPECT_EQ(logits->data.shape()[1], seq_len);
    EXPECT_EQ(logits->data.shape()[2], config.vocab_size);

    Tensor out = logits->data.copy();
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FALSE(std::isnan(out[i]));
        EXPECT_FALSE(std::isinf(out[i]));
    }

    delete token_ids;
}

TEST(TransformerOptimTest, PagedForward)
{
    ModelConfig config;
    config.n_layers = 2;
    config.d_model = 32;
    config.n_heads = 4;
    config.vocab_size = 100;

    TransformerModel model(config);

    size_t num_blocks = 20;
    size_t block_size = 4;
    auto manager =
        std::make_shared<BlockManager>(num_blocks, block_size, config.n_heads, config.d_model / config.n_heads);

    // We only support batch=1 with single PagedKVCache per layer for now in this test
    std::vector<PagedKVCache *> caches;
    for (size_t i = 0; i < config.n_layers; ++i) {
        caches.push_back(new PagedKVCache(manager));
    }

    // Generation step by step (B=1, S=1)
    for (size_t step = 0; step < 5; ++step) {
        Tensor ids_t({1, 1});  // B=1, S=1
        ids_t.at(0, 0) = (float)(step + 1);
        Node *token_ids = new Node(std::move(ids_t));

        Node *logits = model.forward(token_ids, &caches);
        EXPECT_EQ(logits->data.shape().size(), 3);
        EXPECT_EQ(logits->data.shape()[0], 1);
        EXPECT_EQ(logits->data.shape()[1], 1);

        for (auto *c : caches) {
            EXPECT_EQ(c->current_len(), step + 1);
        }

        delete token_ids;
    }

    for (auto *c : caches) delete c;
}

}  // namespace test
}  // namespace serius
