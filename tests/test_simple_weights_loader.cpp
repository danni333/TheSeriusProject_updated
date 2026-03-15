#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "autograd/engine.h"
#include "autograd/node.h"
#include "io/simple_weights_loader.h"
#include "transformer/model.h"

namespace {

using serius::core::autograd::Node;
using serius::io::load_simple_weights;
using serius::io::save_simple_weights;
using serius::model::ModelConfig;
using serius::model::TransformerModel;

ModelConfig tiny_config()
{
    ModelConfig c;
    c.vocab_size = 256;
    c.d_model = 64;
    c.n_heads = 4;
    c.n_layers = 2;
    c.d_ff = 128;
    c.max_seq_len = 64;
    return c;
}

}  // namespace

TEST(SimpleWeightsLoader, SaveAndLoadRoundTrip)
{
    ModelConfig config = tiny_config();
    TransformerModel model_a(config);
    const char *path = "test_simple_weights_roundtrip.bin";

    ASSERT_TRUE(save_simple_weights(path, model_a));

    TransformerModel model_b(config);
    ASSERT_TRUE(load_simple_weights(path, &model_b));

    std::vector<Node *> params_a = model_a.parameters();
    std::vector<Node *> params_b = model_b.parameters();
    ASSERT_EQ(params_a.size(), params_b.size());

    for (size_t i = 0; i < params_a.size(); ++i) {
        const auto &ta = params_a[i]->data;
        const auto &tb = params_b[i]->data;
        ASSERT_EQ(ta.shape(), tb.shape()) << "param index " << i;
        for (size_t j = 0; j < ta.size(); ++j) {
            EXPECT_FLOAT_EQ(ta[j], tb[j]) << "param " << i << " element " << j;
        }
    }

    std::remove(path);
}

TEST(SimpleWeightsLoader, LoadMissingFileReturnsFalse)
{
    ModelConfig config = tiny_config();
    TransformerModel model(config);
    EXPECT_FALSE(load_simple_weights("nonexistent_weights_file_12345.bin", &model));
}

TEST(SimpleWeightsLoader, LoadInvalidMagicReturnsFalse)
{
    ModelConfig config = tiny_config();
    TransformerModel model(config);
    const char *path = "test_simple_weights_bad_magic.bin";
    std::FILE *f = std::fopen(path, "wb");
    ASSERT_NE(f, nullptr);
    uint32_t wrong_magic = 0x12345678;
    std::fwrite(&wrong_magic, sizeof(wrong_magic), 1, f);
    std::fclose(f);

    EXPECT_FALSE(load_simple_weights(path, &model));
    std::remove(path);
}

TEST(SimpleWeightsLoader, ForwardAfterLoadSucceeds)
{
    ModelConfig config = tiny_config();
    TransformerModel model(config);
    const char *path = "test_simple_weights_forward.bin";

    ASSERT_TRUE(save_simple_weights(path, model));
    ASSERT_TRUE(load_simple_weights(path, &model));

    const size_t seq_len = 4;
    std::vector<float> token_ids(seq_len, 0.0f);
    token_ids[0] = 1.0f;
    token_ids[1] = 2.0f;
    token_ids[2] = 3.0f;
    token_ids[3] = 4.0f;

    serius::core::Tensor ids_tensor({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        ids_tensor[i] = token_ids[i];
    }
    serius::core::autograd::Node input_node(std::move(ids_tensor), false);

    serius::core::autograd::Node *logits_node = model.forward(&input_node);
    ASSERT_NE(logits_node, nullptr);
    EXPECT_EQ(logits_node->data.shape().size(), 2u);
    EXPECT_EQ(logits_node->data.shape()[0], seq_len);
    EXPECT_EQ(logits_node->data.shape()[1], config.vocab_size);

    std::remove(path);
}

TEST(SimpleWeightsLoader, LoadNullModelReturnsFalse)
{
    EXPECT_FALSE(load_simple_weights("any_path.bin", nullptr));
}
