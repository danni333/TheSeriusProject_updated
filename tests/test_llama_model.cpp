#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <vector>

#include "autograd/node.h"
#include "io/hf_llama_loader.h"
#include "transformer/llama_model.h"

namespace {

using serius::model::LLamaConfig;
using serius::model::LLaMAModel;

LLamaConfig tiny_llama_config()
{
    LLamaConfig c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 4;
    c.num_hidden_layers = 2;
    c.intermediate_size = 128;
    c.max_position_embeddings = 128;
    c.rms_norm_eps = 1e-5f;
    c.rope_theta = 10000.0f;
    return c;
}

}  // namespace

TEST(LLaMAModel, ForwardShape)
{
    LLamaConfig config = tiny_llama_config();
    LLaMAModel model(config);
    const size_t seq_len = 4;
    serius::core::Tensor ids_tensor({seq_len});
    ids_tensor[0] = 1.0f;
    ids_tensor[1] = 2.0f;
    ids_tensor[2] = 3.0f;
    ids_tensor[3] = 4.0f;
    serius::core::autograd::Node input_node(std::move(ids_tensor), false);

    serius::core::autograd::Node *logits = model.forward(&input_node);
    ASSERT_NE(logits, nullptr);
    ASSERT_EQ(logits->data.shape().size(), 2u);
    EXPECT_EQ(logits->data.shape()[0], seq_len);
    EXPECT_EQ(logits->data.shape()[1], config.vocab_size);
}

TEST(LLaMAModel, LoadConfigFromString)
{
    std::string json = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_hidden_layers": 22,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5
    })";
    std::string path = "test_llama_config.json";
    std::ofstream f(path);
    ASSERT_TRUE(f);
    f << json;
    f.close();
    serius::model::LLamaConfig c = serius::io::load_llama_config(path);
    EXPECT_EQ(c.vocab_size, 32000u);
    EXPECT_EQ(c.hidden_size, 2048u);
    EXPECT_EQ(c.num_key_value_heads, 4u);
    std::remove(path.c_str());
}
