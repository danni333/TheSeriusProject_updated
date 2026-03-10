#include "io/config_loader.h"
#include "transformer/model.h"
#include <gtest/gtest.h>

TEST(ConfigLoader, LoadFromString)
{
    std::string json = R"({"vocab_size":32000,"hidden_size":512,"num_attention_heads":8,
        "num_layers":6,"intermediate_size":2048,"max_position_embeddings":4096})";
    auto config = serius::io::load_model_config_from_string(json);
    EXPECT_EQ(config.vocab_size, 32000u);
    EXPECT_EQ(config.d_model, 512u);
    EXPECT_EQ(config.n_heads, 8u);
    EXPECT_EQ(config.n_layers, 6u);
    EXPECT_EQ(config.d_ff, 2048u);
    EXPECT_EQ(config.max_seq_len, 4096u);
}

TEST(ConfigLoader, InvalidJsonReturnsDefaults)
{
    auto config = serius::io::load_model_config_from_string("not json");
    EXPECT_EQ(config.vocab_size, serius::model::ModelConfig::DEFAULT_VOCAB_SIZE);
    EXPECT_EQ(config.d_model, serius::model::ModelConfig::DEFAULT_D_MODEL);
}
