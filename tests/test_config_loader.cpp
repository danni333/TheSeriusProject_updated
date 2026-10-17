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

TEST(ConfigLoader, NestedAndFlatMoEConfigAreNormalized)
{
    const std::string json = R"({
        "vocab_size": 32000,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_layers": 6,
        "intermediate_size": 2048,
        "max_position_embeddings": 4096,
        "moe_enabled": true,
        "num_experts": 8,
        "top_k": 2,
        "shared_experts": 1,
        "load_balance_alpha": 0.02,
        "load_balance": true,
        "moe": {
            "enabled": true,
            "num_experts": 8,
            "top_k": 2,
            "shared_experts": 1,
            "load_balance_alpha": 0.02,
            "load_balance": true
        }
    })";

    auto config = serius::io::load_model_config_from_string(json);
    EXPECT_TRUE(config.moe_enabled);
    EXPECT_TRUE(config.has_moe());
    EXPECT_EQ(config.num_experts, 8u);
    EXPECT_EQ(config.top_k, 2u);
    EXPECT_EQ(config.shared_experts, 1u);
    EXPECT_FLOAT_EQ(config.load_balance_alpha, 0.02f);
    EXPECT_TRUE(config.load_balance);
    EXPECT_TRUE(config.moe.enabled);
}

TEST(ConfigLoader, ExplicitlyDisabledMoEStaysDisabled)
{
    const std::string json = R"({
        "moe_enabled": false,
        "moe": {
            "enabled": true,
            "num_experts": 4,
            "top_k": 2
        }
    })";

    const auto config = serius::io::load_model_config_from_string(json);
    EXPECT_FALSE(config.moe_enabled);
    EXPECT_FALSE(config.has_moe());
    EXPECT_EQ(config.num_experts, 4u);
    EXPECT_EQ(config.top_k, 2u);
}
