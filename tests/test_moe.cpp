#include <gtest/gtest.h>

#include <cmath>

#include "autograd/engine.h"
#include "io/config_loader.h"
#include "moe/moe_layer.h"
#include "model/moe.h"
#include "transformer/model.h"

TEST(MoE, RouterSelectsTopKAndNormalizesWeights)
{
    serius::model::Router router(4, 3, 2, 7);
    serius::core::Tensor x({4});
    x.set_item(0, 1.0f);
    x.set_item(1, 0.5f);
    x.set_item(2, -0.25f);
    x.set_item(3, 2.0f);

    const auto decision = router.route(x);
    EXPECT_EQ(decision.expert_indices.size(), 2u);
    EXPECT_EQ(decision.expert_weights.size(), 2u);
    EXPECT_NEAR(decision.expert_weights[0] + decision.expert_weights[1], 1.0f, 1e-5f);
}

TEST(MoE, MoELayerProducesCorrectOutputShape)
{
    serius::model::MoELayer moe(4, 4, 2, 1, 11);
    serius::core::Tensor input({4});
    input.set_item(0, 1.0f);
    input.set_item(1, 2.0f);
    input.set_item(2, 3.0f);
    input.set_item(3, 4.0f);

    auto *node = new serius::core::autograd::Node(std::move(input), false);
    auto *out = moe.forward(node);

    ASSERT_EQ(out->data.ndim(), 1u);
    EXPECT_EQ(out->data.shape()[0], 4u);
    delete out;
    delete node;
}

TEST(MoE, BackwardUpdatesExpertAndRouterParameters)
{
    serius::model::MoELayer moe(4, 3, 2, 1, 13, 8);
    serius::core::Tensor input_data({2, 4});
    input_data.set_item(0, 0, 1.0f);
    input_data.set_item(0, 1, -0.5f);
    input_data.set_item(0, 2, 0.25f);
    input_data.set_item(0, 3, 2.0f);
    input_data.set_item(1, 0, -1.0f);
    input_data.set_item(1, 1, 0.75f);
    input_data.set_item(1, 2, 1.5f);
    input_data.set_item(1, 3, -0.25f);

    serius::core::autograd::Node input(std::move(input_data), false);
    serius::core::autograd::Node *output = moe.forward(&input);

    float loss_value = 0.0f;
    for (size_t i = 0; i < output->data.size(); ++i) {
        loss_value += output->data[i];
    }
    serius::core::Tensor loss_data({1});
    loss_data[0] = loss_value;
    serius::core::autograd::Node loss(std::move(loss_data));
    loss.children = {output};
    loss.backward_fn = [output]() {
        for (size_t i = 0; i < output->grad.size(); ++i) {
            output->grad[i] += 1.0f;
        }
    };
    serius::core::autograd::backward(&loss);

    const auto parameters = moe.parameters();
    float router_grad = 0.0f;
    for (size_t i = 0; i < parameters[0]->grad.size(); ++i) {
        router_grad += std::fabs(parameters[0]->grad[i]);
    }
    for (size_t i = 0; i < parameters[1]->grad.size(); ++i) {
        router_grad += std::fabs(parameters[1]->grad[i]);
    }
    float expert_grad = 0.0f;
    for (size_t parameter = 2; parameter < parameters.size(); ++parameter) {
        for (size_t i = 0; i < parameters[parameter]->grad.size(); ++i) {
            expert_grad += std::fabs(parameters[parameter]->grad[i]);
        }
    }
    EXPECT_GT(router_grad, 0.0f);
    EXPECT_GT(expert_grad, 0.0f);
}

TEST(MoE, TransformerModelRunsWithConfigEnabledMoE)
{
    serius::model::ModelConfig config;
    config.vocab_size = 320;
    config.d_model = 16;
    config.n_heads = 4;
    config.n_layers = 1;
    config.d_ff = 32;
    config.max_seq_len = 16;
    config.moe.enabled = true;
    config.moe.num_experts = 4;
    config.moe.top_k = 2;
    config.moe.shared_experts = 1;
    config.normalize_moe();

    serius::model::TransformerModel model(config);
    ASSERT_EQ(model.blocks.size(), 1u);
    EXPECT_TRUE(model.blocks[0]->has_moe());
    const auto moe_parameters = model.blocks[0]->moe_parameters();
    ASSERT_EQ(moe_parameters.size(), 22u);
    EXPECT_EQ(moe_parameters[2]->data.shape()[1], 32u);

    serius::core::Tensor token_ids({2});
    token_ids.set_item(0, 5.0f);
    token_ids.set_item(1, 8.0f);
    serius::core::autograd::Node input(std::move(token_ids), false);
    serius::core::autograd::Node *logits = model.forward(&input);

    ASSERT_NE(logits, nullptr);
    EXPECT_EQ(logits->data.shape(), (std::vector<size_t>{2, 320}));
    delete logits;

    serius::core::Tensor batched_token_ids({1, 2});
    batched_token_ids.set_item(0, 0, 5.0f);
    batched_token_ids.set_item(0, 1, 8.0f);
    serius::core::autograd::Node batched_input(std::move(batched_token_ids), false);
    serius::core::autograd::Node *batched_logits = model.forward(&batched_input);
    ASSERT_NE(batched_logits, nullptr);
    EXPECT_EQ(batched_logits->data.shape(), (std::vector<size_t>{1, 2, 320}));
    delete batched_logits;
}

TEST(MoE, LoadBalanceLossIsNonNegative)
{
    serius::core::Tensor probs({4, 4});
    probs.fill(0.25f);

    std::vector<std::vector<size_t>> assignments = {
        {0, 1}, {1, 2}, {2, 3}, {0, 3}
    };

    const float loss = serius::model::load_balance_loss(probs, assignments, 4, 0.01f);
    EXPECT_GE(loss, 0.0f);
}

TEST(MoE, LoadBalanceLossUsesRouterProbabilities)
{
    serius::core::Tensor probs({2, 2});
    probs.set_item(0, 0, 0.9f);
    probs.set_item(0, 1, 0.1f);
    probs.set_item(1, 0, 0.8f);
    probs.set_item(1, 1, 0.2f);
    const std::vector<std::vector<size_t>> assignments = {{0}, {0}};

    EXPECT_NEAR(serius::model::load_balance_loss(probs, assignments, 2, 0.01f), 0.017f, 1e-6f);
}

TEST(ConfigLoader, MoEConfigIsParsed)
{
    const std::string json = R"({
        "vocab_size":32000,
        "hidden_size":512,
        "num_attention_heads":8,
        "num_layers":6,
        "intermediate_size":2048,
        "max_position_embeddings":4096,
        "moe": {
            "enabled": true,
            "num_experts": 8,
            "top_k": 2,
            "shared_experts": 1,
            "load_balance_alpha": 0.02
        }
    })";

    auto config = serius::io::load_model_config_from_string(json);
    EXPECT_TRUE(config.moe.enabled);
    EXPECT_EQ(config.moe.num_experts, 8u);
    EXPECT_EQ(config.top_k, 2u);
    EXPECT_TRUE(config.moe_enabled);
}
