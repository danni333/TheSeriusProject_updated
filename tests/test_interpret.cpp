#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "interpret/activation_probe.h"
#include "interpret/attention_capture.h"
#include "interpret/export.h"
#include "interpret/logit_lens.h"
#include "tensor/tensor.h"

using serius::core::Tensor;
using serius::interpret::ActivationProbe;
using serius::interpret::AttentionCapture;
using serius::interpret::LogitLens;

TEST(AttentionCapture, StartStopAndRecord)
{
    AttentionCapture cap;
    EXPECT_FALSE(cap.is_capturing());

    cap.start();
    EXPECT_TRUE(cap.is_capturing());

    Tensor attn({2, 2});
    attn.fill(0.5f);
    cap.record(0, 0, attn);

    EXPECT_EQ(cap.num_layers(), 1u);
    EXPECT_EQ(cap.num_heads(0), 1u);

    Tensor retrieved = cap.get(0, 0);
    EXPECT_EQ(retrieved.shape()[0], 2u);
    EXPECT_EQ(retrieved.shape()[1], 2u);

    cap.stop();
    EXPECT_FALSE(cap.is_capturing());
}

TEST(AttentionCapture, SoftmaxValidation)
{
    AttentionCapture cap;
    cap.start();

    // Create a valid softmax attention matrix (row sums = 1.0)
    Tensor attn({3, 3});
    float data[] = {0.5f, 0.3f, 0.2f, 0.1f, 0.6f, 0.3f, 0.25f, 0.25f, 0.5f};
    for (size_t i = 0; i < 9; ++i) attn[i] = data[i];

    cap.record(0, 0, attn);
    EXPECT_TRUE(cap.validate_softmax(0, 0));

    // Invalid softmax (rows don't sum to 1)
    Tensor bad_attn({2, 2});
    bad_attn.fill(0.7f);  // rows sum to 1.4
    cap.record(1, 0, bad_attn);
    EXPECT_FALSE(cap.validate_softmax(1, 0));
}

TEST(ActivationProbe, HookAndTopNeurons)
{
    ActivationProbe probe;

    Tensor act({4, 8});
    for (size_t i = 0; i < 32; ++i) act[i] = static_cast<float>(i % 8) * 0.1f;
    probe.hook("ffn", act);

    auto top = probe.top_neurons("ffn", 0, 3);
    EXPECT_EQ(top.size(), 3u);
    // Top neuron should have highest activation
    EXPECT_GE(top[0].second, top[1].second);
    EXPECT_GE(top[1].second, top[2].second);
}

TEST(ActivationProbe, ExpertRouting)
{
    ActivationProbe probe;
    std::vector<int> experts = {2, 5};
    std::vector<float> weights = {0.6f, 0.4f};
    probe.log_expert_routing(3, 0, experts, weights);

    EXPECT_EQ(probe.routing_log().size(), 1u);
    EXPECT_EQ(probe.routing_log()[0].layer, 3);
    EXPECT_EQ(probe.routing_log()[0].token_pos, 0);
    EXPECT_EQ(probe.routing_log()[0].experts[0], 2);
    EXPECT_FLOAT_EQ(probe.routing_log()[0].weights[0], 0.6f);
}

TEST(LogitLens, BasicPrediction)
{
    LogitLens lens;
    std::vector<std::string> vocab = {"the", "cat", "sat", "on", "mat"};
    lens.set_vocab(vocab);

    // hidden: (4,) small representation
    Tensor hidden({4});
    hidden[0] = 1.0f;
    hidden[1] = 0.0f;
    hidden[2] = 0.0f;
    hidden[3] = 0.0f;

    // lm_head: (5, 4) — identity-like for "the" (token 0)
    Tensor lm_head({5, 4});
    lm_head.fill(0.0f);
    lm_head[0] = 1.0f;  // token "the" has weight 1.0 on dim 0

    auto preds = lens.compute(hidden, lm_head, 3);
    EXPECT_EQ(preds.size(), 3u);
    EXPECT_EQ(preds[0].token_id, 0);
    EXPECT_EQ(preds[0].token, "the");
}

TEST(Export, AttentionJsonExport)
{
    AttentionCapture cap;
    cap.start();

    Tensor attn({2, 2});
    attn[0] = 0.7f; attn[1] = 0.3f;
    attn[2] = 0.4f; attn[3] = 0.6f;
    cap.record(0, 0, attn);

    std::vector<std::string> tokens = {"hello", "world"};
    std::string json = export_attention_json(cap, tokens);

    EXPECT_NE(json.find("hello"), std::string::npos);
    EXPECT_NE(json.find("world"), std::string::npos);
    EXPECT_NE(json.find("0.7"), std::string::npos);
}

TEST(Export, AttentionCsvExport)
{
    AttentionCapture cap;
    cap.start();

    Tensor attn({2, 2});
    attn[0] = 0.7f; attn[1] = 0.3f;
    attn[2] = 0.4f; attn[3] = 0.6f;
    cap.record(0, 0, attn);

    std::string csv = export_attention_csv(cap, 0, 0);
    EXPECT_NE(csv.find("t0"), std::string::npos);
    EXPECT_NE(csv.find("0.7"), std::string::npos);
}

TEST(Export, ProbeJsonExport)
{
    ActivationProbe probe;
    std::vector<int> experts = {1, 3};
    std::vector<float> weights = {0.8f, 0.2f};
    probe.log_expert_routing(2, 1, experts, weights);

    std::string json = export_probe_json(probe);
    EXPECT_NE(json.find("expert"), std::string::npos);
    EXPECT_NE(json.find("0.8"), std::string::npos);
}