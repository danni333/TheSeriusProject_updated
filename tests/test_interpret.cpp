#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "infer/llama_inference.h"
#include "interpret/activation_probe.h"
#include "interpret/attention_capture.h"
#include "interpret/export.h"
#include "interpret/inference_capture.h"
#include "interpret/logit_lens.h"
#include "tensor/tensor.h"
#include "transformer/llama_model.h"

using serius::core::Tensor;
using serius::interpret::ActivationProbe;
using serius::interpret::AttentionCapture;
using serius::interpret::InferenceCapture;
using serius::interpret::LogitLens;
using serius::model::LLaMAModel;
using serius::model::LLamaConfig;

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
// ---- InferenceCapture: real LLaMA forward instrumentation ----

namespace {

LLaMAModel make_capture_model()
{
    LLamaConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 4;
    cfg.num_hidden_layers = 2;
    cfg.intermediate_size = 64;
    cfg.max_position_embeddings = 64;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    return LLaMAModel(cfg);
}

Tensor make_prompt(size_t n)
{
    Tensor ids({n});
    for (size_t i = 0; i < n; ++i) { ids[i] = static_cast<float>((i * 7 + 3) % 64); }
    return ids;
}

}  // namespace

TEST(InferenceCapture, CapturesAllLayersHiddenAndAttention)
{
    auto model = make_capture_model();
    const Tensor ids = make_prompt(6);

    InferenceCapture cap;
    cap.run(model, ids);

    ASSERT_EQ(cap.num_layers(), 2u);
    for (size_t l = 0; l < 2; ++l) {
        const Tensor hidden = cap.hidden_last_position(static_cast<int>(l));
        ASSERT_EQ(hidden.shape().size(), 2u);
        EXPECT_EQ(hidden.shape()[0], 1u);
        EXPECT_EQ(hidden.shape()[1], 16u);
        // Finite values
        for (size_t i = 0; i < hidden.size(); ++i) {
            EXPECT_TRUE(std::isfinite(hidden[i]));
        }

        const size_t heads = cap.num_attention_heads(static_cast<int>(l));
        EXPECT_EQ(heads, 4u);
        for (size_t h = 0; h < heads; ++h) {
            const auto &rows = cap.attention_rows(static_cast<int>(l), static_cast<int>(h));
            ASSERT_FALSE(rows.empty());
            // One row per query position (6 prompt tokens).
            EXPECT_EQ(rows.size(), 6u);
            const Tensor &last = rows.back();
            ASSERT_EQ(last.shape()[0], 6u);  // attends to all 6 positions
            float sum = 0.0f;
            for (size_t k = 0; k < last.shape()[0]; ++k) {
                EXPECT_TRUE(std::isfinite(last[k]));
                EXPECT_GE(last[k], 0.0f);
                sum += last[k];
            }
            EXPECT_NEAR(sum, 1.0f, 1e-4f);
        }
    }
    // Hooks are removed after run().
    EXPECT_FALSE(serius::infer::activation_hooks_enabled());
}

TEST(InferenceCapture, LogitLensOnCapturedHidden)
{
    auto model = make_capture_model();
    const Tensor ids = make_prompt(5);

    InferenceCapture cap;
    cap.run(model, ids);

    const Tensor head = serius::interpret::logit_lens_head(model);
    ASSERT_GT(head.size(), 0u);
    EXPECT_EQ(head.shape()[0], 64u);   // vocab
    EXPECT_EQ(head.shape()[1], 16u);   // hidden

    LogitLens lens;
    const auto preds = lens.compute(cap.hidden_last_position(0), head, 3);
    ASSERT_EQ(preds.size(), 3u);
    for (const auto &p : preds) {
        EXPECT_GE(p.token_id, 0);
        EXPECT_LT(p.token_id, 64);
    }
    // Predictions are sorted by descending logit.
    EXPECT_GE(preds[0].logit, preds[1].logit);
    EXPECT_GE(preds[1].logit, preds[2].logit);
}

TEST(InferenceCapture, HooksDoNotChangeLogits)
{
    auto model = make_capture_model();
    const Tensor ids = make_prompt(4);

    // Baseline: plain forward with the same KV-cache setup as capture.run.
    serius::infer::KVCache cache;
    const Tensor baseline = serius::infer::llama_forward_inference(model, ids, &cache);

    InferenceCapture cap;
    const Tensor captured = cap.run(model, ids);

    ASSERT_EQ(baseline.size(), captured.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_FLOAT_EQ(baseline[i], captured[i]);
    }
}

TEST(InferenceCapture, FeedsAttentionCapture)
{
    auto model = make_capture_model();
    const Tensor ids = make_prompt(4);

    InferenceCapture cap;
    cap.run(model, ids);

    // Bridge captured rows into the classic AttentionCapture store.
    AttentionCapture store;
    store.start();
    const auto &rows = cap.attention_rows(0, 0);
    ASSERT_FALSE(rows.empty());
    store.record(0, 0, rows.back());
    store.stop();

    const Tensor got = store.get(0, 0);
    ASSERT_EQ(got.shape()[0], 4u);
    float sum = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) { sum += got[i]; }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
}
