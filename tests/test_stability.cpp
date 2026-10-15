#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "autograd/node.h"
#include "init/weight_init.h"
#include "stability/ema.h"
#include "stability/grad_clip.h"
#include "stability/loss_scaler.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::core;

// ============================================================
// Weight Initialization Tests (Fase 9)
// ============================================================

TEST(WeightInit, XavierUniformRange)
{
    size_t fan_in = 256, fan_out = 512;
    Tensor w({fan_in, fan_out});
    init::xavier_uniform(w, fan_in, fan_out);

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    const float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        EXPECT_GE(data[i], -limit) << "Value below -limit at index " << i;
        EXPECT_LE(data[i], limit) << "Value above +limit at index " << i;
    }
}

TEST(WeightInit, XavierNormalMeanStd)
{
    size_t fan_in = 512, fan_out = 512;
    Tensor w({fan_in, fan_out});
    init::xavier_normal(w, fan_in, fan_out);

    float expected_std = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));

    // Compute empirical mean and std
    const float *data = w.data();
    double sum = 0, sum_sq = 0;
    size_t n = w.size();
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    float mean = static_cast<float>(sum / n);
    float var = static_cast<float>(sum_sq / n - mean * mean);
    float std_val = std::sqrt(var);

    EXPECT_NEAR(mean, 0.0f, 0.01f) << "Mean should be near 0";
    EXPECT_NEAR(std_val, expected_std, 0.01f) << "Std should match expected";
}

TEST(WeightInit, HeNormalStd)
{
    size_t fan_in = 512;
    Tensor w({fan_in, 256});
    init::he_normal(w, fan_in);

    float expected_std = std::sqrt(2.0f / static_cast<float>(fan_in));

    const float *data = w.data();
    double sum = 0, sum_sq = 0;
    size_t n = w.size();
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    float mean = static_cast<float>(sum / n);
    float var = static_cast<float>(sum_sq / n - mean * mean);
    float std_val = std::sqrt(var);

    EXPECT_NEAR(mean, 0.0f, 0.02f);
    EXPECT_NEAR(std_val, expected_std, 0.01f);
}

TEST(WeightInit, HeUniformRange)
{
    size_t fan_in = 256;
    Tensor w({fan_in, 128});
    init::he_uniform(w, fan_in);

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in));
    const float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        EXPECT_GE(data[i], -limit);
        EXPECT_LE(data[i], limit);
    }
}

TEST(WeightInit, EmbeddingInit)
{
    Tensor w({1000, 64});
    init::embedding_init(w);

    // Should be N(0, 0.02)
    const float *data = w.data();
    double sum = 0, sum_sq = 0;
    size_t n = w.size();
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    float mean = static_cast<float>(sum / n);
    float var = static_cast<float>(sum_sq / n - mean * mean);
    float std_val = std::sqrt(var);

    EXPECT_NEAR(mean, 0.0f, 0.005f);
    EXPECT_NEAR(std_val, 0.02f, 0.005f);
}

TEST(WeightInit, LayerNormInit)
{
    Tensor gamma({128});
    Tensor beta({128});
    init::layernorm_init(gamma, beta);

    for (size_t i = 0; i < gamma.size(); i++) {
        EXPECT_FLOAT_EQ(gamma[i], 1.0f);
        EXPECT_FLOAT_EQ(beta[i], 0.0f);
    }
}

// ============================================================
// Gradient Clipping Tests (Fase 9)
// ============================================================

TEST(GradClip, ClipNorm)
{
    // Create a parameter with large gradient
    autograd::Node param(Tensor({4}));
    param.grad = Tensor({4});
    float *g = param.grad.data();
    g[0] = 3.0f;
    g[1] = 4.0f;
    g[2] = 0.0f;
    g[3] = 0.0f;
    // norm = sqrt(9+16) = 5

    std::vector<autograd::Node *> params = {&param};
    float original_norm = stability::clip_grad_norm(params, 1.0f);

    EXPECT_NEAR(original_norm, 5.0f, 1e-5f);

    // After clipping, norm should be <= 1.0
    float new_norm_sq = 0;
    for (size_t i = 0; i < 4; i++) {
        new_norm_sq += g[i] * g[i];
    }
    EXPECT_NEAR(std::sqrt(new_norm_sq), 1.0f, 1e-4f);
}

TEST(GradClip, NoChangeWhenBelowThreshold)
{
    autograd::Node param(Tensor({3}));
    param.grad = Tensor({3});
    float *g = param.grad.data();
    g[0] = 0.1f;
    g[1] = 0.2f;
    g[2] = 0.3f;
    // norm = sqrt(0.01+0.04+0.09) = sqrt(0.14) ~ 0.374

    float g0 = g[0], g1 = g[1], g2 = g[2];
    std::vector<autograd::Node *> params = {&param};
    stability::clip_grad_norm(params, 1.0f);

    // No change expected
    EXPECT_FLOAT_EQ(g[0], g0);
    EXPECT_FLOAT_EQ(g[1], g1);
    EXPECT_FLOAT_EQ(g[2], g2);
}

TEST(GradClip, ClipValue)
{
    autograd::Node param(Tensor({4}));
    param.grad = Tensor({4});
    float *g = param.grad.data();
    g[0] = 5.0f;
    g[1] = -3.0f;
    g[2] = 0.5f;
    g[3] = -0.1f;

    std::vector<autograd::Node *> params = {&param};
    stability::clip_grad_value(params, 1.0f);

    EXPECT_FLOAT_EQ(g[0], 1.0f);
    EXPECT_FLOAT_EQ(g[1], -1.0f);
    EXPECT_FLOAT_EQ(g[2], 0.5f);
    EXPECT_FLOAT_EQ(g[3], -0.1f);
}

TEST(GradClip, DetectNanInf)
{
    autograd::Node param(Tensor({3}));
    param.grad = Tensor({3});
    float *g = param.grad.data();
    g[0] = 1.0f;
    g[1] = std::numeric_limits<float>::infinity();
    g[2] = 0.0f;

    std::vector<autograd::Node *> params = {&param};
    EXPECT_TRUE(stability::has_nan_or_inf(params));

    g[1] = 2.0f;
    EXPECT_FALSE(stability::has_nan_or_inf(params));

    g[2] = std::nanf("");
    EXPECT_TRUE(stability::has_nan_or_inf(params));
}

// ============================================================
// Dynamic Loss Scaler Tests (Fase 9)
// ============================================================

TEST(LossScaler, ScaleLoss)
{
    stability::DynamicLossScaler scaler;
    EXPECT_FLOAT_EQ(scaler.scale(), 65536.0f);
    EXPECT_FLOAT_EQ(scaler.scale_loss(1.0f), 65536.0f);
}

TEST(LossScaler, OverflowReducesScale)
{
    stability::DynamicLossScaler scaler;
    float initial_scale = scaler.scale();

    scaler.update(true);  // overflow detected
    EXPECT_FLOAT_EQ(scaler.scale(), initial_scale * 0.5f);
    EXPECT_EQ(scaler.overflow_count(), 1);
}

TEST(LossScaler, GrowthAfterGoodSteps)
{
    stability::DynamicLossScaler::Config cfg;
    cfg.initial_scale = 1024.0f;
    cfg.growth_interval = 3;
    stability::DynamicLossScaler scaler(cfg);

    // 3 good steps -> scale should double
    scaler.update(false);
    scaler.update(false);
    EXPECT_FLOAT_EQ(scaler.scale(), 1024.0f);  // not yet
    scaler.update(false);
    EXPECT_FLOAT_EQ(scaler.scale(), 2048.0f);  // doubled after 3 good steps
}

TEST(LossScaler, UnscaleGrads)
{
    stability::DynamicLossScaler::Config cfg;
    cfg.initial_scale = 100.0f;
    stability::DynamicLossScaler scaler(cfg);

    autograd::Node param(Tensor({2}));
    param.grad = Tensor({2});
    param.grad.data()[0] = 500.0f;  // scaled gradient
    param.grad.data()[1] = 300.0f;

    std::vector<autograd::Node *> params = {&param};
    bool ok = scaler.unscale_grads(params);

    EXPECT_TRUE(ok);
    EXPECT_NEAR(param.grad.data()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(param.grad.data()[1], 3.0f, 1e-5f);
}

// ============================================================
// EMA Tests (Fase 9)
// ============================================================

TEST(EMA, InitAndUpdate)
{
    autograd::Node param(Tensor({4}));
    float *p = param.data.data();
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;

    std::vector<autograd::Node *> params = {&param};

    stability::EMAWeights ema(0.9f);
    ema.init(params);

    // After init, shadow should equal current params
    EXPECT_FLOAT_EQ(ema.shadow(0)[0], 1.0f);
    EXPECT_FLOAT_EQ(ema.shadow(0)[3], 4.0f);

    // Change params and update
    p[0] = 10.0f;
    p[1] = 20.0f;
    p[2] = 30.0f;
    p[3] = 40.0f;
    ema.update(params);

    // shadow = 0.9 * old + 0.1 * new
    EXPECT_NEAR(ema.shadow(0)[0], 0.9f * 1.0f + 0.1f * 10.0f, 1e-5f);
    EXPECT_NEAR(ema.shadow(0)[3], 0.9f * 4.0f + 0.1f * 40.0f, 1e-5f);
}

TEST(EMA, ApplyAndRestore)
{
    autograd::Node param(Tensor({2}));
    param.data.data()[0] = 5.0f;
    param.data.data()[1] = 10.0f;

    std::vector<autograd::Node *> params = {&param};

    stability::EMAWeights ema(0.5f);
    ema.init(params);

    // Change weights
    param.data.data()[0] = 100.0f;
    param.data.data()[1] = 200.0f;
    ema.update(params);

    // shadow = 0.5 * 5 + 0.5 * 100 = 52.5 and 0.5 * 10 + 0.5 * 200 = 105
    EXPECT_NEAR(ema.shadow(0)[0], 52.5f, 1e-5f);

    // Apply EMA to model
    ema.apply(params);
    EXPECT_NEAR(param.data.data()[0], 52.5f, 1e-5f);
    EXPECT_NEAR(param.data.data()[1], 105.0f, 1e-5f);

    // Restore original
    ema.restore(params);
    EXPECT_FLOAT_EQ(param.data.data()[0], 100.0f);
    EXPECT_FLOAT_EQ(param.data.data()[1], 200.0f);
}

TEST(EMA, Convergence)
{
    // After many updates with same value, EMA should converge to that value
    autograd::Node param(Tensor({1}));
    param.data.data()[0] = 0.0f;

    std::vector<autograd::Node *> params = {&param};

    stability::EMAWeights ema(0.99f);
    ema.init(params);

    // Set to target and update many times
    param.data.data()[0] = 42.0f;
    for (int i = 0; i < 1000; i++) {
        ema.update(params);
    }

    EXPECT_NEAR(ema.shadow(0)[0], 42.0f, 0.01f);
}
