#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/liquid/adaptive.h"
#include "../src/liquid/ltc.h"
#include "../src/liquid/neural_ode.h"

using namespace serius::liquid;
using serius::core::Tensor;

// ---------------------------------------------------------------------------
// Neural ODE Tests
// ---------------------------------------------------------------------------

TEST(NeuralODETest, ConvergenceWithHigherSteps)
{
    // Exponential decay ODE: dh/dt = -2.0 * h
    // Analytical solution: h(t) = h(0) * exp(-2 * t)
    // For h(0) = 1.0, at t = 1.0 -> h(1) = exp(-2.0) = 0.13533528f
    const float true_h1 = std::exp(-2.0f);

    auto decay_func = [](const Tensor &h, float /*t*/) -> Tensor {
        Tensor dh(h.shape());
        for (size_t i = 0; i < h.size(); ++i) {
            dh[i] = -2.0f * h[i];
        }
        return dh;
    };

    NeuralODE ode(decay_func, ODESolverType::RK4);

    Tensor h0({1});
    h0[0] = 1.0f;

    Tensor res_2 = ode.forward(h0, 0.0f, 1.0f, 2);
    Tensor res_10 = ode.forward(h0, 0.0f, 1.0f, 10);
    Tensor res_50 = ode.forward(h0, 0.0f, 1.0f, 50);

    float err_2 = std::abs(res_2[0] - true_h1);
    float err_10 = std::abs(res_10[0] - true_h1);
    float err_50 = std::abs(res_50[0] - true_h1);

    // Error must decrease strictly as step count increases
    EXPECT_LT(err_10, err_2);
    EXPECT_LT(err_50, err_10);
    EXPECT_NEAR(res_50[0], true_h1, 1e-5f);
}

TEST(NeuralODETest, RK4VsEulerAccuracy)
{
    const float true_h1 = std::exp(-2.0f);

    auto decay_func = [](const Tensor &h, float /*t*/) -> Tensor {
        Tensor dh(h.shape());
        for (size_t i = 0; i < h.size(); ++i) {
            dh[i] = -2.0f * h[i];
        }
        return dh;
    };

    NeuralODE ode_euler(decay_func, ODESolverType::EULER);
    NeuralODE ode_rk4(decay_func, ODESolverType::RK4);

    Tensor h0({1});
    h0[0] = 1.0f;

    const int steps = 5;
    Tensor out_euler = ode_euler.forward(h0, 0.0f, 1.0f, steps);
    Tensor out_rk4 = ode_rk4.forward(h0, 0.0f, 1.0f, steps);

    float err_euler = std::abs(out_euler[0] - true_h1);
    float err_rk4 = std::abs(out_rk4[0] - true_h1);

    // RK4 4th-order method must be significantly more accurate than Euler 1st-order
    EXPECT_LT(err_rk4, err_euler * 0.05f);
}

TEST(NeuralODETest, TrajectoryRecording)
{
    NeuralODE ode(/*hidden_dim=*/4, ODESolverType::RK4);
    Tensor h0({4});
    for (size_t i = 0; i < 4; ++i) h0[i] = static_cast<float>(i + 1);

    const int steps = 8;
    auto traj = ode.forward_trajectory(h0, 0.0f, 1.0f, steps);

    ASSERT_EQ(traj.size(), static_cast<size_t>(steps + 1));
    // Initial state matches
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(traj[0][i], h0[i]);
    }
}

// ---------------------------------------------------------------------------
// Liquid Time-Constant (LTC) Network Tests
// ---------------------------------------------------------------------------

TEST(LTCNetworkTest, AdaptiveTauDynamics)
{
    const size_t in_dim = 2;
    const size_t h_dim = 4;
    const float tau_min = 0.2f;
    const float tau_max = 4.0f;

    LTCNetwork ltc(in_dim, h_dim, tau_min, tau_max);

    Tensor x_low({in_dim});
    x_low[0] = -5.0f; x_low[1] = -5.0f;

    Tensor x_high({in_dim});
    x_high[0] = 5.0f; x_high[1] = 5.0f;

    Tensor tau_low = ltc.compute_tau(x_low);
    Tensor tau_high = ltc.compute_tau(x_high);

    // Tau must stay bounded within [tau_min, tau_max]
    for (size_t i = 0; i < h_dim; ++i) {
        EXPECT_GE(tau_low[i], tau_min - 1e-5f);
        EXPECT_LE(tau_low[i], tau_max + 1e-5f);

        EXPECT_GE(tau_high[i], tau_min - 1e-5f);
        EXPECT_LE(tau_high[i], tau_max + 1e-5f);
    }
}

TEST(LTCNetworkTest, ContinuousIrregularTime)
{
    const size_t in_dim = 3;
    const size_t h_dim = 6;
    LTCNetwork ltc(in_dim, h_dim);

    // Sequence observed at irregular time intervals
    std::vector<float> timestamps = {0.1f, 0.25f, 1.5f, 1.55f, 4.2f};
    std::vector<Tensor> inputs;
    for (size_t k = 0; k < timestamps.size(); ++k) {
        Tensor x({in_dim});
        for (size_t i = 0; i < in_dim; ++i) {
            x[i] = std::sin(static_cast<float>(k * in_dim + i));
        }
        inputs.push_back(std::move(x));
    }

    Tensor h({h_dim});
    h.zeros();

    auto trajectory = ltc.forward_sequence(inputs, timestamps, h);
    ASSERT_EQ(trajectory.size(), timestamps.size());

    // States must remain bounded within [-1, 1] due to continuous contractive dynamics
    for (const auto &state : trajectory) {
        for (size_t i = 0; i < state.size(); ++i) {
            EXPECT_TRUE(std::isfinite(state[i]));
            EXPECT_GE(state[i], -1.05f);
            EXPECT_LE(state[i], 1.05f);
        }
    }
}

TEST(LTCNetworkTest, SequenceMemory)
{
    const size_t in_dim = 2;
    const size_t h_dim = 4;
    LTCNetwork ltc(in_dim, h_dim);

    Tensor h({h_dim});
    h.zeros();

    Tensor x1({in_dim}); x1[0] = 1.0f; x1[1] = 0.0f;
    Tensor x2({in_dim}); x2[0] = 0.0f; x2[1] = 1.0f;

    Tensor h1 = ltc.step(x1, h, 0.5f);
    Tensor h2 = ltc.step(x2, h1, 0.5f);

    // Hidden state evolves and differs between steps
    float diff = 0.0f;
    for (size_t i = 0; i < h_dim; ++i) {
        diff += std::abs(h2[i] - h1[i]);
    }
    EXPECT_GT(diff, 1e-4f);
}

// ---------------------------------------------------------------------------
// Adaptive Step Controller Tests
// ---------------------------------------------------------------------------

TEST(AdaptiveControllerTest, SimpleVsComplexSteps)
{
    AdaptiveStepController controller(/*tolerance=*/1e-3f, /*min_steps=*/2, /*max_steps=*/20);

    // Simple input: uniform constant values (low variance, low fluctuation)
    Tensor x_simple({16});
    for (size_t i = 0; i < 16; ++i) x_simple[i] = 0.5f;

    // Complex input: oscillating high-frequency signal (high variance and delta)
    Tensor x_complex({16});
    for (size_t i = 0; i < 16; ++i) {
        x_complex[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    float c_simple = AdaptiveStepController::estimate_complexity(x_simple);
    float c_complex = AdaptiveStepController::estimate_complexity(x_complex);

    EXPECT_LT(c_simple, c_complex);

    int steps_simple = controller.compute_steps(x_simple);
    int steps_complex = controller.compute_steps(x_complex);

    EXPECT_LT(steps_simple, steps_complex);
}

TEST(AdaptiveControllerTest, ConvergenceStopsEarly)
{
    AdaptiveStepController controller(/*tolerance=*/0.05f, /*min_steps=*/2, /*max_steps=*/40);
    LTCNetwork ltc(4, 4);

    Tensor x({4});
    x.zeros();

    Tensor h({4});
    h.zeros();

    // Since x=0 and h=0, it should converge in very few steps
    int steps = controller.solve_until_convergence(ltc, x, h);
    EXPECT_LT(steps, 40);
    EXPECT_GE(steps, 2);
}
