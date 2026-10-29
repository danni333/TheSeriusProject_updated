#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/quantum/circuit.h"
#include "../src/quantum/state.h"
#include "../src/quantum/vqe.h"

using namespace serius::quantum;
using serius::core::Tensor;

// ---------------------------------------------------------------------------
// Quantum State & Gate Tests
// ---------------------------------------------------------------------------

TEST(QuantumStateTest, HadamardEqualSuperposition)
{
    QuantumState state(1);
    // Initial state |0>
    auto probs0 = state.probabilities();
    EXPECT_NEAR(probs0[0], 1.0f, 1e-6f);
    EXPECT_NEAR(probs0[1], 0.0f, 1e-6f);

    // Apply Hadamard gate
    state.apply_gate(gates::H(), 0);
    auto probs = state.probabilities();

    EXPECT_NEAR(probs[0], 0.5f, 1e-5f);
    EXPECT_NEAR(probs[1], 0.5f, 1e-5f);
}

TEST(QuantumStateTest, BellStateEntanglement)
{
    QuantumState state(2);

    // Create Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    state.apply_gate(gates::H(), 0);
    state.apply_cnot(0, 1);

    auto probs = state.probabilities();
    ASSERT_EQ(probs.size(), 4u);

    // |00> has 50% probability
    EXPECT_NEAR(probs[0], 0.5f, 1e-5f);
    // |01> has 0%
    EXPECT_NEAR(probs[1], 0.0f, 1e-5f);
    // |10> has 0%
    EXPECT_NEAR(probs[2], 0.0f, 1e-5f);
    // |11> has 50% probability
    EXPECT_NEAR(probs[3], 0.5f, 1e-5f);
}

TEST(QuantumStateTest, PauliGatesAndMeasurement)
{
    QuantumState state(1);

    // Apply Pauli-X: |0> -> |1>
    state.apply_gate(gates::X(), 0);
    EXPECT_NEAR(state.expectation_z(0), -1.0f, 1e-5f);

    // Deterministic measurement on |1> must return 1
    int outcome = state.measure(0, 0.5f);
    EXPECT_EQ(outcome, 1);
}

// ---------------------------------------------------------------------------
// Quantum Circuit & Parameter Shift Rule Tests
// ---------------------------------------------------------------------------

TEST(QuantumCircuitTest, ParameterShiftGradientMatchesFiniteDifference)
{
    const size_t n_qubits = 2;
    const size_t n_layers = 2;
    QuantumCircuit circuit(n_qubits, n_layers);

    // Arbitrary diagonal cost Hamiltonian
    Tensor cost({4});
    cost[0] = 1.0f;
    cost[1] = 2.0f;
    cost[2] = -1.5f;
    cost[3] = -3.0f;

    std::vector<float> params = {0.4f, -0.7f, 1.2f, 0.5f};

    // Analytical gradient from quantum parameter-shift rule
    std::vector<float> ps_grads = circuit.parameter_shift_gradient(params, cost);

    // Finite-difference numerical gradient
    const float eps = 1e-3f;
    std::vector<float> fd_grads(params.size(), 0.0f);
    for (size_t i = 0; i < params.size(); ++i) {
        std::vector<float> p_plus = params;
        std::vector<float> p_minus = params;
        p_plus[i] += eps;
        p_minus[i] -= eps;

        float e_plus = circuit.evaluate(p_plus, cost);
        float e_minus = circuit.evaluate(p_minus, cost);
        fd_grads[i] = (e_plus - e_minus) / (2.0f * eps);
    }

    // Compare analytical vs finite difference
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_NEAR(ps_grads[i], fd_grads[i], 1e-3f)
            << "Parameter shift gradient discrepancy at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Variational Quantum Eigensolver (VQE) Tests
// ---------------------------------------------------------------------------

TEST(VQETest, FindsGroundStateMinimum)
{
    const size_t n_qubits = 2;
    const size_t n_layers = 3;
    QuantumOptimizer optimizer(n_qubits, n_layers);

    // Cost Hamiltonian where state |11> (index 3) is the distinct ground state
    Tensor cost({4});
    cost[0] = 2.0f;   // |00>
    cost[1] = 1.5f;   // |01>
    cost[2] = 1.0f;   // |10>
    cost[3] = -2.5f;  // |11> (Ground state minimum)

    VQEResult res = optimizer.optimize(cost, /*max_iterations=*/40, /*lr=*/0.2f);

    // Energy must decrease from initial state
    EXPECT_LT(res.minimum_energy, res.energy_history.front());

    // The optimized state must have significantly higher probability for state |11>
    EXPECT_GT(res.final_probabilities[3], 0.7f)
        << "Probability of ground state |11> is " << res.final_probabilities[3];

    // Final minimum energy must approach ground state eigenvalue -2.5
    EXPECT_NEAR(res.minimum_energy, -2.5f, 0.5f);
}
