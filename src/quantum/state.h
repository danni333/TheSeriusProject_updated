#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace quantum {

using Complex = std::complex<float>;

struct Matrix2x2
{
    Complex m00, m01;
    Complex m10, m11;
};

namespace gates {
Matrix2x2 H();                    // Hadamard
Matrix2x2 X();                    // Pauli-X (NOT)
Matrix2x2 Y();                    // Pauli-Y
Matrix2x2 Z();                    // Pauli-Z
Matrix2x2 RY(float theta);        // Rotation around Y
Matrix2x2 RX(float theta);        // Rotation around X
Matrix2x2 RZ(float theta);        // Rotation around Z
}  // namespace gates

/// Quantum statevector representing a system of n qubits with 2^n complex amplitudes.
class QuantumState
{
   public:
    explicit QuantumState(size_t n_qubits);

    /// Apply a 2x2 single-qubit unitary gate to the qubit at target_idx.
    void apply_gate(const Matrix2x2 &gate, size_t target_idx);

    /// Apply a controlled-NOT (CNOT) gate.
    void apply_cnot(size_t control_idx, size_t target_idx);

    /// Apply a controlled-Z (CZ) gate.
    void apply_cz(size_t control_idx, size_t target_idx);

    /// Get probability distribution across all 2^n basis states.
    std::vector<float> probabilities() const;

    /// Measure a specific qubit, collapsing the statevector.
    /// @param qubit_idx Target qubit
    /// @param sample_rand Uniform random value in [0, 1] used for probabilistic outcome
    /// @return 0 or 1
    int measure(size_t qubit_idx, float sample_rand = 0.5f);

    /// Compute expectation value of Pauli-Z operator on a given qubit: <Z_k> in [-1, 1].
    float expectation_z(size_t qubit_idx) const;

    /// Compute expectation value <psi|H|psi> where H is represented as a diagonal cost tensor.
    float expectation_diagonal(const core::Tensor &cost_diag) const;

    /// Reset state to ground state |00...0>.
    void reset();

    size_t num_qubits() const { return n_qubits_; }
    size_t dim() const { return amplitudes_.size(); }

    const std::vector<Complex> &amplitudes() const { return amplitudes_; }
    std::vector<Complex> &amplitudes() { return amplitudes_; }

   private:
    size_t n_qubits_;
    std::vector<Complex> amplitudes_;
};

}  // namespace quantum
}  // namespace serius
