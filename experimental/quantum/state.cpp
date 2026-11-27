#include "state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace serius {
namespace quantum {

namespace gates {

Matrix2x2 H()
{
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    return Matrix2x2{
        Complex(inv_sqrt2, 0.0f), Complex(inv_sqrt2, 0.0f),
        Complex(inv_sqrt2, 0.0f), Complex(-inv_sqrt2, 0.0f)
    };
}

Matrix2x2 X()
{
    return Matrix2x2{
        Complex(0.0f, 0.0f), Complex(1.0f, 0.0f),
        Complex(1.0f, 0.0f), Complex(0.0f, 0.0f)
    };
}

Matrix2x2 Y()
{
    return Matrix2x2{
        Complex(0.0f, 0.0f), Complex(0.0f, -1.0f),
        Complex(0.0f, 1.0f), Complex(0.0f, 0.0f)
    };
}

Matrix2x2 Z()
{
    return Matrix2x2{
        Complex(1.0f, 0.0f), Complex(0.0f, 0.0f),
        Complex(0.0f, 0.0f), Complex(-1.0f, 0.0f)
    };
}

Matrix2x2 RY(float theta)
{
    float half = theta * 0.5f;
    float c = std::cos(half);
    float s = std::sin(half);
    return Matrix2x2{
        Complex(c, 0.0f), Complex(-s, 0.0f),
        Complex(s, 0.0f), Complex(c, 0.0f)
    };
}

Matrix2x2 RX(float theta)
{
    float half = theta * 0.5f;
    float c = std::cos(half);
    float s = std::sin(half);
    return Matrix2x2{
        Complex(c, 0.0f), Complex(0.0f, -s),
        Complex(0.0f, -s), Complex(c, 0.0f)
    };
}

Matrix2x2 RZ(float theta)
{
    float half = theta * 0.5f;
    return Matrix2x2{
        std::exp(Complex(0.0f, -half)), Complex(0.0f, 0.0f),
        Complex(0.0f, 0.0f), std::exp(Complex(0.0f, half))
    };
}

}  // namespace gates

namespace {
size_t checked_state_dim(size_t n_qubits)
{
    // Bound check must run during member init: 2^n amplitudes must fit in
    // memory and n must stay < 64 (shift UB). This is a teaching simulator.
    if (n_qubits == 0 || n_qubits > 20) {
        throw std::invalid_argument("QuantumState: n_qubits must be in [1, 20]");
    }
    return size_t(1) << n_qubits;
}
}  // namespace

QuantumState::QuantumState(size_t n_qubits)
    : n_qubits_(n_qubits),
      amplitudes_(checked_state_dim(n_qubits), Complex(0.0f, 0.0f))
{
    reset();
}

void QuantumState::reset()
{
    std::fill(amplitudes_.begin(), amplitudes_.end(), Complex(0.0f, 0.0f));
    if (!amplitudes_.empty()) {
        amplitudes_[0] = Complex(1.0f, 0.0f);  // |00...0>
    }
}

void QuantumState::apply_gate(const Matrix2x2 &gate, size_t target_idx)
{
    if (target_idx >= n_qubits_) return;

    size_t total_dim = amplitudes_.size();
    size_t bit_mask = size_t(1) << target_idx;

    for (size_t i = 0; i < total_dim; ++i) {
        if ((i & bit_mask) == 0) {
            size_t i0 = i;
            size_t i1 = i | bit_mask;

            Complex a0 = amplitudes_[i0];
            Complex a1 = amplitudes_[i1];

            amplitudes_[i0] = gate.m00 * a0 + gate.m01 * a1;
            amplitudes_[i1] = gate.m10 * a0 + gate.m11 * a1;
        }
    }
    // Unitary gates preserve the norm in exact arithmetic; renormalize to
    // absorb floating-point drift over long gate sequences.
    normalize();
}

void QuantumState::normalize()
{
    float norm_sq = 0.0f;
    for (const auto &a : amplitudes_) {
        norm_sq += std::norm(a);
    }
    if (norm_sq <= 0.0f) {
        return;
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto &a : amplitudes_) {
        a *= inv_norm;
    }
}

void QuantumState::apply_cnot(size_t control_idx, size_t target_idx)
{
    if (control_idx >= n_qubits_ || target_idx >= n_qubits_ || control_idx == target_idx) {
        return;
    }

    size_t total_dim = amplitudes_.size();
    size_t ctrl_mask = size_t(1) << control_idx;
    size_t target_mask = size_t(1) << target_idx;

    for (size_t i = 0; i < total_dim; ++i) {
        if ((i & ctrl_mask) != 0 && (i & target_mask) == 0) {
            size_t i0 = i;
            size_t i1 = i | target_mask;
            std::swap(amplitudes_[i0], amplitudes_[i1]);
        }
    }
}

void QuantumState::apply_cz(size_t control_idx, size_t target_idx)
{
    // control == target would collapse the mask to a single bit and silently
    // apply a Z on that qubit instead of a two-qubit gate: reject like CNOT.
    if (control_idx >= n_qubits_ || target_idx >= n_qubits_ ||
        control_idx == target_idx) {
        return;
    }

    size_t total_dim = amplitudes_.size();
    size_t mask = (size_t(1) << control_idx) | (size_t(1) << target_idx);

    for (size_t i = 0; i < total_dim; ++i) {
        if ((i & mask) == mask) {
            amplitudes_[i] = -amplitudes_[i];
        }
    }
}

std::vector<float> QuantumState::probabilities() const
{
    std::vector<float> probs(amplitudes_.size(), 0.0f);
    for (size_t i = 0; i < amplitudes_.size(); ++i) {
        probs[i] = std::norm(amplitudes_[i]);  // |psi|^2
    }
    return probs;
}

int QuantumState::measure(size_t qubit_idx, float sample_rand)
{
    if (qubit_idx >= n_qubits_) return 0;

    size_t bit_mask = size_t(1) << qubit_idx;
    float prob_1 = 0.0f;

    for (size_t i = 0; i < amplitudes_.size(); ++i) {
        if ((i & bit_mask) != 0) {
            prob_1 += std::norm(amplitudes_[i]);
        }
    }

    // FP accumulation can push prob_1 slightly outside [0,1]; clamp so
    // (1-prob_1) can't go negative and flip the outcome deterministically.
    prob_1 = std::clamp(prob_1, 0.0f, 1.0f);
    float clamped_rand = std::max(0.0f, std::min(1.0f, sample_rand));
    int outcome = (clamped_rand < (1.0f - prob_1)) ? 0 : 1;

    float norm_factor = (outcome == 1) ? std::sqrt(std::max(1e-8f, prob_1))
                                       : std::sqrt(std::max(1e-8f, 1.0f - prob_1));

    for (size_t i = 0; i < amplitudes_.size(); ++i) {
        int bit = ((i & bit_mask) != 0) ? 1 : 0;
        if (bit == outcome) {
            amplitudes_[i] /= norm_factor;
        } else {
            amplitudes_[i] = Complex(0.0f, 0.0f);
        }
    }

    return outcome;
}

float QuantumState::expectation_z(size_t qubit_idx) const
{
    if (qubit_idx >= n_qubits_) return 0.0f;

    size_t bit_mask = size_t(1) << qubit_idx;
    float exp_val = 0.0f;

    for (size_t i = 0; i < amplitudes_.size(); ++i) {
        float prob = std::norm(amplitudes_[i]);
        if ((i & bit_mask) == 0) {
            exp_val += prob;   // eigenvalue +1 for |0>
        } else {
            exp_val -= prob;   // eigenvalue -1 for |1>
        }
    }

    return exp_val;
}

float QuantumState::expectation_diagonal(const core::Tensor &cost_diag) const
{
    float total = 0.0f;
    size_t limit = std::min(amplitudes_.size(), cost_diag.size());

    for (size_t i = 0; i < limit; ++i) {
        total += std::norm(amplitudes_[i]) * cost_diag[i];
    }

    return total;
}

}  // namespace quantum
}  // namespace serius
