#pragma once

#include <string>

#include "../autograd/node.h"
#include "../blas/blas.h"
#include "../tensor/tensor.h"

namespace serius {
namespace lora {

/// LoRA (Low-Rank Adaptation) linear layer.
/// Wraps a frozen weight matrix W and adds trainable low-rank matrices A, B.
/// Output = W*x + (alpha/rank) * B*(A*x)
class LoRALinear
{
   public:
    /// Create a LoRA adapter for a weight matrix.
    /// @param W Reference to the frozen weight matrix (d_out x d_in)
    /// @param rank Low-rank dimension (typically 8-64)
    /// @param alpha Scaling factor (typically equal to rank)
    LoRALinear(core::autograd::Node &W, int rank, float alpha = 0.0f);

    /// Forward pass: base + lora delta.
    /// x has shape (*, d_in), returns (*, d_out).
    core::Tensor forward(const core::Tensor &x) const;

    /// Merge LoRA weights into the base weight matrix.
    /// After merge, W = W + (alpha/rank) * B * A
    /// This makes inference faster (no LoRA overhead).
    void merge();

    /// Un-merge: subtract the LoRA delta from W.
    void unmerge();

    /// Get trainable parameters (A and B).
    std::vector<core::autograd::Node *> parameters();

    /// Access the adapter matrices.
    core::autograd::Node &A() { return *A_; }
    core::autograd::Node &B() { return *B_; }
    const core::autograd::Node &A() const { return *A_; }
    const core::autograd::Node &B() const { return *B_; }

    int rank() const { return rank_; }
    float alpha() const { return alpha_; }
    float scaling() const { return alpha_ / static_cast<float>(rank_); }
    bool is_merged() const { return merged_; }

   private:
    core::autograd::Node &W_;  // frozen base weight (d_out x d_in)
    core::autograd::Node *A_;  // down-projection (rank x d_in), trainable
    core::autograd::Node *B_;  // up-projection (d_out x rank), trainable
    int rank_;
    float alpha_;
    bool merged_ = false;
};

}  // namespace lora
}  // namespace serius
