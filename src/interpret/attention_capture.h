#pragma once

#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace interpret {

using core::Tensor;

/// Captures attention weights from each layer and head during forward pass.
/// Enables visualization of which tokens the model attends to.
class AttentionCapture
{
   public:
    AttentionCapture() = default;

    /// Start capturing attention weights
    void start() { capturing_ = true; captures_.clear(); }

    /// Stop capturing
    void stop() { capturing_ = false; }

    /// Is capturing enabled?
    bool is_capturing() const { return capturing_; }

    /// Record attention weights for a specific layer and head
    /// weights: (seq, seq) attention matrix for one head
    void record(int layer, int head, const Tensor &weights);

    /// Get captured attention weights
    /// Returns: vector of layers, each containing vector of heads
    const std::vector<std::vector<Tensor>> &captures() const { return captures_; }

    /// Get attention for specific layer/head
    Tensor get(int layer, int head) const;

    /// Number of layers captured
    size_t num_layers() const { return captures_.size(); }

    /// Number of heads in a layer
    size_t num_heads(int layer) const
    {
        return layer < captures_.size() ? captures_[layer].size() : 0;
    }

    /// Check if row sums are approximately 1.0 (valid softmax)
    bool validate_softmax(int layer, int head, float tolerance = 1e-4f) const;

    /// Clear all captures
    void clear();

   private:
    bool capturing_ = false;
    std::vector<std::vector<Tensor>> captures_;
};

}  // namespace interpret
}  // namespace serius