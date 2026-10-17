#pragma once

#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace safety {

using core::Tensor;

/// Lightweight toxicity classifier for real-time output filtering.
/// Uses a simple linear model: embedding -> linear -> relu -> linear -> sigmoid.
class ToxicityClassifier
{
public:
    struct Config {
        size_t vocab_size = 50000;
        size_t embedding_dim = 256;
        size_t hidden_dim = 128;
        float threshold = 0.8f;
    };

    explicit ToxicityClassifier(const Config& config = Config{});
    ~ToxicityClassifier();

    /// Score the toxicity of a token sequence.
    /// @param tokens Token IDs to classify
    /// @return Toxicity score in [0, 1], where higher is more toxic
    float score(const std::vector<int>& tokens) const;

    /// Set the toxicity threshold.
    void set_threshold(float threshold) { threshold_ = threshold; }

    /// Get the current threshold.
    float threshold() const { return threshold_; }

    /// Check if the sequence exceeds the toxicity threshold.
    bool is_toxic(const std::vector<int>& tokens) const;

private:
    Config config_;
    float threshold_;

    // Model weights (simple linear layers)
    Tensor embedding_;      // (vocab_size, embedding_dim)
    Tensor hidden_weight_;  // (embedding_dim, hidden_dim)
    Tensor hidden_bias_;    // (hidden_dim,)
    Tensor output_weight_;  // (hidden_dim, 1)
    Tensor output_bias_;    // (1,)

    // Helper: compute embedding mean
    float compute_embedding_mean(const std::vector<int>& tokens) const;
};

}  // namespace safety
}  // namespace serius
