#pragma once

#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace interpret {

using core::Tensor;

/// Projects intermediate hidden states onto the vocabulary to understand
/// what the model "predicts" at each layer (logit lens technique).
class LogitLens
{
   public:
    struct Prediction
    {
        int token_id;
        float logit;
        std::string token;
    };

    LogitLens() = default;

    /// Set the vocabulary mapping (token_id -> token string)
    void set_vocab(const std::vector<std::string> &vocab) { vocab_ = vocab; }

    /// Compute logit lens predictions for a hidden state
    /// hidden: (d_model,) single token representation
    /// lm_head: (vocab_size, d_model) projection matrix
    /// top_k: number of top predictions to return
    std::vector<Prediction> compute(const Tensor &hidden, const Tensor &lm_head,
                                    int top_k = 5) const;

    /// Get human-readable token for a token id
    std::string token_text(int token_id) const;

   private:
    std::vector<std::string> vocab_;
};

}  // namespace interpret
}  // namespace serius