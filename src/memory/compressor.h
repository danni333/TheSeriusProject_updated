#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace memory {

/// Result of a compression operation.
struct CompressionResult
{
    core::Tensor compressed;   // (summary_tokens, d_model) summary vectors
    float information_ratio;  // compressed_size / original_size
};

/// State compressor: compresses long conversations into a fixed set of
/// summary vectors using cross-attention-style pooling.
class StateCompressor
{
   public:
    StateCompressor(size_t d_model, size_t summary_tokens, unsigned seed = 42);

    /// Compress a sequence of hidden states into summary vectors.
    /// @param hidden_states (seq_len, d_model) input hidden states.
    /// @return CompressionResult with compressed vectors and ratio.
    CompressionResult compress(const core::Tensor &hidden_states);

    /// Build context from multiple compressed sessions.
    /// @param session_paths List of paths to saved session summaries.
    /// @return Concatenated context tensor.
    core::Tensor build_context(const std::vector<std::string> &session_paths);

    /// Save a compressed state to disk.
    bool save_memory(const std::string &path, const core::Tensor &state) const;

    /// Load a compressed state from disk.
    core::Tensor load_memory(const std::string &path) const;

    size_t d_model() const { return d_model_; }
    size_t summary_tokens() const { return summary_tokens_; }

   private:
    /// Learned summary query vectors used for pooling.
    core::Tensor summary_queries_;

    size_t d_model_;
    size_t summary_tokens_;
};

}  // namespace memory
}  // namespace serius