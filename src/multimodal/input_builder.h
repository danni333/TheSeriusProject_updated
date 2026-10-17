#pragma once

#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace multimodal {

using core::Tensor;

/// Concatenate 2D embeddings along the sequence axis. All tensors must share the last dim.
Tensor concat_sequence(const std::vector<Tensor> &parts);

/// Builds a single embedding sequence from optional image/audio tokens and text embeddings.
class MultimodalInputBuilder
{
   public:
    explicit MultimodalInputBuilder(size_t d_model, unsigned seed = 7);

    /// text_embeds: (seq_text, d_model)
    /// visual_tokens / audio_tokens: (n_tokens, d_model) or nullptr
    Tensor build(const Tensor &text_embeds, const Tensor *visual_tokens, const Tensor *audio_tokens) const;

    size_t d_model() const { return d_model_; }

   private:
    Tensor special_token(const Tensor &embed) const;

    size_t d_model_;
    Tensor img_start_;
    Tensor img_end_;
    Tensor aud_start_;
    Tensor aud_end_;
};

}  // namespace multimodal
}  // namespace serius
