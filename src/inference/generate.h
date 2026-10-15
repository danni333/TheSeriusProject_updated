#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../transformer/model.h"
#include "sampler.h"

namespace serius {
namespace inference {

/// Configuration for the generation loop.
struct GenerateConfig
{
    int max_new_tokens = 256;    // Maximum tokens to generate
    int eos_token_id = 2;       // End-of-sequence token ID
    int pad_token_id = 0;       // Padding token ID
    SamplerConfig sampler;       // Sampling configuration
};

/// Result of a generation call.
struct GenerateResult
{
    std::vector<int> token_ids;  // All tokens (prompt + generated)
    int prompt_len = 0;          // Length of the original prompt
    int generated_len = 0;       // Number of new tokens generated
    bool hit_eos = false;        // Whether generation stopped due to EOS
};

/// Callback invoked after each token is generated.
/// token: the newly generated token ID
/// is_done: true if this is the last token (EOS or max_new_tokens reached)
/// Return false from callback to stop generation early.
using StreamCallback = std::function<bool(int token, bool is_done)>;

/// Generate tokens autoregressively from a transformer model.
/// Supports streaming via optional callback.
GenerateResult generate(model::TransformerModel &model, const std::vector<int> &prompt_ids,
                        const GenerateConfig &config = GenerateConfig(),
                        StreamCallback callback = nullptr);

/// Beam search generation (basic implementation).
struct Beam
{
    std::vector<int> tokens;
    float score = 0.0f;    // log-probability
    bool finished = false;
};

/// Generate using beam search.
/// Returns beam_width candidate sequences sorted by score.
std::vector<Beam> beam_search(model::TransformerModel &model, const std::vector<int> &prompt_ids, int beam_width = 4,
                              int max_new_tokens = 256, int eos_token_id = 2);

}  // namespace inference
}  // namespace serius
