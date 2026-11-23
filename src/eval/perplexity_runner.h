#pragma once

#include <cstddef>
#include <string>

namespace serius {
namespace model { class LLaMAModel; }
class BPETokenizer;

namespace eval {

struct PerplexityOptions
{
    /// Max positions fed to the model per forward pass. Each chunk is scored
    /// with up to `context_tokens - 1` tokens of carry-over context from the
    /// previous chunk (fresh KV cache per chunk, positions restart at 0).
    /// Clamped to [2, min(model context, KV-cache capacity)].
    size_t context_tokens = 512;

    /// Stop after this many scored target tokens. 0 = score the whole text.
    size_t max_tokens = 0;
};

struct PerplexityResult
{
    bool ok = false;
    std::string error;

    double mean_nll = 0.0;      ///< mean negative log-likelihood per scored token
    double perplexity = 1.0;    ///< exp(mean_nll)

    size_t scored_tokens = 0;   ///< tokens that contributed to the metric
    size_t total_tokens = 0;    ///< tokens produced by the tokenizer
    size_t chunks = 0;          ///< forward-pass chunks used
    double elapsed_seconds = 0.0;
};

/// Teacher-forced perplexity of `text` under a real LLaMAModel.
///
/// Uses the production decode path: `infer::llama_forward_single_token` with a
/// KV cache, scoring each token by the logits produced for its predecessor.
/// Token ids outside [0, vocab_size) are fed clamped but not scored.
PerplexityResult evaluate_perplexity(const model::LLaMAModel &model,
                                     const BPETokenizer &tokenizer,
                                     const std::string &text,
                                     const PerplexityOptions &opts = {});

/// Same as evaluate_perplexity but reads the corpus from a file.
PerplexityResult evaluate_perplexity_file(const model::LLaMAModel &model,
                                          const BPETokenizer &tokenizer,
                                          const std::string &path,
                                          const PerplexityOptions &opts = {});

}  // namespace eval
}  // namespace serius
