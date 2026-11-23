#include "perplexity_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

#include "infer/llama_inference.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"

namespace serius {
namespace eval {

namespace {

using Clock = std::chrono::steady_clock;

/// Log-probability of `target` under a softmax over `row` (length vocab).
float token_logprob(const float *row, size_t vocab, int target)
{
    if (target < 0 || static_cast<size_t>(target) >= vocab) {
        return 0.0f;
    }
    float max_logit = row[0];
    for (size_t v = 1; v < vocab; ++v) {
        max_logit = std::max(max_logit, row[v]);
    }
    double sum_exp = 0.0;
    for (size_t v = 0; v < vocab; ++v) {
        sum_exp += std::exp(static_cast<double>(row[v]) - static_cast<double>(max_logit));
    }
    const double log_sum = std::log(sum_exp) + static_cast<double>(max_logit);
    return static_cast<float>(static_cast<double>(row[static_cast<size_t>(target)]) - log_sum);
}

PerplexityResult evaluate_ids(const model::LLaMAModel &model,
                              const std::vector<int64_t> &ids,
                              const PerplexityOptions &opts)
{
    PerplexityResult result;
    const auto t0 = Clock::now();

    const auto &cfg = model.config();
    const size_t vocab = cfg.vocab_size;
    if (vocab == 0) {
        result.error = "model has zero vocab_size";
        return result;
    }

    // Clamp feed ids into range; remember which stream positions are scorable.
    std::vector<int> feed(ids.size());
    std::vector<char> scorable(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= 0 && static_cast<size_t>(ids[i]) < vocab) {
            feed[i] = static_cast<int>(ids[i]);
            scorable[i] = 1;
        } else {
            feed[i] = 0;
        }
    }

    // Context per forward: bounded by model positions and the fixed KV-cache
    // capacity (DEFAULT_MAX_SEQ_LEN = 4096 in llama_inference.cpp).
    const size_t max_ctx = std::min(cfg.max_position_embeddings, static_cast<size_t>(4096));
    size_t context = std::max<size_t>(2, std::min(opts.context_tokens, max_ctx));
    if (context < 2) {
        context = 2;
    }

    double sum_nll = 0.0;
    size_t scored = 0;
    size_t pos = 0;                 // next unconsumed stream token
    std::vector<int> prefix;        // carry-over context (unscored), <= context-1
    bool done = false;

    while (pos < ids.size() && !done) {
        const size_t prefix_len = prefix.size();
        const size_t body_cap = context - prefix_len;  // >= 1 by construction
        const size_t body_len = std::min(body_cap, ids.size() - pos);

        // input = prefix + body
        const size_t L = prefix_len + body_len;
        std::vector<int> input;
        input.reserve(L);
        input.insert(input.end(), prefix.begin(), prefix.end());
        input.insert(input.end(), feed.begin() + static_cast<std::ptrdiff_t>(pos),
                     feed.begin() + static_cast<std::ptrdiff_t>(pos + body_len));

        // Stream index of input[j] is pos - prefix_len + j for j >= prefix_len.
        infer::KVCache cache;
        for (size_t k = 0; k + 1 < L && !done; ++k) {
            core::Tensor logits = infer::llama_forward_single_token(
                model, input[k], k, &cache);
            if (logits.size() < vocab) {
                continue;
            }
            const size_t target_j = k + 1;
            if (target_j < prefix_len) {
                continue;  // target is a prefix (context) token — not scored
            }
            const size_t stream_idx = pos - prefix_len + target_j;
            if (!scorable[stream_idx]) {
                continue;
            }
            if (opts.max_tokens > 0 && scored >= opts.max_tokens) {
                done = true;
                break;
            }
            const float *row = logits.data() + (logits.size() - vocab);
            const float lp = token_logprob(row, vocab, feed[stream_idx]);
            sum_nll += -static_cast<double>(lp);
            ++scored;
        }

        // Carry-over: last context-1 tokens of the input become next prefix.
        const size_t keep = std::min(context - 1, L);
        prefix.assign(input.end() - static_cast<std::ptrdiff_t>(keep), input.end());
        pos += body_len;
        ++result.chunks;
    }

    result.total_tokens = ids.size();
    result.scored_tokens = scored;
    result.mean_nll = scored > 0 ? sum_nll / static_cast<double>(scored) : 0.0;
    result.perplexity = scored > 0 ? std::exp(result.mean_nll) : 1.0;
    result.elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - t0).count();
    result.ok = scored > 0;
    if (!result.ok) {
        result.error = "no scorable tokens (text too short or empty)";
    }
    return result;
}

}  // namespace

PerplexityResult evaluate_perplexity(const model::LLaMAModel &model,
                                     const BPETokenizer &tokenizer,
                                     const std::string &text,
                                     const PerplexityOptions &opts)
{
    const auto ids_token = tokenizer.encode(text);
    std::vector<int64_t> ids;
    ids.reserve(ids_token.size());
    for (TokenId id : ids_token) {
        ids.push_back(static_cast<int64_t>(id));
    }
    return evaluate_ids(model, ids, opts);
}

PerplexityResult evaluate_perplexity_file(const model::LLaMAModel &model,
                                          const BPETokenizer &tokenizer,
                                          const std::string &path,
                                          const PerplexityOptions &opts)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        PerplexityResult r;
        r.error = "cannot open file: " + path;
        return r;
    }
    const std::string text((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    return evaluate_perplexity(model, tokenizer, text, opts);
}

}  // namespace eval
}  // namespace serius
