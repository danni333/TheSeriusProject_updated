// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "common/text_embed.h"
#include "mcts.h"

namespace serius {
namespace reasoning {

/// Simple keyword-based verifier: higher score if the answer contains expected
/// keywords and no obvious failure markers.
class KeywordVerifier : public Verifier
{
   public:
    KeywordVerifier(std::string expected_keyword,
                    std::vector<std::string> failure_keywords);
    KeywordVerifier(std::string expected_keyword);

    float score(const std::string &reasoning) override;

   private:
    std::string expected_keyword_;
    std::vector<std::string> failure_keywords_;
};

/// Semantic verifier: scores reasoning by similarity to a reference answer.
///
/// When an embedder is installed, the score is the cosine similarity between
/// the reasoning embedding and the reference embedding, rescaled from the
/// expected "related" cosine range [`cosine_lo`, `cosine_hi`]
/// (default 0.2..0.9) to [0, 1], minus a penalty for each failure keyword
/// found. Without an embedder it falls back to lexical overlap against the
/// reference text.
///
/// The embedder is any callable `std::vector<float>(const std::string&)`
/// (same type as `common::TextEmbedFn` / `rag::EmbedFn`), e.g. a lambda
/// wrapping `hub::NomicBertEmbedder::embed`.
class EmbeddingVerifier : public Verifier
{
   public:
    EmbeddingVerifier(common::TextEmbedFn embedder,
                      std::string reference,
                      std::vector<std::string> failure_keywords = {});

    /// Cosine range considered "unrelated".."fully matching" (default
    /// 0.2..0.9). Requires lo < hi.
    void set_cosine_range(float lo, float hi)
    {
        if (hi > lo) {
            cosine_lo_ = lo;
            cosine_hi_ = hi;
        }
    }

    float score(const std::string &reasoning) override;

   private:
    float lexical_fallback_score(const std::string &reasoning) const;

    common::TextEmbedFn embedder_;
    std::string reference_;
    std::vector<float> reference_emb_;
    std::vector<std::string> failure_keywords_;
    float cosine_lo_ = 0.2f;
    float cosine_hi_ = 0.9f;
};

}  // namespace reasoning
}  // namespace serius
