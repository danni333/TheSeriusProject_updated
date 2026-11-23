// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/text_embed.h"

namespace serius {
namespace grounding {

/// A citation token like [1], [2] and the document it references.
struct Citation
{
    int doc_id;
    std::string token;
};

/// Extract citations like [1], [2] from text, in order of appearance.
std::vector<Citation> extract_citations(const std::string &text);

/// Lexical support score of a claim against a document (0..1).
/// Word-level overlap on content words (stopwords removed); used as the
/// fallback when no embedder is installed. Equivalent to
/// `SupportScorer().score(claim, document)`.
float support_score(const std::string &claim, const std::string &document);

/// Claim/document support scorer.
///
/// Combines two signals:
///  - lexical overlap on content words with morphological tolerance
///    (e.g. "mammals" matches "mammal"), always available and deterministic;
///  - cosine similarity between text embeddings, when an embedder is
///    installed. The embedder is any callable with signature
///    `std::vector<float>(const std::string&)` — same type as
///    `common::TextEmbedFn` / `rag::EmbedFn`; e.g. a lambda wrapping
///    `hub::NomicBertEmbedder::embed` or `hub::EmbeddingManager`.
///
/// The cosine is rescaled from an expected "related" range
/// [`cosine_lo`, `cosine_hi`] (default 0.25..0.9, typical for
/// nomic-embed-style models) to [0, 1] via `common::cosine_unit`, so
/// unrelated texts (~0.3 cosine) do not inflate the score. Tune with
/// `set_cosine_range()` and `set_semantic_weight()`.
class SupportScorer
{
   public:
    SupportScorer();
    explicit SupportScorer(common::TextEmbedFn embedder);

    void set_embedder(common::TextEmbedFn embedder);

    /// Weight of the semantic signal in [0, 1] (default 0.5). 0 = lexical
    /// only, 1 = embeddings only.
    void set_semantic_weight(float w) { semantic_weight_ = std::clamp(w, 0.0f, 1.0f); }

    /// Cosine range considered "unrelated".."fully related". Requires lo < hi.
    void set_cosine_range(float lo, float hi)
    {
        if (hi > lo) {
            cosine_lo_ = lo;
            cosine_hi_ = hi;
        }
    }

    /// Support score of `claim` given `document` in [0, 1].
    float score(const std::string &claim, const std::string &document);

    /// Score one claim against many documents; the claim embedding is
    /// computed once and reused for every document.
    std::vector<float> score_claim_against_documents(const std::string &claim,
                                                     const std::vector<std::string> &documents);

   private:
    float lexical_score(const std::string &claim, const std::string &document) const;
    float semantic_score(const std::string &claim, const std::string &document);
    const std::vector<float> &embed_cached(const std::string &text);

    common::TextEmbedFn embedder_;
    float semantic_weight_ = 0.5f;
    float cosine_lo_ = 0.25f;
    float cosine_hi_ = 0.9f;
    std::unordered_map<std::string, std::vector<float>> emb_cache_;
};

}  // namespace grounding
}  // namespace serius
