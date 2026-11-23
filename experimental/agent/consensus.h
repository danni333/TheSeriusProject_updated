#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/text_embed.h"
#include "model_interface.h"

namespace serius {
namespace agent {

/// Result of a consensus vote.
struct ConsensusResult
{
    std::string final_answer;
    int rounds;
    bool converged;
    std::vector<std::string> all_responses;
};

/// Delphi consensus: multiple agents debate and converge on an answer.
class DelphiConsensus
{
   public:
    DelphiConsensus(std::vector<agent::Model *> agents, int max_rounds = 3);

    /// Optional text embedder (e.g. NomicBertEmbedder::embed_fn()). When set,
    /// response similarity uses embedding cosine mapped through
    /// [embed_lo_, embed_hi_] instead of character-trigram Jaccard.
    void set_embed_fn(common::TextEmbedFn fn) { embed_fn_ = std::move(fn); }
    /// Expected cosine range for "related" responses when using an embedder
    /// (text embeddings typically score ~0.3-0.4 unrelated, ~0.6-0.9 related).
    void set_embed_range(float lo, float hi) { embed_lo_ = lo; embed_hi_ = hi; }

    /// Run the Delphi debate on a question.
    ConsensusResult debate(const std::string &question);

    size_t num_agents() const { return agents_.size(); }

   private:
    std::vector<agent::Model *> agents_;
    int max_rounds_;
    common::TextEmbedFn embed_fn_;
    float embed_lo_ = 0.3f;
    float embed_hi_ = 0.9f;
    // Lazily filled embedding cache so each response is embedded once per
    // DelphiConsensus instance even across many pairwise comparisons.
    mutable std::unordered_map<std::string, std::vector<float>> embed_cache_;

    /// Check if responses have converged (cosine similarity > 0.9).
    bool check_consensus(const std::vector<std::string> &responses) const;

    /// Majority vote fallback.
    std::string majority_vote(const std::vector<std::string> &responses) const;

    /// Response similarity in [0, 1]: embedding cosine (when an embedder is
    /// set) or character n-gram Jaccard otherwise.
    float cosine_sim(const std::string &a, const std::string &b) const;
};

}  // namespace agent
}  // namespace serius