#pragma once

#include <string>
#include <vector>

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

    /// Run the Delphi debate on a question.
    ConsensusResult debate(const std::string &question);

    size_t num_agents() const { return agents_.size(); }

   private:
    std::vector<agent::Model *> agents_;
    int max_rounds_;

    /// Check if responses have converged (cosine similarity > 0.9).
    bool check_consensus(const std::vector<std::string> &responses) const;

    /// Majority vote fallback.
    std::string majority_vote(const std::vector<std::string> &responses) const;

    /// Simple embedding-based cosine similarity (using character n-gram overlap).
    float cosine_sim(const std::string &a, const std::string &b) const;
};

}  // namespace agent
}  // namespace serius