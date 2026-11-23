#include "consensus.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace serius {
namespace agent {

DelphiConsensus::DelphiConsensus(std::vector<agent::Model *> agents, int max_rounds)
    : agents_(std::move(agents)), max_rounds_(max_rounds)
{
}

bool DelphiConsensus::check_consensus(const std::vector<std::string> &responses) const
{
    if (responses.size() < 2) {
        return true;
    }
    // Compare all pairs; require average cosine similarity > 0.9.
    float total_sim = 0.0f;
    size_t pairs = 0;
    for (size_t i = 0; i < responses.size(); ++i) {
        for (size_t j = i + 1; j < responses.size(); ++j) {
            total_sim += cosine_sim(responses[i], responses[j]);
            pairs++;
        }
    }
    if (pairs == 0) return true;
    float avg = total_sim / static_cast<float>(pairs);
    return avg > 0.9f;
}

std::string DelphiConsensus::majority_vote(const std::vector<std::string> &responses) const
{
    // Cluster responses by cosine similarity and pick the largest cluster's answer.
    if (responses.empty()) return "";
    if (responses.size() == 1) return responses[0];

    std::vector<std::string> cluster_representative;
    std::vector<size_t> cluster_sizes;

    for (const auto &resp : responses) {
        bool found = false;
        for (size_t c = 0; c < cluster_representative.size(); ++c) {
            if (cosine_sim(resp, cluster_representative[c]) > 0.7f) {
                cluster_sizes[c]++;
                found = true;
                break;
            }
        }
        if (!found) {
            cluster_representative.push_back(resp);
            cluster_sizes.push_back(1);
        }
    }

    // Pick the largest cluster.
    size_t best = 0;
    for (size_t i = 1; i < cluster_sizes.size(); ++i) {
        if (cluster_sizes[i] > cluster_sizes[best]) {
            best = i;
        }
    }
    return cluster_representative[best];
}

float DelphiConsensus::cosine_sim(const std::string &a, const std::string &b) const
{
    // Simple character trigram overlap as a cheap embedding proxy.
    std::set<std::string> trigrams_a, trigrams_b;

    auto build_trigrams = [](const std::string &s, std::set<std::string> &out) {
        for (size_t i = 0; i + 2 < s.size(); ++i) {
            out.insert(s.substr(i, 3));
        }
    };

    build_trigrams(a, trigrams_a);
    build_trigrams(b, trigrams_b);

    if (trigrams_a.empty() && trigrams_b.empty()) return 1.0f;
    if (trigrams_a.empty() || trigrams_b.empty()) return 0.0f;

    size_t intersection = 0;
    for (const auto &tri : trigrams_a) {
        if (trigrams_b.count(tri)) intersection++;
    }

    float union_size = static_cast<float>(trigrams_a.size() + trigrams_b.size() - intersection);
    if (union_size <= 0.0f) return 0.0f;
    return static_cast<float>(intersection) / union_size;
}

ConsensusResult DelphiConsensus::debate(const std::string &question)
{
    ConsensusResult result;
    result.converged = false;
    result.rounds = 0;

    if (agents_.empty()) {
        result.final_answer = "";
        return result;
    }

    std::vector<std::string> responses(agents_.size());

    for (int round = 0; round < max_rounds_; ++round) {
        result.rounds = round + 1;

        for (size_t i = 0; i < agents_.size(); ++i) {
            std::string prompt = question;
            if (round > 0) {
                prompt += "\n\nOther participants' responses:\n";
                for (size_t j = 0; j < agents_.size(); ++j) {
                    if (j != i) {
                        prompt += "Agent " + std::to_string(j) + ": " + responses[j] + "\n";
                    }
                }
                prompt += "\nReconsider your response:";
            }

            SamplingParams params;
            params.temperature = 0.7f;
            params.max_tokens = 512;
            responses[i] = agents_[i]->generate(prompt, params);
        }

        result.all_responses = responses;

        if (check_consensus(responses)) {
            result.final_answer = responses[0];
            result.converged = true;
            return result;
        }
    }

    // Fallback: majority vote.
    result.final_answer = majority_vote(responses);
    result.converged = false;
    return result;
}

}  // namespace agent
}  // namespace serius