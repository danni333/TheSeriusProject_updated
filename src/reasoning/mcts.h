#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace serius {
namespace reasoning {

/// A node in the reasoning search tree.
struct MCTSNode
{
    std::string text;       // reasoning text generated up to this node
    float value = 0.0f;     // accumulated value from rollouts
    int visits = 0;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode *parent = nullptr;

    explicit MCTSNode(std::string t, MCTSNode *p = nullptr)
        : text(std::move(t)), parent(p)
    {
    }

    /// UCB1 selection score.
    float ucb1(float c = 1.414f) const
    {
        if (visits == 0) {
            return std::numeric_limits<float>::infinity();
        }
        if (parent == nullptr || parent->visits == 0) {
            return value / static_cast<float>(visits);
        }
        return value / static_cast<float>(visits) +
               c * std::sqrt(std::log(static_cast<float>(parent->visits)) /
                             static_cast<float>(visits));
    }
};

/// Verifier interface: scores a reasoning trace in [0, 1].
class Verifier
{
   public:
    virtual ~Verifier() = default;
    virtual float score(const std::string &reasoning) = 0;
};

/// Model interface used by MCTS.
class MCTSModel
{
   public:
    virtual ~MCTSModel() = default;
    virtual std::string generate(const std::string &prompt,
                                 int max_tokens = 256,
                                 float temperature = 0.7f) = 0;
};

/// Monte Carlo Tree Search reasoner for test-time compute.
class MCTSReasoner
{
   public:
    MCTSReasoner(MCTSModel &model, Verifier &verifier, int simulations = 10);

    /// Search for the best reasoning path for a problem.
    std::string search(const std::string &problem);

   private:
    MCTSModel &model_;
    Verifier &verifier_;
    int simulations_;

    void expand(MCTSNode *node);
    float evaluate(MCTSNode *node);
    void backpropagate(MCTSNode *node, float score);
    std::string best_path(MCTSNode *root);
};

}  // namespace reasoning
}  // namespace serius
