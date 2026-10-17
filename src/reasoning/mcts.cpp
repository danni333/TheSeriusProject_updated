#include "mcts.h"

#include <algorithm>
#include <limits>

namespace serius {
namespace reasoning {

MCTSReasoner::MCTSReasoner(MCTSModel &model, Verifier &verifier, int simulations)
    : model_(model), verifier_(verifier), simulations_(simulations)
{
}

std::string MCTSReasoner::search(const std::string &problem)
{
    auto root = std::make_unique<MCTSNode>(problem);
    MCTSNode *root_ptr = root.get();

    for (int sim = 0; sim < simulations_; ++sim) {
        // SELECT: walk down the tree by UCB1 until a leaf.
        MCTSNode *node = root_ptr;
        while (!node->children.empty()) {
            MCTSNode *best = nullptr;
            float best_score = -std::numeric_limits<float>::infinity();
            for (auto &child : node->children) {
                float s = child->ucb1();
                if (s > best_score) {
                    best_score = s;
                    best = child.get();
                }
            }
            if (best == nullptr) {
                break;
            }
            node = best;
        }

        // EXPAND: generate a few candidate next steps.
        expand(node);

        // If expansion produced children, pick one to evaluate.
        if (!node->children.empty()) {
            node = node->children[0].get();
        }

        // EVALUATE + BACKPROPAGATE.
        float score = evaluate(node);
        backpropagate(node, score);
    }

    return best_path(root_ptr);
}

void MCTSReasoner::expand(MCTSNode *node)
{
    for (int i = 0; i < 3; ++i) {
        std::string next = model_.generate(node->text + "\nStep: ", 256, 0.5f + i * 0.1f);
        node->children.push_back(std::make_unique<MCTSNode>(node->text + next, node));
    }
}

float MCTSReasoner::evaluate(MCTSNode *node)
{
    std::string completion =
        model_.generate(node->text + "\nAnswer: ", 512, 0.5f);
    return verifier_.score(node->text + completion);
}

void MCTSReasoner::backpropagate(MCTSNode *node, float score)
{
    MCTSNode *cur = node;
    while (cur != nullptr) {
        cur->visits += 1;
        cur->value += score;
        cur = cur->parent;
    }
}

std::string MCTSReasoner::best_path(MCTSNode *root)
{
    MCTSNode *cur = root;
    while (!cur->children.empty()) {
        MCTSNode *best = nullptr;
        float best_value = -std::numeric_limits<float>::infinity();
        for (auto &child : cur->children) {
            float v = (child->visits > 0)
                          ? child->value / static_cast<float>(child->visits)
                          : child->value;
            if (v > best_value) {
                best_value = v;
                best = child.get();
            }
        }
        if (best == nullptr) {
            break;
        }
        cur = best;
    }
    return cur->text;
}

}  // namespace reasoning
}  // namespace serius
