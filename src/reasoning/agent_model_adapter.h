// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#pragma once

#include "cot.h"
#include "mcts.h"

#include "../agent/model_interface.h"

namespace serius {
namespace reasoning {

/// Adapts agent::Model (e.g. LlamaAgentModel over a real LLaMAModel) to the
/// reasoning::Model interface used by ChainOfThought.
class AgentReasoningModel final : public Model
{
   public:
    explicit AgentReasoningModel(agent::Model &model) : model_(model) {}

    std::string generate(const std::string &prompt,
                         int max_tokens = 512,
                         const std::string &stop_token = "") override
    {
        agent::SamplingParams params;
        params.max_tokens = max_tokens;
        params.temperature = 0.7f;
        if (!stop_token.empty()) {
            params.stop_tokens = {stop_token};
        }
        return model_.generate(prompt, params);
    }

   private:
    agent::Model &model_;
};

/// Adapts agent::Model to the reasoning::MCTSModel interface used by
/// MCTSReasoner. Temperature is forwarded to SamplingParams.
class AgentMCTSModel final : public MCTSModel
{
   public:
    explicit AgentMCTSModel(agent::Model &model) : model_(model) {}

    std::string generate(const std::string &prompt,
                         int max_tokens = 256,
                         float temperature = 0.7f) override
    {
        agent::SamplingParams params;
        params.max_tokens = max_tokens;
        params.temperature = temperature;
        return model_.generate(prompt, params);
    }

   private:
    agent::Model &model_;
};

}  // namespace reasoning
}  // namespace serius
