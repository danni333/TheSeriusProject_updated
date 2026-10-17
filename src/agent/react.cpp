#include "react.h"
#include "action_parser.h"
#include "context_manager.h"

#include <sstream>

namespace serius {
namespace agent {

ReActEngine::ReActEngine(Model &model, int max_steps)
    : model_(model), max_steps_(max_steps)
{
}

std::string ReActEngine::run(const std::string &user_query,
                             const std::string &system_prompt,
                             const ToolResult *tool_results,
                             size_t n_tool_results)
{
    std::string context = system_prompt;
    if (!context.empty() && context.back() != '\n') {
        context += "\n";
    }
    context += "[USER] " + user_query + "\n";

    for (int step = 0; step < max_steps_; ++step) {
        SamplingParams params;
        params.temperature = 0.7f;
        params.max_tokens = 256;
        params.stop_tokens = {"[ACTION]", "[ANSWER]"};

        std::string output = model_.generate(context, params);
        context += output;

        if (ActionParser::has_answer(output)) {
            std::string answer = ActionParser::parse_answer(output);
            return answer.empty() ? output : answer;
        }

        if (ActionParser::has_action(output)) {
            ActionCall call = ActionParser::parse(output);
            std::string observation;

            // Look up a precomputed tool result (mock) if available.
            bool found = false;
            for (size_t i = 0; i < n_tool_results; ++i) {
                if (tool_results[i].name == call.name) {
                    observation = tool_results[i].output;
                    found = true;
                    break;
                }
            }
            if (!found) {
                observation = "Tool " + call.name + " not available.";
            }

            observation = ContextManager::truncate_observation(observation, 200);
            context += "\n[OBSERVATION] " + observation + "\n[THOUGHT] ";
        }
    }

    // Force a final answer if the loop exhausted all steps.
    SamplingParams params;
    params.temperature = 0.5f;
    params.max_tokens = 512;
    return model_.generate(context + "\n[ANSWER] ", params);
}

}  // namespace agent
}  // namespace serius
