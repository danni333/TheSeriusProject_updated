#include "react.h"
#include "action_parser.h"
#include "context_manager.h"

#include "tools/registry.h"

#include <nlohmann/json.hpp>

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
                             size_t n_tool_results,
                             const tools::ToolRegistry *tool_registry)
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

            bool found = false;

            // Real tool execution via the registry takes precedence over
            // precomputed mock results.
            if (tool_registry != nullptr && tool_registry->has_tool(call.name)) {
                nlohmann::json args = nlohmann::json::parse(call.args_json, nullptr, false);
                if (!args.is_object()) {
                    args = nlohmann::json::object();
                }
                nlohmann::json result = tool_registry->execute(call.name, args);
                observation = result.is_string() ? result.get<std::string>() : result.dump();
                found = true;
            }

            // Fall back to a precomputed tool result (mock) if available.
            if (!found) {
                for (size_t i = 0; i < n_tool_results; ++i) {
                    if (tool_results[i].name == call.name) {
                        observation = tool_results[i].output;
                        found = true;
                        break;
                    }
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
