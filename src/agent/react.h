#pragma once

#include "model_interface.h"

#include <string>

namespace serius {
namespace agent {

/// Minimal tool result: name + JSON output.
struct ToolResult
{
    std::string name;
    std::string output;
};

/// Tool interface: a callable tool with a name.
class Tool
{
   public:
    virtual ~Tool() = default;
    virtual std::string name() const = 0;
    virtual std::string call(const std::string &args_json) = 0;
};

/// ReAct (Reasoning + Acting) loop engine.
/// Thought → Action → Observation → Answer, up to max_steps.
class ReActEngine
{
   public:
    ReActEngine(Model &model, int max_steps = 5);

    ReActEngine(const ReActEngine &) = delete;
    ReActEngine &operator=(const ReActEngine &) = delete;

    /// Run the ReAct loop for a user query.
    /// tool_results: optional map of tool name -> precomputed output (mock).
    std::string run(const std::string &user_query,
                    const std::string &system_prompt,
                    const ToolResult *tool_results = nullptr,
                    size_t n_tool_results = 0);

    int max_steps() const { return max_steps_; }

   private:
    Model &model_;
    int max_steps_;
};

}  // namespace agent
}  // namespace serius
