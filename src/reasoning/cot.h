#pragma once

#include <string>

namespace serius {
namespace reasoning {

/// Result of a System-2 generation: hidden thinking + final answer.
struct ThinkingResult
{
    std::string thinking;  // hidden buffer (not shown to the user)
    std::string answer;    // final answer
};

/// Model interface used by reasoning components.
class Model
{
   public:
    virtual ~Model() = default;
    virtual std::string generate(const std::string &prompt,
                                 int max_tokens = 512,
                                 const std::string &stop_token = "") = 0;
};

/// Chain of Thought with a separate thinking buffer (o1 / R1 style).
class ChainOfThought
{
   public:
    ChainOfThought(Model &model, int max_thinking_tokens = 2048);

    /// Generate thinking in a hidden buffer, then the final answer.
    ThinkingResult generate_with_thinking(const std::string &prompt);

   private:
    Model &model_;
    int max_thinking_tokens_;
};

}  // namespace reasoning
}  // namespace serius
