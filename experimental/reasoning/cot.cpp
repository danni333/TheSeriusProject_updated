#include "cot.h"

namespace serius {
namespace reasoning {

ChainOfThought::ChainOfThought(Model &model, int max_thinking_tokens)
    : model_(model), max_thinking_tokens_(max_thinking_tokens)
{
}

ThinkingResult ChainOfThought::generate_with_thinking(const std::string &prompt)
{
    ThinkingResult result;

    // Phase 1: produce thinking in a hidden buffer.
    std::string thinking_prompt = prompt + "\n```\n";
    result.thinking = model_.generate(thinking_prompt, max_thinking_tokens_, "```");

    // Phase 2: produce the final answer conditioned on the thinking.
    std::string answer_prompt = thinking_prompt + result.thinking + "```\n<answer>\n";
    result.answer = model_.generate(answer_prompt, 1024, "</answer>");

    return result;
}

}  // namespace reasoning
}  // namespace serius
