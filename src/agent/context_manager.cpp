#include "context_manager.h"

#include <string>

namespace serius {
namespace agent {

std::string ContextManager::truncate_observation(const std::string &obs,
                                                 size_t max_tokens)
{
    if (obs.size() <= max_tokens) {
        return obs;
    }
    return obs.substr(0, max_tokens) + "\n[...TRUNCATED...]";
}

std::string ContextManager::manage_context(const std::string &context,
                                           size_t system_prompt_length,
                                           size_t max_ctx_tokens)
{
    if (context.size() <= max_ctx_tokens) {
        return context;
    }
    size_t keep_end = max_ctx_tokens > system_prompt_length
                          ? max_ctx_tokens - system_prompt_length
                          : 0;
    std::string head = context.substr(0, system_prompt_length);
    std::string tail = keep_end > 0 ? context.substr(context.size() - keep_end) : std::string{};
    return head + "\n[...]\n" + tail;
}

}  // namespace agent
}  // namespace serius
