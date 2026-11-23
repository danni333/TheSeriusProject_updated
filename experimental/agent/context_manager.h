#pragma once

#include <string>

namespace serius {
namespace agent {

/// Context window management: truncation and summarization helpers.
class ContextManager
{
   public:
    /// Truncate `obs` to at most `max_tokens` tokens by character count
    /// approximation, appending a truncation marker.
    static std::string truncate_observation(const std::string &obs,
                                            size_t max_tokens);

    /// Manage the total context: keep the system prompt head and the tail,
    /// dropping the middle when over `max_ctx_tokens`.
    static std::string manage_context(const std::string &context,
                                      size_t system_prompt_length,
                                      size_t max_ctx_tokens);
};

}  // namespace agent
}  // namespace serius
