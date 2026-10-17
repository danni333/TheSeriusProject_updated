#include "sanitizer.h"

#include <algorithm>

namespace serius {
namespace safety {

Sanitizer::Sanitizer(const Config& config)
    : config_(config)
{
    // Add patterns based on config
    if (config_.remove_sql_injection) {
        custom_patterns_.emplace_back(std::regex(R"(\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|UNION)\s+[A-Z]+\b)", std::regex_constants::icase), "[SQL_REDACTED]");
        custom_patterns_.emplace_back(std::regex(R"(;\s*(DROP|DELETE|INSERT|UPDATE))", std::regex_constants::icase), "[SQL_REDACTED]");
    }

    if (config_.remove_shell_commands) {
        custom_patterns_.emplace_back(std::regex(R"(rm\s+-rf\s+[/\w])"), "[SHELL_REDACTED]");
        custom_patterns_.emplace_back(std::regex(R"(format\s+[cC]:)"), "[SHELL_REDACTED]");
        custom_patterns_.emplace_back(std::regex(R"(del\s+[/\\])"), "[SHELL_REDACTED]");
    }

    if (config_.remove_code_blocks) {
        custom_patterns_.emplace_back(std::regex(R"(```[\s\S]*?```)"), "[CODE_BLOCK]");
    }

    if (config_.remove_urls) {
        custom_patterns_.emplace_back(std::regex(R"(https?://[^\s]+)"), "[URL]");
    }
}

std::string Sanitizer::sanitize(const std::string& text) const
{
    std::string result = text;

    // Apply length limit
    if (result.length() > config_.max_response_length) {
        result = result.substr(0, config_.max_response_length) + "...";
    }

    // Apply custom patterns
    for (const auto& [pattern, replacement] : custom_patterns_) {
        result = std::regex_replace(result, pattern, replacement);
    }

    return result;
}

void Sanitizer::add_pattern(const std::string& pattern, const std::string& replacement)
{
    try {
        custom_patterns_.emplace_back(std::regex(pattern), replacement);
    } catch (const std::regex_error&) {
        // Invalid regex, ignore
    }
}

void Sanitizer::clear_patterns()
{
    custom_patterns_.clear();
}

}  // namespace safety
}  // namespace serius
