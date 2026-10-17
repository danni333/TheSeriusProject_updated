#pragma once

#include <regex>
#include <string>
#include <vector>

namespace serius {
namespace safety {

/// Output sanitization to remove dangerous patterns from generated text.
class Sanitizer
{
public:
    struct Config {
        size_t max_response_length = 10000;
        bool remove_code_blocks = true;
        bool remove_sql_injection = true;
        bool remove_shell_commands = true;
        bool remove_urls = false;
    };

    explicit Sanitizer(const Config& config = Config{});
    ~Sanitizer() = default;

    /// Sanitize the output text.
    /// @param text Input text to sanitize
    /// @return Sanitized text with dangerous patterns removed or redacted
    std::string sanitize(const std::string& text) const;

    /// Add a custom regex pattern to redact.
    /// @param pattern Regex pattern to match
    /// @param replacement Text to replace matches with (default: "[REDACTED]")
    void add_pattern(const std::string& pattern, const std::string& replacement = "[REDACTED]");

    /// Clear all custom patterns.
    void clear_patterns();

private:
    Config config_;
    std::vector<std::pair<std::regex, std::string>> custom_patterns_;
};

}  // namespace safety
}  // namespace serius
