#pragma once

#include <regex>
#include <string>
#include <vector>

namespace serius {
namespace mcp {

/// Scrubber removes sensitive tokens (API keys, passwords, secrets) from text.
class Scrubber
{
   public:
    Scrubber();

    /// Replace sensitive patterns with [REDACTED].
    std::string scrub(const std::string &text) const;

    /// Add a custom regex pattern to scrub.
    void add_pattern(const std::string &pattern);

   private:
    std::vector<std::string> keywords_;
    std::vector<std::regex> complex_;
};

}  // namespace mcp
}  // namespace serius
