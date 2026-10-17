#pragma once

#include <optional>
#include <string>

#include "nlohmann/json.hpp"

namespace serius {
namespace tools {

using json = nlohmann::json;

/// Parsed function call from model output.
struct FunctionCall {
    std::string name;
    json arguments;

    FunctionCall() = default;
    FunctionCall(const std::string& n, const json& a) : name(n), arguments(a) {}
};

/// Parser for detecting function calls in model output.
class FunctionCallParser
{
public:
    FunctionCallParser() = default;
    ~FunctionCallParser() = default;

    /// Parse function call from text.
    /// Looks for JSON format: {"function": "...", "arguments": {...}}
    /// @param text Model output text
    /// @return Parsed function call (empty if not found)
    std::optional<FunctionCall> parse(const std::string& text) const;

    /// Check if text contains a function call.
    /// @param text Model output text
    /// @return true if function call pattern detected
    bool has_function_call(const std::string& text) const;

private:
    /// Extract JSON from text.
    /// @param text Input text
    /// @return Extracted JSON string (empty if not found)
    std::string extract_json(const std::string& text) const;
};

}  // namespace tools
}  // namespace serius
