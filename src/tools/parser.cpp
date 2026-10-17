#include "parser.h"

#include <regex>

namespace serius {
namespace tools {

std::string FunctionCallParser::extract_json(const std::string& text) const
{
    // Look for JSON object pattern: { ... }
    std::regex json_pattern(R"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})");
    std::smatch match;

    if (std::regex_search(text, match, json_pattern)) {
        return match.str();
    }

    return "";
}

bool FunctionCallParser::has_function_call(const std::string& text) const
{
    std::string json_str = extract_json(text);
    if (json_str.empty()) {
        return false;
    }

    try {
        json j = json::parse(json_str);
        return j.contains("function") && j.contains("arguments");
    } catch (...) {
        return false;
    }
}

std::optional<FunctionCall> FunctionCallParser::parse(const std::string& text) const
{
    std::string json_str = extract_json(text);
    if (json_str.empty()) {
        return std::nullopt;
    }

    try {
        json j = json::parse(json_str);

        if (j.contains("function") && j.contains("arguments")) {
            FunctionCall call;
            call.name = j["function"];
            call.arguments = j["arguments"];
            return call;
        }
    } catch (...) {
        // Invalid JSON, return nullopt
    }

    return std::nullopt;
}

}  // namespace tools
}  // namespace serius
