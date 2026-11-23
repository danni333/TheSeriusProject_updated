#include "action_parser.h"

#include <string>

namespace serius {
namespace agent {

namespace {

std::string extract_between(const std::string &text, const std::string &start,
                            const std::string &end)
{
    size_t s = text.find(start);
    if (s == std::string::npos) {
        return {};
    }
    s += start.size();
    size_t e = text.find(end, s);
    if (e == std::string::npos) {
        return text.substr(s);
    }
    return text.substr(s, e - s);
}

}  // namespace

ActionCall ActionParser::parse(const std::string &text)
{
    ActionCall call;
    std::string block = extract_between(text, "[ACTION]", "[/ACTION]");
    if (block.empty()) {
        // Also accept [ACTION] ... up to next marker or end of line.
        block = extract_between(text, "[ACTION]", "\n");
    }
    // Minimal JSON parse: extract "tool" and "args" string values.
    size_t tool_pos = block.find("\"tool\"");
    if (tool_pos != std::string::npos) {
        size_t colon = block.find(':', tool_pos);
        size_t q1 = block.find('"', colon + 1);
        size_t q2 = block.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            call.name = block.substr(q1 + 1, q2 - q1 - 1);
        }
    }
    size_t args_pos = block.find("\"args\"");
    if (args_pos != std::string::npos) {
        size_t brace = block.find('{', args_pos);
        size_t end_brace = block.find('}', brace);
        if (brace != std::string::npos && end_brace != std::string::npos) {
            call.args_json = block.substr(brace, end_brace - brace + 1);
        }
    }
    return call;
}

std::string ActionParser::parse_answer(const std::string &text)
{
    return extract_between(text, "[ANSWER]", "[/ANSWER]");
}

bool ActionParser::has_action(const std::string &text)
{
    return text.find("[ACTION]") != std::string::npos;
}

bool ActionParser::has_answer(const std::string &text)
{
    return text.find("[ANSWER]") != std::string::npos;
}

}  // namespace agent
}  // namespace serius
