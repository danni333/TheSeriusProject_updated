#pragma once

#include "model_interface.h"

#include <string>

namespace serius {
namespace agent {

/// Parses an [ACTION] JSON block from model output, e.g.:
///   [ACTION] {"tool":"search","args":{"query":"..."}}
class ActionParser
{
   public:
    /// Extract the first [ACTION] block from text. Returns empty if none found.
    static ActionCall parse(const std::string &text);

    /// Extract the [ANSWER] block from text. Returns empty if none found.
    static std::string parse_answer(const std::string &text);

    /// Check whether text contains an [ACTION] marker.
    static bool has_action(const std::string &text);

    /// Check whether text contains an [ANSWER] marker.
    static bool has_answer(const std::string &text);
};

}  // namespace agent
}  // namespace serius
