#pragma once

#include <string>
#include <vector>

#include "mcts.h"

namespace serius {
namespace reasoning {

/// Simple keyword-based verifier: higher score if the answer contains expected
/// keywords and no obvious failure markers.
class KeywordVerifier : public Verifier
{
   public:
    KeywordVerifier(std::string expected_keyword,
                    std::vector<std::string> failure_keywords = {});

    float score(const std::string &reasoning) override;

   private:
    std::string expected_keyword_;
    std::vector<std::string> failure_keywords_;
};

}  // namespace reasoning
}  // namespace serius
