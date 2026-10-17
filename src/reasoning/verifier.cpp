#include "verifier.h"

#include <string>

namespace serius {
namespace reasoning {

KeywordVerifier::KeywordVerifier(std::string expected_keyword,
                                 std::vector<std::string> failure_keywords)
    : expected_keyword_(std::move(expected_keyword)),
      failure_keywords_(std::move(failure_keywords))
{
}

float KeywordVerifier::score(const std::string &reasoning)
{
    float score = 0.0f;
    if (reasoning.find(expected_keyword_) != std::string::npos) {
        score += 0.7f;
    }

    int failures = 0;
    for (const auto &fk : failure_keywords_) {
        if (reasoning.find(fk) != std::string::npos) {
            ++failures;
        }
    }
    score -= 0.2f * static_cast<float>(failures);

    if (score < 0.0f) {
        score = 0.0f;
    }
    if (score > 1.0f) {
        score = 1.0f;
    }
    return score;
}

}  // namespace reasoning
}  // namespace serius
