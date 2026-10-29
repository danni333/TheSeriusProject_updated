#include "synthetic.h"

#include <cctype>
#include <algorithm>

namespace serius {
namespace data {

float HeuristicJudge::score(const std::string &instruction,
                            const std::string &output)
{
    (void)instruction;
    if (output.empty()) {
        return 0.0f;
    }
    float score = 0.0f;
    // Length component: longer outputs (up to a point) score higher.
    size_t len = output.size();
    score += std::min(static_cast<float>(len) / 200.0f, 0.5f);
    // Ends with sentence punctuation.
    if (!output.empty() && (output.back() == '.' || output.back() == '!' ||
                            output.back() == '?')) {
        score += 0.2f;
    }
    // Contains at least one space (multi-word).
    if (output.find(' ') != std::string::npos) {
        score += 0.2f;
    }
    // Not all the same character (trivial repetition).
    bool all_same = true;
    char first = output[0];
    for (char c : output) {
        if (c != first) {
            all_same = false;
            break;
        }
    }
    if (!all_same) {
        score += 0.1f;
    }
    return std::min(score, 1.0f);
}

SyntheticGenerator::SyntheticGenerator(InstructionGen gen_instruction,
                                       OutputGen gen_output,
                                       QualityJudge &judge,
                                       float quality_threshold)
    : gen_instruction_(std::move(gen_instruction)),
      gen_output_(std::move(gen_output)),
      judge_(judge),
      quality_threshold_(quality_threshold)
{
}

std::vector<SyntheticSample> SyntheticGenerator::generate(int n_samples)
{
    std::vector<SyntheticSample> samples;
    samples.reserve(static_cast<size_t>(n_samples));
    int accepted = 0;
    int attempts = 0;
    const int max_attempts = n_samples * 10 + 100;  // avoid infinite loop

    while (accepted < n_samples && attempts < max_attempts) {
        ++attempts;
        std::string instruction = gen_instruction_();
        std::string output = gen_output_(instruction);
        float quality = judge_.score(instruction, output);
        if (quality >= quality_threshold_) {
            samples.push_back({instruction, output, quality});
            ++accepted;
        }
    }
    return samples;
}

}  // namespace data
}  // namespace serius
