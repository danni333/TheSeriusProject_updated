#pragma once

#include <map>
#include <string>
#include <vector>

namespace serius {
namespace safety {

struct BiasReport
{
    std::map<std::string, float> group_sentiment;   // group -> average sentiment
    float max_disparity = 0.0f;                     // max difference between groups
    bool passes_threshold = false;                  // disparity < 0.1
};

/// Sentiment scorer: simple heuristic returning a value in [0, 1].
class SentimentScorer
{
   public:
    float score(const std::string &text) const;
};

/// Language model interface for bias analysis.
class BiasTarget
{
   public:
    virtual ~BiasTarget() = default;
    virtual std::string generate(const std::string &prompt, float temperature = 0.1f) = 0;
};

/// Statistical bias analysis across demographic groups.
class BiasScorer
{
   public:
    BiasScorer(BiasTarget &target, SentimentScorer &sentiment);

    /// Analyze responses across groups for given prompt templates.
    BiasReport analyze(const std::vector<std::string> &prompts,
                       const std::vector<std::string> &groups);

   private:
    BiasTarget &target_;
    SentimentScorer &sentiment_;
};

}  // namespace safety
}  // namespace serius
