#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/text_embed.h"

namespace serius {
namespace safety {

enum class AttackCategory { Jailbreak, PromptInjection, DataExtraction, MaliciousCode };

struct AttackResult
{
    std::string prompt;
    std::string response;
    AttackCategory category;
    float severity;  // 0-1
    bool success;    // true = attack succeeded (unsafe)
};

/// Toxicity scorer (0 = safe, 1 = toxic).
///
/// Without an embedder, falls back to keyword matching. When an embedder is
/// installed (via constructor or set_embedder — any callable with signature
/// `std::vector<float>(const std::string&)`, e.g. a lambda wrapping
/// `hub::NomicBertEmbedder::embed` or `hub::EmbeddingManager`), the score
/// blends the lexical signal with a prototype-based semantic signal: max
/// cosine similarity to toxic prototypes minus max cosine similarity to
/// benign prototypes, mapped to [0, 1]. Prototype embeddings are cached, so
/// repeated score() calls only embed the input text.
///
/// If one prototype side is empty, its similarity is replaced by
/// `cosine_lo` (the "neutral" cosine) so scores stay calibrated.
class ToxicityClassifier
{
   public:
    ToxicityClassifier();
    explicit ToxicityClassifier(common::TextEmbedFn embedder);

    void set_embedder(common::TextEmbedFn embedder);

    /// Replace the default toxic/benign prototype texts used by the
    /// semantic path. Pass an empty vector to disable that side.
    void set_prototypes(std::vector<std::string> toxic, std::vector<std::string> benign);

    /// Weight of the semantic signal in [0, 1] (default 0.5).
    void set_semantic_weight(float w) { semantic_weight_ = std::clamp(w, 0.0f, 1.0f); }

    /// Cosine range considered "unrelated".."fully matching" for prototype
    /// similarity (default 0.25..0.9). Requires lo < hi.
    void set_cosine_range(float lo, float hi)
    {
        if (hi > lo) {
            cosine_lo_ = lo;
            cosine_hi_ = hi;
        }
    }

    float score(const std::string &text) const;

   private:
    ToxicityClassifier(common::TextEmbedFn embedder, std::vector<std::string> toxic, std::vector<std::string> benign);

    float lexical_score(const std::string &text) const;
    float semantic_score(const std::string &text) const;
    float max_prototype_similarity(const std::vector<float> &emb, const std::vector<std::string> &prototypes) const;
    const std::vector<float> &prototype_embedding(const std::string &text) const;

    common::TextEmbedFn embedder_;
    std::vector<std::string> toxic_prototypes_;
    std::vector<std::string> benign_prototypes_;
    float semantic_weight_ = 0.5f;
    float cosine_lo_ = 0.25f;
    float cosine_hi_ = 0.9f;
    mutable std::unordered_map<std::string, std::vector<float>> proto_cache_;
};

/// Target model interface for red teaming.
class RedTeamTarget
{
   public:
    virtual ~RedTeamTarget() = default;
    virtual std::string generate(const std::string &prompt, float temperature = 0.3f) = 0;
};

/// Adversarial testing suite (red teaming).
class RedTeamSuite
{
   public:
    RedTeamSuite(RedTeamTarget &target, ToxicityClassifier &toxicity);

    std::vector<AttackResult> run_all();

   private:
    RedTeamTarget &target_;
    ToxicityClassifier &toxicity_;

    AttackResult test_one(const std::string &prompt, AttackCategory category);

    static std::vector<std::string> jailbreak_templates();
    static std::vector<std::string> injection_templates();
    static std::vector<std::string> extraction_templates();
    static std::vector<std::string> malicious_code_templates();
};

}  // namespace safety
}  // namespace serius
