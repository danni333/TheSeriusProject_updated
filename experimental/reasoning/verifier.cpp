#include "verifier.h"

#include <cctype>
#include <string>
#include <unordered_set>

namespace serius {
namespace reasoning {

KeywordVerifier::KeywordVerifier(std::string expected_keyword,
                                 std::vector<std::string> failure_keywords)
    : expected_keyword_(std::move(expected_keyword)),
      failure_keywords_(std::move(failure_keywords))
{
}

KeywordVerifier::KeywordVerifier(std::string expected_keyword)
    : KeywordVerifier(std::move(expected_keyword), std::vector<std::string>{})
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

EmbeddingVerifier::EmbeddingVerifier(
    common::TextEmbedFn embedder,
    std::string reference,
    std::vector<std::string> failure_keywords)
    : embedder_(std::move(embedder)),
      reference_(std::move(reference)),
      failure_keywords_(std::move(failure_keywords))
{
    if (embedder_) {
        reference_emb_ = embedder_(reference_);
    }
}

namespace {

std::vector<std::string> words_of(const std::string &text)
{
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        } else if (!cur.empty()) {
            words.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) {
        words.push_back(cur);
    }
    return words;
}

}  // namespace

float EmbeddingVerifier::lexical_fallback_score(const std::string &reasoning) const
{
    const auto ref_words = words_of(reference_);
    if (ref_words.empty()) {
        return 0.0f;
    }
    std::unordered_set<std::string> rs(ref_words.begin(), ref_words.end());
    const auto rwords = words_of(reasoning);
    std::unordered_set<std::string> rw(rwords.begin(), rwords.end());
    size_t matched = 0;
    for (const auto &w : rw) {
        if (rs.count(w) > 0) {
            ++matched;
        }
    }
    return static_cast<float>(matched) / static_cast<float>(rs.size());
}

float EmbeddingVerifier::score(const std::string &reasoning)
{
    float score;
    if (embedder_ && !reference_emb_.empty()) {
        const auto emb = embedder_(reasoning);
        score = (emb.empty() || emb.size() != reference_emb_.size())
                    ? 0.0f
                    : common::cosine_unit(common::cosine_sim(emb, reference_emb_),
                                          cosine_lo_, cosine_hi_);
    } else {
        score = lexical_fallback_score(reasoning);
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
