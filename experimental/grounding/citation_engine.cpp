#include "citation_engine.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace serius {
namespace grounding {

namespace {

const std::unordered_set<std::string> &stopwords()
{
    static const std::unordered_set<std::string> words = {
        "il",   "lo",   "la",   "i",    "gli",  "le",   "di",   "del",  "della",
        "dei",  "delle", "un",  "una",  "uno",  "e",    "ed",   "o",    "ma",
        "che",  "chi",  "come", "non",  "si",   "per",  "con",  "su",   "in",
        "a",    "da",   "al",   "allo", "ai",   "agli", "alle", "nel",  "nello",
        "the",  "a",    "an",   "and",  "or",   "of",   "to",   "in",   "is",
        "are",  "was",  "were", "for",  "on",   "with", "as",   "by",   "at",
        "it",   "its",  "this", "that", "these", "those", "be",  "been", "there",
    };
    return words;
}

/// Tokenize into lowercase alphanumeric words, dropping stopwords and
/// single-character tokens.
std::vector<std::string> content_words(const std::string &text)
{
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        std::string clean;
        clean.reserve(word.size());
        for (auto c : word) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                clean.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
        }
        if (clean.size() <= 1 || stopwords().count(clean) > 0) {
            continue;
        }
        words.push_back(clean);
    }
    return words;
}

/// Two content words match if identical or if one is a prefix of the other
/// with at least 4 shared characters — covers common morphology
/// ("mammals"/"mammal", "running"/"run") without needing a stemmer.
bool words_match(const std::string &a, const std::string &b)
{
    if (a == b) {
        return true;
    }
    const size_t min_len = std::min(a.size(), b.size());
    if (min_len < 4) {
        return false;
    }
    return a.compare(0, min_len, b, 0, min_len) == 0;
}

}  // namespace

std::vector<Citation> extract_citations(const std::string &text)
{
    std::vector<Citation> citations;
    std::regex re(R"(\[(\d+)\])");
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
        std::smatch match = *it;
        int doc_id = std::stoi(match.str(1));
        citations.push_back({doc_id, match.str()});
    }
    return citations;
}

float support_score(const std::string &claim, const std::string &document)
{
    return SupportScorer().score(claim, document);
}

SupportScorer::SupportScorer() = default;

SupportScorer::SupportScorer(common::TextEmbedFn embedder)
    : embedder_(std::move(embedder))
{
}

void SupportScorer::set_embedder(common::TextEmbedFn embedder)
{
    embedder_ = std::move(embedder);
    emb_cache_.clear();
}

const std::vector<float> &SupportScorer::embed_cached(const std::string &text)
{
    auto it = emb_cache_.find(text);
    if (it != emb_cache_.end()) {
        return it->second;
    }
    auto [ins, _] = emb_cache_.emplace(text, embedder_(text));
    return ins->second;
}

float SupportScorer::lexical_score(const std::string &claim, const std::string &document) const
{
    const auto claim_words = content_words(claim);
    if (claim_words.empty()) {
        return 0.0f;
    }
    const auto doc_words = content_words(document);
    if (doc_words.empty()) {
        return 0.0f;
    }
    size_t matched = 0;
    for (const auto &w : claim_words) {
        for (const auto &d : doc_words) {
            if (words_match(w, d)) {
                ++matched;
                break;
            }
        }
    }
    return static_cast<float>(matched) / static_cast<float>(claim_words.size());
}

float SupportScorer::semantic_score(const std::string &claim, const std::string &document)
{
    const auto &a = embed_cached(claim);
    const auto &b = embed_cached(document);
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }
    return common::cosine_unit(common::cosine_sim(a, b), cosine_lo_, cosine_hi_);
}

float SupportScorer::score(const std::string &claim, const std::string &document)
{
    const float lexical = lexical_score(claim, document);
    if (!embedder_ || semantic_weight_ <= 0.0f) {
        return lexical;
    }
    const float semantic = semantic_score(claim, document);
    return (1.0f - semantic_weight_) * lexical + semantic_weight_ * semantic;
}

std::vector<float> SupportScorer::score_claim_against_documents(
    const std::string &claim, const std::vector<std::string> &documents)
{
    std::vector<float> out;
    out.reserve(documents.size());
    // embed_cached() memoizes, so the claim embedding is computed once.
    for (const auto &doc : documents) {
        out.push_back(score(claim, doc));
    }
    return out;
}

}  // namespace grounding
}  // namespace serius
