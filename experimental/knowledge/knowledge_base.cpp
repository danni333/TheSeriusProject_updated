#include "knowledge_base.h"

#include <cmath>

#include "tensor/tensor.h"

namespace serius {
namespace knowledge {

namespace {

core::Tensor to_tensor(const std::vector<float> &v)
{
    core::Tensor t({v.size()});
    for (size_t i = 0; i < v.size(); ++i) {
        t[i] = v[i];
    }
    return t;
}

}  // namespace

KnowledgeBase::KnowledgeBase(size_t embedding_dim, size_t hash_bits, uint32_t seed)
    : kf_(embedding_dim, hash_bits, seed), embedding_dim_(embedding_dim)
{
}

std::vector<float> KnowledgeBase::embed(const std::string &text) const
{
    if (!embedder_) {
        return {};
    }
    std::vector<float> v = embedder_(text);
    if (v.size() != embedding_dim_) {
        return {};
    }
    // L2-normalize: keeps hyperplane sign decisions scale-invariant.
    double norm = 0.0;
    for (float x : v) {
        norm += static_cast<double>(x) * x;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        const float inv = static_cast<float>(1.0 / norm);
        for (float &x : v) {
            x *= inv;
        }
    }
    return v;
}

size_t KnowledgeBase::add_document(const std::string &text, const std::string &source)
{
    if (!embedder_) {
        return 0;
    }
    const auto facts = extractor_.extract_facts(text, source);
    if (facts.empty()) {
        return 0;
    }

    std::vector<Fact> stored_facts;
    std::vector<core::Tensor> embeddings;
    stored_facts.reserve(facts.size());
    embeddings.reserve(facts.size());

    for (const auto &fact : facts) {
        std::vector<float> v = embed(fact.text);
        if (v.empty()) {
            continue;  // embedder failed or wrong dimension — skip this fact
        }
        const uint64_t h = kf_.hasher().hash(to_tensor(v));
        fact_texts_[h].push_back(fact.text);
        stored_facts.push_back(fact);
        embeddings.push_back(to_tensor(v));
    }

    if (stored_facts.empty()) {
        return 0;
    }
    kf_.build(stored_facts, embeddings);
    return stored_facts.size();
}

KnowledgeBase::QueryResult KnowledgeBase::query(const std::string &question,
                                                size_t fallback_levels) const
{
    QueryResult result;
    if (!embedder_) {
        return result;
    }
    std::vector<float> v = embed(question);
    if (v.empty()) {
        return result;
    }
    const core::Tensor qemb = to_tensor(v);

    result.entry = kf_.query_with_fallback(qemb, fallback_levels);
    if (!result.entry) {
        return result;
    }
    result.found = true;

    // Fact texts are only tracked for the exact hash bucket.
    const uint64_t h = kf_.hasher().hash(qemb);
    auto it = fact_texts_.find(h);
    if (it != fact_texts_.end()) {
        result.fact_texts = it->second;
    }
    return result;
}

void KnowledgeBase::clear()
{
    kf_.clear();
    fact_texts_.clear();
}

}  // namespace knowledge
}  // namespace serius
