#include "knowledge_fn.h"

#include <numeric>

namespace serius {
namespace knowledge {

KnowledgeFunction::KnowledgeFunction(size_t embedding_dim, size_t hash_bits, uint32_t seed)
    : hasher_(embedding_dim, hash_bits, seed), embedding_dim_(embedding_dim)
{
}

void KnowledgeFunction::build(const std::vector<Fact> &facts, const std::vector<core::Tensor> &embeddings)
{
    if (facts.size() != embeddings.size()) {
        return;
    }

    for (size_t i = 0; i < facts.size(); ++i) {
        const Fact &fact = facts[i];
        const core::Tensor &emb = embeddings[i];

        uint64_t h = hasher_.hash(emb);

        CompressedKnowledge new_entry;
        new_entry.latent = emb;
        new_entry.confidence = fact.importance;
        new_entry.sources.push_back(fact.source);
        new_entry.fact_count = 1;

        auto it = table_.find(h);
        if (it != table_.end()) {
            // Merge with existing entry
            it->second = merge(it->second, new_entry);
        } else {
            table_[h] = new_entry;
        }
    }
}

const CompressedKnowledge *KnowledgeFunction::query(const core::Tensor &question_embedding) const
{
    uint64_t h = hasher_.hash(question_embedding);
    auto it = table_.find(h);
    if (it != table_.end()) {
        return &(it->second);
    }
    return nullptr;
}

const CompressedKnowledge *KnowledgeFunction::query_with_fallback(const core::Tensor &question_embedding,
                                                                   size_t fallback_levels) const
{
    // Try full resolution first
    const CompressedKnowledge *result = query(question_embedding);
    if (result) {
        return result;
    }

    // Fallback: try with smaller hash bits (lower resolution)
    // We simulate this by masking off higher bits
    uint64_t h = hasher_.hash(question_embedding);
    size_t current_bits = hasher_.hash_bits();

    for (size_t level = 1; level <= fallback_levels; ++level) {
        size_t reduced_bits = current_bits - level;
        if (reduced_bits == 0) break;

        // Mask to keep only lower bits
        uint64_t mask = (1ULL << reduced_bits) - 1;
        uint64_t reduced_hash = h & mask;

        // Try to find a match in the table with this reduced hash
        for (const auto &entry : table_) {
            uint64_t table_hash = entry.first & mask;
            if (table_hash == reduced_hash) {
                return &(entry.second);
            }
        }
    }

    return nullptr;
}

CompressedKnowledge KnowledgeFunction::merge(const CompressedKnowledge &a, const CompressedKnowledge &b)
{
    CompressedKnowledge merged;

    // Merge latents by averaging
    merged.latent = merge_latents(a.latent, b.latent);

    // Average confidence weighted by fact count
    float total_weight = static_cast<float>(a.fact_count + b.fact_count);
    merged.confidence = (a.confidence * a.fact_count + b.confidence * b.fact_count) / total_weight;

    // Combine sources
    merged.sources = a.sources;
    for (const auto &src : b.sources) {
        bool found = false;
        for (const auto &existing : merged.sources) {
            if (existing == src) {
                found = true;
                break;
            }
        }
        if (!found) {
            merged.sources.push_back(src);
        }
    }

    // Sum fact counts
    merged.fact_count = a.fact_count + b.fact_count;

    return merged;
}

core::Tensor KnowledgeFunction::merge_latents(const core::Tensor &a, const core::Tensor &b)
{
    if (a.size() != b.size()) {
        return a;  // Cannot merge, return first
    }

    core::Tensor merged(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        merged[i] = (a[i] + b[i]) * 0.5f;
    }

    return merged;
}

KnowledgeFunction::Stats KnowledgeFunction::get_stats() const
{
    Stats stats;
    stats.total_entries = table_.size();

    size_t total_facts = 0;
    float total_confidence = 0.0f;
    size_t collisions = 0;

    for (const auto &entry : table_) {
        total_facts += entry.second.fact_count;
        total_confidence += entry.second.confidence;
        if (entry.second.fact_count > 1) {
            collisions++;
        }
    }

    stats.total_facts = total_facts;
    stats.collisions = collisions;

    if (stats.total_entries > 0) {
        stats.avg_confidence = total_confidence / static_cast<float>(stats.total_entries);
    }

    return stats;
}

}  // namespace knowledge
}  // namespace serius
