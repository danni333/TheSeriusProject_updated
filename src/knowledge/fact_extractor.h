#pragma once

#include <string>
#include <vector>

#include "knowledge_fn.h"

namespace serius {
namespace knowledge {

/// Extract atomic facts from documents for knowledge indexing.
/// Uses simple heuristics and NLP patterns to identify factual statements.
class FactExtractor
{
   public:
    FactExtractor() = default;

    /// Extract facts from a document text.
    /// @param text Document text
    /// @param source Source identifier (e.g., filename, URL)
    /// @return List of extracted facts
    std::vector<Fact> extract_facts(const std::string &text, const std::string &source = "unknown");

    /// Set minimum fact length (in characters).
    void set_min_length(size_t min_len) { min_length_ = min_len; }

    /// Set maximum fact length (in characters).
    void set_max_length(size_t max_len) { max_length_ = max_len; }

    /// Enable/disable sentence-based extraction.
    void set_extract_sentences(bool enable) { extract_sentences_ = enable; }

    /// Enable/disable extraction of definitions (X is Y patterns).
    void set_extract_definitions(bool enable) { extract_definitions_ = enable; }

   private:
    size_t min_length_ = 10;
    size_t max_length_ = 500;
    bool extract_sentences_ = true;
    bool extract_definitions_ = true;

    /// Split text into sentences.
    std::vector<std::string> split_sentences(const std::string &text) const;

    /// Check if a sentence is a definition (X is Y, X refers to Y, etc.).
    bool is_definition(const std::string &sentence) const;

    /// Clean and normalize a fact string.
    std::string clean_fact(const std::string &text) const;

    /// Check if a fact is valid (not too short, not garbage).
    bool is_valid_fact(const std::string &fact) const;
};

}  // namespace knowledge
}  // namespace serius
