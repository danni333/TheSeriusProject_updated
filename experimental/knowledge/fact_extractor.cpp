#include "fact_extractor.h"

#include <algorithm>
#include <cctype>

namespace serius {
namespace knowledge {

std::vector<std::string> FactExtractor::split_sentences(const std::string &text) const
{
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        // Sentence terminators
        if (c == '.' || c == '!' || c == '?') {
            // Check if next char is space or end of string
            if (i + 1 >= text.size() || std::isspace(text[i + 1])) {
                std::string sentence = clean_fact(current);
                if (!sentence.empty()) {
                    sentences.push_back(sentence);
                }
                current.clear();
            }
        }
    }

    // Add remaining text
    if (!current.empty()) {
        std::string sentence = clean_fact(current);
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }

    return sentences;
}

bool FactExtractor::is_definition(const std::string &sentence) const
{
    std::string lower = sentence;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Common definition patterns
    std::vector<std::string> patterns = {
        " is ",
        " are ",
        " refers to ",
        " defined as ",
        " means ",
        " can be described as "
    };

    for (const auto &pattern : patterns) {
        if (lower.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
}

std::string FactExtractor::clean_fact(const std::string &text) const
{
    std::string cleaned;

    // Trim leading/trailing whitespace
    size_t start = 0;
    while (start < text.size() && std::isspace(text[start])) {
        start++;
    }

    size_t end = text.size();
    while (end > start && std::isspace(text[end - 1])) {
        end--;
    }

    cleaned = text.substr(start, end - start);

    // Remove multiple spaces
    std::string normalized;
    bool prev_space = false;
    for (char c : cleaned) {
        if (std::isspace(c)) {
            if (!prev_space) {
                normalized += ' ';
                prev_space = true;
            }
        } else {
            normalized += c;
            prev_space = false;
        }
    }

    return normalized;
}

bool FactExtractor::is_valid_fact(const std::string &fact) const
{
    if (fact.empty()) {
        return false;
    }

    size_t len = fact.size();
    if (len < min_length_ || len > max_length_) {
        return false;
    }

    // Check for minimum alphabetic content
    size_t alpha_count = 0;
    for (char c : fact) {
        if (std::isalpha(c)) {
            alpha_count++;
        }
    }

    if (alpha_count < min_length_ / 2) {
        return false;
    }

    return true;
}

std::vector<Fact> FactExtractor::extract_facts(const std::string &text, const std::string &source)
{
    std::vector<Fact> facts;

    if (text.empty()) {
        return facts;
    }

    if (extract_sentences_) {
        std::vector<std::string> sentences = split_sentences(text);

        for (const auto &sentence : sentences) {
            if (!is_valid_fact(sentence)) {
                continue;
            }

            // If definition extraction is enabled, prioritize definitions
            float importance = 1.0f;
            if (extract_definitions_ && is_definition(sentence)) {
                importance = 1.5f;  // Boost importance of definitions
            }

            Fact fact;
            fact.text = sentence;
            fact.source = source;
            fact.importance = importance;

            facts.push_back(fact);
        }
    }

    return facts;
}

}  // namespace knowledge
}  // namespace serius
