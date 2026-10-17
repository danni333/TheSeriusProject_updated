#pragma once

#include <string>
#include <vector>

namespace serius {
namespace grounding {

/// A citation token like [1], [2] and the document it references.
struct Citation
{
    int doc_id;
    std::string token;
};

/// Extract citations like [1], [2] from text, in order of appearance.
std::vector<Citation> extract_citations(const std::string &text);

/// Simple keyword-overlap support score of a claim against a document (0..1).
float support_score(const std::string &claim, const std::string &document);

}  // namespace grounding
}  // namespace serius
