#include "citation_engine.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

namespace serius {
namespace grounding {

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
    std::istringstream iss(claim);
    std::string word;
    int total = 0;
    int matched = 0;
    while (iss >> word) {
        for (auto &c : word) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        ++total;
        if (document.find(word) != std::string::npos) {
            ++matched;
        }
    }
    if (total == 0) {
        return 0.0f;
    }
    return static_cast<float>(matched) / static_cast<float>(total);
}

}  // namespace grounding
}  // namespace serius
