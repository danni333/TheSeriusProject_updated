#include "guardrails.h"

#include <algorithm>
#include <cctype>

namespace serius {
namespace safety {

void Guardrails::add_blocklist_token(int token_id)
{
    blocklist_tokens_.insert(token_id);
}

void Guardrails::remove_blocklist_token(int token_id)
{
    blocklist_tokens_.erase(token_id);
}

bool Guardrails::is_blocked_token(int token_id) const
{
    return blocklist_tokens_.count(token_id) != 0;
}

void Guardrails::add_blocklist_phrase(const std::string &phrase)
{
    if (!phrase.empty()) {
        blocklist_phrases_.push_back(phrase);
    }
}

bool Guardrails::check_response(const std::string &text) const
{
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    for (const std::string &phrase : blocklist_phrases_) {
        std::string pl;
        pl.reserve(phrase.size());
        for (char c : phrase) {
            pl.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (lower.find(pl) != std::string::npos) {
            return false;
        }
    }
    return true;
}

void Guardrails::clear()
{
    blocklist_tokens_.clear();
    blocklist_phrases_.clear();
}

}  // namespace safety
}  // namespace serius
