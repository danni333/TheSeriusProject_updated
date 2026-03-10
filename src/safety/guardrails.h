#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace serius {
namespace safety {

/// Minimal guardrails: blocklist/allowlist for response content.
class Guardrails
{
public:
    Guardrails() = default;

    /// Block a token ID (e.g. from tokenizer).
    void add_blocklist_token(int token_id);
    void remove_blocklist_token(int token_id);
    bool is_blocked_token(int token_id) const;

    /// Block a substring in decoded text (e.g. harmful phrase).
    void add_blocklist_phrase(const std::string &phrase);
    bool check_response(const std::string &text) const;

    /// Clear all blocklists.
    void clear();

private:
    std::unordered_set<int> blocklist_tokens_;
    std::vector<std::string> blocklist_phrases_;
};

}  // namespace safety
}  // namespace serius
