#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>


namespace serius {

using TokenId = int32_t;

struct Vocab
{
    std::unordered_map<std::string, TokenId> token_to_id;
    std::unordered_map<TokenId, std::string> id_to_token;

    // Merge rules: pair of token IDs -> merged token ID
    std::map<std::pair<TokenId, TokenId>, TokenId> merges;

    void add_token(const std::string &token, TokenId id)
    {
        token_to_id[token] = id;
        id_to_token[id] = token;
    }

    size_t size() const { return token_to_id.size(); }
};

}  // namespace serius
