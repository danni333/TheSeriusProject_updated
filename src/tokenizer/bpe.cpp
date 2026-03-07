#include "bpe.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>


namespace serius {

BPETokenizer::BPETokenizer()
{
    init_basic_vocab();
}

void BPETokenizer::init_basic_vocab()
{
    vocab_.token_to_id.clear();
    vocab_.id_to_token.clear();
    vocab_.merges.clear();

    vocab_.add_token("<PAD>", PAD_ID);
    vocab_.add_token("<BOS>", BOS_ID);
    vocab_.add_token("<EOS>", EOS_ID);
    vocab_.add_token("<UNK>", UNK_ID);

    for (int i = 0; i < 256; ++i) {
        std::string s_token(1, static_cast<char>(i));
        vocab_.add_token(s_token, 4 + i);
    }
}

void BPETokenizer::train(const std::string &text, size_t target_vocab_size)
{
    std::vector<TokenId> corpus;
    for (unsigned char c_char : text) {
        corpus.push_back(4 + c_char);
    }

    size_t current_vocab_size = vocab_.size();
    while (current_vocab_size < target_vocab_size) {
        std::map<std::pair<TokenId, TokenId>, size_t> pair_counts;
        for (size_t i = 0; i + 1 < corpus.size(); ++i) {
            pair_counts[{corpus[i], corpus[i + 1]}]++;
        }

        if (pair_counts.empty()) {
            break;
        }

        auto best_pair_it =
            std::max_element(pair_counts.begin(), pair_counts.end(),
                             [](const auto &pair_a, const auto &pair_b) { return pair_a.second < pair_b.second; });

        if (best_pair_it->second < 1) {
            break;
        }

        std::pair<TokenId, TokenId> best_pair = best_pair_it->first;
        TokenId new_token_id = static_cast<TokenId>(current_vocab_size);

        vocab_.merges[best_pair] = new_token_id;

        std::string new_token_str = vocab_.id_to_token[best_pair.first] + vocab_.id_to_token[best_pair.second];
        vocab_.add_token(new_token_str, new_token_id);

        std::vector<TokenId> new_corpus;
        for (size_t i = 0; i < corpus.size(); ++i) {
            if (i + 1 < corpus.size() && corpus[i] == best_pair.first && corpus[i + 1] == best_pair.second) {
                new_corpus.push_back(new_token_id);
                i++;
            } else {
                new_corpus.push_back(corpus[i]);
            }
        }
        corpus = std::move(new_corpus);
        current_vocab_size++;
    }
}

std::vector<TokenId> BPETokenizer::encode(const std::string &text) const
{
    std::vector<TokenId> ids;
    for (unsigned char c_char : text) {
        ids.push_back(4 + c_char);
    }

    return merge_tokens(ids);
}

std::vector<TokenId> BPETokenizer::merge_tokens(std::vector<TokenId> ids) const
{
    if (ids.empty()) {
        return ids;
    }

    while (true) {
        std::pair<TokenId, TokenId> best_pair = {-1, -1};
        TokenId best_new_id = -1;

        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it_merge = vocab_.merges.find({ids[i], ids[i + 1]});
            if (it_merge != vocab_.merges.end()) {
                if (best_new_id == -1 || it_merge->second < best_new_id) {
                    best_new_id = it_merge->second;
                    best_pair = it_merge->first;
                }
            }
        }

        if (best_new_id == -1) {
            break;
        }

        std::vector<TokenId> new_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == best_pair.first && ids[i + 1] == best_pair.second) {
                new_ids.push_back(best_new_id);
                i++;
            } else {
                new_ids.push_back(ids[i]);
            }
        }
        ids = std::move(new_ids);
    }
    return ids;
}

std::string BPETokenizer::decode(const std::vector<TokenId> &ids) const
{
    std::string result;
    for (TokenId token_id : ids) {
        auto it_token = vocab_.id_to_token.find(token_id);
        if (it_token != vocab_.id_to_token.end()) {
            result += it_token->second;
        }
    }
    return result;
}

TokenId BPETokenizer::get_id(const std::string &token_str) const
{
    auto it_id = vocab_.token_to_id.find(token_str);
    if (it_id != vocab_.token_to_id.end()) {
        return it_id->second;
    }
    return UNK_ID;
}

std::string BPETokenizer::get_token(TokenId token_id) const
{
    auto it_token = vocab_.id_to_token.find(token_id);
    if (it_token != vocab_.id_to_token.end()) {
        return it_token->second;
    }
    return "<UNK>";
}

void BPETokenizer::save(const std::string &path_str) const
{
    nlohmann::json json_out;

    // Use an array of objects to avoid UTF-8 string keys issues
    nlohmann::json vocab_array = nlohmann::json::array();
    for (const auto &vocab_entry : vocab_.token_to_id) {
        std::vector<uint8_t> token_bytes(vocab_entry.first.begin(), vocab_entry.first.end());
        vocab_array.push_back({{"t", token_bytes}, {"i", vocab_entry.second}});
    }
    json_out["vocab"] = vocab_array;

    nlohmann::json merges_json = nlohmann::json::array();
    for (const auto &merge_entry : vocab_.merges) {
        merges_json.push_back({merge_entry.first.first, merge_entry.first.second, merge_entry.second});
    }
    json_out["merges"] = merges_json;

    std::ofstream file_out(path_str);
    file_out << json_out.dump(4);
}

void BPETokenizer::load(const std::string &path_str)
{
    std::ifstream file_in(path_str);
    if (!file_in.is_open()) {
        return;
    }

    nlohmann::json json_in;
    file_in >> json_in;

    vocab_.token_to_id.clear();
    vocab_.id_to_token.clear();

    for (const auto &vocab_item : json_in["vocab"]) {
        std::vector<uint8_t> token_bytes = vocab_item["t"].get<std::vector<uint8_t>>();
        std::string token_str(token_bytes.begin(), token_bytes.end());
        TokenId token_id = vocab_item["i"].get<TokenId>();
        vocab_.add_token(token_str, token_id);
    }

    vocab_.merges.clear();
    for (const auto &merge_item : json_in["merges"]) {
        vocab_.merges[{merge_item[0], merge_item[1]}] = merge_item[2];
    }
}

}  // namespace serius
