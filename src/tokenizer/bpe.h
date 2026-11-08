#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vocab.h"

namespace serius {

namespace io { struct GGUFTokenizerData; }  // forward declare

class BPETokenizer
{
   public:
    BPETokenizer();
    ~BPETokenizer() = default;

    // Special token IDs
    static constexpr TokenId PAD_ID = 0;
    static constexpr TokenId BOS_ID = 1;
    static constexpr TokenId EOS_ID = 2;
    static constexpr TokenId UNK_ID = 3;

    void train(const std::string &text, size_t vocab_size);

    std::vector<TokenId> encode(const std::string &text) const;
    std::string decode(const std::vector<TokenId> &ids) const;

    TokenId get_id(const std::string &token) const;
    std::string get_token(TokenId id) const;

    size_t vocab_size() const { return vocab_.size(); }

    bool is_gpt2_byte_level() const { return is_gpt2_byte_level_; }
    const std::vector<std::string> &special_tokens() const { return special_tokens_; }

    // Logic to save/load (optional but recommended for later)
    void save(const std::string &path) const;
    void load(const std::string &path);

    /// Load vocabulary and merge rules from a GGUFTokenizerData struct.
    /// Supports both BPE (gpt2-style) and SentencePiece-style GGUF tokenizers.
    void load_from_gguf(const io::GGUFTokenizerData &data);

    void init_basic_vocab();

   private:
    Vocab vocab_;
    std::vector<std::string> special_tokens_;
    bool is_gpt2_byte_level_ = false;
    bool is_sentencepiece_ = false;

    std::vector<TokenId> merge_tokens(std::vector<TokenId> ids) const;
};

}  // namespace serius
