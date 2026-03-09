#include <gtest/gtest.h>

#include <filesystem>

#include "tokenizer/bpe.h"

namespace serius {
namespace test {

TEST(TokenizerTest, BasicInit)
{
    BPETokenizer tokenizer;
    EXPECT_EQ(tokenizer.vocab_size(), 256 + 4);
    EXPECT_EQ(tokenizer.get_token(BPETokenizer::PAD_ID), "<PAD>");
}

TEST(TokenizerTest, TrainAndEncode)
{
    BPETokenizer tokenizer;
    std::string text = "abababacacac";
    tokenizer.train(text, 265);  // Add some merges

    auto ids = tokenizer.encode("abac");
    EXPECT_LT(ids.size(), 4);  // Should have merged some

    std::string decoded = tokenizer.decode(ids);
    EXPECT_EQ(decoded, "abac");
}

TEST(TokenizerTest, RoundTrip)
{
    BPETokenizer tokenizer;
    std::string text = "the quick brown fox jumps over the lazy dog";
    tokenizer.train(text, 300);

    std::string test_str = "the fox is quick";
    auto ids = tokenizer.encode(test_str);
    EXPECT_EQ(tokenizer.decode(ids), test_str);
}

TEST(TokenizerTest, SpecialTokens)
{
    BPETokenizer tokenizer;
    EXPECT_EQ(tokenizer.get_id("<PAD>"), BPETokenizer::PAD_ID);
    EXPECT_EQ(tokenizer.get_id("<BOS>"), BPETokenizer::BOS_ID);
    EXPECT_EQ(tokenizer.get_id("<EOS>"), BPETokenizer::EOS_ID);
}

TEST(TokenizerTest, SaveLoad)
{
    BPETokenizer tokenizer;
    tokenizer.train("hello world hello", 265);

    std::string path = "test_vocab.json";
    tokenizer.save(path);

    BPETokenizer tokenizer2;
    tokenizer2.load(path);

    EXPECT_EQ(tokenizer.vocab_size(), tokenizer2.vocab_size());
    EXPECT_EQ(tokenizer.encode("hello"), tokenizer2.encode("hello"));

    std::filesystem::remove(path);
}
}  // namespace test
}  // namespace serius
