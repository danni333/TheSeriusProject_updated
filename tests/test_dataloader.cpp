#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "data/dataloader.h"

namespace serius {
namespace test {

class DataLoaderTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        // Create a predictable test file
        std::ofstream f1("test_data1.txt");
        f1 << "ABCDE";  // Byte tokens: 4+65, 4+66, 4+67, 4+68, 4+69
        f1.close();

        tokenizer = std::make_shared<BPETokenizer>();
        // No merges, just use raw bytes for predictability
    }

    void TearDown() override { std::filesystem::remove("test_data1.txt"); }

    std::shared_ptr<BPETokenizer> tokenizer;
};

TEST_F(DataLoaderTest, BasicBatch)
{
    size_t batch_size = 2;
    size_t seq_len = 8;
    DataLoader loader(tokenizer, batch_size, seq_len, 1);
    loader.load_files({"test_data1.txt"});

    auto b_opt = loader.next_batch();
    ASSERT_TRUE(b_opt.has_value());
    auto &b = *b_opt;

    EXPECT_EQ(b.input_ids.shape()[0], batch_size);
    EXPECT_EQ(b.input_ids.shape()[1], seq_len);
}

TEST_F(DataLoaderTest, TargetShiftAndPadding)
{
    size_t batch_size = 1;
    size_t seq_len = 4;  // "ABCDE" -> input "ABCD", target "BCDE"
    DataLoader loader(tokenizer, batch_size, seq_len, 1);
    loader.load_files({"test_data1.txt"});

    auto b_opt = loader.next_batch();
    ASSERT_TRUE(b_opt.has_value());
    auto &b = *b_opt;

    // Input: 'A', 'B', 'C', 'D' (IDs: 69, 70, 71, 72) - Base ID 4 + ASCII
    // Target: 'B', 'C', 'D', 'E' (IDs: 70, 71, 72, 73)

    EXPECT_EQ(b.input_ids.at(0, 0), static_cast<float>(4 + 'A'));
    EXPECT_EQ(b.input_ids.at(0, 1), static_cast<float>(4 + 'B'));
    EXPECT_EQ(b.input_ids.at(0, 2), static_cast<float>(4 + 'C'));
    EXPECT_EQ(b.input_ids.at(0, 3), static_cast<float>(4 + 'D'));

    EXPECT_EQ(b.target_ids.at(0, 0), static_cast<float>(4 + 'B'));
    EXPECT_EQ(b.target_ids.at(0, 1), static_cast<float>(4 + 'C'));
    EXPECT_EQ(b.target_ids.at(0, 2), static_cast<float>(4 + 'D'));
    EXPECT_EQ(b.target_ids.at(0, 3), static_cast<float>(4 + 'E'));

    // Check padding/mask if we had shorter sequence
    std::ofstream f2("test_short.txt");
    f2 << "AB";
    f2.close();

    DataLoader loader2(tokenizer, 1, 4, 1);
    loader2.load_files({"test_short.txt"});
    auto b2_opt = loader2.next_batch();
    auto &b2 = *b2_opt;

    // "AB" -> input ['A', 'B', PAD, PAD], target ['B', PAD, PAD, PAD]
    EXPECT_EQ(b2.input_ids.at(0, 0), static_cast<float>(4 + 'A'));
    EXPECT_EQ(b2.input_ids.at(0, 1), static_cast<float>(4 + 'B'));
    EXPECT_EQ(b2.input_ids.at(0, 2), static_cast<float>(BPETokenizer::PAD_ID));

    EXPECT_EQ(b2.padding_mask.at(0, 0), 1.0f);
    EXPECT_EQ(b2.padding_mask.at(0, 1), 1.0f);
    EXPECT_EQ(b2.padding_mask.at(0, 2), 0.0f);

    std::filesystem::remove("test_short.txt");
}

TEST_F(DataLoaderTest, Stopper)
{
    DataLoader loader(tokenizer, 1, 4, 1);
    loader.load_files({"test_data1.txt"});
    loader.stop();
}
}  // namespace test
}  // namespace serius
