#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>

#include "io/safetensors.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::core;
using namespace serius::io;

namespace {
const std::string TEST_FILE = "test_output.safetensors";

void cleanup()
{
    std::filesystem::remove(TEST_FILE);
}
}  // namespace

class SafetensorsTest : public ::testing::Test
{
   protected:
    void TearDown() override { cleanup(); }
};

TEST_F(SafetensorsTest, WriteAndReadSingleTensor)
{
    // Write
    Tensor original({3, 4});
    for (size_t i = 0; i < original.size(); i++) {
        original.data()[i] = static_cast<float>(i) * 0.5f;
    }

    SafetensorsWriter writer;
    writer.add_tensor("weight", original);
    writer.save(TEST_FILE);

    // Read
    SafetensorsReader reader(TEST_FILE);
    ASSERT_EQ(reader.num_tensors(), 1u);
    ASSERT_TRUE(reader.has_tensor("weight"));
    ASSERT_FALSE(reader.has_tensor("nonexistent"));

    auto names = reader.tensor_names();
    ASSERT_EQ(names.size(), 1u);

    const auto &info = reader.tensor_info("weight");
    EXPECT_EQ(info.dtype, "F32");
    EXPECT_EQ(info.shape.size(), 2u);
    EXPECT_EQ(info.shape[0], 3u);
    EXPECT_EQ(info.shape[1], 4u);

    Tensor loaded = reader.get_tensor("weight");
    ASSERT_EQ(loaded.shape(), original.shape());
    for (size_t i = 0; i < original.size(); i++) {
        EXPECT_FLOAT_EQ(loaded.data()[i], original.data()[i]);
    }
}

TEST_F(SafetensorsTest, WriteAndReadMultipleTensors)
{
    Tensor w1({2, 3});
    Tensor w2({4, 5});
    w1.fill(1.0f);
    w2.fill(2.0f);

    SafetensorsWriter writer;
    writer.add_tensor("layer1.weight", w1);
    writer.add_tensor("layer2.weight", w2);
    writer.save(TEST_FILE);

    SafetensorsReader reader(TEST_FILE);
    ASSERT_EQ(reader.num_tensors(), 2u);

    Tensor r1 = reader.get_tensor("layer1.weight");
    Tensor r2 = reader.get_tensor("layer2.weight");

    EXPECT_EQ(r1.shape()[0], 2u);
    EXPECT_EQ(r1.shape()[1], 3u);
    EXPECT_FLOAT_EQ(r1.data()[0], 1.0f);

    EXPECT_EQ(r2.shape()[0], 4u);
    EXPECT_EQ(r2.shape()[1], 5u);
    EXPECT_FLOAT_EQ(r2.data()[0], 2.0f);
}

TEST_F(SafetensorsTest, Metadata)
{
    Tensor t({2});
    t.fill(42.0f);

    SafetensorsWriter writer;
    writer.add_tensor("data", t);
    writer.add_metadata("format", "serius");
    writer.add_metadata("version", "1.0");
    writer.save(TEST_FILE);

    SafetensorsReader reader(TEST_FILE);
    auto &meta = reader.metadata();
    EXPECT_EQ(meta.at("format"), "serius");
    EXPECT_EQ(meta.at("version"), "1.0");
}

TEST_F(SafetensorsTest, RoundTripPreservesData)
{
    // Create tensor with specific pattern
    Tensor original({10, 20});
    for (size_t i = 0; i < original.size(); i++) {
        original.data()[i] = static_cast<float>(i) * 0.123f - 50.0f;
    }

    SafetensorsWriter writer;
    writer.add_tensor("test", original);
    writer.save(TEST_FILE);

    SafetensorsReader reader(TEST_FILE);
    Tensor loaded = reader.get_tensor("test");

    // Bit-exact comparison
    ASSERT_EQ(loaded.size(), original.size());
    EXPECT_EQ(std::memcmp(loaded.data(), original.data(), original.size() * sizeof(float)), 0);
}

TEST_F(SafetensorsTest, EmptyTensorShape)
{
    Tensor t({1});
    t.data()[0] = 99.0f;

    SafetensorsWriter writer;
    writer.add_tensor("scalar", t);
    writer.save(TEST_FILE);

    SafetensorsReader reader(TEST_FILE);
    Tensor loaded = reader.get_tensor("scalar");
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_FLOAT_EQ(loaded.data()[0], 99.0f);
}

TEST_F(SafetensorsTest, LargerTensor)
{
    Tensor t({128, 256});
    t.rand();

    SafetensorsWriter writer;
    writer.add_tensor("big", t);
    writer.save(TEST_FILE);

    SafetensorsReader reader(TEST_FILE);
    Tensor loaded = reader.get_tensor("big");
    ASSERT_EQ(loaded.size(), t.size());
    EXPECT_EQ(std::memcmp(loaded.data(), t.data(), t.size() * sizeof(float)), 0);
}
