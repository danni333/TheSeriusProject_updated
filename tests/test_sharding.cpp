#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "io/safetensors_writer.h"
#include "io/sharding.h"
#include "tensor/tensor.h"

namespace fs = std::filesystem;

TEST(ShardingTest, WriteAndReadMultiShardIndex)
{
    const auto root = fs::temp_directory_path() / ("serius_sharding_" + std::to_string(std::rand()));
    fs::remove_all(root);
    fs::create_directories(root);

    serius::core::Tensor weight_a({2, 3});
    serius::core::Tensor weight_b({3, 2});
    for (size_t i = 0; i < weight_a.size(); ++i) {
        weight_a.data()[i] = static_cast<float>(i + 1) * 0.5f;
    }
    for (size_t i = 0; i < weight_b.size(); ++i) {
        weight_b.data()[i] = static_cast<float>(i + 10) * 1.25f;
    }

    std::vector<std::pair<std::string, serius::core::Tensor>> tensors;
    tensors.emplace_back("layer1.weight", weight_a);
    tensors.emplace_back("layer2.weight", weight_b);

    ASSERT_TRUE(serius::io::write_sharded_safetensors(root.string(), "model", tensors, 64));
    const auto index_path = root / "model.safetensors.index.json";
    ASSERT_TRUE(fs::exists(index_path));

    std::map<std::string, serius::core::Tensor> loaded;
    ASSERT_TRUE(serius::io::load_sharded_safetensors(index_path.string(), &loaded));
    ASSERT_EQ(loaded.size(), 2u);

    auto &first = loaded.at("layer1.weight");
    ASSERT_EQ(first.shape().size(), 2u);
    ASSERT_EQ(first.shape()[0], 2u);
    ASSERT_EQ(first.shape()[1], 3u);
    for (size_t i = 0; i < weight_a.size(); ++i) {
        EXPECT_FLOAT_EQ(first.data()[i], weight_a.data()[i]);
    }

    auto &second = loaded.at("layer2.weight");
    ASSERT_EQ(second.shape().size(), 2u);
    ASSERT_EQ(second.shape()[0], 3u);
    ASSERT_EQ(second.shape()[1], 2u);
    for (size_t i = 0; i < weight_b.size(); ++i) {
        EXPECT_FLOAT_EQ(second.data()[i], weight_b.data()[i]);
    }

    fs::remove_all(root);
}
