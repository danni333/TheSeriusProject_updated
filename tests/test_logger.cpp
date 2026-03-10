#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "../src/debug/tensor_logger.h"
#include "autograd/node.h"

using namespace serius;
using namespace serius::debug;

class TensorLoggerTest : public ::testing::Test
{
   protected:
    void SetUp() override { get_tensor_logger().clear(); }
    void TearDown() override
    {
        if (std::filesystem::exists("test_log.csv")) std::filesystem::remove("test_log.csv");
        if (std::filesystem::exists("test_log.json")) std::filesystem::remove("test_log.json");
    }
};

TEST_F(TensorLoggerTest, BasicLogging)
{
    core::Tensor t = core::Tensor::zeros({10});
    get_tensor_logger().log(t, "zeros", 0);

    auto history = get_tensor_logger().history();
    EXPECT_EQ(history.size(), 1);
    EXPECT_EQ(history[0].name, "zeros");
    EXPECT_FLOAT_EQ(history[0].mean, 0.0f);
}

TEST_F(TensorLoggerTest, ExportCSV)
{
    core::Tensor t = core::Tensor::ones({5});
    get_tensor_logger().log(t, "ones", 1);
    get_tensor_logger().export_csv("test_log.csv");

    EXPECT_TRUE(std::filesystem::exists("test_log.csv"));
    std::ifstream f("test_log.csv");
    std::string line;
    std::getline(f, line);  // header
    std::getline(f, line);  // data
    EXPECT_TRUE(line.find("ones,1,1.00000000") != std::string::npos);
}

TEST_F(TensorLoggerTest, ExportJSON)
{
    core::Tensor t = core::Tensor::ones({5});
    get_tensor_logger().log(t, "ones_json", 2);
    get_tensor_logger().export_json("test_log.json");

    EXPECT_TRUE(std::filesystem::exists("test_log.json"));
    std::ifstream f("test_log.json");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("\"name\": \"ones_json\"") != std::string::npos);
}

TEST_F(TensorLoggerTest, GradientHealth)
{
    // Mock parameters
    core::Tensor data = core::Tensor::ones({2});
    core::autograd::Node n(std::move(data), true);
    n.name = "weight";
    n.grad = core::Tensor::ones({2});  // Set some grad

    std::vector<core::autograd::Node *> params = {&n};
    get_tensor_logger().log_gradients(params, 10);

    auto history = get_tensor_logger().history();
    EXPECT_EQ(history.size(), 1);
    EXPECT_EQ(history[0].name, "weight.grad");
    EXPECT_FLOAT_EQ(history[0].mean, 1.0f);
}
