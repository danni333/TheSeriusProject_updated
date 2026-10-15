#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numeric>

#include "autograd/node.h"
#include "blas/blas.h"
#include "lora/lora_adapter.h"
#include "lora/lora_linear.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::core;
using namespace serius::lora;

namespace {
const std::string TEST_LORA_FILE = "test_lora.bin";

void cleanup()
{
    std::filesystem::remove(TEST_LORA_FILE);
}
}  // namespace

class LoRATest : public ::testing::Test
{
   protected:
    void TearDown() override { cleanup(); }
};

TEST_F(LoRATest, InitBZeroMeansNoChange)
{
    // B is initialized to zero, so initially LoRA delta = 0
    // Output should equal base W*x
    autograd::Node W(Tensor({4, 3}));  // d_out=4, d_in=3
    W.data.fill(1.0f);

    LoRALinear lora(W, 2);  // rank=2

    // B should be zero
    for (size_t i = 0; i < lora.B().data.size(); i++) {
        EXPECT_FLOAT_EQ(lora.B().data.data()[i], 0.0f);
    }

    // Forward should match base
    Tensor x({1, 3});
    x.fill(1.0f);

    Tensor base_out = blas::matmul(x, W.data.transpose());
    Tensor lora_out = lora.forward(x);

    ASSERT_EQ(base_out.shape(), lora_out.shape());
    for (size_t i = 0; i < base_out.size(); i++) {
        EXPECT_NEAR(base_out.data()[i], lora_out.data()[i], 1e-5f);
    }
}

TEST_F(LoRATest, LoRAChangesOutput)
{
    autograd::Node W(Tensor({4, 3}));
    W.data.fill(1.0f);

    LoRALinear lora(W, 2);

    // Set B to non-zero so LoRA actually changes output
    lora.B().data.fill(0.5f);

    Tensor x({1, 3});
    x.fill(1.0f);

    Tensor base_out = blas::matmul(x, W.data.transpose());
    Tensor lora_out = lora.forward(x);

    // Should be different from base
    bool different = false;
    for (size_t i = 0; i < base_out.size(); i++) {
        if (std::abs(base_out.data()[i] - lora_out.data()[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "LoRA with non-zero B should change output";
}

TEST_F(LoRATest, MergeAndUnmerge)
{
    autograd::Node W(Tensor({4, 3}));
    W.data.fill(1.0f);

    LoRALinear lora(W, 2, 2.0f);

    // Set A and B to known values
    lora.A().data.fill(0.1f);
    lora.B().data.fill(0.5f);

    // Get output before merge
    Tensor x({1, 3});
    x.fill(1.0f);
    Tensor before_merge = lora.forward(x);

    // Save original W
    Tensor W_original = W.data.copy();

    // Merge
    lora.merge();
    EXPECT_TRUE(lora.is_merged());

    // After merge, forward should give same result (but through W directly)
    Tensor after_merge = lora.forward(x);
    for (size_t i = 0; i < before_merge.size(); i++) {
        EXPECT_NEAR(before_merge.data()[i], after_merge.data()[i], 1e-4f);
    }

    // W should have changed
    bool w_changed = false;
    for (size_t i = 0; i < W.data.size(); i++) {
        if (std::abs(W.data.data()[i] - W_original.data()[i]) > 1e-6f) {
            w_changed = true;
            break;
        }
    }
    EXPECT_TRUE(w_changed);

    // Unmerge
    lora.unmerge();
    EXPECT_FALSE(lora.is_merged());

    // W should be back to original
    for (size_t i = 0; i < W.data.size(); i++) {
        EXPECT_NEAR(W.data.data()[i], W_original.data()[i], 1e-5f);
    }
}

TEST_F(LoRATest, Parameters)
{
    autograd::Node W(Tensor({8, 4}));
    LoRALinear lora(W, 2);

    auto params = lora.parameters();
    EXPECT_EQ(params.size(), 2u);  // A and B

    // A: (rank=2, d_in=4) = 8 params
    // B: (d_out=8, rank=2) = 16 params
    size_t total = 0;
    for (auto *p : params) total += p->data.size();
    EXPECT_EQ(total, 8u + 16u);
}

TEST_F(LoRATest, RankConstraint)
{
    autograd::Node W(Tensor({16, 8}));
    LoRALinear lora(W, 4);

    EXPECT_EQ(lora.rank(), 4);
    EXPECT_EQ(lora.A().data.shape()[0], 4u);   // rank
    EXPECT_EQ(lora.A().data.shape()[1], 8u);   // d_in
    EXPECT_EQ(lora.B().data.shape()[0], 16u);  // d_out
    EXPECT_EQ(lora.B().data.shape()[1], 4u);   // rank
}

// ============================================================
// LoRA Adapter Tests
// ============================================================

TEST_F(LoRATest, AdapterAddLayer)
{
    LoRAConfig config;
    config.rank = 4;
    config.target_modules = {"q_proj", "v_proj"};

    LoRAAdapter adapter(config);

    autograd::Node W1(Tensor({8, 4}));
    autograd::Node W2(Tensor({8, 4}));

    adapter.add_layer("layer0.q_proj", W1);
    adapter.add_layer("layer0.v_proj", W2);

    EXPECT_EQ(adapter.num_layers(), 2u);
    EXPECT_NE(adapter.get_layer("layer0.q_proj"), nullptr);
    EXPECT_NE(adapter.get_layer("layer0.v_proj"), nullptr);
    EXPECT_EQ(adapter.get_layer("nonexistent"), nullptr);
}

TEST_F(LoRATest, AdapterShouldAdapt)
{
    LoRAConfig config;
    config.target_modules = {"q_proj", "v_proj"};
    LoRAAdapter adapter(config);

    EXPECT_TRUE(adapter.should_adapt("layer0.q_proj.weight"));
    EXPECT_TRUE(adapter.should_adapt("layer5.v_proj"));
    EXPECT_FALSE(adapter.should_adapt("layer0.k_proj"));
    EXPECT_FALSE(adapter.should_adapt("embedding"));
}

TEST_F(LoRATest, AdapterTrainableParams)
{
    LoRAConfig config;
    config.rank = 2;
    LoRAAdapter adapter(config);

    autograd::Node W(Tensor({8, 4}));
    adapter.add_layer("test", W);

    // A: 2*4=8, B: 8*2=16, total = 24
    EXPECT_EQ(adapter.num_trainable_params(), 24u);
    EXPECT_EQ(adapter.trainable_bytes(), 24u * sizeof(float));
}

TEST_F(LoRATest, AdapterSaveLoad)
{
    LoRAConfig config;
    config.rank = 2;
    LoRAAdapter adapter(config);

    autograd::Node W(Tensor({4, 3}));
    W.data.fill(1.0f);
    auto &layer = adapter.add_layer("proj", W);

    // Set specific values
    layer.A().data.fill(0.3f);
    layer.B().data.fill(0.7f);

    // Save
    adapter.save(TEST_LORA_FILE);

    // Create new adapter with same structure
    LoRAConfig config2;
    config2.rank = 2;
    LoRAAdapter adapter2(config2);

    autograd::Node W2(Tensor({4, 3}));
    W2.data.fill(1.0f);
    adapter2.add_layer("proj", W2);

    // Load
    adapter2.load(TEST_LORA_FILE);

    // Verify values match
    auto *loaded = adapter2.get_layer("proj");
    ASSERT_NE(loaded, nullptr);

    for (size_t i = 0; i < loaded->A().data.size(); i++) {
        EXPECT_NEAR(loaded->A().data.data()[i], 0.3f, 1e-5f);
    }
    for (size_t i = 0; i < loaded->B().data.size(); i++) {
        EXPECT_NEAR(loaded->B().data.data()[i], 0.7f, 1e-5f);
    }
}

TEST_F(LoRATest, AdapterMergeAll)
{
    LoRAConfig config;
    config.rank = 2;
    LoRAAdapter adapter(config);

    autograd::Node W1(Tensor({4, 3}));
    autograd::Node W2(Tensor({4, 3}));
    W1.data.fill(1.0f);
    W2.data.fill(2.0f);

    adapter.add_layer("layer1", W1);
    adapter.add_layer("layer2", W2);

    adapter.merge_all();

    EXPECT_TRUE(adapter.get_layer("layer1")->is_merged());
    EXPECT_TRUE(adapter.get_layer("layer2")->is_merged());

    adapter.unmerge_all();

    EXPECT_FALSE(adapter.get_layer("layer1")->is_merged());
    EXPECT_FALSE(adapter.get_layer("layer2")->is_merged());
}
