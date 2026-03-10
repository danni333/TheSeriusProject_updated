#include "debug/nan_debugger.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>
#include <limits>

TEST(NanDebugger, HasNanOrInf)
{
    serius::core::Tensor t({4});
    t.fill(1.0f);
    EXPECT_FALSE(serius::debug::has_nan_or_inf(t));
    t[2] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(serius::debug::has_nan_or_inf(t));
}

TEST(NanDebugger, FindFirstNan)
{
    serius::core::Tensor a({2});
    a.fill(1.0f);
    serius::core::Tensor b({2});
    b.fill(2.0f);
    b[1] = std::numeric_limits<float>::quiet_NaN();
    std::vector<serius::debug::NamedTensor> tensors = {{"a", &a}, {"b", &b}};
    EXPECT_EQ(serius::debug::find_first_nan(tensors), "b");
    std::vector<serius::debug::NamedTensor> tensors_clean = {{"a", &a}};
    EXPECT_TRUE(serius::debug::find_first_nan(tensors_clean).empty());
}
