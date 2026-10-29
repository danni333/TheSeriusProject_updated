#include <gtest/gtest.h>
#include "tensor/tensor.h"

using namespace serius;

TEST(CorePlaceholder, TensorDefaultConstruction)
{
    core::Tensor t;
    EXPECT_EQ(t.device_type(), hal::DeviceType::CPU);
}

TEST(CorePlaceholder, BasicScalar)
{
    core::Tensor t = core::Tensor::zeros({1});
    EXPECT_EQ(t.numel(), 1);
}
