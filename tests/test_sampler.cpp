#include <gtest/gtest.h>

#include "infer/sampler.h"
#include "tensor/tensor.h"


TEST(Sampler, Greedy)
{
    serius::infer::Sampler sampler(1.0f, 0, 1.0f, 42u);
    std::vector<size_t> shape = {10};
    serius::core::Tensor logits(shape);
    logits.fill(0.f);
    logits[3] = 1.f;
    int tok = sampler.sample_greedy(logits);
    EXPECT_EQ(tok, 3);
}

TEST(Sampler, SampleFromLogits)
{
    serius::infer::Sampler sampler(0.5f, 5, 0.9f, 123u);
    std::vector<size_t> shape = {1, 4};
    serius::core::Tensor logits(shape);
    logits.at(0, 0) = 0.1f;
    logits.at(0, 1) = 0.2f;
    logits.at(0, 2) = 10.f;
    logits.at(0, 3) = 0.1f;
    int tok = sampler.sample(logits);
    EXPECT_GE(tok, 0);
    EXPECT_LT(tok, 4);
}
