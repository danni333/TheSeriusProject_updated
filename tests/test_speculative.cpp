#include <gtest/gtest.h>

#include <vector>

#include "infer/speculative.h"
#include "tensor/tensor.h"

namespace {

serius::core::Tensor make_logits(std::size_t vocab_size, int chosen_token)
{
    serius::core::Tensor logits({vocab_size});
    logits.fill(0.0f);
    logits[static_cast<size_t>(chosen_token)] = 12.0f;
    return logits;
}

}  // namespace

TEST(SpeculativeDecoder, GeneratesAdditionalTokens)
{
    auto draft = [](const std::vector<int> &sequence) {
        static_cast<void>(sequence);
        return make_logits(16, 7);
    };

    auto target = [](const std::vector<int> &sequence) {
        static_cast<void>(sequence);
        return make_logits(16, 7);
    };

    serius::infer::SpeculativeDecoder decoder(draft, target, serius::infer::Sampler(1.0f, 0, 1.0f, 123u), 2, 2, 123u);

    std::vector<int> prompt = {1, 2};
    auto result = decoder.generate(prompt, 4);

    EXPECT_GE(result.size(), prompt.size() + 1);
    EXPECT_GT(decoder.metrics().draft_steps, 0u);
}

TEST(SpeculativeDecoder, AcceptsWhenDraftAndTargetMatch)
{
    auto model = [](const std::vector<int> &sequence) {
        static_cast<void>(sequence);
        return make_logits(16, 9);
    };

    serius::infer::SpeculativeDecoder decoder(model, model, serius::infer::Sampler(1.0f, 0, 1.0f, 42u), 4, 2, 42u);

    std::vector<int> prompt = {1};
    auto result = decoder.generate(prompt, 3);

    EXPECT_GT(result.size(), prompt.size());
    EXPECT_GT(decoder.metrics().accepted_tokens, 0u);
}
