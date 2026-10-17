#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include "server/batcher.h"

using serius::server::ContinuousBatcher;

static std::vector<int> next_token_mock(const std::vector<std::vector<int>> &batch)
{
    // Deterministic mock: returns token = sequence length mod 10, or -1 (EOS) at length 8
    std::vector<int> out;
    out.reserve(batch.size());
    for (const auto &seq : batch) {
        if (static_cast<int>(seq.size()) >= 8) {
            out.push_back(-1);  // EOS
        } else {
            out.push_back(static_cast<int>(seq.size()) % 10);
        }
    }
    return out;
}

TEST(ContinuousBatcher, SlotFillAndParallel)
{
    ContinuousBatcher b(4);

    std::vector<std::string> seen_a, seen_b;
    b.enqueue(1, {10, 11}, [&](std::string t, bool done) { seen_a.push_back(t); });
    b.enqueue(2, {20, 21}, [&](std::string t, bool done) { seen_b.push_back(t); });

    // Forward mock returns next token for each active slot
    auto forward = [](const std::vector<std::vector<int>> &batch) -> std::vector<int> {
        std::vector<int> out;
        out.reserve(batch.size());
        for (const auto &seq : batch) {
            out.push_back(static_cast<int>(seq.size()) % 10);
        }
        return out;
    };

    b.iteration(forward);
    EXPECT_EQ(b.active_slots(), 2u);
    EXPECT_EQ(b.pending(), 0u);
}

TEST(ContinuousBatcher, SlotReuseAfterEOS)
{
    ContinuousBatcher b(2);

    bool done_a = false;
    bool done_b = false;
    b.enqueue(1, {1}, [&](std::string, bool done) { done_a = done; });
    b.enqueue(2, {1}, [&](std::string, bool done) { done_b = done; });

    // Iteration 1: both slots active, no EOS yet (seq length 2 < 8)
    b.iteration(next_token_mock);
    EXPECT_EQ(b.active_slots(), 2u);

    // Iterate until EOS (length reaches 8)
    int iters = 0;
    while (b.active_slots() > 0 && iters < 20) {
        b.iteration(next_token_mock);
        ++iters;
    }
    EXPECT_EQ(b.active_slots(), 0u);  // both slots freed after EOS

    // Slots can be reused for new requests
    bool done_c = false;
    b.enqueue(3, {1}, [&](std::string, bool done) { done_c = done; });
    b.iteration(next_token_mock);
    EXPECT_EQ(b.active_slots(), 1u);
}

TEST(ContinuousBatcher, PerSlotTokens)
{
    ContinuousBatcher b(2);

    std::vector<int> tokens_a;
    std::vector<int> tokens_b;
    b.enqueue(1, {1}, [&](std::string t, bool) { tokens_a.push_back(std::stoi(t)); });
    b.enqueue(2, {2}, [&](std::string t, bool) { tokens_b.push_back(std::stoi(t)); });

    b.iteration(next_token_mock);
    b.iteration(next_token_mock);
    b.iteration(next_token_mock);

    // Each slot received its own tokens
    EXPECT_GT(tokens_a.size(), 0u);
    EXPECT_GT(tokens_b.size(), 0u);
}

TEST(ContinuousBatcher, PendingQueue)
{
    ContinuousBatcher b(1);
    b.enqueue(1, {1}, [](std::string, bool) {});
    b.iteration(next_token_mock);  // move request into active slot
    EXPECT_EQ(b.pending(), 0u);    // 1 active, 0 waiting
    b.enqueue(2, {2}, [](std::string, bool) {});
    EXPECT_EQ(b.pending(), 1u);    // 1 active, 1 waiting
}
