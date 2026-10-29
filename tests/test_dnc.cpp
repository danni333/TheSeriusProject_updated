#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/memory/dnc.h"

using serius::core::Tensor;
using serius::memory::DNCMemory;

namespace {

// Cosine similarity between two 1-D tensors.
float cosine_sim(const Tensor &a, const Tensor &b, size_t d)
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
}

}  // namespace

// ---------------------------------------------------------------------------
// Test: write a vector and read it back — cosine similarity > 0.95.
// ---------------------------------------------------------------------------
TEST(DNCMemoryTest, WriteReadCosineSimilarity)
{
    const size_t N = 16;
    const size_t D = 32;
    DNCMemory mem(N, D);

    // Create a distinctive key/vector pair.
    Tensor key({D});
    Tensor vec({D});
    for (size_t i = 0; i < D; ++i) {
        key[i] = static_cast<float>(i) / static_cast<float>(D);
        vec[i] = static_cast<float>(i + 1) / static_cast<float>(D);
    }

    mem.write(key, vec, /*write_strength=*/10.0f, /*free_gate=*/0.0f);

    Tensor readback = mem.read(key, /*read_strength=*/10.0f);
    float sim = cosine_sim(readback, vec, D);
    EXPECT_GT(sim, 0.95f) << "Readback cosine similarity is " << sim;
}

// ---------------------------------------------------------------------------
// Test: N distinct write/read cycles — each recovered independently.
// ---------------------------------------------------------------------------
TEST(DNCMemoryTest, CapacityDistinctSlots)
{
    const size_t N = 8;
    const size_t D = 16;
    DNCMemory mem(N, D);

    std::vector<Tensor> keys;
    std::vector<Tensor> values;

    // Write N distinct (key, value) pairs.
    for (size_t s = 0; s < N; ++s) {
        Tensor k({D});
        Tensor v({D});
        for (size_t i = 0; i < D; ++i) {
            // Make each key/value pair unique via a slot-dependent offset.
            k[i] = std::sin(static_cast<float>(s * D + i));
            v[i] = std::cos(static_cast<float>(s * D + i + 7));
        }
        keys.push_back(std::move(k));
        values.push_back(std::move(v));
    }

    // Write them all into memory, preferring allocation (free_gate=0).
    for (size_t s = 0; s < N; ++s) {
        mem.write(keys[s], values[s], /*write_strength=*/10.0f, /*free_gate=*/0.0f);
    }

    // Read back: each should still be recognisable.
    int recovered = 0;
    for (size_t s = 0; s < N; ++s) {
        Tensor readback = mem.read(keys[s], /*read_strength=*/10.0f);
        float sim = cosine_sim(readback, values[s], D);
        if (sim > 0.7f) ++recovered;
    }
    // At least half of the slots should be faithfully recovered.
    EXPECT_GE(recovered, static_cast<int>(N) / 2)
        << "Only " << recovered << "/" << N << " slots recovered";
}

// ---------------------------------------------------------------------------
// Test: writing with high free_gate routes through content addressing.
// ---------------------------------------------------------------------------
TEST(DNCMemoryTest, ContentAddressingVsAllocation)
{
    const size_t N = 8;
    const size_t D = 16;
    DNCMemory mem(N, D);

    Tensor key({D});
    Tensor vec({D});
    for (size_t i = 0; i < D; ++i) {
        key[i] = 1.0f;
        vec[i] = static_cast<float>(i);
    }

    // First write with pure allocation (free_gate=0).
    mem.write(key, vec, 5.0f, 0.0f);

    // Overwrite via content addressing (free_gate=1).
    Tensor vec2({D});
    for (size_t i = 0; i < D; ++i) vec2[i] = static_cast<float>(D - i);
    mem.write(key, vec2, 5.0f, 1.0f);

    // Read back — should be closer to vec2 now.
    Tensor readback = mem.read(key, 5.0f);
    float sim2 = cosine_sim(readback, vec2, D);
    EXPECT_GT(sim2, 0.5f) << "Content-addressed overwrite not reflected";
}

// ---------------------------------------------------------------------------
// Test: memory matrix starts as zeros.
// ---------------------------------------------------------------------------
TEST(DNCMemoryTest, InitialStateIsZero)
{
    const size_t N = 4;
    const size_t D = 8;
    DNCMemory mem(N, D);

    const Tensor &M = mem.matrix();
    for (size_t i = 0; i < M.size(); ++i) {
        EXPECT_FLOAT_EQ(M[i], 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Test: reading from an empty (all-zeros) memory returns zeros.
// ---------------------------------------------------------------------------
TEST(DNCMemoryTest, ReadFromEmptyMemoryReturnsZero)
{
    const size_t N = 4;
    const size_t D = 8;
    DNCMemory mem(N, D);

    Tensor key({D});
    for (size_t i = 0; i < D; ++i) key[i] = 1.0f;

    Tensor readback = mem.read(key, 5.0f);
    float total = 0.0f;
    for (size_t i = 0; i < D; ++i) total += std::abs(readback[i]);
    EXPECT_NEAR(total, 0.0f, 1e-6f);
}
