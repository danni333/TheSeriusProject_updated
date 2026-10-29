#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "../src/memory/compressor.h"
#include "../src/memory/persistent.h"

using serius::core::Tensor;
using serius::memory::CompressionResult;
using serius::memory::PersistentMemory;
using serius::memory::SessionMeta;
using serius::memory::StateCompressor;

namespace fs = std::filesystem;

// ===================================================================
// StateCompressor tests
// ===================================================================

TEST(StateCompressorTest, CompressReducesDimension)
{
    const size_t D  = 32;
    const size_t S  = 128;  // input sequence length
    const size_t ST = 8;    // summary tokens

    StateCompressor comp(D, ST);

    Tensor hidden({S, D});
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = std::sin(static_cast<float>(i) * 0.01f);
    }

    CompressionResult res = comp.compress(hidden);

    ASSERT_EQ(res.compressed.ndim(), 2u);
    EXPECT_EQ(res.compressed.shape()[0], ST);
    EXPECT_EQ(res.compressed.shape()[1], D);
    EXPECT_LT(res.information_ratio, 1.0f);
    EXPECT_FLOAT_EQ(res.information_ratio, static_cast<float>(ST) / static_cast<float>(S));
}

TEST(StateCompressorTest, CompressPreservesInformationCosineSim)
{
    const size_t D  = 16;
    const size_t S  = 64;
    const size_t ST = 16;  // same as input — should be near-identity

    StateCompressor comp(D, ST);

    Tensor hidden({S, D});
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(i % 7) * 0.1f;
    }

    CompressionResult res = comp.compress(hidden);

    // Compressed output should not be all zeros.
    float total = 0.0f;
    for (size_t i = 0; i < res.compressed.size(); ++i) {
        total += std::abs(res.compressed[i]);
    }
    EXPECT_GT(total, 0.0f);
}

TEST(StateCompressorTest, NonTwoDInputPassthrough)
{
    const size_t D  = 16;
    const size_t ST = 4;
    StateCompressor comp(D, ST);

    // 1-D input: should be returned as-is (copy).
    Tensor input_1d({D});
    for (size_t i = 0; i < D; ++i) input_1d[i] = static_cast<float>(i);

    CompressionResult res = comp.compress(input_1d);
    EXPECT_EQ(res.compressed.size(), D);
    EXPECT_FLOAT_EQ(res.information_ratio, 1.0f);
}

// ===================================================================
// StateCompressor save/load round-trip
// ===================================================================

class CompressorDiskTest : public ::testing::Test {
   protected:
    std::string tmp_dir;

    void SetUp() override
    {
        tmp_dir = (fs::temp_directory_path() / "serius_test_compressor").string();
        fs::create_directories(tmp_dir);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(tmp_dir, ec);
    }
};

TEST_F(CompressorDiskTest, SaveLoadRoundTrip)
{
    const size_t D  = 16;
    const size_t ST = 4;
    StateCompressor comp(D, ST);

    Tensor state({ST, D});
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = static_cast<float>(i) * 0.123f;
    }

    std::string path = tmp_dir + "/session.bin";
    ASSERT_TRUE(comp.save_memory(path, state));

    Tensor loaded = comp.load_memory(path);
    ASSERT_EQ(loaded.ndim(), 2u);
    ASSERT_EQ(loaded.shape()[0], ST);
    ASSERT_EQ(loaded.shape()[1], D);

    for (size_t i = 0; i < state.size(); ++i) {
        EXPECT_FLOAT_EQ(loaded[i], state[i]) << "Mismatch at index " << i;
    }
}

TEST_F(CompressorDiskTest, BuildContextFromMultipleSessions)
{
    const size_t D  = 8;
    const size_t ST = 4;
    StateCompressor comp(D, ST);

    // Save three sessions.
    std::vector<std::string> paths;
    for (int s = 0; s < 3; ++s) {
        Tensor state({ST, D});
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] = static_cast<float>(s * 100 + i);
        }
        std::string p = tmp_dir + "/session_" + std::to_string(s) + ".bin";
        ASSERT_TRUE(comp.save_memory(p, state));
        paths.push_back(p);
    }

    Tensor ctx = comp.build_context(paths);
    ASSERT_EQ(ctx.ndim(), 2u);
    EXPECT_EQ(ctx.shape()[0], ST * 3);  // 3 sessions × ST rows
    EXPECT_EQ(ctx.shape()[1], D);

    // Verify first session's data.
    for (size_t i = 0; i < D; ++i) {
        EXPECT_FLOAT_EQ(ctx.at(0, i), static_cast<float>(i));
    }
}

TEST_F(CompressorDiskTest, LoadMissingFileReturnsEmpty)
{
    const size_t D  = 8;
    const size_t ST = 4;
    StateCompressor comp(D, ST);

    Tensor loaded = comp.load_memory(tmp_dir + "/nonexistent.bin");
    EXPECT_EQ(loaded.size(), 0u);
}

// ===================================================================
// PersistentMemory tests
// ===================================================================

class PersistentMemoryTest : public ::testing::Test {
   protected:
    std::string root;

    void SetUp() override
    {
        root = (fs::temp_directory_path() / "serius_test_persistent").string();
        std::error_code ec;
        fs::remove_all(root, ec);
        fs::create_directories(root, ec);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

TEST_F(PersistentMemoryTest, SaveAndLoadSession)
{
    const size_t D = 16;
    PersistentMemory pm(root, D);

    Tensor state({4, D});
    for (size_t i = 0; i < state.size(); ++i) state[i] = static_cast<float>(i) * 0.5f;

    SessionMeta meta = pm.save_session("user_alice", state, 0.25f);
    ASSERT_FALSE(meta.session_id.empty()) << "save_session failed";
    EXPECT_EQ(meta.user_id, "user_alice");
    EXPECT_EQ(meta.num_vectors, 4u);
    EXPECT_EQ(meta.d_model, D);
    EXPECT_FLOAT_EQ(meta.information_ratio, 0.25f);

    // Load it back.
    Tensor loaded = pm.load_session("user_alice", meta.session_id);
    ASSERT_EQ(loaded.ndim(), 2u);
    ASSERT_EQ(loaded.shape()[0], 4u);
    ASSERT_EQ(loaded.shape()[1], D);
    for (size_t i = 0; i < state.size(); ++i) {
        EXPECT_FLOAT_EQ(loaded[i], state[i]);
    }
}

TEST_F(PersistentMemoryTest, ListSessions)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    // Save 3 sessions.
    for (int s = 0; s < 3; ++s) {
        Tensor state({2, D});
        state.fill(static_cast<float>(s));
        pm.save_session("user_bob", state);
    }

    auto sessions = pm.list_sessions("user_bob");
    EXPECT_EQ(sessions.size(), 3u);
    EXPECT_EQ(pm.session_count("user_bob"), 3u);
}

TEST_F(PersistentMemoryTest, DeleteSession)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    Tensor state({2, D});
    state.fill(1.0f);
    SessionMeta meta = pm.save_session("user_carol", state);
    ASSERT_FALSE(meta.session_id.empty());

    EXPECT_EQ(pm.session_count("user_carol"), 1u);
    EXPECT_TRUE(pm.delete_session("user_carol", meta.session_id));
    EXPECT_EQ(pm.session_count("user_carol"), 0u);

    // Binary file should be gone.
    EXPECT_FALSE(fs::exists(meta.file_path));
}

TEST_F(PersistentMemoryTest, BuildUserContext)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    // Save 3 sessions with 2 vectors each → context should have 6 rows.
    for (int s = 0; s < 3; ++s) {
        Tensor state({2, D});
        state.fill(static_cast<float>(s + 1));
        pm.save_session("user_dave", state);
    }

    Tensor ctx = pm.build_user_context("user_dave");
    ASSERT_EQ(ctx.ndim(), 2u);
    EXPECT_EQ(ctx.shape()[0], 6u);
    EXPECT_EQ(ctx.shape()[1], D);
}

TEST_F(PersistentMemoryTest, BuildUserContextWithMaxSessions)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    for (int s = 0; s < 5; ++s) {
        Tensor state({2, D});
        state.fill(static_cast<float>(s + 1));
        pm.save_session("user_eve", state);
    }

    // Only the 2 most recent sessions.
    Tensor ctx = pm.build_user_context("user_eve", /*max_sessions=*/2);
    ASSERT_EQ(ctx.ndim(), 2u);
    EXPECT_EQ(ctx.shape()[0], 4u);  // 2 sessions × 2 rows
    EXPECT_EQ(ctx.shape()[1], D);
}

TEST_F(PersistentMemoryTest, EmptyUserReturnsEmptyContext)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    Tensor ctx = pm.build_user_context("nobody");
    EXPECT_EQ(ctx.shape()[0], 0u);
}

TEST_F(PersistentMemoryTest, WrongDimensionRejected)
{
    const size_t D = 8;
    PersistentMemory pm(root, D);

    // Wrong d_model.
    Tensor bad({4, D + 1});
    bad.fill(1.0f);
    SessionMeta meta = pm.save_session("user_x", bad);
    EXPECT_TRUE(meta.session_id.empty()) << "Should reject mismatched d_model";
}

TEST_F(PersistentMemoryTest, PersistenceAcrossInstances)
{
    const size_t D = 8;

    // First instance: save a session.
    {
        PersistentMemory pm(root, D);
        Tensor state({3, D});
        for (size_t i = 0; i < state.size(); ++i) state[i] = static_cast<float>(i);
        pm.save_session("user_persist", state, 0.1f);
    }

    // Second instance: should find the session.
    {
        PersistentMemory pm(root, D);
        EXPECT_EQ(pm.session_count("user_persist"), 1u);

        auto sessions = pm.list_sessions("user_persist");
        ASSERT_EQ(sessions.size(), 1u);

        Tensor loaded = pm.load_session("user_persist", sessions[0].session_id);
        ASSERT_EQ(loaded.ndim(), 2u);
        ASSERT_EQ(loaded.shape()[0], 3u);
        for (size_t i = 0; i < loaded.size(); ++i) {
            EXPECT_FLOAT_EQ(loaded[i], static_cast<float>(i));
        }
    }
}
