#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "automl/auto_tune.h"
#include "automl/pruner.h"
#include "automl/tune_cache.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::automl;
using namespace serius::core;

// ---- AutoTune tests ----

TEST(AutoTune, SelectsBestConfig)
{
    AutoTune tuner;
    auto result = tuner.tune_matmul(64, 64, 64);

    EXPECT_GT(result.best_time_ms, 0.0f);
    EXPECT_GT(result.best_config.block_x, 0);
    EXPECT_GT(result.best_config.block_y, 0);
    EXPECT_GT(result.best_config.tile_size, 0);
    EXPECT_GE(result.speedup, 1.0f);
}

TEST(AutoTune, EvaluatesAllCandidates)
{
    AutoTune tuner;
    std::vector<KernelConfig> candidates = {
        {16, 16, 32, true},
        {32, 8, 64, true},
        {32, 32, 32, false},
    };

    auto result = tuner.tune("test", candidates,
        [](const KernelConfig &cfg) -> float {
            // Simulated timing: larger tiles are faster.
            return 10.0f / static_cast<float>(cfg.tile_size);
        });

    EXPECT_EQ(tuner.evaluations(), candidates.size());
    EXPECT_GE(result.speedup, 1.0f);
}

TEST(AutoTune, EmptyCandidates)
{
    AutoTune tuner;
    auto result = tuner.tune("empty", {}, [](const KernelConfig &) { return 1.0f; });

    EXPECT_EQ(result.best_config.block_x, 0);
    EXPECT_EQ(result.best_time_ms, 0.0f);
    EXPECT_EQ(result.speedup, 1.0f);
}

// ---- Pruner tests ----

TEST(Pruner, PrunesAllButKeepsAtLeastOne)
{
    Pruner pruner(0.5f);
    Tensor W({4, 8});
    for (size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>(i);

    std::vector<float> stats = {0.01f, 0.02f, 0.03f, 0.04f};
    auto result = pruner.prune(W, stats);

    // All stats below threshold, but at least one row is kept.
    EXPECT_EQ(result.kept_ratio, 0.25f);
    EXPECT_GT(result.estimated_speedup, 1.0f);
}

TEST(Pruner, PrunesLowActivationRows)
{
    Pruner pruner(0.1f);
    Tensor W({4, 8});
    for (size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>(i);

    // Row 0 has low activation → should be pruned.
    std::vector<float> stats = {0.01f, 0.5f, 0.6f, 0.7f};
    auto result = pruner.prune(W, stats);

    EXPECT_LT(result.kept_ratio, 1.0f);
    EXPECT_GT(result.estimated_speedup, 1.0f);
}

TEST(Pruner, PruneWeightsReturnsCorrectShape)
{
    Pruner pruner(0.1f);
    Tensor W({4, 8});
    for (size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>(i);

    std::vector<float> stats = {0.01f, 0.5f, 0.6f, 0.7f};
    Tensor pruned = pruner.prune_weights(W, stats);

    EXPECT_EQ(pruned.shape()[0], 3u);  // 1 row pruned
    EXPECT_EQ(pruned.shape()[1], 8u);
}

TEST(Pruner, ThresholdAccessor)
{
    Pruner pruner(0.3f);
    EXPECT_FLOAT_EQ(pruner.threshold(), 0.3f);
    pruner.set_threshold(0.5f);
    EXPECT_FLOAT_EQ(pruner.threshold(), 0.5f);
}

// ---- TuneCache tests ----

TEST(TuneCache, PutAndGet)
{
    TuneCache cache;
    TuneCacheKey key = std::make_tuple(64, 64, 64);
    KernelConfig cfg{16, 16, 32, true};
    float time = 1.5f;

    cache.put(key, cfg, time);

    KernelConfig out_cfg;
    float out_time;
    EXPECT_TRUE(cache.get(key, out_cfg, out_time));
    EXPECT_EQ(out_cfg.block_x, 16);
    EXPECT_EQ(out_cfg.block_y, 16);
    EXPECT_EQ(out_cfg.tile_size, 32);
    EXPECT_FLOAT_EQ(out_time, time);
}

TEST(TuneCache, MissReturnsFalse)
{
    TuneCache cache;
    TuneCacheKey key = std::make_tuple(128, 128, 128);
    KernelConfig cfg;
    float time;
    EXPECT_FALSE(cache.get(key, cfg, time));
}

TEST(TuneCache, SaveAndLoad)
{
    std::string tmp_file = "test_tune_cache_tmp.json";
    {
        TuneCache cache(tmp_file);
        cache.put(std::make_tuple(32, 32, 32), KernelConfig{8, 8, 64, true}, 2.0f);
        cache.put(std::make_tuple(64, 64, 64), KernelConfig{16, 16, 32, true}, 1.5f);
        EXPECT_TRUE(cache.save());
    }

    {
        TuneCache cache(tmp_file);
        EXPECT_TRUE(cache.load());
        EXPECT_EQ(cache.size(), 2u);

        KernelConfig cfg;
        float time;
        EXPECT_TRUE(cache.get(std::make_tuple(32, 32, 32), cfg, time));
        EXPECT_EQ(cfg.block_x, 8);
        EXPECT_FLOAT_EQ(time, 2.0f);
    }
}

TEST(TuneCache, Clear)
{
    TuneCache cache;
    cache.put(std::make_tuple(1, 1, 1), KernelConfig{1, 1, 1, false}, 1.0f);
    EXPECT_EQ(cache.size(), 1u);
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}