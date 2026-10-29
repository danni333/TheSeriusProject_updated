#include "auto_tune.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <tuple>

namespace serius {
namespace automl {

namespace {

using Clock = std::chrono::high_resolution_clock;

float now_ms()
{
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count()) /
    1000.0f;
}

}  // namespace

AutoTune::AutoTune() : evaluations_(0)
{
}

TuneResult AutoTune::tune_matmul(int M, int N, int K)
{
    // Standard candidate configurations for matmul kernels.
    std::vector<KernelConfig> candidates = {
        {16, 16, 32, true},
        {32, 8, 64, true},
        {32, 32, 32, false},
        {16, 16, 64, true},
        {8, 8, 128, true},
        {16, 8, 64, true},
        {8, 16, 64, true},
    };

    // Reference: naive scalar implementation (informational only).
    auto reference_fn = [&](const KernelConfig & /*cfg*/) -> float {
        // Simulated naive matmul timing proportional to M*N*K.
        float ops = static_cast<float>(M) * static_cast<float>(N) * static_cast<float>(K);
        return ops / 1.0e9f;  // synthetic baseline
    };
    (void)reference_fn;

    // Simulated benchmark: each config has a "speed" factor based on tile size
    // and shared memory usage. Larger tiles with shared memory are faster.
    auto launch_fn = [&](const KernelConfig &cfg) -> float {
        float base = static_cast<float>(M) * static_cast<float>(N) * static_cast<float>(K) / 5.0e9f;
        float tile_factor = static_cast<float>(cfg.tile_size) / 32.0f;
        float shared_factor = cfg.use_shared_mem ? 0.7f : 1.0f;
        float block_factor = (static_cast<float>(cfg.block_x * cfg.block_y) / 256.0f);
        return base / (tile_factor * shared_factor * block_factor);
    };

    return tune("matmul", candidates, launch_fn);
}

TuneResult AutoTune::tune(const std::string &name,
                          const std::vector<KernelConfig> &candidates,
                          std::function<float(const KernelConfig &)> launch_fn)
{
    if (candidates.empty()) {
        return TuneResult{{0, 0, 0, false}, 0.0f, 0.0f, 1.0f};
    }

    KernelConfig best = candidates[0];
    float best_time = std::numeric_limits<float>::max();
    float first_time = 0.0f;
    bool first_set = false;

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto &cfg = candidates[i];

        // Warmup run.
        launch_fn(cfg);

        // Benchmark: average of 3 runs.
        float total = 0.0f;
        constexpr int kRuns = 3;
        for (int r = 0; r < kRuns; ++r) {
            total += launch_fn(cfg);
        }
        float elapsed = total / static_cast<float>(kRuns);

        if (!first_set) {
            first_time = elapsed;
            first_set = true;
        }

        if (elapsed < best_time) {
            best_time = elapsed;
            best = cfg;
        }

        evaluations_++;

        std::cout << "[AutoTune] " << name << " candidate " << i << " (block=" << cfg.block_x
                  << "x" << cfg.block_y << ", tile=" << cfg.tile_size
                  << ", shared=" << (cfg.use_shared_mem ? "yes" : "no")
                  << "): " << elapsed << " ms\n";
    }

    float speedup = first_time > 0.0f ? first_time / best_time : 1.0f;

    std::cout << "[AutoTune] " << name << " best: block=" << best.block_x << "x" << best.block_y
              << ", tile=" << best.tile_size << ", time=" << best_time
              << " ms, speedup=" << speedup << "x\n";

    return TuneResult{best, best_time, first_time, speedup};
}

}  // namespace automl
}  // namespace serius