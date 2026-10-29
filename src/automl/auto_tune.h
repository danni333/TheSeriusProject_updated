#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>
#include <string>

namespace serius {
namespace automl {

/// Configuration for a CUDA kernel launch.
struct KernelConfig
{
    int block_x;
    int block_y;
    int tile_size;
    bool use_shared_mem;
};

/// Result of a kernel auto-tuning run.
struct TuneResult
{
    KernelConfig best_config;
    float best_time_ms;
    float reference_time_ms;
    float speedup;
};

/// Auto-tuning engine for CUDA kernels.
/// Tests multiple kernel configurations and selects the fastest for the given problem size.
class AutoTune
{
   public:
    AutoTune();

    /// Auto-tune a matrix multiplication kernel for the given dimensions.
    /// @param M Number of rows in output matrix.
    /// @param N Number of columns in output matrix.
    /// @param K Inner dimension of the matmul.
    /// @return TuneResult with the best configuration found.
    TuneResult tune_matmul(int M, int N, int K);

    /// Auto-tune a generic kernel given a launch function.
    /// @param name Human-readable name for logging.
    /// @param candidates List of configurations to benchmark.
    /// @param launch_fn Callable that launches the kernel given a config. Returns elapsed ms.
    /// @return TuneResult with the best configuration.
    TuneResult tune(const std::string &name,
                    const std::vector<KernelConfig> &candidates,
                    std::function<float(const KernelConfig &)> launch_fn);

    /// Get the number of candidates evaluated.
    size_t evaluations() const { return evaluations_; }

   private:
    size_t evaluations_;
};

}  // namespace automl
}  // namespace serius