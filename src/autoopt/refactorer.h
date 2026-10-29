#pragma once

#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "git_manager.h"
#include "hot_swap.h"
#include "llm_client.h"
#include "sandbox.h"

namespace serius {
namespace autoopt {

struct RefactorReport
{
    bool optimized = false;
    std::string kernel_name;
    float old_latency_ms = 0.0f;
    float new_latency_ms = 0.0f;
    float speedup = 1.0f;
    std::string reason;
    std::string generated_code;  // LLM-generated code (if applicable)
};

/// Autonomous recursive refactoring pipeline.
/// Validates, benchmarks, hot-swaps and commits verified kernel improvements.
class AutoRefactorer
{
   public:
    AutoRefactorer(HotSwapManager &hot_swap,
                   Sandbox &sandbox,
                   GitManager &git_mgr,
                   float min_speedup = 1.05f);

    /// Attempt an optimization on a target hotspot kernel.
    /// @param kernel_name       Name of the registered active kernel
    /// @param candidate_kernel  Proposed optimized implementation
    /// @param candidate_source  C++ source code of the candidate for sandbox compilation
    /// @param test_inputs       Set of test tensors for numerical verification and latency testing
    /// @return RefactorReport with outcome details
    RefactorReport optimize_hotspot(const std::string &kernel_name,
                                   const KernelFn &candidate_kernel,
                                   const std::string &candidate_source,
                                   const std::vector<core::Tensor> &test_inputs);

    /// Attempt LLM-guided optimization on a target hotspot kernel.
    /// Uses LLM to generate optimized code, then compiles, verifies, and benchmarks.
    /// @param kernel_name       Name of the registered active kernel
    /// @param current_source    Current implementation source code
    /// @param performance_info  Performance metrics (latency, bottleneck info)
    /// @param test_inputs       Set of test tensors for numerical verification and latency testing
    /// @return RefactorReport with outcome details
    RefactorReport optimize_with_llm(const std::string &kernel_name,
                                     const std::string &current_source,
                                     const std::string &performance_info,
                                     const std::vector<core::Tensor> &test_inputs);

    float min_speedup() const { return min_speedup_; }
    void set_min_speedup(float s) { min_speedup_ = s; }

    const LLMClient &llm_client() const { return llm_client_; }
    LLMClient &llm_client() { return llm_client_; }

   private:
    HotSwapManager &hot_swap_;
    Sandbox &sandbox_;
    GitManager &git_mgr_;
    LLMClient llm_client_;
    float min_speedup_;
};

}  // namespace autoopt
}  // namespace serius
