#include "refactorer.h"

namespace serius {
namespace autoopt {

AutoRefactorer::AutoRefactorer(HotSwapManager &hot_swap,
                               Sandbox &sandbox,
                               GitManager &git_mgr,
                               float min_speedup)
    : hot_swap_(hot_swap),
      sandbox_(sandbox),
      git_mgr_(git_mgr),
      min_speedup_(min_speedup)
{
}

RefactorReport AutoRefactorer::optimize_hotspot(const std::string &kernel_name,
                                               const KernelFn &candidate_kernel,
                                               const std::string &candidate_source,
                                               const std::vector<core::Tensor> &test_inputs)
{
    RefactorReport report;
    report.kernel_name = kernel_name;
    report.optimized = false;

    if (!hot_swap_.has_kernel(kernel_name)) {
        report.reason = "Kernel not found in HotSwapManager: " + kernel_name;
        return report;
    }

    // Step 1: Sandbox compile check
    CompilationResult comp = sandbox_.compile(candidate_source, kernel_name + "_v2");
    if (!comp.success) {
        report.reason = "Compilation failed: " + comp.error_message;
        return report;
    }

    // Step 2 & 3: Verification and benchmark, then swap/commit on success.
    return evaluate_candidate(kernel_name, candidate_kernel, test_inputs, std::move(report));
}

RefactorReport AutoRefactorer::evaluate_candidate(const std::string &kernel_name,
                                                  const KernelFn &candidate_kernel,
                                                  const std::vector<core::Tensor> &test_inputs,
                                                  RefactorReport report)
{
    // Capture current reference
    KernelFn reference = [this, kernel_name](const core::Tensor &in) {
        return hot_swap_.execute(kernel_name, in);
    };

    VerificationResult bench = sandbox_.benchmark(reference, candidate_kernel, test_inputs);

    report.old_latency_ms = bench.reference_time_ms;
    report.new_latency_ms = bench.candidate_time_ms;
    report.speedup = bench.speedup;

    if (!bench.passed) {
        report.reason = "Verification failed: numerical discrepancy exceeding tolerance";
        return report;
    }

    if (bench.speedup < min_speedup_) {
        report.reason = "Insufficient speedup (" + std::to_string(bench.speedup) + "x < " +
                        std::to_string(min_speedup_) + "x threshold)";
        return report;
    }

    hot_swap_.hot_swap(kernel_name, candidate_kernel);
    git_mgr_.commit_optimization(kernel_name, report.old_latency_ms, report.new_latency_ms, /*dry_run=*/true);

    report.optimized = true;
    report.reason = "Optimization successful and hot-swapped";
    return report;
}

RefactorReport AutoRefactorer::optimize_with_llm(const std::string &kernel_name,
                                                const std::string &current_source,
                                                const std::string &performance_info,
                                                const std::vector<core::Tensor> &test_inputs)
{
    RefactorReport report;
    report.kernel_name = kernel_name;
    report.optimized = false;

    if (!hot_swap_.has_kernel(kernel_name)) {
        report.reason = "Kernel not found in HotSwapManager: " + kernel_name;
        return report;
    }

    // Step 1: Generate optimized code using LLM
    std::string generated_code = llm_client_.generate_optimized_code(kernel_name, current_source, performance_info);
    if (generated_code.empty()) {
        report.reason = "LLM failed to generate code (check API endpoint and model)";
        return report;
    }

    report.generated_code = generated_code;

    // Step 2: Sandbox compile — produces a real shared library when the
    // sandbox was configured with use_real_compiler.
    CompilationResult comp = sandbox_.compile(generated_code, kernel_name + "_v2");
    if (!comp.success) {
        report.reason = "Compilation failed: " + comp.error_message;
        return report;
    }

    // Step 3: Load the compiled kernel through the C ABI and run it through
    // the same verify/benchmark/hot-swap pipeline as optimize_hotspot.
    if (comp.library_path.empty()) {
        report.reason = "Sandbox compiled without a loadable library "
                        "(use_real_compiler is disabled)";
        return report;
    }

    KernelFn candidate = Sandbox::load_kernel(comp.library_path);
    if (!candidate) {
        report.reason = "Compiled library does not export 'kernel_entry' "
                        "with the expected C ABI signature";
        return report;
    }

    return evaluate_candidate(kernel_name, candidate, test_inputs, std::move(report));
}

}  // namespace autoopt
}  // namespace serius
