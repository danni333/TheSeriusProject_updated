#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace autoopt {

using KernelFn = std::function<core::Tensor(const core::Tensor &)>;

struct CompilationResult
{
    bool success = false;
    std::string artifact_name;
    std::string library_path;  // Path to compiled .dll/.so
    std::string error_message;
};

struct VerificationResult
{
    bool passed = false;
    float max_error = 0.0f;
    float reference_time_ms = 0.0f;
    float candidate_time_ms = 0.0f;
    float speedup = 1.0f;
};

/// Sandbox environment for safely compiling, verifying, and benchmarking candidate kernels.
class Sandbox
{
   public:
    explicit Sandbox(float atol = 1e-5f, bool use_real_compiler = false);

    /// Compile and syntax-validate C++ source code in sandbox.
    /// If use_real_compiler is true, uses system compiler (cl/g++/clang++) to build a shared library.
    CompilationResult compile(const std::string &source_code, const std::string &target_name) const;

    /// Verify numerical correctness of candidate kernel against a trusted reference implementation.
    VerificationResult verify(const KernelFn &reference,
                             const KernelFn &candidate,
                             const std::vector<core::Tensor> &test_inputs) const;

    /// Benchmark candidate execution speed vs reference on representative test inputs.
    VerificationResult benchmark(const KernelFn &reference,
                                 const KernelFn &candidate,
                                 const std::vector<core::Tensor> &test_inputs,
                                 size_t iterations = 10) const;

    float atol() const { return atol_; }
    void set_atol(float tol) { atol_ = tol; }

    bool use_real_compiler() const { return use_real_compiler_; }
    void set_use_real_compiler(bool use) { use_real_compiler_ = use; }

   private:
    float atol_;
    bool use_real_compiler_;

    /// Create temporary directory for compilation artifacts
    std::string create_temp_dir() const;

    /// Get system compiler command (cl for Windows, g++/clang++ for Unix)
    std::string get_compiler_command() const;

    /// Compile source to shared library using system compiler
    bool compile_to_library(const std::string &source, const std::string &output_path, std::string &error) const;
};

}  // namespace autoopt
}  // namespace serius
