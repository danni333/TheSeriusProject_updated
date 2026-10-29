#include <gtest/gtest.h>
#include "autoopt/refactorer.h"
#include "autoopt/sandbox.h"
#include "autoopt/hot_swap.h"
#include "autoopt/git_manager.h"
#include "autoopt/llm_client.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::autoopt;
using namespace serius::core;

// Simple identity kernel for testing
Tensor identity_kernel(const Tensor& in) {
    Tensor out(in.shape());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i];
    }
    return out;
}

// Slightly slower version (simulate unoptimized)
Tensor slow_identity(const Tensor& in) {
    Tensor out(in.shape());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i] * 1.0f;  // unnecessary multiplication
    }
    return out;
}

// Faster version (simulate optimized)
Tensor fast_identity(const Tensor& in) {
    Tensor out(in.shape());
    // std::copy is typically faster than manual loop
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i];
    }
    return out;
}

// Incorrect kernel (simulates bug)
Tensor incorrect_kernel(const Tensor& in) {
    Tensor out(in.shape());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i] + 1.0f;  // wrong!
    }
    return out;
}

TEST(Sandbox, TestSandboxCompile) {
    Sandbox sandbox;  // Real compiler disabled by default

    // Valid code (balanced braces)
    std::string valid_code = R"(
        void kernel() {
            int x = 1;
        }
    )";
    CompilationResult comp = sandbox.compile(valid_code, "test_kernel");
    EXPECT_TRUE(comp.success);
    EXPECT_EQ(comp.artifact_name, "test_kernel");

    // Invalid code (unmatched braces)
    std::string invalid_code = R"(
        void kernel() {
            int x = 1;
    )";
    comp = sandbox.compile(invalid_code, "test_kernel");
    EXPECT_FALSE(comp.success);
    EXPECT_FALSE(comp.error_message.empty());

    // Empty code
    comp = sandbox.compile("", "test_kernel");
    EXPECT_FALSE(comp.success);
    EXPECT_EQ(comp.error_message, "Empty source code");
}

TEST(Sandbox, TestRealCompilerDisabled) {
    Sandbox sandbox(1e-5f, false);  // Real compiler disabled

    std::string code = R"(
        void kernel() {
            int x = 1;
        }
    )";
    CompilationResult comp = sandbox.compile(code, "test_kernel");
    EXPECT_TRUE(comp.success);
    EXPECT_TRUE(comp.library_path.empty());  // No library compiled
}

TEST(Sandbox, TestRealCompilerEnabled) {
    Sandbox sandbox(1e-5f, true);  // Real compiler enabled

    std::string code = R"(
        #include <cstdint>
        extern "C" core::Tensor kernel_entry(const core::Tensor& in) {
            return in;
        }
    )";
    CompilationResult comp = sandbox.compile(code, "test_kernel");
    // May fail if compiler not available, but should not crash
    // In CI environments without compiler, we expect failure
    if (!comp.success) {
        EXPECT_FALSE(comp.error_message.empty());
    } else {
        EXPECT_FALSE(comp.library_path.empty());
    }
}

TEST(Sandbox, TestCorrectness) {
    Sandbox sandbox(1e-5f);

    std::vector<Tensor> test_inputs;
    Tensor input1({3});
    input1[0] = 1.0f; input1[1] = 2.0f; input1[2] = 3.0f;
    test_inputs.push_back(input1);

    Tensor input2({3});
    input2[0] = 0.5f; input2[1] = -1.5f; input2[2] = 2.5f;
    test_inputs.push_back(input2);

    // Correct kernel should pass
    VerificationResult res = sandbox.verify(identity_kernel, identity_kernel, test_inputs);
    EXPECT_TRUE(res.passed);
    EXPECT_LE(res.max_error, 1e-5f);

    // Incorrect kernel should fail
    res = sandbox.verify(identity_kernel, incorrect_kernel, test_inputs);
    EXPECT_FALSE(res.passed);
    EXPECT_GT(res.max_error, 1e-5f);
}

TEST(Sandbox, TestSpeedup) {
    Sandbox sandbox(1e-5f);

    std::vector<Tensor> test_inputs;
    for (int i = 0; i < 10; ++i) {
        Tensor input({5});
        input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f; input[3] = 4.0f; input[4] = 5.0f;
        test_inputs.push_back(input);
    }

    // Benchmark same kernel (speedup should be ~1.0)
    VerificationResult res = sandbox.benchmark(identity_kernel, identity_kernel, test_inputs, 5);
    EXPECT_TRUE(res.passed);
    EXPECT_FLOAT_EQ(res.speedup, 1.0f);

    // Benchmark different kernels (speedup may vary, but should be positive)
    res = sandbox.benchmark(slow_identity, fast_identity, test_inputs, 5);
    EXPECT_TRUE(res.passed);
    EXPECT_GT(res.speedup, 0.0f);
}

TEST(HotSwap, TestHotSwap) {
    HotSwapManager mgr;

    std::string kernel_name = "test_kernel";

    // Register initial kernel
    mgr.register_kernel(kernel_name, identity_kernel);
    EXPECT_TRUE(mgr.has_kernel(kernel_name));
    EXPECT_EQ(mgr.get_version(kernel_name), 1);

    // Execute initial kernel
    Tensor input({3});
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f;
    Tensor out1 = mgr.execute(kernel_name, input);
    EXPECT_EQ(out1.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(out1[i], input[i]);
    }

    // Hot-swap to new kernel
    bool swapped = mgr.hot_swap(kernel_name, fast_identity);
    EXPECT_TRUE(swapped);
    EXPECT_EQ(mgr.get_version(kernel_name), 2);

    // Execute swapped kernel
    Tensor out2 = mgr.execute(kernel_name, input);
    EXPECT_EQ(out2.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(out2[i], input[i]);
    }
}

TEST(HotSwap, TestRollback) {
    HotSwapManager mgr;

    std::string kernel_name = "test_kernel";

    // Register initial kernel
    mgr.register_kernel(kernel_name, identity_kernel);
    EXPECT_EQ(mgr.get_version(kernel_name), 1);

    // Hot-swap to new kernel
    mgr.hot_swap(kernel_name, fast_identity);
    EXPECT_EQ(mgr.get_version(kernel_name), 2);

    // Rollback to previous version
    bool rolled_back = mgr.rollback(kernel_name);
    EXPECT_TRUE(rolled_back);
    EXPECT_EQ(mgr.get_version(kernel_name), 1);

    // Cannot rollback initial version
    rolled_back = mgr.rollback(kernel_name);
    EXPECT_FALSE(rolled_back);
    EXPECT_EQ(mgr.get_version(kernel_name), 1);
}

TEST(GitManager, TestFormatCommitMessage) {
    GitManager git_mgr;

    std::string msg = git_mgr.format_commit_message("matmul_avx", 12.4f, 8.1f);
    EXPECT_FALSE(msg.empty());
    EXPECT_TRUE(msg.find("matmul_avx") != std::string::npos);
    EXPECT_TRUE(msg.find("12.4") != std::string::npos);
    EXPECT_TRUE(msg.find("8.1") != std::string::npos);
    EXPECT_TRUE(msg.find("perf:") != std::string::npos);
}

TEST(GitManager, TestCommitOptimizationDryRun) {
    GitManager git_mgr;

    bool success = git_mgr.commit_optimization("test_kernel", 10.0f, 8.0f, /*dry_run=*/true);
    EXPECT_TRUE(success);

    const auto& history = git_mgr.commit_history();
    EXPECT_EQ(history.size(), 1);
    EXPECT_TRUE(history[0].find("test_kernel") != std::string::npos);
}

TEST(AutoRefactorer, TestOptimizeHotspotSuccess) {
    HotSwapManager hot_swap;
    Sandbox sandbox(1e-5f);
    GitManager git_mgr;

    AutoRefactorer refactorer(hot_swap, sandbox, git_mgr, 1.05f);

    std::string kernel_name = "test_kernel";
    hot_swap.register_kernel(kernel_name, slow_identity);

    std::vector<Tensor> test_inputs;
    Tensor input1({3});
    input1[0] = 1.0f; input1[1] = 2.0f; input1[2] = 3.0f;
    test_inputs.push_back(input1);

    Tensor input2({3});
    input2[0] = 0.5f; input2[1] = -1.5f; input2[2] = 2.5f;
    test_inputs.push_back(input2);

    std::string dummy_source = "void kernel() { int x = 1; }";

    // Try to optimize (may or may not succeed depending on actual timing)
    RefactorReport report = refactorer.optimize_hotspot(
        kernel_name, fast_identity, dummy_source, test_inputs);

    // Should at least not crash
    EXPECT_FALSE(report.kernel_name.empty());
}

TEST(AutoRefactorer, TestOptimizeHotspotVerificationFailure) {
    HotSwapManager hot_swap;
    Sandbox sandbox(1e-5f);
    GitManager git_mgr;

    AutoRefactorer refactorer(hot_swap, sandbox, git_mgr, 1.05f);

    std::string kernel_name = "test_kernel";
    hot_swap.register_kernel(kernel_name, identity_kernel);

    std::vector<Tensor> test_inputs;
    Tensor input({3});
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f;
    test_inputs.push_back(input);

    std::string dummy_source = "void kernel() { int x = 1; }";

    // Try with incorrect kernel (should fail verification)
    RefactorReport report = refactorer.optimize_hotspot(
        kernel_name, incorrect_kernel, dummy_source, test_inputs);

    EXPECT_FALSE(report.optimized);
    EXPECT_TRUE(report.reason.find("Verification failed") != std::string::npos ||
                report.reason.find("numerical discrepancy") != std::string::npos);
}

TEST(AutoRefactorer, TestOptimizeHotspotKernelNotFound) {
    HotSwapManager hot_swap;
    Sandbox sandbox(1e-5f);
    GitManager git_mgr;

    AutoRefactorer refactorer(hot_swap, sandbox, git_mgr, 1.05f);

    std::vector<Tensor> test_inputs;
    Tensor input({3});
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f;
    test_inputs.push_back(input);

    std::string dummy_source = "void kernel() { int x = 1; }";

    // Try with non-existent kernel
    RefactorReport report = refactorer.optimize_hotspot(
        "nonexistent_kernel", identity_kernel, dummy_source, test_inputs);

    EXPECT_FALSE(report.optimized);
    EXPECT_TRUE(report.reason.find("not found") != std::string::npos);
}

TEST(LLMClient, TestBuildPrompt) {
    LLMClient client;
    // Just test config, don't actually call LLM (requires running service)
    // In production, mock the HTTP client
    EXPECT_FALSE(client.config().model_name.empty());
}

TEST(LLMClient, TestConfig) {
    LLMConfig config;
    config.model_name = "codellama";
    config.temperature = 0.3f;

    LLMClient client(config);
    EXPECT_EQ(client.config().model_name, "codellama");
    EXPECT_FLOAT_EQ(client.config().temperature, 0.3f);
}
