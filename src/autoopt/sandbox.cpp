#include "sandbox.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace serius {
namespace autoopt {

Sandbox::Sandbox(float atol, bool use_real_compiler)
    : atol_(atol), use_real_compiler_(use_real_compiler)
{
}

std::string Sandbox::create_temp_dir() const
{
    namespace fs = std::filesystem;

    // Create a unique temp directory
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    std::string temp_dir_name = "serius_sandbox_" + std::to_string(dis(gen));
    fs::path temp_path = fs::temp_directory_path() / temp_dir_name;

    if (!fs::create_directory(temp_path)) {
        return "";
    }

    return temp_path.string();
}

std::string Sandbox::get_compiler_command() const
{
#ifdef _WIN32
    // Try to find MSVC cl.exe
    char* vswhere = nullptr;
    size_t len = 0;
    _dupenv_s(&vswhere, &len, "VSWhere");
    if (vswhere) {
        free(vswhere);
        return "cl.exe";
    }
    // Fallback: check PATH
    return "cl.exe";
#else
    // Try g++ first, then clang++
    if (system("which g++ > /dev/null 2>&1") == 0) {
        return "g++";
    }
    if (system("which clang++ > /dev/null 2>&1") == 0) {
        return "clang++";
    }
    return "g++";  // Default fallback
#endif
}

bool Sandbox::compile_to_library(const std::string &source, const std::string &output_path, std::string &error) const
{
    namespace fs = std::filesystem;

    // Write source to temp file
    std::string source_path = output_path + ".cpp";
    std::ofstream out(source_path);
    if (!out) {
        error = "Failed to create source file: " + source_path;
        return false;
    }
    out << source;
    out.close();

    // Build compiler command
    std::string compiler = get_compiler_command();
    std::ostringstream cmd;

#ifdef _WIN32
    // MSVC: cl.exe /LD /EHsc source.cpp /OUT:output.dll
    cmd << "\"" << compiler << "\" /LD /EHsc \"" << source_path << "\" /Fe:\"" << output_path << "\" 2>&1";
#else
    // g++/clang++: -shared -fPIC -o output.so source.cpp
    cmd << compiler << " -shared -fPIC -std=c++20 -o \"" << output_path << "\" \"" << source_path << "\" 2>&1";
#endif

    // Execute compilation with timeout
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        error = "Failed to execute compiler: " + compiler;
        return false;
    }

    // Read compiler output
    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int status = pclose(pipe);
    (void)status;  // Suppress unused warning

    // Check if library was created
    if (!fs::exists(output_path)) {
        error = "Compilation failed: " + output;
        return false;
    }

    return true;
}

CompilationResult Sandbox::compile(const std::string &source_code, const std::string &target_name) const
{
    CompilationResult res;
    res.artifact_name = target_name;

    if (source_code.empty()) {
        res.success = false;
        res.error_message = "Empty source code";
        return res;
    }

    // Basic syntax safety checks (always run)
    int brace_count = 0;
    for (char c : source_code) {
        if (c == '{') brace_count++;
        if (c == '}') brace_count--;
    }

    if (brace_count != 0) {
        res.success = false;
        res.error_message = "Unmatched braces in generated source";
        return res;
    }

    // If real compiler is disabled, just return success after syntax check
    if (!use_real_compiler_) {
        res.success = true;
        return res;
    }

    // Create temp directory
    std::string temp_dir = create_temp_dir();
    if (temp_dir.empty()) {
        res.success = false;
        res.error_message = "Failed to create temp directory";
        return res;
    }

    // Determine library extension
#ifdef _WIN32
    std::string lib_ext = ".dll";
#else
    std::string lib_ext = ".so";
#endif

    std::string lib_path = temp_dir + "/" + target_name + lib_ext;

    // Compile to shared library
    std::string error;
    if (!compile_to_library(source_code, lib_path, error)) {
        res.success = false;
        res.error_message = error;
        return res;
    }

    res.success = true;
    res.library_path = lib_path;
    return res;
}

VerificationResult Sandbox::verify(const KernelFn &reference,
                                   const KernelFn &candidate,
                                   const std::vector<core::Tensor> &test_inputs) const
{
    VerificationResult res;
    res.passed = true;
    res.max_error = 0.0f;

    if (!reference || !candidate) {
        res.passed = false;
        return res;
    }

    for (const auto &input : test_inputs) {
        core::Tensor ref_out = reference(input);
        core::Tensor cand_out = candidate(input);

        if (ref_out.size() != cand_out.size()) {
            res.passed = false;
            return res;
        }

        for (size_t i = 0; i < ref_out.size(); ++i) {
            float err = std::abs(ref_out[i] - cand_out[i]);
            if (err > res.max_error) {
                res.max_error = err;
            }
            if (err > atol_) {
                res.passed = false;
            }
        }
    }

    return res;
}

VerificationResult Sandbox::benchmark(const KernelFn &reference,
                                      const KernelFn &candidate,
                                      const std::vector<core::Tensor> &test_inputs,
                                      size_t iterations) const
{
    VerificationResult res = verify(reference, candidate, test_inputs);
    if (!res.passed) {
        return res;
    }

    size_t iters = std::max<size_t>(1, iterations);

    // Warmup
    for (const auto &input : test_inputs) {
        reference(input);
        candidate(input);
    }

    // Benchmark Reference
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iters; ++it) {
        for (const auto &input : test_inputs) {
            reference(input);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    res.reference_time_ms = static_cast<float>(ref_us / 1000.0);

    // Benchmark Candidate
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iters; ++it) {
        for (const auto &input : test_inputs) {
            candidate(input);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double cand_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    res.candidate_time_ms = static_cast<float>(cand_us / 1000.0);

    if (res.candidate_time_ms > 1e-6f) {
        res.speedup = res.reference_time_ms / res.candidate_time_ms;
    } else {
        res.speedup = 1.0f;
    }

    return res;
}

}  // namespace autoopt
}  // namespace serius
