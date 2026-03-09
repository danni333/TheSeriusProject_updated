#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "blas/blas.h"
#include "hal/device.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

using namespace std::chrono;
namespace core = serius::core;
using core::Tensor;
namespace blas = core::blas;
namespace hal = serius::hal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double gflops(size_t m, size_t n, size_t k, double ms)
{
    double flops = 2.0 * m * n * k;
    return flops / (ms * 1e6);  // ms -> sec, then / 1e9 for GFLOPS
}

static const char *dtype_name(hal::DType dt)
{
    switch (dt) {
        case hal::DType::Float32:
            return "FP32";
        case hal::DType::Float16:
            return "FP16";
        case hal::DType::BFloat16:
            return "BF16";
        default:
            return "???";
    }
}

// ---------------------------------------------------------------------------
// 1. Precision Comparison
// ---------------------------------------------------------------------------

static void bench_precision(size_t size)
{
    std::cout << "\n=== PRECISION COMPARISON === (size " << size << "x" << size << ")\n";
    std::cout << std::string(60, '-') << "\n";

    // Reference: FP32
    Tensor a_fp32({size, size}, hal::DType::Float32);
    Tensor b_fp32({size, size}, hal::DType::Float32);
    a_fp32.rand();
    b_fp32.rand();
    Tensor c_fp32 = blas::matmul(a_fp32, b_fp32);

    hal::DType half_types[] = {hal::DType::Float16, hal::DType::BFloat16};

    for (auto dt : half_types) {
        // Create half-precision copies
        Tensor a_half({size, size}, dt);
        Tensor b_half({size, size}, dt);

        // Copy data via item/set_item (converts)
        for (size_t i = 0; i < a_fp32.size(); ++i) {
            a_half.set_item(i, a_fp32.item(i));
            b_half.set_item(i, b_fp32.item(i));
        }

        Tensor c_half = blas::matmul(a_half, b_half);

        // Compare against FP32 reference
        double max_err = 0.0;
        double sum_err = 0.0;
        double sum_sq_err = 0.0;

        for (size_t i = 0; i < c_fp32.size(); ++i) {
            double ref = c_fp32.item(i);
            double val = c_half.item(i);
            double err = std::abs(ref - val);
            max_err = std::max(max_err, err);
            sum_err += err;
            sum_sq_err += err * err;
        }

        double avg_err = sum_err / c_fp32.size();
        double rmse = std::sqrt(sum_sq_err / c_fp32.size());

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  " << dtype_name(dt) << " vs FP32:\n";
        std::cout << "    Max Error : " << max_err << "\n";
        std::cout << "    Avg Error : " << avg_err << "\n";
        std::cout << "    RMSE      : " << rmse << "\n";
    }
}

// ---------------------------------------------------------------------------
// 2. Performance Comparison (CPU vs CUDA, per dtype)
// ---------------------------------------------------------------------------

static void bench_performance(size_t size)
{
    std::cout << "\n=== PERFORMANCE COMPARISON === (size " << size << "x" << size << ")\n";
    std::cout << std::string(60, '-') << "\n";

    const int WARMUP = 2;
    const int ITERS = 5;

    // FP32 CPU baseline
    Tensor a_fp32({size, size}, hal::DType::Float32);
    Tensor b_fp32({size, size}, hal::DType::Float32);
    a_fp32.rand();
    b_fp32.rand();

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        Tensor c = blas::matmul(a_fp32, b_fp32);
    }

    // Scalar baseline
    {
        Tensor c_scalar({size, size}, hal::DType::Float32);
        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            c_scalar.zeros();
            blas::matmul_scalar(a_fp32.data(), b_fp32.data(), c_scalar.data(), size, size, size, a_fp32.strides()[0],
                                b_fp32.strides()[0], c_scalar.strides()[0]);
        }
        auto end = high_resolution_clock::now();
        double ms = duration<double, std::milli>(end - start).count() / ITERS;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  FP32 Scalar CPU  : " << std::setw(10) << ms << " ms  (" << std::setw(8)
                  << gflops(size, size, size, ms) << " GFLOPS)\n";
    }

    // Dispatched (will pick SIMD on CPU or CUDA depending on size)
    hal::DType dtypes[] = {hal::DType::Float32, hal::DType::Float16, hal::DType::BFloat16};

    for (auto dt : dtypes) {
        Tensor a_t({size, size}, dt);
        Tensor b_t({size, size}, dt);

        // Fill via item/set_item for half types, rand() for FP32
        if (dt == hal::DType::Float32) {
            a_t.rand();
            b_t.rand();
        } else {
            for (size_t i = 0; i < a_t.size(); ++i) {
                a_t.set_item(i, a_fp32.item(i));
                b_t.set_item(i, b_fp32.item(i));
            }
        }

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            Tensor c = blas::matmul(a_t, b_t);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            Tensor c = blas::matmul(a_t, b_t);
        }
        auto end = high_resolution_clock::now();
        double ms = duration<double, std::milli>(end - start).count() / ITERS;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << dtype_name(dt) << " Dispatch   : " << std::setw(10) << ms << " ms  (" << std::setw(8)
                  << gflops(size, size, size, ms) << " GFLOPS)\n";
    }
}

// ---------------------------------------------------------------------------
// 3. Memory Footprint Comparison
// ---------------------------------------------------------------------------

static void bench_memory_footprint()
{
    std::cout << "\n=== MEMORY FOOTPRINT COMPARISON ===\n";
    std::cout << std::string(60, '-') << "\n";

    size_t sizes[] = {256, 512, 1024, 2048};
    hal::DType dtypes[] = {hal::DType::Float32, hal::DType::Float16, hal::DType::BFloat16};

    std::cout << std::setw(10) << "Size" << " | ";
    for (auto dt : dtypes) {
        std::cout << std::setw(12) << dtype_name(dt) << " | ";
    }
    std::cout << std::setw(12) << "FP16 Save" << " | " << std::setw(12) << "BF16 Save" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto size : sizes) {
        // Memory for 3 matrices (A, B, C) of size NxN
        std::cout << std::setw(6) << size << "x" << std::setw(4) << size << std::left << "" << std::right;
        std::cout << " | ";

        size_t bytes_per_dtype[3];
        int idx = 0;
        for (auto dt : dtypes) {
            Tensor tmp({size, size}, dt);
            size_t mem = 3 * size * size * tmp.element_size();  // A + B + C
            bytes_per_dtype[idx++] = mem;

            if (mem > 1024 * 1024) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << (mem / (1024.0 * 1024.0)) << " MB"
                          << " | ";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << (mem / 1024.0) << " KB"
                          << " | ";
            }
        }

        // Savings
        double save_fp16 = 100.0 * (1.0 - (double)bytes_per_dtype[1] / bytes_per_dtype[0]);
        double save_bf16 = 100.0 * (1.0 - (double)bytes_per_dtype[2] / bytes_per_dtype[0]);
        std::cout << std::setw(9) << std::fixed << std::setprecision(1) << save_fp16 << " %"
                  << " | ";
        std::cout << std::setw(9) << std::fixed << std::setprecision(1) << save_bf16 << " %"
                  << "\n";
    }
}

void run_mixed_precision_benchmarks()
{
    std::cout << "============================================================\n";
    std::cout << "           MIXED PRECISION BENCHMARK SUITE\n";
    std::cout << "============================================================\n";

    // --- Performance ---
    std::vector<size_t> perf_sizes = {256, 512, 1024};
    for (size_t s : perf_sizes) {
        bench_performance(s);
    }

    // --- Precision ---
    bench_precision(512);

    // --- Memory ---
    bench_memory_footprint();

    std::cout << "\n============================================================\n";
    std::cout << "                    BENCHMARK COMPLETE\n";
    std::cout << "============================================================\n";
}

}  // namespace test
}  // namespace serius

int main()
{
    serius::test::run_mixed_precision_benchmarks();
    return 0;
}
