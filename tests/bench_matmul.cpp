#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "blas/blas.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

using namespace std::chrono;
namespace core = serius::core;
using core::Tensor;
namespace blas = core::blas;

void run_matmul_benchmark(size_t size)
{
    std::cout << "Benchmarking matrix size: " << size << "x" << size << "\n";

    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C_naive({size, size});
    Tensor C_simd({size, size});

    A.rand();
    B.rand();

    // Warmup memory
    C_naive.zeros();
    C_simd.zeros();

    // --- Naive Benchmark ---
    auto start_naive = high_resolution_clock::now();
    blas::matmul_scalar(A.data(), B.data(), C_naive.data(), size, size, size, A.strides()[0], B.strides()[0],
                        C_naive.strides()[0]);
    auto end_naive = high_resolution_clock::now();
    duration<double, std::milli> time_naive = end_naive - start_naive;

    // --- SIMD Benchmark ---
    auto start_simd = high_resolution_clock::now();
    blas::matmul(A, B, C_simd);
    auto end_simd = high_resolution_clock::now();
    duration<double, std::milli> time_simd = end_simd - start_simd;

    // --- Results ---
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Naive / Scalar : " << time_naive.count() << " ms\n";
    std::cout << "  SIMD (Dispatch): " << time_simd.count() << " ms\n";
    std::cout << "  Speedup        : " << time_naive.count() / time_simd.count() << "x\n";
    std::cout << std::string(40, '-') << "\n";
}

void run_matmul_benchmarks()
{
    std::cout << "--- MATMUL BENCHMARK ---\n\n";

    auto info = blas::get_cpu_info();
    std::cout << "CPU Info:\n";
    std::cout << "  AVX2 Support:   " << (info.has_avx2 ? "Yes" : "No") << "\n";
    std::cout << "  AVX512 Support: " << (info.has_avx512f ? "Yes" : "No") << "\n";
    std::cout << std::string(40, '-') << "\n";

    std::vector<size_t> sizes = {256, 512, 1024};
    for (size_t s : sizes) {
        run_matmul_benchmark(s);
    }
}

}  // namespace test
}  // namespace serius

int main()
{
    serius::test::run_matmul_benchmarks();
    return 0;
}
