#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "blas/blas.h"
#include "common/serius_check.h"
#include "data/dataloader.h"
#include "memory/memory_pool.h"
#include "tensor/tensor.h"
#include "tokenizer/bpe.h"
#include "transformer/model.h"

namespace serius {
namespace test {

using namespace std::chrono;
namespace core = serius::core;
using core::Tensor;
namespace blas = core::blas;
namespace hal = serius::hal;
namespace memory = core::memory;
namespace model = serius::model;
using model::ModelConfig;
using model::TransformerModel;

// ---------------------------------------------------------------------------
// Performance Metrics
// ---------------------------------------------------------------------------

struct BenchResult
{
    std::string name;
    double time_ms;
    double metric_value;
    std::string metric_unit;
};

void print_result(const BenchResult &res)
{
    std::cout << "  " << std::left << std::setw(30) << res.name << ": " << std::right << std::setw(10) << std::fixed
              << std::setprecision(2) << res.time_ms << " ms"
              << " (" << res.metric_value << " " << res.metric_unit << ")\n";
}

// ---------------------------------------------------------------------------
// 1. BLAS Benchmarks
// ---------------------------------------------------------------------------

BenchResult bench_matmul(size_t size)
{
    Tensor A({size, size});
    Tensor B({size, size});
    A.rand();
    B.rand();

    auto start = high_resolution_clock::now();
    Tensor C = blas::matmul(A, B);
    auto end = high_resolution_clock::now();

    double ms = duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * size * size * size) / (ms * 1e6);

    return {"MatMul " + std::to_string(size) + "x" + std::to_string(size), ms, gflops, "GFLOPS"};
}

// ---------------------------------------------------------------------------
// 2. Memory Benchmarks
// ---------------------------------------------------------------------------

BenchResult bench_memory_pool(size_t count, size_t size)
{
    memory::MemoryPool pool(count * size + 1024 * 1024);

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < count; ++i) {
        pool.allocate(size);
    }
    pool.reset();
    auto end = high_resolution_clock::now();

    double ms = duration<double, std::milli>(end - start).count();
    double gb_s = (static_cast<double>(count) * size) / (ms * 1e6);  // Simplified "rate"

    return {"MemoryPool Alloc/Reset (" + std::to_string(count) + ")", ms, gb_s, "GB/s equivalent"};
}

// ---------------------------------------------------------------------------
// 3. Transformer Throughput
// ---------------------------------------------------------------------------

BenchResult bench_transformer_forward(size_t batch, size_t seq, size_t d_model)
{
    ModelConfig config;
    config.d_model = d_model;
    config.n_heads = 8;
    config.n_layers = 6;
    config.vocab_size = 32000;
    config.max_seq_len = seq;

    TransformerModel model(config);
    Tensor tokens({batch, seq});
    for (size_t i = 0; i < tokens.size(); ++i) tokens.data()[i] = static_cast<float>(i % config.vocab_size);

    core::autograd::Node input_node(tokens, false);

    auto start = high_resolution_clock::now();
    model.forward(&input_node);
    auto end = high_resolution_clock::now();

    double ms = duration<double, std::milli>(end - start).count();
    double tokens_per_sec = (batch * seq) / (ms / 1000.0);

    return {"Transformer Forward (B=" + std::to_string(batch) + ", S=" + std::to_string(seq) + ")", ms, tokens_per_sec,
            "tok/s"};
}

void benchmark_full_pipeline()
{
    std::cout << "============================================================\n";
    std::cout << "             SERIUS UNIFIED BENCHMARK SUITE\n";
    std::cout << "============================================================\n\n";

    auto info = blas::get_cpu_info();
    std::cout << "System Info:\n";
    std::cout << "  AVX2: " << (info.has_avx2 ? "Yes" : "No") << " | AVX512: " << (info.has_avx512f ? "Yes" : "No")
              << "\n\n";

    std::cout << "--- 1. BLAS Performance ---\n";
    print_result(bench_matmul(512));
    print_result(bench_matmul(1024));

    std::cout << "\n--- 2. Memory Subsystem ---\n";
    print_result(bench_memory_pool(1000000, 64));
    print_result(bench_memory_pool(100000, 1024));

    std::cout << "\n--- 3. Transformer Throughput ---\n";
    print_result(bench_transformer_forward(1, 128, 512));
    print_result(bench_transformer_forward(8, 128, 512));

    std::cout << "\n============================================================\n";
    std::cout << "                    BENCHMARKS COMPLETE\n";
    std::cout << "============================================================\n";
}

}  // namespace test
}  // namespace serius

int main()
{
    serius::test::benchmark_full_pipeline();
    return 0;
}
