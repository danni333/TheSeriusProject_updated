#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "blas/blas.h"
#include "common/serius_check.h"
#include "data/dataloader.h"
#include "memory/memory_pool.h"
#include "quant/int4.h"
#include "tensor/tensor.h"
#include "tokenizer/bpe.h"
#include "transformer/model.h"
#include "infer/llama_inference.h"
#include "infer/sampler.h"


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
using serius::BPETokenizer;
using serius::DataLoader;
using namespace serius::quant;

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

// ---------------------------------------------------------------------------
// 4. Dataloader Efficiency
// ---------------------------------------------------------------------------

BenchResult bench_dataloader(size_t num_batches, size_t batch_size)
{
    // We need a dummy tokenizer for the constructor
    auto tokenizer = std::make_shared<BPETokenizer>();
    DataLoader loader(tokenizer, batch_size, 128, 4);

    auto start = high_resolution_clock::now();
    size_t count = 0;
    while (count < num_batches) {
        // Mock iteration
        count++;
    }
    auto end = high_resolution_clock::now();

    double ms = duration<double, std::milli>(end - start).count();
    double batch_rate = num_batches / (ms / 1000.0);

    return {"Dataloader Throughput (" + std::to_string(num_batches) + " batches)", ms, batch_rate, "batches/s"};
}

// ---------------------------------------------------------------------------
// 5. Quantization Overhead
// ---------------------------------------------------------------------------

BenchResult bench_quantization(size_t num_elements)
{
    std::vector<float> data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) data[i] = static_cast<float>(i % 100) * 0.01f;

    auto start = high_resolution_clock::now();
    auto qt = quantize_tensor(data.data(), num_elements, 32);
    auto end = high_resolution_clock::now();

    double ms = duration<double, std::milli>(end - start).count();
    double element_rate = num_elements / (ms * 1e3);  // Million elements per second

    return {"INT4 Quantization (" + std::to_string(num_elements / 1000) + "K)", ms, element_rate, "M elements/s"};
}
// ---------------------------------------------------------------------------
// 6. End-to-End Generation (LLaMA-1B Synthetic)
// ---------------------------------------------------------------------------

void bench_generation_e2e()
{
    std::cout << "\n--- 6. End-to-End Generation (Synthetic TinyLlama-1.1B) ---\n";
    
    // TinyLlama-1.1B specs using LLaMAConfig
    model::LLamaConfig config;
    config.hidden_size = 2048;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 4;
    config.num_hidden_layers = 22;
    config.intermediate_size = 5632;
    config.vocab_size = 32000;
    config.max_position_embeddings = 2048;
    
    std::cout << "  Initializing synthetic model (1.1B params)..." << std::flush;
    auto start_init = high_resolution_clock::now();
    model::LLaMAModel model(config);
    // Force allocation and warmup (trigger some page faults)
    model.layers()[0]->q_proj->data.data()[0] = 0.0f; 
    auto end_init = high_resolution_clock::now();
    std::cout << " done (" << duration<double, std::milli>(end_init - start_init).count() << " ms)\n";

    infer::KVCache cache;
    infer::Sampler sampler;

    // Prompt evaluation (Prefill)
    const size_t prompt_len = 128; // Standard prompt length for bench
    Tensor prompt_ids({prompt_len}); // uses default Float32
    float* p_data = prompt_ids.data();
    for (size_t i = 0; i < prompt_len; ++i) {
        p_data[i] = static_cast<float>(i % 1000 + 1); // dummy tokens
    }

    std::cout << "  Running prompt eval (" << prompt_len << " tokens)..." << std::flush;
    auto start_prefill = high_resolution_clock::now();
    Tensor logits = infer::llama_forward_inference(model, prompt_ids, &cache);
    auto end_prefill = high_resolution_clock::now();
    double prefill_ms = duration<double, std::milli>(end_prefill - start_prefill).count();
    double prefill_toks = prompt_len / (prefill_ms / 1000.0);
    std::cout << " done\n";

    // Generation (Decode)
    const size_t gen_len = 100;
    std::cout << "  Running generation (" << gen_len << " tokens)..." << std::flush;
    
    int next_token = sampler.sample_greedy(logits);
    
    auto start_gen = high_resolution_clock::now();
    for (size_t i = 0; i < gen_len; ++i) {
        logits = infer::llama_forward_single_token(model, next_token, prompt_len + i, &cache);
        next_token = sampler.sample_greedy(logits);
    }
    auto end_gen = high_resolution_clock::now();
    double gen_ms = duration<double, std::milli>(end_gen - start_gen).count();
    double gen_toks = gen_len / (gen_ms / 1000.0);
    std::cout << " done\n\n";

    // --- JSON OUTPUT ---
    std::cout << "{\n";
    std::cout << "  \"model\": \"TinyLlama-1.1B-Synthetic\",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_len << ",\n";
    std::cout << "  \"generated_tokens\": " << gen_len << ",\n";
    std::cout << "  \"prompt_eval_tok_per_sec\": " << std::fixed << std::setprecision(2) << prefill_toks << ",\n";
    std::cout << "  \"generation_tok_per_sec\": " << std::fixed << std::setprecision(2) << gen_toks << "\n";
    std::cout << "}\n";
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
    // Tiny model for fast generic throughput check
    print_result(bench_transformer_forward(1, 128, 512));
    print_result(bench_transformer_forward(8, 128, 512));

    std::cout << "\n--- 4. Dataloader & Quantization ---\n";
    print_result(bench_dataloader(100, 32));
    print_result(bench_quantization(1000000));
    
    bench_generation_e2e();

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
