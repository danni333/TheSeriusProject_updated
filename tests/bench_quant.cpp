#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "quant/awq.h"
#include "quant/conversions.h"
#include "quant/dequant_matmul.h"
#include "quant/fp8.h"
#include "quant/gptq.h"
#include "quant/int4.h"

using namespace serius::quant;

static double now_ms()
{
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// ============================================================
// bench_int4_vs_fp32: Speedup of INT4 dequant matmul vs FP32 matmul
// ============================================================
static void bench_int4_vs_fp32()
{
    printf("\n=== bench_int4_vs_fp32 ===\n");

    const int M = 32, N = 128, K = 256;
    const int block_size = 32;
    const int iterations = 20;
    int blocks_per_row = (K + block_size - 1) / block_size;

    // Create weight matrix
    std::vector<float> W(N * K);
    for (int i = 0; i < N * K; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.01f) * 2.0f;
    }

    // Quantize weights to INT4
    std::vector<float> scales(N * blocks_per_row);
    std::vector<uint8_t> zeros(N * blocks_per_row);
    std::vector<uint8_t> W_q4((N * K + 1) / 2);

    for (int j = 0; j < N; ++j) {
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const float *block_data = W.data() + j * K + blk * block_size;
            int count = std::min(block_size, K - blk * block_size);
            QuantizedBlock qb = quantize_block(block_data, count);
            scales[j * blocks_per_row + blk] = qb.scale;
            zeros[j * blocks_per_row + blk] = qb.zero_point;
            for (int p = 0; p < (count + 1) / 2; ++p) {
                W_q4[(j * K + blk * block_size) / 2 + p] = qb.packed[p];
            }
        }
    }

    // Input matrix
    std::vector<float> X(M * K);
    for (int i = 0; i < M * K; ++i) {
        X[i] = std::cos(static_cast<float>(i) * 0.02f);
    }

    std::vector<float> Y(M * N);

    // Benchmark FP32 matmul
    double t0 = now_ms();
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += X[i * K + k] * W[j * K + k];
                }
                Y[i * N + j] = sum;
            }
        }
    }
    double fp32_ms = (now_ms() - t0) / iterations;

    // Benchmark INT4 dequant matmul
    t0 = now_ms();
    for (int iter = 0; iter < iterations; ++iter) {
        dequant_matmul(W_q4.data(), scales.data(), zeros.data(),
                       X.data(), Y.data(), M, N, K, block_size);
    }
    double int4_ms = (now_ms() - t0) / iterations;

    printf("  Matrix: X[%d,%d] * W[%d,%d]^T\n", M, K, N, K);
    printf("  FP32 matmul:         %.3f ms\n", fp32_ms);
    printf("  INT4 dequant matmul: %.3f ms\n", int4_ms);
    printf("  Speedup: %.2fx\n", fp32_ms / int4_ms);

    // Memory comparison
    size_t fp32_bytes = static_cast<size_t>(N * K) * sizeof(float);
    size_t int4_bytes = W_q4.size() + scales.size() * sizeof(float) + zeros.size();
    printf("  FP32 weight memory: %zu bytes\n", fp32_bytes);
    printf("  INT4 weight memory: %zu bytes (%.1fx compression)\n",
           int4_bytes, static_cast<double>(fp32_bytes) / static_cast<double>(int4_bytes));
}

// ============================================================
// bench_fp8_vs_fp32: FP8 conversion throughput
// ============================================================
static void bench_fp8_vs_fp32()
{
    printf("\n=== bench_fp8_vs_fp32 ===\n");

    const size_t N = 1024 * 1024;  // 1M elements
    const int iterations = 10;

    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.001f) * 200.0f;
    }

    // FP16 roundtrip
    std::vector<uint16_t> fp16_buf(N);
    std::vector<float> fp16_out(N);
    double t0 = now_ms();
    for (int iter = 0; iter < iterations; ++iter) {
        convert_float_to_half(data.data(), fp16_buf.data(), N);
        convert_half_to_float(fp16_buf.data(), fp16_out.data(), N);
    }
    double fp16_ms = (now_ms() - t0) / iterations;

    // FP8 E4M3 roundtrip
    std::vector<uint8_t> fp8_buf(N);
    std::vector<float> fp8_out(N);
    t0 = now_ms();
    for (int iter = 0; iter < iterations; ++iter) {
        convert_float_to_fp8_e4m3(data.data(), fp8_buf.data(), N);
        convert_fp8_e4m3_to_float(fp8_buf.data(), fp8_out.data(), N);
    }
    double fp8_e4m3_ms = (now_ms() - t0) / iterations;

    // FP8 E5M2 roundtrip
    t0 = now_ms();
    for (int iter = 0; iter < iterations; ++iter) {
        convert_float_to_fp8_e5m2(data.data(), fp8_buf.data(), N);
        convert_fp8_e5m2_to_float(fp8_buf.data(), fp8_out.data(), N);
    }
    double fp8_e5m2_ms = (now_ms() - t0) / iterations;

    printf("  Elements: %zu\n", N);
    printf("  FP16 roundtrip: %.3f ms\n", fp16_ms);
    printf("  FP8 E4M3 roundtrip: %.3f ms\n", fp8_e4m3_ms);
    printf("  FP8 E5M2 roundtrip: %.3f ms\n", fp8_e5m2_ms);
    printf("  Memory: FP32=%zu B, FP16=%zu B, FP8=%zu B\n",
           N * 4, N * 2, N * 1);
}

// ============================================================
// bench_memory: Memory usage comparison for model weights
// ============================================================
static void bench_memory()
{
    printf("\n=== bench_memory (model weight sizes) ===\n");

    // Simulate 125M parameter model
    size_t params_125m = 125'000'000;
    printf("  Model: 125M parameters\n");
    printf("  FP32: %.1f MB\n", static_cast<double>(params_125m * 4) / (1024.0 * 1024.0));
    printf("  FP16: %.1f MB\n", static_cast<double>(params_125m * 2) / (1024.0 * 1024.0));
    printf("  FP8:  %.1f MB\n", static_cast<double>(params_125m * 1) / (1024.0 * 1024.0));
    printf("  INT4: %.1f MB (+ scales/zeros overhead)\n",
           static_cast<double>(params_125m / 2) / (1024.0 * 1024.0));

    // 7B parameter model
    size_t params_7b = 7'000'000'000ULL;
    printf("\n  Model: 7B parameters\n");
    printf("  FP32: %.1f GB\n", static_cast<double>(params_7b * 4) / (1024.0 * 1024.0 * 1024.0));
    printf("  FP16: %.1f GB\n", static_cast<double>(params_7b * 2) / (1024.0 * 1024.0 * 1024.0));
    printf("  FP8:  %.1f GB\n", static_cast<double>(params_7b * 1) / (1024.0 * 1024.0 * 1024.0));
    printf("  INT4: %.1f GB (+ scales/zeros overhead)\n",
           static_cast<double>(params_7b / 2) / (1024.0 * 1024.0 * 1024.0));
}

// ============================================================
// bench_awq_vs_gptq: Compare AWQ and GPTQ quantization quality
// ============================================================
static void bench_awq_vs_gptq()
{
    printf("\n=== bench_awq_vs_gptq (quality comparison) ===\n");

    const size_t rows = 64, cols = 128, num_tokens = 256;
    const size_t block_size = 32;

    // Generate realistic weight matrix
    std::vector<float> W(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.01f) * 2.0f;
    }

    // Generate calibration data
    std::vector<float> activations(num_tokens * cols);
    for (size_t i = 0; i < num_tokens * cols; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.005f) * 3.0f;
    }

    // Naive INT4
    QuantizedTensor naive_qt = quantize_tensor(W.data(), rows * cols, block_size);
    std::vector<float> naive_out(rows * cols);
    dequantize_tensor(naive_qt, naive_out.data());

    float naive_mse = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float d = naive_out[i] - W[i];
        naive_mse += d * d;
    }
    naive_mse /= static_cast<float>(rows * cols);

    // AWQ
    double t0 = now_ms();
    ActivationStats stats = collect_activation_stats(activations.data(), num_tokens, cols);
    std::vector<float> awq_scales = compute_awq_scales(stats, 0.5f);
    AWQQuantizedWeight awq = awq_quantize(W.data(), rows, cols, awq_scales, block_size);
    double awq_ms = now_ms() - t0;

    std::vector<float> awq_out(rows * cols);
    awq_dequantize(awq, awq_out.data());

    float awq_mse = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float d = awq_out[i] - W[i];
        awq_mse += d * d;
    }
    awq_mse /= static_cast<float>(rows * cols);

    // GPTQ
    t0 = now_ms();
    std::vector<float> H = compute_hessian(activations.data(), num_tokens, cols);
    GPTQConfig gptq_config;
    gptq_config.block_size = static_cast<size_t>(block_size);
    gptq_config.group_size = 32;
    gptq_config.damp_percent = 0.01f;
    GPTQQuantizedWeight gptq = gptq_quantize(W.data(), rows, cols, H, gptq_config);
    double gptq_ms = now_ms() - t0;

    std::vector<float> gptq_out(rows * cols);
    gptq_dequantize(gptq, gptq_out.data());

    float gptq_mse = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float d = gptq_out[i] - W[i];
        gptq_mse += d * d;
    }
    gptq_mse /= static_cast<float>(rows * cols);

    printf("  Weight matrix: [%zu, %zu], block_size=%zu\n", rows, cols, block_size);
    printf("  Naive INT4 MSE: %.6f\n", naive_mse);
    printf("  AWQ MSE:        %.6f (quantize time: %.1f ms)\n", awq_mse, awq_ms);
    printf("  GPTQ MSE:       %.6f (quantize time: %.1f ms)\n", gptq_mse, gptq_ms);
}

int main()
{
    printf("TheSeriusProject Quantization Benchmarks\n");
    printf("========================================\n");

    bench_int4_vs_fp32();
    bench_fp8_vs_fp32();
    bench_memory();
    bench_awq_vs_gptq();

    printf("\nDone.\n");
    return 0;
}
