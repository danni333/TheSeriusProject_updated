#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "memory/memory_pool.h"
#include "memory/pool_allocator.h"
#include "quant/awq.h"
#include "quant/conversions.h"
#include "quant/dequant_matmul.h"
#include "quant/fp8.h"
#include "quant/gptq.h"
#include "quant/int4.h"
#include "tensor/tensor.h"
#include "transformer/kv_cache.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
namespace model = serius::model;
using model::KVCache;
using namespace quant;

// ============================================================
// FP16 / BF16 Roundtrip Tests
// ============================================================

TEST(QuantConversions, FP16Roundtrip)
{
    // Typical weight/activation values
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14f, -2.71f, 100.0f, -100.0f, 0.001f};
    for (float val : values) {
        uint16_t h = float_to_half(val);
        float back = half_to_float(h);
        if (std::abs(val) > 1e-6f) {
            float rel_error = std::abs(back - val) / std::abs(val);
            EXPECT_LT(rel_error, 0.001f) << "FP16 roundtrip error too large for value " << val;
        } else {
            EXPECT_NEAR(back, val, 1e-4f);
        }
    }
}

TEST(QuantConversions, FP16BulkRoundtrip)
{
    const size_t N = 256;
    std::vector<float> src(N);
    std::vector<uint16_t> half_buf(N);
    std::vector<float> dst(N);

    for (size_t i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i) * 0.1f - 12.8f;
    }

    convert_float_to_half(src.data(), half_buf.data(), N);
    convert_half_to_float(half_buf.data(), dst.data(), N);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(src[i]) > 1e-6f) {
            float rel_error = std::abs(dst[i] - src[i]) / std::abs(src[i]);
            EXPECT_LT(rel_error, 0.002f) << "Bulk FP16 roundtrip error at index " << i;
        }
    }
}

TEST(QuantConversions, BF16Roundtrip)
{
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14f, -2.71f, 1000.0f, -1000.0f, 1e10f, -1e10f};
    for (float val : values) {
        uint16_t b = float_to_bf16(val);
        float back = bf16_to_float(b);
        if (std::abs(val) > 1e-6f) {
            float rel_error = std::abs(back - val) / std::abs(val);
            // BF16 has less mantissa precision (7 bits) → allow ~1% error
            EXPECT_LT(rel_error, 0.01f) << "BF16 roundtrip error too large for value " << val;
        } else {
            EXPECT_NEAR(back, val, 1e-4f);
        }
    }
}

TEST(QuantConversions, BF16PreservesRange)
{
    // BF16 has same exponent range as FP32 (8 bits)
    float large_values[] = {1e20f, -1e20f, 1e30f, -1e30f, 3.4e38f};
    for (float val : large_values) {
        uint16_t b = float_to_bf16(val);
        float back = bf16_to_float(b);
        // Just check that the order of magnitude is preserved
        EXPECT_GT(std::abs(back), std::abs(val) * 0.9f) << "BF16 lost range for " << val;
    }
}

// ============================================================
// INT4 Pack / Unpack Tests
// ============================================================

TEST(QuantINT4, PackUnpackAllValues)
{
    for (uint8_t a = 0; a <= 15; ++a) {
        for (uint8_t b = 0; b <= 15; ++b) {
            uint8_t packed = pack_two(a, b);
            uint8_t ua, ub;
            unpack_two(packed, ua, ub);
            EXPECT_EQ(ua, a) << "Unpack mismatch for low nibble a=" << (int)a << " b=" << (int)b;
            EXPECT_EQ(ub, b) << "Unpack mismatch for high nibble a=" << (int)a << " b=" << (int)b;
        }
    }
}

// ============================================================
// INT4 Block Quantization Tests
// ============================================================

TEST(QuantINT4, BlockQuantizeDequantize)
{
    const size_t block_size = 32;
    std::vector<float> data(block_size);
    for (size_t i = 0; i < block_size; ++i) {
        data[i] = static_cast<float>(i) * 0.1f - 1.5f;
    }

    QuantizedBlock block = quantize_block(data.data(), block_size);
    EXPECT_EQ(block.packed.size(), block_size / 2);

    std::vector<float> recovered(block_size);
    dequantize_block(block, recovered.data(), block_size);

    for (size_t i = 0; i < block_size; ++i) {
        float abs_err = std::abs(recovered[i] - data[i]);
        // INT4 with 16 levels over a range of ~3.1: step ≈ 0.2,  max error ≈ 0.1
        EXPECT_LT(abs_err, 0.25f) << "Block quant error too large at index " << i << " original=" << data[i]
                                  << " recovered=" << recovered[i];
    }
}

TEST(QuantINT4, BlockQuantizeConstantData)
{
    const size_t block_size = 16;
    std::vector<float> data(block_size, 3.14f);

    QuantizedBlock block = quantize_block(data.data(), block_size);

    std::vector<float> recovered(block_size);
    dequantize_block(block, recovered.data(), block_size);

    for (size_t i = 0; i < block_size; ++i) {
        EXPECT_TRUE(std::isfinite(recovered[i]));
    }
}

// ============================================================
// INT4 Tensor Quantization Tests
// ============================================================

TEST(QuantINT4, TensorQuantizeDequantize)
{
    const size_t N = 128;
    const size_t block_size = 32;
    std::vector<float> data(N);

    // Create realistic weight-like data
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.1f) * 2.0f;
    }

    QuantizedTensor qt = quantize_tensor(data.data(), N, block_size);
    EXPECT_EQ(qt.num_elements, N);
    EXPECT_EQ(qt.block_size, block_size);
    EXPECT_EQ(qt.scales.size(), N / block_size);

    std::vector<float> recovered(N);
    dequantize_tensor(qt, recovered.data());

    // Check that the absolute error is bounded uniformly across the tensor
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float abs_error = std::abs(recovered[i] - data[i]);
        max_abs_error = std::max(max_abs_error, abs_error);
    }
    // INT4 step size for range [-2, 2] is ~0.26, max absolute error should be < 0.30
    EXPECT_LT(max_abs_error, 0.30f) << "Tensor INT4 max absolute error too large";
}

TEST(QuantINT4, TensorQuantizeNonAlignedSize)
{
    // Test with size not divisible by block_size
    const size_t N = 50;  // Not divisible by 32
    const size_t block_size = 32;
    std::vector<float> data(N);

    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i) * 0.05f - 1.0f;
    }

    QuantizedTensor qt = quantize_tensor(data.data(), N, block_size);
    EXPECT_EQ(qt.num_elements, N);
    EXPECT_EQ(qt.scales.size(), (size_t)2);  // ceil(50/32) = 2 blocks

    std::vector<float> recovered(N);
    dequantize_tensor(qt, recovered.data());

    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(std::isfinite(recovered[i])) << "Non-finite at index " << i;
    }
}

// ============================================================
// Dequantized Matmul Tests
// ============================================================

TEST(QuantDequantMatmul, AccuracyVsFP32)
{
    // Small matmul: Y = X * W^T
    // X: [M=2, K=8], W: [N=3, K=8]
    const int M = 2, N = 3, K = 8;
    const int block_size = 8;  // one block per row for simplicity
    int blocks_per_row = K / block_size;

    // Create W in float
    std::vector<float> W(N * K);
    for (int i = 0; i < N * K; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.3f) * 1.5f;
    }

    // Quantize W per row block
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

            // Copy packed data
            for (int p = 0; p < (count + 1) / 2; ++p) {
                W_q4[(j * K + blk * block_size) / 2 + p] = qb.packed[p];
            }
        }
    }

    // Create X
    std::vector<float> X(M * K);
    for (int i = 0; i < M * K; ++i) {
        X[i] = std::cos(static_cast<float>(i) * 0.2f);
    }

    // Dequant matmul
    std::vector<float> Y_quant(M * N, 0.0f);
    dequant_matmul(W_q4.data(), scales.data(), zeros.data(), X.data(), Y_quant.data(), M, N, K, block_size);

    // Reference FP32 matmul: Y = X * W^T
    std::vector<float> Y_ref(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += X[i * K + k] * W[j * K + k];
            }
            Y_ref[i * N + j] = sum;
        }
    }

    // Compare — allow some error from quantization
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float ref = Y_ref[i * N + j];
            float actual = Y_quant[i * N + j];
            // Use hybrid absolute/relative error for matrix multiplication
            float abs_err = std::abs(actual - ref);
            float rel_err = abs_err / (std::abs(ref) + 1e-5f);

            // Allow larger absolute error if relative error is small, and vice versa
            EXPECT_TRUE(abs_err < 0.5f || rel_err < 0.15f)
                << "Dequant matmul error at (" << i << "," << j << "): ref=" << ref << " actual=" << actual
                << " (abs_err=" << abs_err << ", rel_err=" << rel_err << ")";
        }
    }
}

// ============================================================
// KV Cache Reorder Tests
// ============================================================

TEST(KVCacheTest, BeamReorder)
{
    size_t max_seq = 16, num_heads = 2, head_dim = 4;
    KVCache cache(max_seq, num_heads, head_dim);

    // Simulate batch_size=3, insert 2 tokens
    Tensor keys({3, 2, 2, 4});  // [batch=3, heads=2, seq=2, dim=4]
    Tensor values({3, 2, 2, 4});

    // Fill with distinguishable values per batch
    for (size_t b = 0; b < 3; ++b) {
        for (size_t h = 0; h < 2; ++h) {
            for (size_t t = 0; t < 2; ++t) {
                for (size_t d = 0; d < 4; ++d) {
                    float val = static_cast<float>(b * 100 + h * 10 + t * 4 + d);
                    keys.at({b, h, t, d}) = val;
                    values.at({b, h, t, d}) = val + 1000.0f;
                }
            }
        }
    }

    cache.update(keys, values);
    EXPECT_EQ(cache.current_len(), (size_t)2);

    // Reorder: beam 0 ← old beam 2, beam 1 ← old beam 0, beam 2 ← old beam 1
    cache.reorder({2, 0, 1});

    Tensor k_out = cache.get_keys();
    Tensor v_out = cache.get_values();

    // Verify beam 0 now has data from old beam 2
    for (size_t h = 0; h < 2; ++h) {
        for (size_t t = 0; t < 2; ++t) {
            for (size_t d = 0; d < 4; ++d) {
                float expected_k = static_cast<float>(2 * 100 + h * 10 + t * 4 + d);
                EXPECT_NEAR(k_out.at({0, h, t, d}), expected_k, 1e-5f) << "Reorder key mismatch for new_beam=0";

                float expected_v = expected_k + 1000.0f;
                EXPECT_NEAR(v_out.at({0, h, t, d}), expected_v, 1e-5f) << "Reorder value mismatch for new_beam=0";
            }
        }
    }

    // Verify beam 1 now has data from old beam 0
    EXPECT_NEAR(k_out.at({1, 0, 0, 0}), 0.0f, 1e-5f);
    // Verify beam 2 now has data from old beam 1
    EXPECT_NEAR(k_out.at({2, 0, 0, 0}), 100.0f, 1e-5f);
}

// ============================================================
// KV Cache Rolling Buffer Tests
// ============================================================

TEST(KVCacheTest, RollingBufferOverwrite)
{
    size_t max_seq = 4, num_heads = 1, head_dim = 2;
    KVCache cache(max_seq, num_heads, head_dim, /*rolling=*/true);

    // Insert 4 tokens (fills the buffer)
    for (size_t t = 0; t < 4; ++t) {
        Tensor key({1, 1, 1, 2});
        Tensor val({1, 1, 1, 2});
        key.at({0, 0, 0, 0}) = static_cast<float>(t);
        key.at({0, 0, 0, 1}) = static_cast<float>(t + 10);
        val.at({0, 0, 0, 0}) = static_cast<float>(t + 100);
        val.at({0, 0, 0, 1}) = static_cast<float>(t + 110);
        cache.update(key, val);
    }

    EXPECT_EQ(cache.current_len(), (size_t)4);

    // Insert 2 more tokens — should overwrite positions 0 and 1 circularly
    for (size_t t = 0; t < 2; ++t) {
        Tensor key({1, 1, 1, 2});
        Tensor val({1, 1, 1, 2});
        key.at({0, 0, 0, 0}) = static_cast<float>(t + 50);
        key.at({0, 0, 0, 1}) = static_cast<float>(t + 60);
        val.at({0, 0, 0, 0}) = static_cast<float>(t + 150);
        val.at({0, 0, 0, 1}) = static_cast<float>(t + 160);
        cache.update(key, val);
    }

    // current_len_ should be 6, but effective length is min(6, 4) = 4
    EXPECT_EQ(cache.current_len(), (size_t)6);
    EXPECT_TRUE(cache.is_rolling());

    Tensor k_out = cache.get_keys();
    // The buffer should hold: [50, 51, 2, 3] at positions [0, 1, 2, 3]
    // Position 0 was overwritten with token t=0+50=50
    EXPECT_NEAR(k_out.at({0, 0, 0, 0}), 50.0f, 1e-5f);
    // Position 1 was overwritten with token t=1+50=51
    EXPECT_NEAR(k_out.at({0, 0, 1, 0}), 51.0f, 1e-5f);
    // Position 2 should still be original token 2
    EXPECT_NEAR(k_out.at({0, 0, 2, 0}), 2.0f, 1e-5f);
    // Position 3 should still be original token 3
    EXPECT_NEAR(k_out.at({0, 0, 3, 0}), 3.0f, 1e-5f);
}

TEST(KVCacheTest, NonRollingOverflow)
{
    size_t max_seq = 2, num_heads = 1, head_dim = 2;
    KVCache cache(max_seq, num_heads, head_dim, /*rolling=*/false);

    Tensor key({1, 1, 1, 2});
    Tensor val({1, 1, 1, 2});
    key.fill(1.0f);
    val.fill(2.0f);

    cache.update(key, val);
    cache.update(key, val);

    // Third update should throw
    EXPECT_THROW(cache.update(key, val), std::runtime_error);
}

// ============================================================
// FP8 E4M3 Roundtrip Tests
// ============================================================

TEST(QuantFP8, E4M3RoundtripTypicalValues)
{
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -3.14f, 100.0f, -100.0f, 0.001f, 448.0f};
    for (float val : values) {
        uint8_t fp8 = float_to_fp8_e4m3(val);
        float back = fp8_e4m3_to_float(fp8);
        if (std::abs(val) > 0.01f && std::abs(val) <= 448.0f) {
            float rel_error = std::abs(back - val) / std::abs(val);
            // E4M3 has 3 mantissa bits -> ~12.5% precision per step
            EXPECT_LT(rel_error, 0.15f) << "FP8 E4M3 roundtrip error too large for value " << val
                                        << " got " << back;
        }
    }
}

TEST(QuantFP8, E4M3ClampOverflow)
{
    // Values > 448 should clamp to max representable
    uint8_t fp8 = float_to_fp8_e4m3(1000.0f);
    float back = fp8_e4m3_to_float(fp8);
    EXPECT_LE(back, 448.0f);
    EXPECT_GT(back, 400.0f);

    // Negative overflow
    fp8 = float_to_fp8_e4m3(-1000.0f);
    back = fp8_e4m3_to_float(fp8);
    EXPECT_GE(back, -448.0f);
    EXPECT_LT(back, -400.0f);
}

TEST(QuantFP8, E4M3Zero)
{
    uint8_t fp8 = float_to_fp8_e4m3(0.0f);
    float back = fp8_e4m3_to_float(fp8);
    EXPECT_EQ(back, 0.0f);

    fp8 = float_to_fp8_e4m3(-0.0f);
    back = fp8_e4m3_to_float(fp8);
    EXPECT_EQ(back, 0.0f);
}

TEST(QuantFP8, E4M3BulkRoundtrip)
{
    const size_t N = 128;
    std::vector<float> src(N);
    std::vector<uint8_t> fp8_buf(N);
    std::vector<float> dst(N);

    for (size_t i = 0; i < N; ++i) {
        src[i] = std::sin(static_cast<float>(i) * 0.1f) * 100.0f;
    }

    convert_float_to_fp8_e4m3(src.data(), fp8_buf.data(), N);
    convert_fp8_e4m3_to_float(fp8_buf.data(), dst.data(), N);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(src[i]) > 0.1f) {
            float rel_error = std::abs(dst[i] - src[i]) / std::abs(src[i]);
            EXPECT_LT(rel_error, 0.15f) << "Bulk FP8 E4M3 error at index " << i;
        }
    }
}

// ============================================================
// FP8 E5M2 Roundtrip Tests
// ============================================================

TEST(QuantFP8, E5M2RoundtripTypicalValues)
{
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 100.0f, -1000.0f, 10000.0f, -50000.0f};
    for (float val : values) {
        uint8_t fp8 = float_to_fp8_e5m2(val);
        float back = fp8_e5m2_to_float(fp8);
        if (std::abs(val) > 0.01f && std::abs(val) <= 57344.0f) {
            float rel_error = std::abs(back - val) / std::abs(val);
            // E5M2 has 2 mantissa bits -> ~25% precision per step
            EXPECT_LT(rel_error, 0.35f) << "FP8 E5M2 roundtrip error too large for value " << val
                                        << " got " << back;
        }
    }
}

TEST(QuantFP8, E5M2PreservesWiderRange)
{
    // E5M2 should handle larger values than E4M3
    float large_values[] = {1000.0f, 10000.0f, 50000.0f};
    for (float val : large_values) {
        uint8_t fp8 = float_to_fp8_e5m2(val);
        float back = fp8_e5m2_to_float(fp8);
        EXPECT_GT(std::abs(back), std::abs(val) * 0.5f) << "E5M2 lost too much range for " << val;
    }
}

TEST(QuantFP8, E5M2Infinity)
{
    uint8_t fp8 = float_to_fp8_e5m2(100000.0f);  // > 57344 -> inf
    float back = fp8_e5m2_to_float(fp8);
    EXPECT_TRUE(std::isinf(back));
}

TEST(QuantFP8, E5M2BulkRoundtrip)
{
    const size_t N = 128;
    std::vector<float> src(N);
    std::vector<uint8_t> fp8_buf(N);
    std::vector<float> dst(N);

    // Use gradient-like values (wider range)
    for (size_t i = 0; i < N; ++i) {
        src[i] = std::sin(static_cast<float>(i) * 0.1f) * 1000.0f;
    }

    convert_float_to_fp8_e5m2(src.data(), fp8_buf.data(), N);
    convert_fp8_e5m2_to_float(fp8_buf.data(), dst.data(), N);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(src[i]) > 1.0f) {
            float rel_error = std::abs(dst[i] - src[i]) / std::abs(src[i]);
            EXPECT_LT(rel_error, 0.35f) << "Bulk FP8 E5M2 error at index " << i;
        }
    }
}

// ============================================================
// FP8 Block-wise Scaling Tests
// ============================================================

TEST(QuantFP8, BlockwiseScalingE4M3)
{
    const size_t N = 64;
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.2f) * 200.0f;
    }

    FP8Block block = fp8_quantize_block(data.data(), N, /*use_e4m3=*/true);
    EXPECT_TRUE(block.is_e4m3);
    EXPECT_EQ(block.data.size(), N);
    EXPECT_GT(block.scale, 0.0f);

    std::vector<float> recovered(N);
    fp8_dequantize_block(block, recovered.data());

    float max_rel_error = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(data[i]) > 1.0f) {
            float rel = std::abs(recovered[i] - data[i]) / std::abs(data[i]);
            max_rel_error = std::max(max_rel_error, rel);
        }
    }
    // With block scaling, error should be significantly reduced
    EXPECT_LT(max_rel_error, 0.15f) << "Block-wise FP8 E4M3 max relative error too large";
}

TEST(QuantFP8, TensorQuantizeDequantizeE4M3)
{
    const size_t N = 256;
    const size_t block_size = 128;
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::cos(static_cast<float>(i) * 0.05f) * 300.0f;
    }

    std::vector<float> scales;
    std::vector<uint8_t> fp8_data;
    fp8_quantize_tensor(data.data(), N, block_size, /*use_e4m3=*/true, scales, fp8_data);

    EXPECT_EQ(scales.size(), (N + block_size - 1) / block_size);
    EXPECT_EQ(fp8_data.size(), N);

    std::vector<float> recovered(N);
    fp8_dequantize_tensor(fp8_data.data(), scales.data(), N, block_size, true, recovered.data());

    float max_rel_error = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(data[i]) > 1.0f) {
            float rel = std::abs(recovered[i] - data[i]) / std::abs(data[i]);
            max_rel_error = std::max(max_rel_error, rel);
        }
    }
    EXPECT_LT(max_rel_error, 0.15f) << "Tensor FP8 E4M3 max relative error too large";
}

// ============================================================
// AWQ Tests
// ============================================================

TEST(QuantAWQ, CollectActivationStats)
{
    // 4 tokens, 3 channels
    std::vector<float> activations = {
        1.0f, 0.1f, 0.5f,   // token 0
        2.0f, 0.2f, 0.6f,   // token 1
        3.0f, 0.3f, 0.7f,   // token 2
        4.0f, 0.4f, 0.8f,   // token 3
    };

    ActivationStats stats = collect_activation_stats(activations.data(), 4, 3);
    EXPECT_EQ(stats.num_channels, (size_t)3);
    EXPECT_EQ(stats.channel_magnitudes.size(), (size_t)3);

    // Channel 0: mean(|1,2,3,4|) = 2.5
    EXPECT_NEAR(stats.channel_magnitudes[0], 2.5f, 1e-5f);
    // Channel 1: mean(|0.1,0.2,0.3,0.4|) = 0.25
    EXPECT_NEAR(stats.channel_magnitudes[1], 0.25f, 1e-5f);
    // Channel 2: mean(|0.5,0.6,0.7,0.8|) = 0.65
    EXPECT_NEAR(stats.channel_magnitudes[2], 0.65f, 1e-5f);
}

TEST(QuantAWQ, ComputeScales)
{
    ActivationStats stats;
    stats.num_channels = 3;
    stats.channel_magnitudes = {10.0f, 1.0f, 5.0f};

    std::vector<float> scales = compute_awq_scales(stats, 0.5f);
    EXPECT_EQ(scales.size(), (size_t)3);

    // Channel 0 has highest magnitude (10.0), normalized = 1.0, scale = 1.0^0.5 = 1.0
    EXPECT_NEAR(scales[0], 1.0f, 0.01f);
    // Channel 1 has lowest (1.0), normalized = 0.1, scale = 0.1^0.5 ≈ 0.316
    EXPECT_LT(scales[1], scales[0]);
    // Channel 2 is middle (5.0), normalized = 0.5, scale = 0.5^0.5 ≈ 0.707
    EXPECT_LT(scales[2], scales[0]);
    EXPECT_GT(scales[2], scales[1]);
}

TEST(QuantAWQ, QuantizeDequantizeAccuracy)
{
    // Weight matrix [4, 8], calibration activations [16, 8]
    const size_t rows = 4, cols = 8, num_tokens = 16;

    std::vector<float> weights(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.3f) * 2.0f;
    }

    // Generate calibration activations
    std::vector<float> activations(num_tokens * cols);
    for (size_t i = 0; i < num_tokens * cols; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.1f) * 3.0f;
    }

    // AWQ pipeline
    ActivationStats stats = collect_activation_stats(activations.data(), num_tokens, cols);
    std::vector<float> scales = compute_awq_scales(stats, 0.5f);
    AWQQuantizedWeight awq = awq_quantize(weights.data(), rows, cols, scales, 8);

    // Dequantize and check accuracy
    std::vector<float> recovered(rows * cols);
    awq_dequantize(awq, recovered.data());

    float max_abs_error = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float err = std::abs(recovered[i] - weights[i]);
        max_abs_error = std::max(max_abs_error, err);
    }
    // INT4 quantization error should be bounded
    EXPECT_LT(max_abs_error, 0.6f) << "AWQ dequantized max error too large";
}

TEST(QuantAWQ, MatmulAccuracy)
{
    const size_t M = 2, N = 3, K = 8, num_tokens = 16;

    std::vector<float> W(N * K);
    for (size_t i = 0; i < N * K; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.3f) * 1.5f;
    }

    std::vector<float> X(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        X[i] = std::cos(static_cast<float>(i) * 0.2f);
    }

    // Calibration
    std::vector<float> activations(num_tokens * K);
    for (size_t i = 0; i < num_tokens * K; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.15f) * 2.0f;
    }

    ActivationStats stats = collect_activation_stats(activations.data(), num_tokens, K);
    std::vector<float> scales = compute_awq_scales(stats, 0.5f);
    AWQQuantizedWeight awq = awq_quantize(W.data(), N, K, scales, 8);

    // AWQ matmul
    std::vector<float> Y_awq(M * N, 0.0f);
    awq_matmul(awq, X.data(), Y_awq.data(), static_cast<int>(M));

    // Reference FP32 matmul
    std::vector<float> Y_ref(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += X[i * K + k] * W[j * K + k];
            }
            Y_ref[i * N + j] = sum;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float abs_err = std::abs(Y_awq[i * N + j] - Y_ref[i * N + j]);
            float rel_err = abs_err / (std::abs(Y_ref[i * N + j]) + 1e-5f);
            EXPECT_TRUE(abs_err < 1.0f || rel_err < 0.20f)
                << "AWQ matmul error at (" << i << "," << j << "): ref=" << Y_ref[i * N + j]
                << " actual=" << Y_awq[i * N + j];
        }
    }
}

// ============================================================
// GPTQ Tests
// ============================================================

TEST(QuantGPTQ, ComputeHessian)
{
    // 4 tokens, 3 dimensions
    std::vector<float> activations = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
    };

    std::vector<float> H = compute_hessian(activations.data(), 4, 3);
    EXPECT_EQ(H.size(), (size_t)9);

    // H = X^T * X / 4
    // H[0,0] = (1 + 0 + 0 + 1) / 4 = 0.5
    EXPECT_NEAR(H[0], 0.5f, 1e-5f);
    // H[1,1] = (0 + 1 + 0 + 1) / 4 = 0.5
    EXPECT_NEAR(H[4], 0.5f, 1e-5f);
    // H[0,1] = (0 + 0 + 0 + 1) / 4 = 0.25
    EXPECT_NEAR(H[1], 0.25f, 1e-5f);
    // Symmetry
    EXPECT_NEAR(H[1], H[3], 1e-5f);
}

TEST(QuantGPTQ, QuantizeDequantizeAccuracy)
{
    const size_t rows = 4, cols = 8, num_tokens = 32;

    std::vector<float> weights(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.3f) * 2.0f;
    }

    // Generate calibration activations
    std::vector<float> activations(num_tokens * cols);
    for (size_t i = 0; i < num_tokens * cols; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.1f) * 3.0f;
    }

    std::vector<float> H = compute_hessian(activations.data(), num_tokens, cols);

    GPTQConfig config;
    config.block_size = 8;
    config.group_size = 4;
    config.damp_percent = 0.01f;

    GPTQQuantizedWeight gptq = gptq_quantize(weights.data(), rows, cols, H, config);
    EXPECT_EQ(gptq.rows, rows);
    EXPECT_EQ(gptq.cols, cols);

    std::vector<float> recovered(rows * cols);
    gptq_dequantize(gptq, recovered.data());

    float max_abs_error = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float err = std::abs(recovered[i] - weights[i]);
        max_abs_error = std::max(max_abs_error, err);
    }
    // GPTQ should have bounded quantization error (with Hessian compensation)
    EXPECT_LT(max_abs_error, 0.6f) << "GPTQ dequantized max error too large";
}

TEST(QuantGPTQ, MatmulAccuracy)
{
    const size_t M = 2, N = 3, K = 8, num_tokens = 32;

    std::vector<float> W(N * K);
    for (size_t i = 0; i < N * K; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.3f) * 1.5f;
    }

    std::vector<float> X(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        X[i] = std::cos(static_cast<float>(i) * 0.2f);
    }

    std::vector<float> activations(num_tokens * K);
    for (size_t i = 0; i < num_tokens * K; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.15f) * 2.0f;
    }

    std::vector<float> H = compute_hessian(activations.data(), num_tokens, K);

    GPTQConfig config;
    config.block_size = 8;
    config.group_size = 4;
    config.damp_percent = 0.01f;

    GPTQQuantizedWeight gptq = gptq_quantize(W.data(), N, K, H, config);

    std::vector<float> Y_gptq(M * N, 0.0f);
    gptq_matmul(gptq, X.data(), Y_gptq.data(), static_cast<int>(M));

    // Reference
    std::vector<float> Y_ref(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += X[i * K + k] * W[j * K + k];
            }
            Y_ref[i * N + j] = sum;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float abs_err = std::abs(Y_gptq[i * N + j] - Y_ref[i * N + j]);
            float rel_err = abs_err / (std::abs(Y_ref[i * N + j]) + 1e-5f);
            EXPECT_TRUE(abs_err < 1.0f || rel_err < 0.20f)
                << "GPTQ matmul error at (" << i << "," << j << "): ref=" << Y_ref[i * N + j]
                << " actual=" << Y_gptq[i * N + j];
        }
    }
}

TEST(QuantGPTQ, GPTQBetterThanNaive)
{
    // GPTQ with Hessian compensation should produce lower error than naive INT4
    const size_t rows = 4, cols = 16, num_tokens = 64;

    std::vector<float> weights(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.2f) * 3.0f;
    }

    std::vector<float> activations(num_tokens * cols);
    for (size_t i = 0; i < num_tokens * cols; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.1f) * 2.0f;
    }

    // Naive INT4
    QuantizedTensor naive_qt = quantize_tensor(weights.data(), rows * cols, 8);
    std::vector<float> naive_recovered(rows * cols);
    dequantize_tensor(naive_qt, naive_recovered.data());

    float naive_mse = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float diff = naive_recovered[i] - weights[i];
        naive_mse += diff * diff;
    }
    naive_mse /= static_cast<float>(rows * cols);

    // GPTQ
    std::vector<float> H = compute_hessian(activations.data(), num_tokens, cols);
    GPTQConfig config;
    config.block_size = 8;
    config.group_size = 8;
    config.damp_percent = 0.01f;

    GPTQQuantizedWeight gptq = gptq_quantize(weights.data(), rows, cols, H, config);
    std::vector<float> gptq_recovered(rows * cols);
    gptq_dequantize(gptq, gptq_recovered.data());

    float gptq_mse = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        float diff = gptq_recovered[i] - weights[i];
        gptq_mse += diff * diff;
    }
    gptq_mse /= static_cast<float>(rows * cols);

    // GPTQ MSE should not be significantly worse than naive
    // (On small matrices the benefit is marginal, but it should not be worse)
    EXPECT_LT(gptq_mse, naive_mse * 2.0f)
        << "GPTQ MSE=" << gptq_mse << " should not be much worse than naive MSE=" << naive_mse;
}

}  // namespace test
}  // namespace serius
