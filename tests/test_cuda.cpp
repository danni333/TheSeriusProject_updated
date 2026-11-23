#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

#include "attention/sparsity_mask.h"
#include "blas/blas.h"
#include "hal/device.h"
#include "tensor/tensor.h"

#ifdef SERIUS_USE_CUDA
#include "blas/matmul_cuda.h"
#endif

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
namespace blas = core::blas;
namespace hal = serius::hal;

void test_cuda_matmul_integration()
{
    std::cout << "Testing CUDA matmul dispatch...\n";

    // Small tensors to test fallback to CPU
    Tensor A_small({2, 2});
    Tensor B_small({2, 2});
    A_small.fill(1.0f);
    B_small.fill(2.0f);

    Tensor C_small({2, 2});
    blas::matmul(A_small, B_small, C_small);  // Should run on CPU

    // Large tensor to force CUDA dispatch (if available)
    size_t dim = 512;
    Tensor A_large({dim, dim});
    Tensor B_large({dim, dim});
    A_large.fill(1.0f);
    B_large.fill(2.0f);

    Tensor C_large({dim, dim});
    blas::matmul(A_large, B_large, C_large);  // Should correctly dispatch to GPU if available and sync back

    // Check results
    C_large.syncToHost();
    const float *c_data = C_large.data();
    assert(std::abs(c_data[0] - (2.0f * dim)) < 1e-4 && "CUDA matmul result incorrect or sync failed");

    std::cout << "CUDA matmul dispatch test ok.\n";
}

void test_cuda_streams_parallel()
{
#ifdef SERIUS_USE_CUDA
    std::cout << "\nTesting CUDA Stream Parallelism...\n";

    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);
    if (!backend || !backend->is_available()) {
        std::cout << "Skipping CUDA stream test: CUDA backend not available.\n";
        return;
    }

    size_t dim = 2048;  // Large enough to see parallel benefits
    Tensor A1({dim, dim}); Tensor B1({dim, dim}); Tensor C1({dim, dim});
    Tensor A2({dim, dim}); Tensor B2({dim, dim}); Tensor C2({dim, dim});
    
    A1.fill(1.0f); B1.fill(1.0f);
    A2.fill(1.0f); B2.fill(1.0f);

    // Pre-sync to device to isolate kernel execution time
    A1.syncToDevice(); B1.syncToDevice(); C1.syncToDevice();
    A2.syncToDevice(); B2.syncToDevice(); C2.syncToDevice();

    // Warmup
    blas::matmul(A1, B1, C1);
    backend->sync_stream(nullptr);

    void *s1 = backend->create_stream();
    void *s2 = backend->create_stream();

    // 1. Sequential Timing
    auto start_seq = std::chrono::high_resolution_clock::now();
    blas::matmul(A1, B1, C1, s1);
    backend->sync_stream(s1);
    blas::matmul(A2, B2, C2, s2);
    backend->sync_stream(s2);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    // 2. Parallel Timing
    auto start_par = std::chrono::high_resolution_clock::now();
    blas::matmul(A1, B1, C1, s1);
    blas::matmul(A2, B2, C2, s2);
    backend->sync_stream(s1);
    backend->sync_stream(s2);
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    std::cout << "Sequential duration: " << seq_ms << " ms\n";
    std::cout << "Parallel duration:   " << par_ms << " ms\n";
    
    double speedup = seq_ms / par_ms;
    std::cout << "Stream speedup:      " << speedup << "x\n";

    // Verify results
    C1.syncToHost();
    C2.syncToHost();
    assert(std::abs(C1.data()[0] - (float)dim) < 1e-4);
    assert(std::abs(C2.data()[0] - (float)dim) < 1e-4);

    backend->destroy_stream(s1);
    backend->destroy_stream(s2);

    if (speedup >= 1.2) { 
        std::cout << "Stream parallelism verification successful!\n";
    } else {
        std::cout << "Note: Low speedup detected (" << speedup << "x). This can happen on single-engine GPUs or if compute-bound.\n";
    }
#else
    std::cout << "Skipping CUDA stream test: SERIUS_USE_CUDA not defined.\n";
#endif
}

#ifdef SERIUS_USE_CUDA
void test_block_sparse_attention_cuda()
{
    std::cout << "\nTesting block-sparse attention CUDA kernel...\n";

    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);
    if (!backend || !backend->is_available()) {
        std::cout << "Skipping block-sparse attention test: CUDA backend not available.\n";
        return;
    }

    namespace attn = serius::attention;
    const size_t seq = 8, d = 8, bs = 4;

    Tensor Q({seq, d}), K({seq, d}), V({seq, d});
    for (size_t i = 0; i < seq; ++i) {
        for (size_t j = 0; j < d; ++j) {
            const float v = static_cast<float>((i * 31 + j * 17) % 23) * 0.05f - 0.5f;
            Q.set_item(i, j, v);
            K.set_item(i, j, static_cast<float>((i * 13 + j * 7) % 19) * 0.05f - 0.4f);
            V.set_item(i, j, static_cast<float>((i * 5 + j * 11) % 17) * 0.05f - 0.3f);
        }
    }

    // CPU reference: causal masked softmax attention restricted to active
    // blocks (same semantics as the kernel: mask AND k <= q).
    auto reference = [&](const Tensor &mask) {
        Tensor ref({seq, d});
        const float scale = 1.0f / std::sqrt(static_cast<float>(d));
        const size_t nb = seq / bs;
        for (size_t q = 0; q < seq; ++q) {
            std::vector<float> scores(seq, -1e30f);
            float m = -1e30f;
            for (size_t k = 0; k <= q; ++k) {
                const bool active = mask.item((q / bs) * nb + (k / bs)) > 0.5f;
                if (!active) continue;
                float dot = 0.f;
                for (size_t j = 0; j < d; ++j) dot += Q.item(q, j) * K.item(k, j);
                scores[k] = dot * scale;
                m = std::max(m, scores[k]);
            }
            float l = 0.f;
            for (size_t k = 0; k <= q; ++k) {
                if (scores[k] <= -1e29f) continue;
                scores[k] = std::exp(scores[k] - m);
                l += scores[k];
            }
            for (size_t j = 0; j < d; ++j) {
                float acc = 0.f;
                for (size_t k = 0; k <= q; ++k) {
                    if (scores[k] <= -1e29f) continue;
                    acc += scores[k] * V.item(k, j);
                }
                ref.set_item(q, j, l > 0.f ? acc / l : 0.f);
            }
        }
        return ref;
    };

    auto check = [&](const Tensor &mask, const char *name) {
        Tensor out = attn::block_sparse_attention_cuda(Q, K, V, mask, bs, 0);
        Tensor ref = reference(mask);
        out.syncToHost();
        double max_err = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            max_err = std::max(max_err, static_cast<double>(std::abs(out[i] - ref[i])));
        }
        std::cout << name << " max abs err: " << max_err << "\n";
        assert(max_err < 1e-4 && "block_sparse_attention_cuda mismatch vs reference");
    };

    // 1. Full causal mask.
    check(attn::causal_block_mask(seq / bs), "causal");

    // 2. Sliding-window mask (each block attends only to itself): exercises
    //    skipped blocks and the causal guard inside active blocks.
    Tensor local({2, 2});
    local.fill(0.0f);
    local.set_item(0, 0, 1.0f);
    local.set_item(1, 1, 1.0f);
    check(local, "local-window");

    // 3. Sparse custom mask: q-block 1 attends only to k-block 0 (all keys
    //    are in the past, so causality does not interfere).
    Tensor past_only({2, 2});
    past_only.fill(0.0f);
    past_only.set_item(0, 0, 1.0f);
    past_only.set_item(1, 0, 1.0f);
    check(past_only, "past-only");

    std::cout << "Block-sparse attention CUDA test ok.\n";
}
#endif

}  // namespace test
}  // namespace serius

int main()
{
    std::cout << "--- CUDA INTEGRATION TESTS ---\n";
    serius::test::test_cuda_matmul_integration();
    serius::test::test_cuda_streams_parallel();
#ifdef SERIUS_USE_CUDA
    serius::test::test_block_sparse_attention_cuda();
#endif
    std::cout << "All integration tests completed.\n";
    return 0;
}
