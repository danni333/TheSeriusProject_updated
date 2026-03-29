#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

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

}  // namespace test
}  // namespace serius

int main()
{
    std::cout << "--- CUDA INTEGRATION TESTS ---\n";
    serius::test::test_cuda_matmul_integration();
    serius::test::test_cuda_streams_parallel();
    std::cout << "All integration tests completed.\n";
    return 0;
}
