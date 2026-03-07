#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "blas/blas.h"
#include "hal/device.h"
#include "tensor/tensor.h"

#ifdef SERIUS_USE_CUDA
#include "blas/matmul_cuda.h"
#endif

using namespace serius;

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

int main()
{
#ifdef SERIUS_USE_CUDA
    hal::IBackend *cuda = hal::get_backend(hal::DeviceType::CUDA);
    if (!cuda || !cuda->is_available()) {
        std::cout << "CUDA backend not available, skipping CUDA tests.\n";
        return 0;
    }

    test_cuda_matmul_integration();
#else
    std::cout << "Compiled without SERIUS_USE_CUDA macro. Skipping CUDA integration tests.\n";
#endif

    return 0;
}
