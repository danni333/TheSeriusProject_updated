#include "cuda_backend.h"

#ifdef SERIUS_USE_CUDA

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


#define CHECK_CUDA(call)                                                                             \
    do {                                                                                             \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" \
                      << cudaGetErrorString(err) << "\"" << std::endl;                               \
            throw std::runtime_error("CUDA error");                                                  \
        }                                                                                            \
    } while (0)

namespace serius {
namespace hal {

void *CudaBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void CudaBackend::free(void *ptr)
{
    if (!ptr) return;
    CHECK_CUDA(cudaFree(ptr));
}

void CudaBackend::memcpy_h2d(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
}

void CudaBackend::memcpy_d2h(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
}

void CudaBackend::memcpy_d2d(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
}

bool CudaBackend::is_available() const
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return false;
    return count > 0;
}

static CudaBackend *g_cuda_backend = nullptr;

IBackend *get_cuda_backend()
{
    if (!g_cuda_backend) {
        g_cuda_backend = new CudaBackend();
    }
    return g_cuda_backend;
}

}  // namespace hal
}  // namespace serius

#endif  // SERIUS_USE_CUDA
