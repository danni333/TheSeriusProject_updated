#include "cuda_backend.h"

#ifdef SERIUS_USE_CUDA

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "cuda_common.h"

namespace serius {
namespace hal {

void *CudaBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void *CudaBackend::malloc_managed(size_t size)
{
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    CHECK_CUDA(cudaMallocManaged(&ptr, size));
    return ptr;
}

void *CudaBackend::malloc_pinned(size_t size)
{
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}

void CudaBackend::free(void *ptr)
{
    if (!ptr) return;
    CHECK_CUDA(cudaFree(ptr));
}

void CudaBackend::free_pinned(void *ptr)
{
    if (!ptr) return;
    CHECK_CUDA(cudaFreeHost(ptr));
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

void *CudaBackend::create_stream()
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return static_cast<void *>(stream);
}

void CudaBackend::destroy_stream(void *stream)
{
    if (stream) {
        CHECK_CUDA(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
    }
}

void CudaBackend::sync_stream(void *stream)
{
    if (stream) {
        CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }
}

void CudaBackend::memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
    }
}

void CudaBackend::memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
    }
}

void CudaBackend::memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream)
{
    if (dst && src && size > 0) {
        CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
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
