#include "cuda_backend.h"

#ifdef SERIUS_USE_CUDA

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cuda_common.h"

namespace serius {
namespace hal {

CudaBackend::~CudaBackend()
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto &kv : free_lists_) {
        for (void *p : kv.second) {
            cudaFree(p);
        }
    }
    free_lists_.clear();
    alloc_sizes_.clear();
    cached_bytes_ = 0;
}

void *CudaBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
    const size_t sz = (size + kAllocAlign - 1) & ~(kAllocAlign - 1);

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = free_lists_.find(sz);
        if (it != free_lists_.end() && !it->second.empty()) {
            void *ptr = it->second.back();
            it->second.pop_back();
            alloc_sizes_[ptr] = sz;
            return ptr;
        }
    }

    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, sz));
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        alloc_sizes_[ptr] = sz;
    }
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

    size_t sz = 0;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = alloc_sizes_.find(ptr);
        if (it == alloc_sizes_.end()) {
            // Not allocated through this backend (or already freed): fall back
            // to a direct cudaFree is unsafe without the size, so try cudaFree anyway.
            cudaFree(ptr);
            return;
        }
        sz = it->second;
        alloc_sizes_.erase(it);

        if (cached_bytes_ + sz <= kCacheLimit) {
            free_lists_[sz].push_back(ptr);
            cached_bytes_ += sz;
            return;
        }
    }
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
    // SERIUS_NO_GPU forces the CPU path even in CUDA builds: without this,
    // BLAS dispatch would still offload large matmuls to the GPU and leave
    // host buffers stale for CPU readers.
    if (std::getenv("SERIUS_NO_GPU") != nullptr) return false;
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
