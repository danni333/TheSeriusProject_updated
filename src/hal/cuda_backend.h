#pragma once

#include "device.h"

#ifdef SERIUS_USE_CUDA

#include <mutex>
#include <unordered_map>
#include <vector>

namespace serius {
namespace hal {

class CudaBackend : public IBackend
{
   public:
    CudaBackend() = default;
    ~CudaBackend() override;

    DeviceType get_type() const override { return DeviceType::CUDA; }

    void *malloc(size_t size) override;
    void *malloc_pinned(size_t size) override;
    void *malloc_managed(size_t size) override;
    void free(void *ptr) override;
    void free_pinned(void *ptr) override;

    void memcpy_h2d(void *dst, const void *src, size_t size) override;
    void memcpy_d2h(void *dst, const void *src, size_t size) override;
    void memcpy_d2d(void *dst, const void *src, size_t size) override;

    // Stream Management
    void *create_stream() override;
    void destroy_stream(void *stream) override;
    void sync_stream(void *stream) override;

    // Async Memory Management
    void memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream) override;
    void memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream) override;
    void memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream) override;

    /// cudaFree every block sitting in the free-list cache (used after large
    /// one-time reallocations so VRAM is actually returned to the driver).
    void flush_cache() override;


    bool is_available() const override;

   private:
    // Simple size-class free-list cache: avoids cudaMalloc/cudaFree churn for
    // the many small temporary tensors created per inference step.
    static constexpr size_t kAllocAlign = 256;
    static constexpr size_t kCacheLimit = size_t(1) << 30;  // 1 GiB

    std::mutex cache_mutex_;
    std::unordered_map<size_t, std::vector<void *>> free_lists_;
    std::unordered_map<void *, size_t> alloc_sizes_;
    size_t cached_bytes_ = 0;
};

IBackend *get_cuda_backend();

}  // namespace hal
}  // namespace serius

#endif  // SERIUS_USE_CUDA
