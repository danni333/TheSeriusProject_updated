#include "cpu_backend.h"
#include "fused_kernels.h"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace serius {
namespace hal {

constexpr size_t ALIGNMENT = 64;

namespace {

size_t dtype_size(DType dtype)
{
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Float16:
        case DType::BFloat16:
            return sizeof(uint16_t);
        case DType::Int8:
            return sizeof(int8_t);
        case DType::Int4:
            return sizeof(uint8_t);
        default:
            return sizeof(float);
    }
}

}  // namespace

#ifndef SERIUS_USE_CUDA
void copy_layout(void *dst, const void *src, const size_t *shape, const size_t *src_strides, size_t dims,
                 DType dtype, cudaStream_t stream)
{
    (void)stream;
    if (dst == nullptr || src == nullptr || dims == 0) {
        return;
    }

    size_t total = 1;
    for (size_t i = 0; i < dims; ++i) {
        total *= shape[i];
    }

    const size_t elem_size = dtype_size(dtype);
    const auto *src_bytes = static_cast<const uint8_t *>(src);
    auto *dst_bytes = static_cast<uint8_t *>(dst);

    for (size_t flat = 0; flat < total; ++flat) {
        size_t rem = flat;
        size_t src_offset = 0;
        for (size_t d = 0; d < dims; ++d) {
            const size_t dim_idx = dims - 1 - d;
            const size_t coord = rem % shape[dim_idx];
            rem /= shape[dim_idx];
            src_offset += coord * src_strides[dim_idx];
        }
        const size_t dst_offset = flat * elem_size;
        std::memcpy(dst_bytes + dst_offset, src_bytes + src_offset * elem_size, elem_size);
    }
}
#endif  // !SERIUS_USE_CUDA

void *CpuBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
#if defined(_MSC_VER)
    return _aligned_malloc(size, ALIGNMENT);
#else
    void *ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0) ptr = nullptr;
    return ptr;
#endif
}

void *CpuBackend::malloc_managed(size_t size)
{
    // For CPU, managed is just local heap.
    return malloc(size);
}

void *CpuBackend::malloc_pinned(size_t size)
{
    // For CPU, all memory is effectively pinned.
    return malloc(size);
}

void CpuBackend::free_pinned(void *ptr)
{
    free(ptr);
}

void CpuBackend::free(void *ptr)
{
    if (ptr != nullptr) {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }
}

void CpuBackend::memcpy_h2d(void *dst, const void *src, size_t size)
{
    // For CPU backend, host is device
    if (dst != nullptr && src != nullptr && size > 0) {
        std::memcpy(dst, src, size);
    }
}

void CpuBackend::memcpy_d2h(void *dst, const void *src, size_t size)
{
    // For CPU backend, host is device
    if (dst != nullptr && src != nullptr && size > 0) {
        std::memcpy(dst, src, size);
    }
}

void CpuBackend::memcpy_d2d(void *dst, const void *src, size_t size)
{
    if (dst != nullptr && src != nullptr && size > 0) {
        std::memcpy(dst, src, size);
    }
}

// CPU "stream": a single worker thread executing enqueued ops in order.
// sync_stream() blocks until all previously enqueued work has run.
struct CpuStream
{
    std::thread worker;
    std::queue<std::function<void()>> queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;

    explicit CpuStream()
    {
        worker = std::thread([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lk(mutex);
                    cv.wait(lk, [this] { return stop || !queue.empty(); });
                    if (stop && queue.empty()) { return; }
                    task = std::move(queue.front());
                    queue.pop();
                }
                task();
            }
        });
    }

    ~CpuStream()
    {
        {
            std::lock_guard<std::mutex> lk(mutex);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }

    void enqueue(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lk(mutex);
            queue.push(std::move(task));
        }
        cv.notify_one();
    }

    void sync()
    {
        // Barrier: enqueue a no-op and wait until the worker has drained
        // everything enqueued before it.
        std::mutex done_mtx;
        std::condition_variable done_cv;
        bool done = false;
        enqueue([&] {
            std::lock_guard<std::mutex> lk(done_mtx);
            done = true;
            done_cv.notify_one();
        });
        std::unique_lock<std::mutex> lk(done_mtx);
        done_cv.wait(lk, [&] { return done; });
    }
};

void *CpuBackend::create_stream()
{
    return new CpuStream();
}

void CpuBackend::destroy_stream(void *stream)
{
    delete static_cast<CpuStream *>(stream);
}

void CpuBackend::sync_stream(void *stream)
{
    if (stream != nullptr) {
        static_cast<CpuStream *>(stream)->sync();
    }
}

void CpuBackend::memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream)
{
    if (stream == nullptr || size == 0) {
        memcpy_h2d(dst, src, size);
        return;
    }
    static_cast<CpuStream *>(stream)->enqueue([dst, src, size] { std::memcpy(dst, src, size); });
}

void CpuBackend::memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream)
{
    if (stream == nullptr || size == 0) {
        memcpy_d2h(dst, src, size);
        return;
    }
    static_cast<CpuStream *>(stream)->enqueue([dst, src, size] { std::memcpy(dst, src, size); });
}

void CpuBackend::memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream)
{
    if (stream == nullptr || size == 0) {
        memcpy_d2d(dst, src, size);
        return;
    }
    static_cast<CpuStream *>(stream)->enqueue([dst, src, size] { std::memcpy(dst, src, size); });
}

// Global accessor singleton implementation
static CpuBackend *g_cpu_backend_instance = nullptr;

IBackend *get_backend(DeviceType type)
{
    if (type == DeviceType::CPU) {
        if (g_cpu_backend_instance == nullptr) {
            g_cpu_backend_instance = new CpuBackend();
        }
        return g_cpu_backend_instance;
    }
#ifdef SERIUS_USE_CUDA
    extern IBackend *get_cuda_backend();
    if (type == DeviceType::CUDA) {
        return get_cuda_backend();
    }
#endif
#ifdef SERIUS_USE_VULKAN
    extern IBackend *get_vulkan_backend();
    if (type == DeviceType::VULKAN) {
        return get_vulkan_backend();
    }
#endif
#ifdef SERIUS_USE_METAL
    extern IBackend *get_metal_backend();
    if (type == DeviceType::METAL) {
        return get_metal_backend();
    }
#endif
    return nullptr;
}

IBackend *get_gpu_backend()
{
#ifdef SERIUS_USE_CUDA
    IBackend *cuda = get_backend(DeviceType::CUDA);
    if (cuda && cuda->is_available()) return cuda;
#endif
#ifdef SERIUS_USE_METAL
    IBackend *metal = get_backend(DeviceType::METAL);
    if (metal && metal->is_available()) return metal;
#endif
#ifdef SERIUS_USE_VULKAN
    IBackend *vulkan = get_backend(DeviceType::VULKAN);
    if (vulkan && vulkan->is_available()) return vulkan;
#endif
    return nullptr;
}

}  // namespace hal
}  // namespace serius
