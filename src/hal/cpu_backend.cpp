#include "cpu_backend.h"

#include <cstdlib>
#include <cstring>

namespace serius {
namespace hal {

constexpr size_t ALIGNMENT = 64;

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

void *CpuBackend::create_stream()
{
    return nullptr;
}

void CpuBackend::destroy_stream(void *stream)
{
    (void)stream;
    // No-op for CPU
}

void CpuBackend::sync_stream(void *stream)
{
    (void)stream;
    // CPU operations are synchronous for now.
}

void CpuBackend::memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream)
{
    (void)stream;
    memcpy_h2d(dst, src, size);
}

void CpuBackend::memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream)
{
    (void)stream;
    memcpy_d2h(dst, src, size);
}

void CpuBackend::memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream)
{
    (void)stream;
    memcpy_d2d(dst, src, size);
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
