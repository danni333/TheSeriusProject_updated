#include "vulkan_backend.h"

#ifdef SERIUS_USE_VULKAN

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace serius {
namespace hal {

void *VulkanBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
    // TODO: Implement Vulkan memory allocation via VMA or native Vulkan APIs
    return nullptr;
}

void *VulkanBackend::malloc_pinned(size_t size)
{
    if (size == 0) return nullptr;
    return nullptr;
}

void *VulkanBackend::malloc_managed(size_t size)
{
    if (size == 0) return nullptr;
    return nullptr;
}

void VulkanBackend::free(void *ptr)
{
    if (!ptr) return;
    // TODO: Implement Vulkan memory free
}

void VulkanBackend::free_pinned(void *ptr)
{
    if (!ptr) return;
}

void VulkanBackend::memcpy_h2d(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        // TODO: Implement Vulkan buffer mapping/copying
    }
}

void VulkanBackend::memcpy_d2h(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        // TODO: Implement Vulkan buffer mapping/copying
    }
}

void VulkanBackend::memcpy_d2d(void *dst, const void *src, size_t size)
{
    if (dst != nullptr && src != nullptr && size > 0) {
        // TODO: Implement Vulkan device-to-device copy via compute/blit
        std::memcpy(dst, src, size);
    }
}

void *VulkanBackend::create_stream()
{
    return nullptr;
}

void VulkanBackend::destroy_stream(void *stream)
{
}

void VulkanBackend::sync_stream(void *stream)
{
}

void VulkanBackend::memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_h2d(dst, src, size);
}

void VulkanBackend::memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_d2h(dst, src, size);
}

void VulkanBackend::memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_d2d(dst, src, size);
}

bool VulkanBackend::is_available() const
{
    // Return true only if Vulkan context is successfully initialized
    return false;  // Stub
}

static VulkanBackend *g_vulkan_backend = nullptr;

IBackend *get_vulkan_backend()
{
    if (!g_vulkan_backend) {
        g_vulkan_backend = new VulkanBackend();
    }
    return g_vulkan_backend;
}

}  // namespace hal
}  // namespace serius

#endif  // SERIUS_USE_VULKAN
