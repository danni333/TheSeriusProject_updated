#include "metal_backend.h"

#ifdef SERIUS_USE_METAL

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace serius {
namespace hal {

void *MetalBackend::malloc(size_t size)
{
    if (size == 0) return nullptr;
    // TODO: Implement Metal MTLBuffer allocation
    return nullptr;
}

void *MetalBackend::malloc_pinned(size_t size)
{
    if (size == 0) return nullptr;
    return nullptr;
}

void *MetalBackend::malloc_managed(size_t size)
{
    if (size == 0) return nullptr;
    return nullptr;
}

void MetalBackend::free(void *ptr)
{
    if (!ptr) return;
    // TODO: Implement Metal buffer release
}

void MetalBackend::free_pinned(void *ptr)
{
    if (!ptr) return;
}

void MetalBackend::memcpy_h2d(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        // Unified Memory allows direct memcpy for Shared buffers usually!
    }
}

void MetalBackend::memcpy_d2h(void *dst, const void *src, size_t size)
{
    if (dst && src && size > 0) {
        // Unified Memory allows direct memcpy
    }
}

void MetalBackend::memcpy_d2d(void *dst, const void *src, size_t size)
{
    if (dst != nullptr && src != nullptr && size > 0) {
        // TODO: Implement Metal device-to-device copy via blit/compute
        std::memcpy(dst, src, size);
    }
}

void *MetalBackend::create_stream()
{
    return nullptr;
}

void MetalBackend::destroy_stream(void *stream)
{
}

void MetalBackend::sync_stream(void *stream)
{
}

void MetalBackend::memcpy_h2d_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_h2d(dst, src, size);
}

void MetalBackend::memcpy_d2h_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_d2h(dst, src, size);
}

void MetalBackend::memcpy_d2d_async(void *dst, const void *src, size_t size, void *stream)
{
    memcpy_d2d(dst, src, size);
}

bool MetalBackend::is_available() const
{
    // Ensure MTLCreateSystemDefaultDevice() works
    return false;  // Stub
}

static MetalBackend *g_metal_backend = nullptr;

IBackend *get_metal_backend()
{
    if (!g_metal_backend) {
        g_metal_backend = new MetalBackend();
    }
    return g_metal_backend;
}

}  // namespace hal
}  // namespace serius

#endif  // SERIUS_USE_METAL
