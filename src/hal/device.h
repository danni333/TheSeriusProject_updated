#pragma once

#include <cstddef>
#include <cstdint>

namespace serius {
namespace hal {

// Forward declarations of common types to avoid pulling heavy headers
enum class DeviceType : std::uint8_t { CPU, CUDA, VULKAN, METAL };

class IBackend
{
   public:
    virtual ~IBackend() = default;

    virtual DeviceType get_type() const = 0;

    // Memory Management
    virtual void *malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;

    virtual void memcpy_h2d(void *dst, const void *src, size_t size) = 0;
    virtual void memcpy_d2h(void *dst, const void *src, size_t size) = 0;
    virtual void memcpy_d2d(void *dst, const void *src, size_t size) = 0;

    // Initialization check
    virtual bool is_available() const = 0;
};

// Global accessor to get backends by specific type
IBackend *get_backend(DeviceType type);

// Helper to get the best available GPU backend (CUDA > Metal > Vulkan), or nullptr if none
IBackend *get_gpu_backend();

}  // namespace hal
}  // namespace serius
