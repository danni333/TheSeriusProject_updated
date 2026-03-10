#pragma once

#include "device.h"

namespace serius {
namespace hal {

class CpuBackend : public IBackend
{
   public:
    CpuBackend() = default;
    ~CpuBackend() override = default;

    DeviceType get_type() const override { return DeviceType::CPU; }

    void *malloc(size_t size) override;
    void *malloc_managed(size_t size) override;
    void free(void *ptr) override;

    void memcpy_h2d(void *dst, const void *src, size_t size) override;
    void memcpy_d2h(void *dst, const void *src, size_t size) override;
    void memcpy_d2d(void *dst, const void *src, size_t size) override;

    bool is_available() const override { return true; }  // CPU is always available
};

}  // namespace hal
}  // namespace serius
