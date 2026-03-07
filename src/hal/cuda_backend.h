#pragma once

#include "device.h"

#ifdef SERIUS_USE_CUDA

namespace serius {
namespace hal {

class CudaBackend : public IBackend
{
   public:
    CudaBackend() = default;
    ~CudaBackend() override = default;

    DeviceType get_type() const override { return DeviceType::CUDA; }

    void *malloc(size_t size) override;
    void free(void *ptr) override;

    void memcpy_h2d(void *dst, const void *src, size_t size) override;
    void memcpy_d2h(void *dst, const void *src, size_t size) override;
    void memcpy_d2d(void *dst, const void *src, size_t size) override;

    bool is_available() const override;
};

IBackend *get_cuda_backend();

}  // namespace hal
}  // namespace serius

#endif  // SERIUS_USE_CUDA
