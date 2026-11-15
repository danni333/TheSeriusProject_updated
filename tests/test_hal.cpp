#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "hal/device.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
namespace hal = serius::hal;

void test_backend_discovery()
{
    std::cout << "Testing backend discovery...\n";

    // CPU should always be available
    hal::IBackend *cpu = hal::get_backend(hal::DeviceType::CPU);
    assert(cpu != nullptr && "CPU backend should be available");
    assert(cpu->get_type() == hal::DeviceType::CPU);

    hal::IBackend *gpu = hal::get_gpu_backend();
    if (gpu) {
        std::cout << "Found active GPU backend of type: " << static_cast<int>(gpu->get_type()) << "\n";
    } else {
        std::cout << "No GPU backend active.\n";
    }
}

void test_tensor_sync()
{
    std::cout << "Testing Tensor sync logic...\n";

    Tensor t({2, 2});
    t.fill(1.5f);

    // Sync to device. If no GPU is present, it's virtually a no-op but shouldn't crash.
    t.syncToDevice();

    // Check values remain accessible on CPU (either syncToHost puts it back or it's still there)
    t.syncToHost();

    const float *data = t.data();
    for (size_t i = 0; i < t.size(); ++i) {
        assert(std::abs(data[i] - 1.5f) < 1e-5 && "Tensor data corruption during sync");
    }

    std::cout << "Tensor sync logic ok.\n";
}

void test_cpu_stream()
{
    std::cout << "Testing CPU backend streams...\n";

    hal::IBackend *be = hal::get_backend(hal::DeviceType::CPU);
    assert(be != nullptr);

    void *stream = be->create_stream();
    assert(stream != nullptr);

    // Enqueue ordered async copies; data must be correct after sync_stream.
    float src_a[4] = {1.f, 2.f, 3.f, 4.f};
    float src_b[4] = {10.f, 20.f, 30.f, 40.f};
    float dst_a[4] = {};
    float dst_b[4] = {};

    be->memcpy_h2d_async(dst_a, src_a, sizeof(src_a), stream);
    be->memcpy_d2d_async(dst_b, src_b, sizeof(src_b), stream);
    // Not synced yet: dst may still be zero (async). Sync then verify.
    be->sync_stream(stream);

    for (int i = 0; i < 4; ++i) {
        assert(dst_a[i] == src_a[i]);
        assert(dst_b[i] == src_b[i]);
    }

    // Ordering: second copy overwrites the first on the same buffer.
    be->memcpy_d2d_async(dst_a, src_b, sizeof(src_b), stream);
    be->sync_stream(stream);
    for (int i = 0; i < 4; ++i) assert(dst_a[i] == src_b[i]);

    be->destroy_stream(stream);
    std::cout << "CPU stream ok.\n";
}

}  // namespace test
}  // namespace serius

int main()
{
    serius::test::test_backend_discovery();
    serius::test::test_tensor_sync();
    serius::test::test_cpu_stream();

    std::cout << "All HAL/Device tests passed.\n";
    return 0;
}
