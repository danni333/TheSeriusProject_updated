#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "hal/device.h"
#include "tensor/tensor.h"

using namespace serius;

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

int main()
{
    test_backend_discovery();
    test_tensor_sync();

    std::cout << "All HAL/Device tests passed.\n";
    return 0;
}
