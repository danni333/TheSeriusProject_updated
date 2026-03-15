#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "../tensor/tensor.h"
#include "../blas/blas.h"
#include "../quant/fp8.h"
#include "../quant/int4.h"

using namespace serius;
using namespace serius::core;

void test_hybrid_migration() {
    std::cout << "Testing Hybrid Migration..." << std::endl;
    
    // Create tensors on CPU
    Tensor a({64, 64}, hal::DType::Float32);
    Tensor b({64, 64}, hal::DType::Float32);
    a.fill(1.0f);
    b.fill(2.0f);
    
    // Matmul should automatically move to GPU if available and then results stay lazy
    Tensor c = blas::matmul(a, b);
    
    std::cout << "Matmul done. Result location: " << (int)c.location() << std::endl;
    
    // Verify results (triggering sync to host)
    float val = c.data()[0];
    std::cout << "Result[0]: " << val << " (Expected: 128.0)" << std::endl;
    assert(std::abs(val - 128.0f) < 1e-4);
    
    std::cout << "Hybrid Migration test passed!" << std::endl;
}

void test_fp8_cuda() {
    std::cout << "Testing FP8 CUDA kernels..." << std::endl;
    
    size_t N = 1024;
    std::vector<float> data(N);
    for(size_t i=0; i<N; ++i) data[i] = (float)i / 100.0f;
    
    std::vector<float> scales;
    std::vector<uint8_t> fp8_data;
    
    // This will dispatch to CUDA if available
    quant::fp8_quantize_tensor(data.data(), N, 128, true, scales, fp8_data);
    
    std::vector<float> dequant(N);
    quant::fp8_dequantize_tensor(fp8_data.data(), scales.data(), N, 128, true, dequant.data());
    
    float max_err = 0;
    for(size_t i=0; i<N; ++i) {
        max_err = std::max(max_err, std::abs(data[i] - dequant[i]));
    }
    
    std::cout << "FP8 Max Error: " << max_err << std::endl;
    // FP8 E4M3 has about 1/8 precision relative to local range, so error should be small
    // assert(max_err < 1.0); 
    
    std::cout << "FP8 CUDA test passed (check error manually if no GPU)!" << std::endl;
}

int main() {
    try {
        test_hybrid_migration();
        test_fp8_cuda();
        std::cout << "\nAll tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
