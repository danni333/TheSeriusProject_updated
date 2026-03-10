enable_testing()

# Make all test targets resolve #include "foo/bar.h" from the src directory
# This is needed because MSVC resolves ../src/ paths differently than gcc/clang
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test_dummy test_dummy.cpp)
target_link_libraries(test_dummy PRIVATE gtest_main serius_core)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE gtest_main serius_core)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE gtest_main serius_core)

add_executable(test_blas test_blas.cpp)
target_link_libraries(test_blas PRIVATE gtest_main serius_core)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd PRIVATE gtest_main serius_core)

add_executable(test_kv_cache_logic test_kv_cache_logic.cpp)
target_link_libraries(test_kv_cache_logic PRIVATE gtest_main serius_core)

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer PRIVATE gtest_main serius_core)

# Benchmarks (no gtest needed)
add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE serius_core)

add_executable(bench_memory bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE serius_core)

add_executable(bench_mixed_precision bench_mixed_precision.cpp)
target_link_libraries(bench_mixed_precision PRIVATE serius_core)

add_executable(bench_full bench_full.cpp)
target_link_libraries(bench_full PRIVATE serius_core)

include(GoogleTest)
gtest_discover_tests(test_dummy DISCOVERY_TIMEOUT 10)
gtest_discover_tests(test_memory DISCOVERY_TIMEOUT 10)
gtest_discover_tests(test_tensor DISCOVERY_TIMEOUT 10)
gtest_discover_tests(test_blas DISCOVERY_TIMEOUT 10)
gtest_discover_tests(test_autograd DISCOVERY_TIMEOUT 10)
gtest_discover_tests(test_transformer DISCOVERY_TIMEOUT 10)

# Tokenizer Tests (converted from catch2 to gtest)
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE gtest_main serius_core)
gtest_discover_tests(test_tokenizer DISCOVERY_TIMEOUT 10)

# DataLoader Tests (converted from catch2 to gtest)
add_executable(test_dataloader test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE gtest_main serius_core)
gtest_discover_tests(test_dataloader DISCOVERY_TIMEOUT 10)

# HAL / Device Tests (standalone, no gtest main needed)
add_executable(test_hal test_hal.cpp)
target_link_libraries(test_hal PRIVATE serius_core)
target_include_directories(test_hal PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_hal COMMAND test_hal)

# CUDA integration tests (standalone, no gtest main needed)
add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE serius_core)
target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_cuda COMMAND test_cuda)

# Training Tests
add_executable(test_training test_training.cpp)
target_link_libraries(test_training PRIVATE gtest_main serius_core)
gtest_discover_tests(test_training DISCOVERY_TIMEOUT 10)

# Fused CUDA Tests
add_executable(test_fused_cuda test_fused_cuda.cpp)
target_link_libraries(test_fused_cuda PRIVATE gtest_main serius_core)
gtest_discover_tests(test_fused_cuda DISCOVERY_TIMEOUT 10)

# Quantization Tests (INT4, FP16/BF16, KV Cache enhancements)
add_executable(test_quant test_quant.cpp)
target_link_libraries(test_quant PRIVATE gtest_main serius_core)
gtest_discover_tests(test_quant DISCOVERY_TIMEOUT 10)

# Stability Tests (Fase 9: Weight Init, Gradient Clipping, Loss Scaling, EMA)
add_executable(test_stability test_stability.cpp)
target_link_libraries(test_stability PRIVATE gtest_main serius_core)
gtest_discover_tests(test_stability DISCOVERY_TIMEOUT 10)

# Sampling Tests (Fase 10: Top-K, Top-P, Temperature, Repetition Penalty)
add_executable(test_sampling test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE gtest_main serius_core)
gtest_discover_tests(test_sampling DISCOVERY_TIMEOUT 10)

# Safetensors Tests (Fase 15: Read/Write safetensors format)
add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE gtest_main serius_core)
gtest_discover_tests(test_safetensors DISCOVERY_TIMEOUT 10)

# LoRA Tests (Fase 19: Low-Rank Adaptation)
add_executable(test_lora test_lora.cpp)
target_link_libraries(test_lora PRIVATE gtest_main serius_core)
gtest_discover_tests(test_lora DISCOVERY_TIMEOUT 10)
