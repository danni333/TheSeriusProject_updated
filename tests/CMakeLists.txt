cmake_minimum_required(VERSION 3.10)

# GTest from root FetchContent (googletest)
include(GoogleTest)

# Serius Core tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE serius_core gtest_main)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd PRIVATE serius_core gtest_main)

add_executable(test_hal test_hal.cpp)
target_link_libraries(test_hal PRIVATE serius_core gtest_main)

add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE serius_core gtest_main)

add_executable(test_paged_kv test_paged_kv.cpp)
target_link_libraries(test_paged_kv PRIVATE serius_core gtest_main)

add_executable(test_logger test_logger.cpp)
target_link_libraries(test_logger PRIVATE serius_core gtest_main)

add_executable(test_transformer_optim test_transformer_optim.cpp)
target_link_libraries(test_transformer_optim PRIVATE serius_core gtest)

# Quantization tests
add_executable(test_quant test_quant.cpp)
target_link_libraries(test_quant PRIVATE serius_core gtest_main)

# Flash Attention tests
add_executable(test_flash_attn test_flash_attn.cpp)
target_link_libraries(test_flash_attn PRIVATE serius_core gtest_main)

add_executable(test_config_loader test_config_loader.cpp)
target_link_libraries(test_config_loader PRIVATE serius_core gtest_main)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE serius_core gtest_main)

add_executable(test_guardrails test_guardrails.cpp)
target_link_libraries(test_guardrails PRIVATE serius_core gtest_main)

add_executable(test_nan_debugger test_nan_debugger.cpp)
target_link_libraries(test_nan_debugger PRIVATE serius_core gtest_main)

# Discovery
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_autograd)
gtest_discover_tests(test_hal)
gtest_discover_tests(test_cuda)
gtest_discover_tests(test_paged_kv)
gtest_discover_tests(test_logger)
gtest_discover_tests(test_quant)
gtest_discover_tests(test_flash_attn)
gtest_discover_tests(test_config_loader)
gtest_discover_tests(test_sampler)
gtest_discover_tests(test_guardrails)
gtest_discover_tests(test_nan_debugger)
gtest_discover_tests(test_transformer_optim)
