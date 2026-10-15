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

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer PRIVATE serius_core gtest_main)

add_executable(test_transformer_optim test_transformer_optim.cpp)
target_link_libraries(test_transformer_optim PRIVATE serius_core gtest_main)

add_executable(test_quant test_quant.cpp)
target_link_libraries(test_quant PRIVATE serius_core gtest_main)

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

add_executable(test_amp test_amp.cpp)
target_link_libraries(test_amp PRIVATE serius_core gtest_main)

add_executable(test_blas test_blas.cpp)
target_link_libraries(test_blas PRIVATE serius_core gtest_main)

add_executable(test_dataloader test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE serius_core gtest_main)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE serius_core gtest_main)

add_executable(test_training test_training.cpp)
target_link_libraries(test_training PRIVATE serius_core gtest_main)

add_executable(test_fused_cuda test_fused_cuda.cpp)
target_link_libraries(test_fused_cuda PRIVATE serius_core gtest_main)

add_executable(test_kv_cache_logic test_kv_cache_logic.cpp)
target_link_libraries(test_kv_cache_logic PRIVATE serius_core gtest_main)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE serius_core gtest_main)

add_executable(test_distributed test_distributed.cpp)
target_link_libraries(test_distributed PRIVATE serius_core gtest_main)

add_executable(test_placeholder test_placeholder.cpp)
target_link_libraries(test_placeholder PRIVATE serius_core gtest_main)

# Benchmarks
add_executable(bench_full bench_full.cpp)
target_link_libraries(bench_full PRIVATE serius_core)

add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE serius_core)

add_executable(bench_memory bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE serius_core)

add_executable(bench_mixed_precision bench_mixed_precision.cpp)
target_link_libraries(bench_mixed_precision PRIVATE serius_core)

add_executable(bench_quant bench_quant.cpp)
target_link_libraries(bench_quant PRIVATE serius_core)

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
gtest_discover_tests(test_transformer)
gtest_discover_tests(test_transformer_optim)
gtest_discover_tests(test_amp)
gtest_discover_tests(test_blas)
gtest_discover_tests(test_dataloader)
gtest_discover_tests(test_tokenizer)
gtest_discover_tests(test_training)
gtest_discover_tests(test_fused_cuda)
gtest_discover_tests(test_kv_cache_logic)
gtest_discover_tests(test_memory)
gtest_discover_tests(test_distributed)
gtest_discover_tests(test_placeholder)
