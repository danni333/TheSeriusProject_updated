enable_testing()

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

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer PRIVATE gtest_main serius_core)

# Benchmarks
add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE serius_core)

add_executable(bench_memory bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE serius_core)

include(GoogleTest)
gtest_discover_tests(test_dummy)
gtest_discover_tests(test_memory)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_blas)
gtest_discover_tests(test_autograd)
gtest_discover_tests(test_transformer)

# Tokenizer Tests
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE serius_core catch2_main)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

# DataLoader Tests
add_executable(test_dataloader test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE serius_core catch2_main)
add_test(NAME test_dataloader COMMAND test_dataloader)

# HAL / GPU Tests
add_executable(test_hal test_hal.cpp)
target_link_libraries(test_hal PRIVATE serius_core)
add_test(NAME test_hal COMMAND test_hal)

add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE serius_core)
add_test(NAME test_cuda COMMAND test_cuda)

add_executable(test_training test_training.cpp)
target_link_libraries(test_training PRIVATE gtest_main serius_core)
gtest_discover_tests(test_training)
