add_library(serius_core 
    core/dummy.cpp
    memory/memory_pool.cpp
    tensor/tensor.cpp
    blas/elementwise.cpp
    blas/matmul_scalar.cpp
    blas/dispatch.cpp
    blas/matmul_avx2.cpp
    blas/matmul_avx512.cpp
    autograd/node.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    transformer/layernorm.cpp
    transformer/positional.cpp
    transformer/attention.cpp
    transformer/ffn.cpp
    transformer/block.cpp
    transformer/model.cpp
    tokenizer/bpe.cpp
    data/dataloader.cpp
    optim/adamw.cpp
    optim/lr_schedule.cpp
    train/checkpoint.cpp
    train/trainer.cpp
    hal/cpu_backend.cpp
)

if(SERIUS_USE_CUDA)
    target_sources(serius_core PRIVATE 
        hal/cuda_backend.cu
        blas/matmul_cuda.cu
    )
endif()

if(SERIUS_USE_VULKAN)
    target_sources(serius_core PRIVATE 
        hal/vulkan_backend.cpp
    )
endif()

if(SERIUS_USE_METAL)
    target_sources(serius_core PRIVATE 
        hal/metal_backend.cpp
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(serius_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set specific compiler flags for SIMD files
if(MSVC)
    set_source_files_properties(blas/matmul_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(blas/matmul_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
    set_source_files_properties(blas/matmul_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(blas/matmul_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
endif()
target_include_directories(serius_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(serius_core PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json)
