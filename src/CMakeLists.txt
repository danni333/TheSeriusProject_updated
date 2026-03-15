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
    transformer/kv_cache.cpp
    transformer/paged_kv_cache.cpp
    transformer/block.cpp
    transformer/model.cpp
    attention/flash_attn.cpp
    tokenizer/bpe.cpp
    data/dataloader.cpp
    optim/adamw.cpp
    optim/lr_schedule.cpp
    train/checkpoint.cpp
    train/trainer.cpp
    hal/cpu_backend.cpp
    quant/conversions.cpp
    quant/int4.cpp
    quant/dequant_matmul.cpp
    quant/fp8.cpp
    quant/awq.cpp
    quant/gptq.cpp
    debug/tensor_logger.cpp
    io/config_loader.cpp
    io/gguf_loader.cpp
    io/simple_weights_loader.cpp
    io/safetensors_loader.cpp
    io/hf_llama_loader.cpp
    transformer/llama_model.cpp
    transformer/qwen_model.cpp
    infer/sampler.cpp
    infer/llama_inference.cpp
    safety/guardrails.cpp
    debug/nan_debugger.cpp
    distributed/local_comm.cpp
    distributed/data_parallel.cpp
    distributed/pipeline.cpp
    distributed/tensor_parallel.cpp
)

if(SERIUS_USE_CUDA)
    target_sources(serius_core PRIVATE 
        hal/cuda_backend.cu
        hal/fused_kernels.cu
        blas/matmul_cuda.cu
        quant/dequant_matmul.cu
        attention/flash_attn_cuda.cu
    )
endif()

if(SERIUS_USE_VULKAN)
    target_sources(serius_core PRIVATE 
        hal/vulkan_backend.cpp
    )
endif()

if(SERIUS_USE_NCCL)
    target_sources(serius_core PRIVATE
        distributed/nccl_comm.cpp
    )
    # Link NCCL library
    target_link_libraries(serius_core PUBLIC nccl)
endif()

if(SERIUS_USE_MPI)
    target_sources(serius_core PRIVATE
        distributed/mpi_comm.cpp
    )
    target_link_libraries(serius_core PUBLIC MPI::MPI_CXX)
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
    target_compile_options(serius_core PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>
    )
    set_source_files_properties(blas/matmul_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(blas/matmul_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
    target_compile_options(serius_core PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror=return-type>
    )
    set_source_files_properties(blas/matmul_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(blas/matmul_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
endif()
target_include_directories(serius_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(serius_core PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json)

# --- Server library ---
add_library(serius_server STATIC server/server.cpp)
target_link_libraries(serius_server PUBLIC serius_core httplib::httplib)
target_include_directories(serius_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- Hub components (Phase 65) ---
add_subdirectory(hub)

# --- CLI executable (serve, chat) ---
add_executable(serius app/main.cpp)
target_link_libraries(serius PRIVATE serius_core serius_server hub)
target_include_directories(serius PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
