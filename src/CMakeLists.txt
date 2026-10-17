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
    finetune/sft_loader.cpp
    finetune/sft_trainer.cpp
    transformer/layernorm.cpp
    transformer/positional.cpp
    transformer/attention.cpp
    transformer/ffn.cpp
    transformer/kv_cache.cpp
    transformer/paged_kv_cache.cpp
    transformer/block.cpp
    transformer/model.cpp
    attention/flash_attn.cpp
    attention/sliding_window.cpp
    attention/sparse_attention.cpp
    attention/sparsity_mask.cpp
    cache/prefix_cache.cpp
    tokenizer/bpe.cpp
    data/dataloader.cpp
    optim/adamw.cpp
    optim/lr_schedule.cpp
    optim/graph_compiler.cpp
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
    io/safetensors.cpp
    io/safetensors_writer.cpp
    io/sharding.cpp
    init/weight_init.cpp
    stability/grad_clip.cpp
    stability/loss_scaler.cpp
    stability/ema.cpp
    inference/sampler.cpp
    inference/generate.cpp
    inference/speculative.cpp
    lora/lora_linear.cpp
    lora/lora_adapter.cpp
    transformer/llama_model.cpp
    transformer/qwen_model.cpp
    moe/router.cpp
    moe/expert.cpp
    moe/moe_layer.cpp
    moe/load_balance.cpp
    model/moe.cpp
    infer/sampler.cpp
    infer/llama_inference.cpp
    infer/speculative.cpp
    safety/guardrails.cpp
    safety/toxicity.cpp
    safety/sanitizer.cpp
    debug/nan_debugger.cpp
    distributed/local_comm.cpp
    distributed/data_parallel.cpp
    distributed/pipeline.cpp
    distributed/tensor_parallel.cpp
    rag/vector_client.cpp
    rag/embeddings.cpp
    rag/retriever.cpp
    tools/registry.cpp
    tools/parser.cpp
    vision/patch_embed.cpp
    vision/vision_encoder.cpp
    audio/mel.cpp
    audio/audio_encoder.cpp
multimodal/cross_attention.cpp
     multimodal/input_builder.cpp
    interpret/attention_capture.cpp
    interpret/activation_probe.cpp
    interpret/logit_lens.cpp
    interpret/export.cpp
    rlhf/reward_model.cpp
    rlhf/ppo.cpp
    grounding/self_consistency.cpp
    grounding/citation_engine.cpp
    grounding/confidence.cpp
    eval/eval_suite.cpp
    eval/perplexity.cpp
    eval/metrics.cpp
    eval/report.cpp
    server/metrics.cpp
    server/batcher.cpp
    server/logging.cpp
    data/minhash.cpp
    data/lsh.cpp
    data/dedup.cpp
    data/synthetic.cpp
    safety/red_team.cpp
    safety/bias_scorer.cpp
    mcp/client.cpp
    mcp/transport_stdio.cpp
    mcp/transport_http.cpp
    agent/action_parser.cpp
    agent/context_manager.cpp
    agent/react.cpp
    mcp/guard.cpp
    mcp/scrubber.cpp
    mcp/audit.cpp
    reasoning/cot.cpp
    reasoning/mcts.cpp
    reasoning/verifier.cpp
    reasoning/budget.cpp
    federated/dp_noise.cpp
    federated/aggregator.cpp
    federated/user_adapter.cpp
)

if(SERIUS_USE_CUDA)
    target_sources(serius_core PRIVATE
        hal/cuda_backend.cu
        hal/fused_kernels.cu
        blas/matmul_cuda.cu
        quant/dequant_matmul.cu
        quant/quant_kernels.cu
        attention/flash_attn_cuda.cu
        attention/block_sparse.cu
    )
    target_compile_definitions(serius_core PRIVATE SERIUS_USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        find_package(CUDAToolkit REQUIRED)
        target_include_directories(serius_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    else()
        target_include_directories(serius_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
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
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CXX>:/FS>
    )
    set_source_files_properties(blas/matmul_avx2.cpp infer/llama_inference.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
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
target_link_libraries(serius PRIVATE serius_core serius_server)
target_include_directories(serius PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
