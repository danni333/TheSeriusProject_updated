cmake_minimum_required(VERSION 3.10)
include_directories(${CMAKE_SOURCE_DIR}/src)

# GTest from root FetchContent (googletest)
include(GoogleTest)

# Serius Core tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE serius_core gtest_main)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd PRIVATE serius_core gtest_main)

add_executable(test_hal test_hal.cpp)
target_link_libraries(test_hal PRIVATE serius_core gtest_main)

add_executable(test_server test_server.cpp)
target_link_libraries(test_server PRIVATE serius_server gtest_main)

if(SERIUS_USE_CUDA)
    add_executable(test_cuda test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE serius_core gtest_main)

    add_executable(bench_flash_attn bench_flash_attn.cu)
    target_link_libraries(bench_flash_attn PRIVATE serius_core)

    add_executable(test_fused_cuda test_fused_cuda.cpp)
    target_link_libraries(test_fused_cuda PRIVATE serius_core gtest_main)

    add_executable(test_llama_inference_cuda test_llama_inference_cuda.cpp)
    target_link_libraries(test_llama_inference_cuda PRIVATE serius_core gtest_main)
endif()

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

add_executable(test_simple_weights_loader test_simple_weights_loader.cpp)
target_link_libraries(test_simple_weights_loader PRIVATE serius_core gtest_main)

add_executable(test_llama_model test_llama_model.cpp)
target_link_libraries(test_llama_model PRIVATE serius_core gtest_main)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE serius_core gtest_main)

add_executable(test_speculative test_speculative.cpp)
target_link_libraries(test_speculative PRIVATE serius_core gtest_main)

add_executable(test_guardrails test_guardrails.cpp)
target_link_libraries(test_guardrails PRIVATE serius_core gtest_main)

add_executable(test_distributed_llama test_distributed_llama.cpp)
target_link_libraries(test_distributed_llama PRIVATE serius_core gtest_main)

add_executable(test_pipeline_llama test_pipeline_llama.cpp)
target_link_libraries(test_pipeline_llama PRIVATE serius_core gtest_main)

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

add_executable(test_sft test_sft.cpp)
target_link_libraries(test_sft PRIVATE serius_core gtest_main)

add_executable(test_kv_cache_logic test_kv_cache_logic.cpp)
target_link_libraries(test_kv_cache_logic PRIVATE serius_core gtest_main)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE serius_core gtest_main)

add_executable(test_distributed test_distributed.cpp)
target_link_libraries(test_distributed PRIVATE serius_core gtest_main)

if(SERIUS_USE_NCCL AND SERIUS_USE_MPI)
    add_executable(test_nccl_bootstrap test_nccl_bootstrap.cpp)
    target_link_libraries(test_nccl_bootstrap PRIVATE serius_core)
    add_test(NAME test_nccl_bootstrap COMMAND test_nccl_bootstrap)
endif()

add_executable(test_thread_pool test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE serius_core gtest_main)

add_executable(test_stability test_stability.cpp)
target_link_libraries(test_stability PRIVATE serius_core gtest_main)

add_executable(test_sampling test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE serius_core gtest_main)

add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE serius_core gtest_main)

add_executable(test_sharding test_sharding.cpp)
target_link_libraries(test_sharding PRIVATE serius_core gtest_main)

add_executable(test_lora test_lora.cpp)
target_link_libraries(test_lora PRIVATE serius_core gtest_main)

add_executable(test_moe test_moe.cpp)
target_link_libraries(test_moe PRIVATE serius_core gtest_main)

add_executable(test_rope test_rope.cpp)
target_link_libraries(test_rope PRIVATE serius_core gtest_main)

add_executable(test_sliding_window test_sliding_window.cpp)
target_link_libraries(test_sliding_window PRIVATE serius_core gtest_main)

add_executable(test_prefix_cache test_prefix_cache.cpp)
target_link_libraries(test_prefix_cache PRIVATE serius_core gtest_main)

add_executable(test_graph_compiler test_graph_compiler.cpp)
target_link_libraries(test_graph_compiler PRIVATE serius_core gtest_main)

add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE serius_core gtest_main)

add_executable(test_rag test_rag.cpp)
target_link_libraries(test_rag PRIVATE serius_core gtest_main)

add_executable(test_tools test_tools.cpp)
target_link_libraries(test_tools PRIVATE serius_core gtest_main)

add_executable(test_vision test_vision.cpp)
target_link_libraries(test_vision PRIVATE serius_core gtest_main)

add_executable(test_multimodal test_multimodal.cpp)
target_link_libraries(test_multimodal PRIVATE serius_core gtest_main)

add_executable(test_interpret test_interpret.cpp)
target_link_libraries(test_interpret PRIVATE serius_core gtest_main)

add_executable(test_rlhf test_rlhf.cpp)
target_link_libraries(test_rlhf PRIVATE serius_core gtest_main)

add_executable(test_grounding test_grounding.cpp)
target_link_libraries(test_grounding PRIVATE serius_core gtest_main)

add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval PRIVATE serius_core gtest_main)

add_executable(test_batcher test_batcher.cpp)
target_link_libraries(test_batcher PRIVATE serius_core gtest_main)

add_executable(test_dedup test_dedup.cpp)
target_link_libraries(test_dedup PRIVATE serius_core gtest_main)

add_executable(test_red_team test_red_team.cpp)
target_link_libraries(test_red_team PRIVATE serius_core gtest_main)

add_executable(test_mcp_client test_mcp_client.cpp)
target_link_libraries(test_mcp_client PRIVATE serius_core gtest_main)

add_executable(test_react test_react.cpp)
target_link_libraries(test_react PRIVATE serius_core gtest_main)

add_executable(test_guard test_guard.cpp)
target_link_libraries(test_guard PRIVATE serius_core gtest_main)

add_executable(test_reasoning test_reasoning.cpp)
target_link_libraries(test_reasoning PRIVATE serius_core gtest_main)

add_executable(test_federated test_federated.cpp)
 target_link_libraries(test_federated PRIVATE serius_core gtest_main)
 
 add_executable(test_automl test_automl.cpp)
 target_link_libraries(test_automl PRIVATE serius_core gtest_main)
 
 add_executable(test_multi_agent test_multi_agent.cpp)
 target_link_libraries(test_multi_agent PRIVATE serius_core gtest_main)

add_executable(test_dnc test_dnc.cpp)
target_link_libraries(test_dnc PRIVATE serius_core gtest_main)

add_executable(test_compression test_compression.cpp)
target_link_libraries(test_compression PRIVATE serius_core gtest_main)

add_executable(test_snn test_snn.cpp)
target_link_libraries(test_snn PRIVATE serius_core gtest_main)

add_executable(test_liquid test_liquid.cpp)
target_link_libraries(test_liquid PRIVATE serius_core gtest_main)

add_executable(test_world test_world.cpp)
target_link_libraries(test_world PRIVATE serius_core gtest_main)

add_executable(test_quantum test_quantum.cpp)
target_link_libraries(test_quantum PRIVATE serius_core gtest_main)

add_executable(test_autoopt test_autoopt.cpp)
target_link_libraries(test_autoopt PRIVATE serius_core gtest_main)

add_executable(test_knowledge test_knowledge.cpp)
target_link_libraries(test_knowledge PRIVATE serius_core gtest_main)

add_executable(test_bench test_bench.cpp)
target_link_libraries(test_bench PRIVATE serius_core gtest_main)

add_executable(test_stress test_stress.cpp)
target_link_libraries(test_stress PRIVATE serius_core gtest_main)

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
gtest_discover_tests(test_server)
if(SERIUS_USE_CUDA)
    gtest_discover_tests(test_fused_cuda)
    gtest_discover_tests(test_llama_inference_cuda)
endif()
gtest_discover_tests(test_paged_kv)
gtest_discover_tests(test_logger)
gtest_discover_tests(test_quant)
gtest_discover_tests(test_flash_attn)
gtest_discover_tests(test_config_loader)
gtest_discover_tests(test_simple_weights_loader)
gtest_discover_tests(test_llama_model)
gtest_discover_tests(test_sampler)
gtest_discover_tests(test_speculative)
gtest_discover_tests(test_guardrails)
gtest_discover_tests(test_nan_debugger)
gtest_discover_tests(test_transformer)
gtest_discover_tests(test_transformer_optim)
gtest_discover_tests(test_amp)
gtest_discover_tests(test_blas)
gtest_discover_tests(test_dataloader)
gtest_discover_tests(test_tokenizer)
gtest_discover_tests(test_training)
gtest_discover_tests(test_sft)
gtest_discover_tests(test_kv_cache_logic)
gtest_discover_tests(test_memory)
gtest_discover_tests(test_distributed)
gtest_discover_tests(test_thread_pool)
gtest_discover_tests(test_stability)
gtest_discover_tests(test_sampling)
gtest_discover_tests(test_safetensors)
gtest_discover_tests(test_sharding)
gtest_discover_tests(test_lora)
gtest_discover_tests(test_moe)
gtest_discover_tests(test_rope)
gtest_discover_tests(test_sliding_window)
gtest_discover_tests(test_prefix_cache)
gtest_discover_tests(test_graph_compiler)
gtest_discover_tests(test_sparse)
gtest_discover_tests(test_rag)
gtest_discover_tests(test_tools)
gtest_discover_tests(test_vision)
gtest_discover_tests(test_multimodal)
gtest_discover_tests(test_interpret)
gtest_discover_tests(test_rlhf)
gtest_discover_tests(test_grounding)
gtest_discover_tests(test_eval)
gtest_discover_tests(test_batcher)
gtest_discover_tests(test_dedup)
gtest_discover_tests(test_red_team)
gtest_discover_tests(test_mcp_client)
gtest_discover_tests(test_react)
gtest_discover_tests(test_guard)
gtest_discover_tests(test_reasoning)
 gtest_discover_tests(test_federated)
 gtest_discover_tests(test_automl)
 gtest_discover_tests(test_multi_agent)
gtest_discover_tests(test_dnc)
gtest_discover_tests(test_compression)
gtest_discover_tests(test_snn)
gtest_discover_tests(test_liquid)
gtest_discover_tests(test_world)
gtest_discover_tests(test_quantum)
gtest_discover_tests(test_autoopt)
gtest_discover_tests(test_knowledge)
gtest_discover_tests(test_bench)
gtest_discover_tests(test_stress)
