cmake_minimum_required(VERSION 3.25)
project(TheSeriusProject VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(SERIUS_VERSION="${PROJECT_VERSION}")

# cpp-httplib defaults its connection thread pool to max(8, cores-1). Under
# concurrent load every in-flight generation occupies a pool slot, so a
# handful of slow requests can starve cheap endpoints like /health. 64 slots
# keeps monitoring responsive; generation itself is serialized by
# inference_mutex_ anyway, so extra slots only wait on I/O/mutex.
add_compile_definitions(CPPHTTPLIB_THREAD_POOL_COUNT=64)

# Force MSVC to use Dynamic (MD) runtime to match dependencies like spdlog/gtest
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Warning rigorosi handled in src/CMakeLists.txt

# Release con simboli di debug, usando flag compatibili con il compilatore.
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")
endif()

# Sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES MATCHES Debug)
  add_compile_definitions(SERIUS_DEBUG)
  if(NOT MSVC)
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined)
  endif()
endif()

include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0)
# cpp-httplib (header only)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.67.0)
# SQLite for hub metadata
FetchContent_Declare(sqlite3
    URL https://www.sqlite.org/2026/sqlite-autoconf-3540100.tar.gz)
# OpenSSL for authentication (optional for now)
# JWT library (optional for now)

FetchContent_MakeAvailable(json spdlog googletest httplib sqlite3)

find_package(OpenMP)
# OpenMP is handled per-target to avoid NVCC compatibility issues.
# On Windows/MSVC, OpenMP is bundled with the compiler – no extra install needed.

# --- SIMD flags (applied globally for Release / RelWithDebInfo) ---
# Per-file SIMD flags are set in src/CMakeLists.txt (matmul_avx2.cpp, matmul_avx512.cpp).
# Here we ensure the dispatcher + inference code get AVX2 by default on MSVC builds.
if(MSVC)
    # /arch:AVX2 enables AVX2 + FMA for the whole project (safe on modern x86-64).
    # Individual files that need AVX-512 get /arch:AVX512 in src/CMakeLists.txt.
    add_compile_options($<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:/arch:AVX2>)
    # CCCL (CUDA 13 CCCL headers) requires the standard-conforming MSVC
    # preprocessor; without /Zc:preprocessor cl.exe triggers a fatal #error.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

enable_testing()

# --- Optional Hardware Backends ---

option(SERIUS_USE_CUDA "Enable CUDA backend" OFF)
option(SERIUS_USE_NCCL "Enable NCCL distributed backend" OFF)
option(SERIUS_USE_MPI "Enable MPI distributed backend" OFF)

# --- Experimental / Didactic Modules (default OFF to keep core lean) ---
# Sources live in experimental/<module>/ (outside src/); see docs/experimental.md.
option(SERIUS_EXPERIMENTAL_QUANTUM "Enable quantum/ (VQE simulator, didactic)" OFF)
option(SERIUS_EXPERIMENTAL_LIQUID "Enable liquid/ (Neural ODE/LTC, didactic)" OFF)
option(SERIUS_EXPERIMENTAL_WORLD "Enable world/ (World model/JEPA, prototype)" OFF)
option(SERIUS_EXPERIMENTAL_NEURO "Enable neuro/ (SNN/LIF, didactic)" OFF)
option(SERIUS_EXPERIMENTAL_SSM "Enable ssm/ (Minimal Mamba-like block)" OFF)
option(SERIUS_EXPERIMENTAL_FEDERATED "Enable federated/ (DP noise/aggregator, no real training loop)" OFF)
option(SERIUS_EXPERIMENTAL_MULTIMODAL "Enable multimodal/vision/audio (Didactic encoders, no HF weights)" OFF)
option(SERIUS_EXPERIMENTAL_REASONING "Enable reasoning/ (CoT/MCTS, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_RLHF "Enable rlhf/ (Reward model/PPO, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_INTERPRET "Enable interpret/ (Activation hooks/LogitLens, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_KNOWLEDGE "Enable knowledge/ (Semantic hash/fact extractor, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_AUTOML "Enable automl/ (Auto-tune with simulated timing)" OFF)
option(SERIUS_EXPERIMENTAL_EVAL "Enable eval/ (PerplexityRunner, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_GROUNDING "Enable grounding/ (Citation/confidence, partially real)" OFF)
option(SERIUS_EXPERIMENTAL_INFERENCE "Enable inference/ (autograd TransformerModel generate/sampler, didactic)" OFF)
option(SERIUS_EXPERIMENTAL_RAG "Enable rag/ (Retriever/hash embeddings, no server integration)" OFF)
option(SERIUS_EXPERIMENTAL_AGENT "Enable agent/ (ReAct/planner/supervisor/multi-agent)" OFF)
option(SERIUS_EXPERIMENTAL_AUTOOPT "Enable autoopt/ (self-modifying hot-swap/refactorer)" OFF)

# reasoning/ adapters wrap agent::Model — agent sources are required.
if(SERIUS_EXPERIMENTAL_REASONING AND NOT SERIUS_EXPERIMENTAL_AGENT)
    set(SERIUS_EXPERIMENTAL_AGENT ON)
endif()

if(SERIUS_USE_CUDA)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        add_compile_definitions(SERIUS_USE_CUDA=1)
        message(STATUS "CUDA backend enabled.")
    else()
        message(WARNING "CUDA compiler not found, disabling CUDA backend.")
        set(SERIUS_USE_CUDA OFF)
    endif()
endif()

if(SERIUS_USE_NCCL)
    # NCCL requires CUDA
    if(SERIUS_USE_CUDA)
        add_compile_definitions(SERIUS_USE_NCCL=1)
        message(STATUS "NCCL distributed backend enabled.")
    else()
        message(WARNING "NCCL requires CUDA. Disabling NCCL backend.")
        set(SERIUS_USE_NCCL OFF)
    endif()
endif()

if(SERIUS_USE_MPI)
    find_package(MPI)
    if(MPI_CXX_FOUND)
        add_compile_definitions(SERIUS_USE_MPI=1)
        message(STATUS "MPI distributed backend enabled.")
    else()
        message(WARNING "MPI not found, disabling MPI backend.")
        set(SERIUS_USE_MPI OFF)
    endif()
endif()

add_subdirectory(src)
add_subdirectory(tests)
