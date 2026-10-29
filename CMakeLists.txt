cmake_minimum_required(VERSION 3.25)
project(TheSeriusProject VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    GIT_TAG v3.11.3)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
# cpp-httplib (header only)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3)
# SQLite for hub metadata
FetchContent_Declare(sqlite3
    URL https://www.sqlite.org/2024/sqlite-autoconf-3450300.tar.gz)
# OpenSSL for authentication (optional for now)
# JWT library (optional for now)

FetchContent_MakeAvailable(json spdlog googletest httplib sqlite3)

find_package(OpenMP)
# OpenMP is handled per-target to avoid NVCC compatibility issues

enable_testing()

# --- Optional Hardware Backends ---

option(SERIUS_USE_CUDA "Enable CUDA backend" OFF)
option(SERIUS_USE_VULKAN "Enable Vulkan backend" OFF)
option(SERIUS_USE_METAL "Enable Metal backend" OFF)
option(SERIUS_USE_NCCL "Enable NCCL distributed backend" OFF)
option(SERIUS_USE_MPI "Enable MPI distributed backend" OFF)

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

if(SERIUS_USE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        add_compile_definitions(SERIUS_USE_VULKAN=1)
        message(STATUS "Vulkan backend enabled.")
    else()
        message(WARNING "Vulkan SDK not found, disabling Vulkan backend.")
        set(SERIUS_USE_VULKAN OFF)
    endif()
endif()

if(SERIUS_USE_METAL)
    if(APPLE)
        add_compile_definitions(SERIUS_USE_METAL=1)
        # Would need to link "-framework Metal -framework Foundation" 
        message(STATUS "Metal backend enabled.")
    else()
        message(WARNING "Metal is only supported on Apple platforms. Disabling.")
        set(SERIUS_USE_METAL OFF)
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
