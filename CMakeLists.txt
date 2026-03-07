cmake_minimum_required(VERSION 3.25)
project(TheSeriusProject VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warning rigorosi
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

# Release con simboli di debug
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")

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

FetchContent_MakeAvailable(json spdlog googletest httplib)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

enable_testing()

# --- Optional Hardware Backends ---

option(SERIUS_USE_CUDA "Enable CUDA backend" OFF)
option(SERIUS_USE_VULKAN "Enable Vulkan backend" OFF)
option(SERIUS_USE_METAL "Enable Metal backend" OFF)

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

add_subdirectory(src)
add_subdirectory(tests)
