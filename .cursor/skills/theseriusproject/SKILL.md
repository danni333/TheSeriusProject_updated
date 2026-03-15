---
name: theseriusproject
description: Build, test, and extend TheSeriusProject C++ LLM engine. Use when building, running tests, adding new sources or tests to CMake, diagnosing build failures, or running benchmarks.
---

# TheSeriusProject — Build & Test

## Build (from repo root)

**Configure (first time or after CMake change):**
```powershell
mkdir build -Force; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Optional: -DSERIUS_USE_CUDA=ON -DSERIUS_USE_NCCL=ON -DSERIUS_USE_MPI=ON -DSERIUS_USE_METAL=ON
```

**Compile:**
```powershell
# From build/
cmake --build . --config Release
```

Or use the helper: `./.cursor/skills/theseriusproject/scripts/build.ps1` (run from repo root).

## Tests

**From `build/`:**
```powershell
ctest --output-on-failure
```

**Single test executable:**
```powershell
./tests/Release/test_tensor.exe
# or Debug: ./tests/Debug/test_tensor.exe
```

**Benchmarks (from build):**
```powershell
./bench_matmul/Release/bench_matmul.exe
./bench_full/Release/bench_full.exe
```

## Adding new sources

1. **New .cpp in `src/`**  
   Add to `src/CMakeLists.txt` in `add_library(serius_core ...)`.

2. **New CUDA .cu**  
   Add inside `if(SERIUS_USE_CUDA)` → `target_sources(serius_core PRIVATE ...)`.

3. **New test**  
   - Add `tests/test_foo.cpp` (link to `serius_core gtest_main`).  
   - In `tests/CMakeLists.txt`:  
     `add_executable(test_foo test_foo.cpp)`  
     `target_link_libraries(test_foo PRIVATE serius_core gtest_main)`  
     `gtest_discover_tests(test_foo)`  
   - If test needs CUDA: wrap in `if(SERIUS_USE_CUDA)` like `test_cuda`.

4. **SIMD (AVX2/AVX512)**  
   Only add to existing `matmul_avx2.cpp` / `matmul_avx512.cpp` or keep same `COMPILE_FLAGS` for new files.

## Common build issues

- **MSVC**: Runtime is MD/MDd (set in root CMakeLists). Link errors → check that new libs are in `target_link_libraries(serius_core PUBLIC ...)`.
- **CUDA**: If NVCC not found, CMake disables CUDA. Ensure CUDA Toolkit is on PATH and `cmake` is re-run from a clean `build/`.
- **Missing symbol in test**: Test must link `serius_core` (and `gtest_main`). New source must be in `serius_core` in `src/CMakeLists.txt`.
- **Build log**: Project often logs to `build_output.txt` / `build_log.txt` in repo root; check there for full output.

## When making code changes

1. Edit sources under `src/` (respect namespaces: `serius::core`, `serius::model`, etc.).  
2. Run build from `build/`.  
3. Run `ctest --output-on-failure`; fix failing tests before considering done.  
4. Apply project clang-format (see `.cursor/rules/cpp-style.mdc`) before committing.

## Reference

- Architecture and modules: `.cursor/rules/theseriusproject.mdc`
- C++ style and CMake details: `.cursor/rules/cpp-style.mdc`
