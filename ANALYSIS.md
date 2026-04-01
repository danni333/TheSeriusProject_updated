# TheSeriusProject - Codebase Analysis

## Overview

**TheSeriusProject** is a C++ LLM inference engine designed for local, reproducible, and educational use. It aims to provide full control over the model lifecycle with optional GPU acceleration (CUDA, Vulkan, Metal).

- **Language**: C++20
- **Build system**: CMake >= 3.25
- **License**: Apache 2.0

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Total source files (.cpp/.h/.hpp/.cu) | 146 |
| Total lines of code | ~18,500 |
| Source modules (under `src/`) | 22 |
| Test files (under `tests/`) | 30 |
| Test LOC | ~5,100 |
| Largest source file | `src/tensor/tensor.cpp` (1,040 LOC) |
| Largest header | `src/tensor/tensor.h` (243 LOC) |
| Largest test file | `tests/test_quant.cpp` (898 LOC) |

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| nlohmann/json | v3.11.3 | JSON parsing |
| spdlog | v1.14.1 | Logging |
| googletest | v1.14.0 | Testing framework |
| cpp-httplib | v0.15.3 | HTTP server |
| SQLite | 3.45.3 | Hub metadata storage |
| OpenMP | (optional) | Parallelism |

## Architecture (22 modules)

### Core Engine
- `src/tensor/` - Custom Tensor type with memory management (largest module, 1,283 LOC)
- `src/autograd/` - Automatic differentiation engine (1,092 LOC)
- `src/blas/` - BLAS operations: scalar, AVX2, AVX-512, CUDA dispatch (960 LOC)
- `src/memory/` - Memory pool allocator (161 LOC)

### Transformer Stack
- `src/transformer/` - Attention, FFN, LayerNorm, KV cache, model, blocks
- `src/attention/` - Flash attention implementation
- `src/tokenizer/` - BPE tokenizer (207 LOC)

### Training & Optimization
- `src/train/` - Training loop, checkpointing
- `src/optim/` - AdamW, learning rate schedules, AMP

### Quantization
- `src/quant/` - INT4, FP8 (E4M3/E5M2), AWQ, GPTQ (1,468 LOC)

### Hardware Abstraction
- `src/hal/` - CPU, CUDA, Vulkan, Metal backends
- `src/distributed/` - Data parallel, pipeline parallel, tensor parallel (800 LOC)

### Infrastructure
- `src/server/` - REST API server (212 LOC)
- `src/hub/` - Model hub, plugin system, auth, search (935 LOC)
- `src/io/` - Config loader, GGUF loader
- `src/debug/` - Tensor logger, NaN debugger
- `src/safety/` - Guardrails
- `src/app/` - CLI entry point (189 LOC)
- `src/infer/` - Sampler (greedy, top-k, top-p, temperature)

## Implemented Phases (0-8+)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundations (Tensor, BLAS, Memory) | Done |
| 1 | Math engine (Autograd, ops) | Done |
| 2 | Transformer architecture | Done |
| 3 | Tokenizer (BPE) | Done |
| 4 | Training loop | Done |
| 5 | Hybrid CPU/GPU (HAL) | Done |
| 6 | Quantization (INT4, FP8, AWQ, GPTQ) | Done |
| 7 | Distributed computing | Done |
| 8 | Stability & sampling | Partial |

## Roadmap

143 phase files in `todo/` covering phases 0 through 72 with extended "+" variants. Phases 9-72 cover: finetuning, flash attention, profiling, deployment, speculative decoding, MoE, safety, LoRA, multimodal, RAG, RLHF, CLI, OpenAI-compatible API, chat UI, model hub, documentation, plugin system, and many more.

## Known Build Issues

1. **MSVC `test_memory` timeout**: `test_memory.exe` terminates due to timeout during GoogleTest discovery (5s limit). Likely caused by heavy memory allocation in test initialization on MSVC Debug builds.

2. **`test_training` compilation error**: `Tensor::get` is not a member of `serius::Tensor` at `tests/test_training.cpp:101`. The `Tensor` class (declared in `src/tensor/tensor.h`) is missing a `get()` accessor method, or the test uses an outdated API.

3. **Build log pollution**: 15+ `build_*.txt` files in project root suggest iterative build attempts without cleanup. These should be in `build/` or `.gitignore`d.

## Code Quality Observations

- **Naming**: Italian comments and documentation; English code identifiers
- **Header hygiene**: Most headers use proper include guards or pragma once
- **Test coverage**: 30 test files covering major subsystems, but `test_memory` can't run on MSVC
- **SIMD dispatch**: Proper runtime detection in `blas/dispatch.cpp` (AVX2, AVX-512F)
- **Modularity**: Clean separation of concerns across 22 modules
- **Build artifacts in repo**: `build_*.txt` files should not be tracked in version control

## Recommendations

1. Fix `Tensor::get` method or update `test_training.cpp` to use the current API
2. Increase `TEST_DISCOVERY_TIMEOUT` for `test_memory` or optimize test initialization
3. Add `build_*.txt` to `.gitignore` and remove from tracking
4. Add CI configuration (GitHub Actions) for automated builds
5. Complete API stability before expanding to later phases
