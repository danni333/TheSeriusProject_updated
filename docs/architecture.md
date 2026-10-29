# Architecture

Internal architecture documentation for developers.

## Overview

Serius is a C++20 LLM engine with:
- Core tensor operations and BLAS
- Automatic differentiation (autograd)
- Transformer model implementations
- Distributed training support
- Quantization and optimization
- HTTP API server with OpenAI compatibility

## Directory Structure

```
src/
├── tensor/          # Core tensor operations, memory management
├── blas/            # BLAS implementations (scalar, AVX2, AVX512, CUDA)
├── autograd/        # Computational graph, backpropagation
├── transformer/     # Transformer models (LLaMA, Qwen, generic)
├── attention/       # Attention mechanisms (Flash Attention, Sparse)
├── tokenizer/       # BPE tokenizer
├── infer/           # Inference engine, KV cache, sampling
├── train/           # Training infrastructure
├── optim/           # Optimizers (AdamW, schedulers)
├── quant/           # Quantization (INT4, FP8, AWQ, GPTQ)
├── distributed/     # Distributed training (DataParallel, TensorParallel)
├── server/          # HTTP API server
├── cli/             # Command-line interface
├── hub/             # Model hub and downloader
├── knowledge/       # Semantic hashing and knowledge function
├── autoopt/         # Auto-refactoring and hot-swapping
└── hal/             # Hardware abstraction layer
```

## Core Components

### Tensor

- N-dimensional contiguous tensors
- Memory pool for efficient allocation
- Shape-aware operations

### BLAS

- Scalar fallback
- AVX2 SIMD optimization
- AVX512 SIMD optimization
- CUDA kernel implementation

### Autograd

- Computational graph with Node
- Topological sort for backprop
- Standard operations (matmul, relu, softmax, etc.)

### Transformer

- LLaMA architecture
- Qwen architecture
- Generic configurable transformer
- MoE (Mixture of Experts) support

### Attention

- Multi-head attention
- Flash Attention (CUDA)
- Sliding window attention
- Sparse attention patterns

### Inference

- KV cache management
- Sampling strategies (temperature, top-k, top-p)
- Speculative decoding support

### Training

- AdamW optimizer
- Learning rate schedulers (warmup, cosine decay)
- Automatic mixed precision (AMP)
- Checkpointing

### Quantization

- INT4, INT8, FP8, FP16
- AWQ (Activation-aware Weight Quantization)
- GPTQ
- Dequantization kernels

### Distributed

- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- Communication backends (NCCL, MPI, local)

## Namespaces

- `serius::core` - Tensor, BLAS, memory, autograd
- `serius::model` - Transformer models
- `serius::optim` - Optimizers and schedulers
- `serius::training` - Training infrastructure
- `serius::infer` - Inference engine
- `serius::server` - HTTP API server
- `serius::hub` - Model hub
- `serius::knowledge` - Semantic hashing
- `serius::autoopt` - Auto-refactoring

## Design Principles

1. **Zero-copy** where possible
2. **SIMD optimization** for CPU operations
3. **CUDA kernels** for GPU acceleration
4. **Header-only** for core tensor operations (flexibility)
5. **OpenAI compatibility** for API server
6. **Extensible** architecture for new models and ops

## Performance Optimization

- **Memory**: Memory pool, paged KV cache
- **Compute**: SIMD, kernel fusion, Flash Attention
- **I/O**: Streaming generation, batching
- **Quantization**: INT4/FP8 for reduced memory

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
