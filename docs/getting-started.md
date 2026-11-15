# Quick Start Guide

Get up and running with Serius LLM Engine in 5 minutes.

## Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2022)
- CMake 3.25+
- (Optional) CUDA 12+ for GPU support

## 1. Build from Source

```bash
git clone https://github.com/danni333/TheSeriusProject_updated.git
cd TheSeriusProject_updated
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
# Binary: ./src/serius (Linux) or src/serius.exe (Windows)
```

## 2. Get a Model

```bash
# Directly from HuggingFace (downloads the GGUF at runtime)
./src/serius serve -m hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q4_k_m

# Or a local GGUF / safetensors dir / config.json
./src/serius serve -m ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf --type llama
```

## 3. Chat

```bash
# Web UI (open http://localhost:8080)
./src/serius serve -m ./models/model.gguf --port 8080 --ui ui

# Or interactive CLI
./src/serius chat -m ./models/model.gguf --temperature 0.7
```

Additional models can be loaded/unloaded while the server runs via `POST /v1/models/load` and `DELETE /v1/models/{id}` (or the Models panel in the UI).

## 4. Use the API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Next Steps

- See [CLI Reference](cli-reference.md) for all commands
- See [API Reference](api-reference.md) for HTTP endpoints
- See [Supported Models](supported-models.md) for model compatibility
- See [GPU Setup](gpu-setup.md) for GPU acceleration
