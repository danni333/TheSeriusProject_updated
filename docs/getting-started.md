# Quick Start Guide

Get up and running with Serius LLM Engine in 5 minutes.

## Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2022)
- CMake 3.25+
- (Optional) CUDA 12+ for GPU support

## 1. Build from Source

```bash
git clone https://github.com/your-org/serius.git
cd serius
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## 2. Download a Model

```bash
# Import a small model from HuggingFace
./serius import --source hf://Qwen/Qwen2.5-0.5B-Instruct --output ./models/
```

## 3. Chat with the Model

```bash
# Interactive chat
./serius chat --model ./models/Qwen2.5-0.5B-Instruct --temperature 0.7
```

## 4. Start the Web UI

```bash
# Start HTTP server with web UI
./serius serve --model ./models/Qwen2.5-0.5B-Instruct --port 8080

# Open http://localhost:8080 in your browser
```

## 5. Use the API

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
