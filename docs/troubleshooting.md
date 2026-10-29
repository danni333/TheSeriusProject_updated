# Troubleshooting

Common issues and solutions.

## Build Issues

### CMake version too old

**Error:** `CMake 3.25 or higher is required`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

### CUDA not found

**Error:** `CUDA Toolkit not found`

**Solution:**
```bash
# Install CUDA from https://developer.nvidia.com/cuda-downloads
# Set environment variable
export CUDA_HOME=/usr/local/cuda-12.0

# Or specify path in CMake
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0
```

### Compilation errors with AVX

**Error:** AVX/AVX512 compilation fails

**Solution:**
```bash
# Disable AVX-specific flags
cmake .. -DSERIUS_DISABLE_AVX=ON
```

## Runtime Issues

### Model not found

**Error:** `Failed to load model: file not found`

**Solution:**
- Check model path is correct
- Ensure model is in the expected format (GGUF, Safetensors)
- Use `serius import` to download models

### Out of memory

**Error:** `CUDA out of memory` or `Out of memory`

**Solution:**
- Use a smaller model or quantization (INT4 instead of FP16)
- Reduce `max_tokens` parameter
- Use a model with smaller context window
- On CPU, ensure sufficient RAM (model size should be < available RAM)

### Slow inference

**Issue:** Generation is very slow

**Solution:**
- Enable GPU support (CUDA/Metal)
- Use quantized model (INT4, FP8)
- Increase `--threads` for CPU inference
- Enable Flash Attention (CUDA)
- Use smaller model

### Segmentation fault

**Error:** Program crashes with segfault

**Solution:**
- Ensure model format is compatible
- Check tensor dimensions match model config
- Try with a different model
- Enable debug build: `cmake .. -DCMAKE_BUILD_TYPE=Debug`

## API Issues

### Connection refused

**Error:** `curl: (7) Failed to connect to localhost port 8080`

**Solution:**
- Ensure server is running: `serius serve --model ...`
- Check port is not in use: `netstat -an | grep 8080`
- Try different port: `serius serve --port 8081`

### 404 Not Found

**Error:** HTTP 404 when accessing endpoint

**Solution:**
- Check endpoint URL is correct (`/v1/chat/completions`, not `/chat/completions`)
- Ensure server is running with a model loaded

### Streaming doesn't work

**Issue:** No response when `stream: true`

**Solution:**
- Ensure client supports SSE (Server-Sent Events)
- Check network doesn't buffer chunks
- Try non-streaming mode first to verify model works

## GPU Issues

### CUDA not detected

**Error:** GPU not found in diagnostics

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Rebuild with CUDA enabled
cmake .. -DSERIUS_USE_CUDA=ON
cmake --build .
```

### Metal not working on macOS

**Issue:** Metal acceleration not working on Apple Silicon

**Solution:**
```bash
# Rebuild with Metal enabled
cmake .. -DSERIUS_USE_METAL=ON
cmake --build .
```

## Getting Help

If you encounter an issue not listed here:

1. Check the [GitHub Issues](https://github.com/your-org/serius/issues)
2. Search existing issues for similar problems
3. Create a new issue with:
   - Serius version (`serius --version`)
   - OS and compiler version
   - Full error message
   - Steps to reproduce
   - System diagnostics (`serius doctor`)
