# Installation Guide

Detailed installation instructions for Serius LLM Engine.

## Building from Source

### Prerequisites

- **C++20 compiler**: GCC 10+, Clang 12+, or MSVC 2022
- **CMake**: Version 3.25 or higher
- **Git**: For cloning the repository

### Optional Dependencies

- **CUDA**: 12.0+ for GPU acceleration (NVIDIA)
- **ROCm**: For AMD GPU support
- **Metal**: For Apple Silicon GPU support

### Build Steps

```bash
# Clone the repository
git clone https://github.com/your-org/serius.git
cd serius

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DSERIUS_USE_CUDA=ON  # Enable CUDA if available

# Build
cmake --build . -j$(nproc)

# (Optional) Install
cmake --install .
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `SERIUS_USE_CUDA` | Enable CUDA support | OFF |
| `SERIUS_USE_METAL` | Enable Metal support (macOS) | OFF |
| `SERIUS_USE_NCCL` | Enable NCCL for distributed training | OFF |
| `SERIUS_USE_MPI` | Enable MPI for distributed training | OFF |

### Troubleshooting Build Errors

**CMake version too old:**
```bash
# Install newer CMake via package manager
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake
```

**CUDA not found:**
```bash
# Set CUDA path manually
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0
```

## Installing Pre-built Binaries

### Linux

```bash
# Download latest release
wget https://github.com/your-org/serius/releases/latest/download/serius-linux-x64.tar.gz

# Extract
tar -xzf serius-linux-x64.tar.gz

# Add to PATH
export PATH=$PATH:$(pwd)/serius-linux-x64/bin
```

### macOS

```bash
# Download latest release
curl -L https://github.com/your-org/serius/releases/latest/download/serius-macos-x64.tar.gz -o serius.tar.gz

# Extract
tar -xzf serius.tar.gz

# Add to PATH
export PATH=$PATH:$(pwd)/serius-macos-x64/bin
```

### Windows

Download the installer from [GitHub Releases](https://github.com/your-org/serius/releases).

## GPU Setup

### NVIDIA (CUDA)

1. Install CUDA Toolkit 12.0+: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN 8.9+: https://developer.nvidia.com/cudnn
3. Build with `SERIUS_USE_CUDA=ON`

### AMD (ROCm)

1. Install ROCm: https://rocm.docs.amd.com/
2. Build with `SERIUS_USE_ROCM=ON` (when available)

### Apple Silicon (Metal)

No additional setup needed on macOS with Apple Silicon. Build with `SERIUS_USE_METAL=ON`.

## Verification

Test your installation:

```bash
# Check version
serius --version

# Run diagnostics
serius doctor

# Try a simple generation (requires model)
serius chat --model ./models/model.gguf --max-tokens 10
```

## Next Steps

- [Quick Start Guide](getting-started.md)
- [CLI Reference](cli-reference.md)
- [Supported Models](supported-models.md)
