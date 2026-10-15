#!/bin/bash
set -e

echo "=== Building TheSeriusProject_updated ==="

# 1. Create build/ directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

# 2. Run CMake configuration
echo "Running CMake configuration..."
if ! cmake -S . -B build; then
    echo "FAILURE: CMake configuration failed"
    exit 1
fi

# 3. Build the project
echo "Building project..."
if ! cmake --build build -j4; then
    echo "FAILURE: Build failed"
    exit 1
fi

# 4. Success
echo "SUCCESS: Build completed successfully"
exit 0