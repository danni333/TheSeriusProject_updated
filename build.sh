#!/bin/bash
set -e

echo "=== Building TheSeriusProject_updated ==="

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

echo "Running CMake configuration..."
if ! cmake -S . -B build; then
    echo "FAILURE: CMake configuration failed"
    exit 1
fi

echo "Building project..."
if ! cmake --build build -j4; then
    echo "FAILURE: Build failed"
    exit 1
fi

echo "SUCCESS: Build completed successfully"
exit 0