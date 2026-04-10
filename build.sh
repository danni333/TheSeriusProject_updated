#!/bin/bash
set -e

echo "Creating build directory..."
mkdir -p build

echo "Running CMake..."
if ! cmake -S . -B build; then
    echo "CMake configuration FAILED"
    exit 1
fi
echo "CMake configuration SUCCEEDED"

echo "Building project..."
if ! cmake --build build -j4; then
    echo "Build FAILED"
    exit 1
fi
echo "Build SUCCEEDED"

echo "All steps completed successfully!"