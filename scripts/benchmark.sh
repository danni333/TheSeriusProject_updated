#!/usr/bin/env bash
# TheSeriusProject - Run benchmarks (matmul, memory, inference)
# Usage: ./scripts/benchmark.sh [build_dir]
# Default build_dir: build

set -e
BUILD_DIR="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Build directory $BUILD_DIR not found. Run: cmake -B $BUILD_DIR && cmake --build $BUILD_DIR"
    exit 1
fi

echo "=== Running benchmarks from $BUILD_DIR ==="

# Run tests that have benchmark-style output (if any)
if [[ -f "$BUILD_DIR/tests/test_blas" ]]; then
    echo "--- test_blas ---"
    "$BUILD_DIR/tests/test_blas" --gtest_filter="*Benchmark*" 2>/dev/null || true
fi
if [[ -f "$BUILD_DIR/tests/test_memory" ]]; then
    echo "--- test_memory ---"
    "$BUILD_DIR/tests/test_memory" --gtest_filter="*Benchmark*" 2>/dev/null || true
fi

# CTest runs all tests; use -R to run only specific tests
echo "--- ctest (all) ---"
ctest --test-dir "$BUILD_DIR" --output-on-failure -V 2>&1 | tail -50

echo "=== Done ==="
