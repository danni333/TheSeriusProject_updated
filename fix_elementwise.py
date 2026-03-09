import re

with open('src/blas/elementwise.cpp', 'r') as f:
    code = f.read()

# Replace A, B, C with tensor_a, tensor_b, tensor_c
code = re.sub(r'\bA\b', 'tensor_a', code)
code = re.sub(r'\bB\b', 'tensor_b', code)
code = re.sub(r'\bC\b', 'tensor_c', code)

# Replace d with dim in the loops: for (int d = ... ; d >= 0; --d)
code = re.sub(r'\bd\b', 'dim', code)

# Fix 1e-7f to 1e-7F
code = code.replace('1e-7f', '1e-7F')

# Braces for the single-line if statement inside the loop
# Original: if (indices[dim] < tensor_a.shape()[dim]) break;
# New: if (indices[dim] < tensor_a.shape()[dim]) {\n                    break;\n                }
code = re.sub(r'if \(indices\[dim\] < ([^)]+)\) break;', r'if (indices[dim] < \1) {\n                    break;\n                }', code)

# Adding immintrin.h at top
old_includes = """#include <cassert>
#include <stdexcept>

#include "blas.h"
"""
new_includes = """#include <cassert>
#include <stdexcept>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#include "blas.h"
"""
code = code.replace(old_includes, new_includes)

# AVX2 for add
old_add = """#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }"""
new_add = """#if defined(__AVX2__) || defined(__AVX__)
        size_t simd_size = size - (size % 8);
#pragma omp parallel for if (size > 10000)
        for (int64_t j = 0; j < static_cast<int64_t>(simd_size); j += 8) {
            __m256 a = _mm256_loadu_ps(a_ptr + j);
            __m256 b = _mm256_loadu_ps(b_ptr + j);
            __m256 c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(c_ptr + j, c);
        }
        for (size_t i = simd_size; i < size; ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
#else
#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
#endif"""
code = code.replace(old_add, new_add)

# AVX2 for mul
old_mul = """#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] * b_ptr[i];
        }"""
new_mul = """#if defined(__AVX2__) || defined(__AVX__)
        size_t simd_size = size - (size % 8);
#pragma omp parallel for if (size > 10000)
        for (int64_t j = 0; j < static_cast<int64_t>(simd_size); j += 8) {
            __m256 a = _mm256_loadu_ps(a_ptr + j);
            __m256 b = _mm256_loadu_ps(b_ptr + j);
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(c_ptr + j, c);
        }
        for (size_t i = simd_size; i < size; ++i) {
            c_ptr[i] = a_ptr[i] * b_ptr[i];
        }
#else
#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] * b_ptr[i];
        }
#endif"""
code = code.replace(old_mul, new_mul)

with open('src/blas/elementwise.cpp', 'w') as f:
    f.write(code)
