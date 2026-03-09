#include <cassert>
#include <stdexcept>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#include "blas.h"

namespace serius {
namespace core {
namespace blas {

void element_wise_add(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c)
{
    if (tensor_a.shape() != tensor_b.shape() || tensor_a.shape() != tensor_c.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_add");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous() && tensor_c.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        const float *b_ptr = tensor_b.data();
        float *c_ptr = tensor_c.data();
        size_t size = tensor_a.size();

// Simple scalar loop (compiler can auto-vectorize this easily)
#if defined(__AVX2__) || defined(__AVX__)
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
#endif
    } else {
        // Fallback for non-contiguous tensors
        // In a real optimized engine we would use an N-dimensional iterator
        // For simplicity and since most parameters are contiguous, we use flat iteration.
        // Actually, since this is an element-wise op, we can just iterate the indices
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            tensor_c.at(indices) = tensor_a.at(indices) + tensor_b.at(indices);

            // Increment indices
            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

void element_wise_mul(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c)
{
    if (tensor_a.shape() != tensor_b.shape() || tensor_a.shape() != tensor_c.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_mul");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous() && tensor_c.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        const float *b_ptr = tensor_b.data();
        float *c_ptr = tensor_c.data();
        size_t size = tensor_a.size();

#if defined(__AVX2__) || defined(__AVX__)
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
#endif
    } else {
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            tensor_c.at(indices) = tensor_a.at(indices) * tensor_b.at(indices);

            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

void element_wise_sub(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c)
{
    if (tensor_a.shape() != tensor_b.shape() || tensor_a.shape() != tensor_c.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_sub");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous() && tensor_c.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        const float *b_ptr = tensor_b.data();
        float *c_ptr = tensor_c.data();
        size_t size = tensor_a.size();

#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    } else {
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            tensor_c.at(indices) = tensor_a.at(indices) - tensor_b.at(indices);

            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

void element_wise_div(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c)
{
    if (tensor_a.shape() != tensor_b.shape() || tensor_a.shape() != tensor_c.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_div");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous() && tensor_c.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        const float *b_ptr = tensor_b.data();
        float *c_ptr = tensor_c.data();
        size_t size = tensor_a.size();

#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            c_ptr[i] = a_ptr[i] / (b_ptr[i] == 0.0f ? 1e-7F : b_ptr[i]);
        }
    } else {
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            float denom = tensor_b.at(indices);
            tensor_c.at(indices) = tensor_a.at(indices) / (denom == 0.0f ? 1e-7F : denom);

            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

void scale(const Tensor &tensor_a, float alpha, Tensor &tensor_b)
{
    if (tensor_a.shape() != tensor_b.shape()) {
        throw std::invalid_argument("Shapes must match for scale");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        float *b_ptr = tensor_b.data();
        size_t size = tensor_a.size();

#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            b_ptr[i] = a_ptr[i] * alpha;
        }
    } else {
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            tensor_b.at(indices) = tensor_a.at(indices) * alpha;

            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

void element_wise_apply(const Tensor &tensor_a, float (*func)(float), Tensor &tensor_b)
{
    if (tensor_a.shape() != tensor_b.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_apply");
    }

    if (tensor_a.is_contiguous() && tensor_b.is_contiguous()) {
        const float *a_ptr = tensor_a.data();
        float *b_ptr = tensor_b.data();
        size_t size = tensor_a.size();

#pragma omp parallel for if (size > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
            b_ptr[i] = func(a_ptr[i]);
        }
    } else {
        std::vector<size_t> indices(tensor_a.shape().size(), 0);
        size_t size = tensor_a.size();
        for (size_t i = 0; i < size; ++i) {
            tensor_b.at(indices) = func(tensor_a.at(indices));

            for (int dim = static_cast<int>(tensor_a.shape().size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensor_a.shape()[dim]) break;
                indices[dim] = 0;
            }
        }
    }
}

// --- Convenience Wrappers ---

Tensor element_wise_add(const Tensor &tensor_a, const Tensor &tensor_b)
{
    Tensor tensor_c(tensor_a.shape());
    element_wise_add(tensor_a, tensor_b, tensor_c);
    return tensor_c;
}

Tensor element_wise_sub(const Tensor &tensor_a, const Tensor &tensor_b)
{
    Tensor tensor_c(tensor_a.shape());
    element_wise_sub(tensor_a, tensor_b, tensor_c);
    return tensor_c;
}

Tensor element_wise_mul(const Tensor &tensor_a, const Tensor &tensor_b)
{
    Tensor tensor_c(tensor_a.shape());
    element_wise_mul(tensor_a, tensor_b, tensor_c);
    return tensor_c;
}

Tensor element_wise_div(const Tensor &tensor_a, const Tensor &tensor_b)
{
    Tensor tensor_c(tensor_a.shape());
    element_wise_div(tensor_a, tensor_b, tensor_c);
    return tensor_c;
}

Tensor scale(const Tensor &tensor_a, float alpha)
{
    Tensor tensor_b(tensor_a.shape());
    scale(tensor_a, alpha, tensor_b);
    return tensor_b;
}

Tensor element_wise_apply(const Tensor &tensor_a, float (*func)(float))
{
    Tensor tensor_b(tensor_a.shape());
    element_wise_apply(tensor_a, func, tensor_b);
    return tensor_b;
}

}  // namespace blas
}  // namespace core
}  // namespace serius
