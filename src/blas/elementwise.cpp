#include <cassert>
#include <stdexcept>

#include "blas.h"

namespace serius {
namespace blas {

void element_wise_add(const Tensor &A, const Tensor &B, Tensor &C)
{
    if (A.shape() != B.shape() || A.shape() != C.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_add");
    }

    if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
        const float *a_ptr = A.data();
        const float *b_ptr = B.data();
        float *c_ptr = C.data();
        size_t size = A.size();

        // Simple scalar loop (compiler can auto-vectorize this easily)
        for (size_t i = 0; i < size; ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    } else {
        // Fallback for non-contiguous tensors
        // In a real optimized engine we would use an N-dimensional iterator
        // For simplicity and since most parameters are contiguous, we use flat iteration.
        // Actually, since this is an element-wise op, we can just iterate the indices
        std::vector<size_t> indices(A.shape().size(), 0);
        size_t size = A.size();
        for (size_t i = 0; i < size; ++i) {
            C.at(indices) = A.at(indices) + B.at(indices);

            // Increment indices
            for (int d = static_cast<int>(A.shape().size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < A.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }
}

void element_wise_mul(const Tensor &A, const Tensor &B, Tensor &C)
{
    if (A.shape() != B.shape() || A.shape() != C.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_mul");
    }

    if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
        const float *a_ptr = A.data();
        const float *b_ptr = B.data();
        float *c_ptr = C.data();
        size_t size = A.size();

        for (size_t i = 0; i < size; ++i) {
            c_ptr[i] = a_ptr[i] * b_ptr[i];
        }
    } else {
        std::vector<size_t> indices(A.shape().size(), 0);
        size_t size = A.size();
        for (size_t i = 0; i < size; ++i) {
            C.at(indices) = A.at(indices) * B.at(indices);

            for (int d = static_cast<int>(A.shape().size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < A.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }
}

void scale(const Tensor &A, float alpha, Tensor &B)
{
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("Shapes must match for scale");
    }

    if (A.is_contiguous() && B.is_contiguous()) {
        const float *a_ptr = A.data();
        float *b_ptr = B.data();
        size_t size = A.size();

        for (size_t i = 0; i < size; ++i) {
            b_ptr[i] = a_ptr[i] * alpha;
        }
    } else {
        std::vector<size_t> indices(A.shape().size(), 0);
        size_t size = A.size();
        for (size_t i = 0; i < size; ++i) {
            B.at(indices) = A.at(indices) * alpha;

            for (int d = static_cast<int>(A.shape().size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < A.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }
}

void element_wise_apply(const Tensor &A, float (*func)(float), Tensor &B)
{
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("Shapes must match for element_wise_apply");
    }

    if (A.is_contiguous() && B.is_contiguous()) {
        const float *a_ptr = A.data();
        float *b_ptr = B.data();
        size_t size = A.size();

        for (size_t i = 0; i < size; ++i) {
            b_ptr[i] = func(a_ptr[i]);
        }
    } else {
        std::vector<size_t> indices(A.shape().size(), 0);
        size_t size = A.size();
        for (size_t i = 0; i < size; ++i) {
            B.at(indices) = func(A.at(indices));

            for (int d = static_cast<int>(A.shape().size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < A.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }
}

// --- Convenience Wrappers ---

Tensor element_wise_add(const Tensor &A, const Tensor &B)
{
    Tensor C(A.shape());
    element_wise_add(A, B, C);
    return C;
}

Tensor element_wise_mul(const Tensor &A, const Tensor &B)
{
    Tensor C(A.shape());
    element_wise_mul(A, B, C);
    return C;
}

Tensor scale(const Tensor &A, float alpha)
{
    Tensor B(A.shape());
    scale(A, alpha, B);
    return B;
}

Tensor element_wise_apply(const Tensor &A, float (*func)(float))
{
    Tensor B(A.shape());
    element_wise_apply(A, func, B);
    return B;
}

}  // namespace blas
}  // namespace serius
