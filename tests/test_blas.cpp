#include <gtest/gtest.h>

#include <cmath>

#include "../src/blas/blas.h"
#include "../src/tensor/tensor.h"

namespace serius {
namespace core {
namespace blas {

void matmul_avx2(const float *a, const float *b, float *c, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);

namespace test {

namespace core = serius::core;
using core::Tensor;

TEST(BlasTest, MatmulNaive)
{
    Tensor A({2, 3});
    Tensor B({3, 2});
    Tensor C({2, 2});
    C.zeros();

    // A = [[1, 2, 3], [4, 5, 6]]
    A.at(0, 0) = 1;
    A.at(0, 1) = 2;
    A.at(0, 2) = 3;
    A.at(1, 0) = 4;
    A.at(1, 1) = 5;
    A.at(1, 2) = 6;

    // B = [[7, 8], [9, 10], [11, 12]]
    B.at(0, 0) = 7;
    B.at(0, 1) = 8;
    B.at(1, 0) = 9;
    B.at(1, 1) = 10;
    B.at(2, 0) = 11;
    B.at(2, 1) = 12;

    matmul(A, B, C);

    EXPECT_FLOAT_EQ(C.at(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(C.at(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(C.at(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(C.at(1, 1), 154.0f);
}

TEST(BlasTest, MatmulLarge)
{
    const size_t M = 128;
    const size_t K = 256;
    const size_t N = 128;

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_expected({M, N});
    Tensor C({M, N});

    A.rand();
    B.rand();
    C.zeros();
    C_expected.zeros();

    matmul_scalar(A.data(), B.data(), C_expected.data(), M, N, K, A.strides()[0], B.strides()[0],
                  C_expected.strides()[0]);

    matmul(A, B, C);

    C.syncToHost();  // Ensure data is synced if matmul ran on GPU

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_NEAR(C[i], C_expected[i], 1e-4f) << " at index " << i;
    }
}

TEST(BlasTest, MatmulCPUAVX2)
{
    const size_t M = 200;
    const size_t K = 256;
    const size_t N = 200;

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_expected({M, N});
    Tensor C({M, N});

    A.rand();
    B.rand();
    C.zeros();
    C_expected.zeros();

    matmul_scalar(A.data(), B.data(), C_expected.data(), M, N, K, A.strides()[0], B.strides()[0],
                  C_expected.strides()[0]);

    // Force CPU AVX2 dispatch by calling it directly
    matmul_avx2(A.data(), B.data(), C.data(), M, N, K, A.strides()[0], B.strides()[0], C.strides()[0]);

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_NEAR(C.data()[i], C_expected.data()[i], 1e-4f) << " at index " << i;
    }
}

TEST(BlasTest, MatVec)
{
    Tensor A({3, 3});
    Tensor x({3});
    Tensor y({3});

    A.zeros();
    A.at(0, 0) = 2;
    A.at(1, 1) = 2;
    A.at(2, 2) = 2;

    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    y.zeros();

    matvec(A, x, y);

    EXPECT_FLOAT_EQ(y[0], 2.0f);
    EXPECT_FLOAT_EQ(y[1], 4.0f);
    EXPECT_FLOAT_EQ(y[2], 6.0f);
}

TEST(BlasTest, ElementWiseAdd)
{
    Tensor A({2, 2});
    Tensor B({2, 2});
    Tensor C({2, 2});

    A.fill(1.5f);
    B.fill(2.5f);

    element_wise_add(A, B, C);

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 4.0f);
    }
}

TEST(BlasTest, ElementWiseMul)
{
    Tensor A({2, 2});
    Tensor B({2, 2});
    Tensor C({2, 2});

    A.fill(3.0f);
    B.fill(4.0f);

    element_wise_mul(A, B, C);

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 12.0f);
    }
}

TEST(BlasTest, Scale)
{
    Tensor A({2, 2});
    Tensor B({2, 2});

    A.fill(2.0f);

    scale(A, 3.5f, B);

    for (size_t i = 0; i < B.size(); ++i) {
        EXPECT_FLOAT_EQ(B[i], 7.0f);
    }
}

float sample_relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

TEST(BlasTest, ElementWiseApply)
{
    Tensor A({2});
    Tensor B({2});

    A[0] = -5.0f;
    A[1] = 5.0f;

    element_wise_apply(A, sample_relu, B);

    EXPECT_FLOAT_EQ(B[0], 0.0f);
    EXPECT_FLOAT_EQ(B[1], 5.0f);
}

TEST(BlasTest, ElementWiseSub)
{
    Tensor A({2, 2});
    Tensor B({2, 2});
    Tensor C({2, 2});

    A.fill(5.0f);
    B.fill(2.5f);

    element_wise_sub(A, B, C);

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 2.5f);
    }
}

TEST(BlasTest, ElementWiseDiv)
{
    Tensor A({2, 2});
    Tensor B({2, 2});
    Tensor C({2, 2});

    A.fill(10.0f);
    B.fill(2.0f);

    element_wise_div(A, B, C);

    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 5.0f);
    }
}

TEST(BlasTest, ElementWiseOpenMP)
{
    size_t large_size = 20000;
    Tensor A({large_size});
    Tensor B({large_size});
    Tensor C({large_size});

    A.fill(1.0f);
    B.fill(2.0f);

    element_wise_add(A, B, C);

    for (size_t i = 0; i < large_size; ++i) {
        if (std::abs(C[i] - 3.0f) > 1e-5f) {
            FAIL() << "ElementWiseOpenMP failed at index " << i << ": expected 3.0, got " << C[i];
        }
    }
}

}  // namespace test
}  // namespace blas
}  // namespace core
}  // namespace serius
