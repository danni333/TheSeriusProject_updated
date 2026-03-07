#include <gtest/gtest.h>

#include <cmath>

#include "../src/blas/blas.h"
#include "../src/tensor/tensor.h"


namespace serius {
namespace blas {
namespace test {

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

    // Dispatch checks capabilities and should run scalar if AVX not present,
    // but results must be completely identical regardless of path taken.
    matmul(A, B, C);

    // Expected C =
    // [[ 1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12 ],
    //  [ 4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12 ]]
    // = [[ 58, 64 ],
    //    [ 139, 154 ]]

    EXPECT_FLOAT_EQ(C.at(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(C.at(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(C.at(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(C.at(1, 1), 154.0f);
}

TEST(BlasTest, MatmulLarge)
{
    // Large matrix test (triggers block tiling optimization seamlessly)
    const size_t M = 128;
    const size_t K = 256;
    const size_t N = 128;

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_expected({M, N});
    Tensor C({M, N});

    // Fill with random
    A.rand();
    B.rand();
    C.zeros();
    C_expected.zeros();

    // Compute scalar reference
    matmul_scalar(A.data(), B.data(), C_expected.data(), M, N, K, A.strides()[0], B.strides()[0],
                  C_expected.strides()[0]);

    // Compute dispatch
    matmul(A, B, C);

    for (size_t i = 0; i < C.size(); ++i) {
        // High floating point tolerance depending on summation order due to FMA vs separated
        EXPECT_NEAR(C[i], C_expected[i], 1e-4f) << " at index " << i;
    }
}

TEST(BlasTest, MatVec)
{
    Tensor A({3, 3});
    Tensor x({3});
    Tensor y({3});

    // A = Identity * 2
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

}  // namespace test
}  // namespace blas
}  // namespace serius
