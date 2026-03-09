#include <gtest/gtest.h>

#include <cmath>

#include "../src/memory/memory_pool.h"
#include "../src/tensor/tensor.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;

TEST(TensorTest, CreateAndShape)
{
    Tensor t({3, 4, 5});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
    EXPECT_EQ(t.shape()[2], 5);
    EXPECT_EQ(t.size(), 60);
    EXPECT_TRUE(t.owns_data());
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, ZeroInitialized)
{
    Tensor t({10, 10});
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST(TensorTest, FillAndAccess)
{
    Tensor t({3, 4});
    t.fill(7.0f);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(t.at(i, j), 7.0f);
        }
    }
}

TEST(TensorTest, StridesRowMajor)
{
    Tensor t({2, 3, 4});
    // Row-major: strides should be [12, 4, 1]
    EXPECT_EQ(t.strides()[0], 12);
    EXPECT_EQ(t.strides()[1], 4);
    EXPECT_EQ(t.strides()[2], 1);
}

TEST(TensorTest, IndexedAccess2D)
{
    Tensor t({3, 4});
    // Write manually
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t.at(i, j) = static_cast<float>(i * 4 + j);
        }
    }
    // Verify with flat access
    for (size_t k = 0; k < 12; ++k) {
        EXPECT_FLOAT_EQ(t[k], static_cast<float>(k));
    }
}

TEST(TensorTest, IndexedAccess3D)
{
    Tensor t({2, 3, 4});
    float val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t.at(i, j, k) = val;
                val += 1.0f;
            }
        }
    }
    // Verify contiguous storage
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], static_cast<float>(i));
    }
}

TEST(TensorTest, Reshape)
{
    Tensor t({3, 4});
    t.fill(1.0f);

    Tensor reshaped = t.reshape({12});
    EXPECT_EQ(reshaped.ndim(), 1);
    EXPECT_EQ(reshaped.shape()[0], 12);
    EXPECT_EQ(reshaped.size(), 12);

    // Should share data (view)
    EXPECT_FALSE(reshaped.owns_data());
    EXPECT_EQ(reshaped.data(), t.data());
}

TEST(TensorTest, ReshapeSizeMismatch)
{
    Tensor t({3, 4});
    EXPECT_THROW(t.reshape({5, 5}), std::invalid_argument);
}

TEST(TensorTest, Slice)
{
    Tensor t({4, 5});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            t.at(i, j) = static_cast<float>(i * 10 + j);
        }
    }

    // Slice rows 1..3 (exclusive)
    Tensor sliced = t.slice(0, 1, 3);
    EXPECT_EQ(sliced.shape()[0], 2);
    EXPECT_EQ(sliced.shape()[1], 5);
    EXPECT_FALSE(sliced.owns_data());

    // sliced[0, 0] should be original [1, 0] = 10.0
    EXPECT_FLOAT_EQ(sliced.at(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(sliced.at(1, 0), 20.0f);
}

TEST(TensorTest, SliceColumns)
{
    Tensor t({3, 6});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            t.at(i, j) = static_cast<float>(i * 6 + j);
        }
    }

    // Slice columns 2..4
    Tensor sliced = t.slice(1, 2, 4);
    EXPECT_EQ(sliced.shape()[0], 3);
    EXPECT_EQ(sliced.shape()[1], 2);

    // sliced[0, 0] = original[0, 2] = 2.0
    EXPECT_FLOAT_EQ(sliced.at(0, 0), 2.0f);
    // sliced[1, 0] = original[1, 2] = 8.0
    EXPECT_FLOAT_EQ(sliced.at(1, 0), 8.0f);
}

TEST(TensorTest, ViewSharesData)
{
    Tensor t({4, 4});
    t.fill(3.0f);

    Tensor view = t.reshape({16});
    EXPECT_EQ(view.data(), t.data());

    // Modify through view
    view[0] = 99.0f;
    EXPECT_FLOAT_EQ(t.at(0, 0), 99.0f);
}

TEST(TensorTest, DeepCopy)
{
    Tensor t({3, 3});
    t.fill(5.0f);

    Tensor c = t.copy();
    EXPECT_NE(c.data(), t.data());
    EXPECT_TRUE(c.owns_data());

    // Modify copy, original unchanged
    c.at(0, 0) = 999.0f;
    EXPECT_FLOAT_EQ(t.at(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(c.at(0, 0), 999.0f);
}

TEST(TensorTest, Transpose)
{
    Tensor t({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t.at(i, j) = static_cast<float>(i * 3 + j);
        }
    }

    Tensor tr = t.transpose();
    EXPECT_EQ(tr.shape()[0], 3);
    EXPECT_EQ(tr.shape()[1], 2);

    // tr[j, i] == t[i, j]
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(tr.at(j, i), t.at(i, j));
        }
    }

    // Transpose is a view, non-contiguous
    EXPECT_FALSE(tr.is_contiguous());
}

TEST(TensorTest, MemoryPoolAllocation)
{
    core::memory::MemoryPool pool(4096);
    Tensor t({4, 4}, &pool);

    t.fill(2.0f);
    EXPECT_FLOAT_EQ(t.at(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(t.at(3, 3), 2.0f);
    EXPECT_TRUE(t.owns_data());
    EXPECT_GT(pool.used(), 0);
}

TEST(TensorTest, MoveSemantics)
{
    Tensor t1({3, 3});
    t1.fill(42.0f);
    float *original_data = t1.data();

    Tensor t2(std::move(t1));
    EXPECT_EQ(t2.data(), original_data);
    EXPECT_EQ(t2.at(1, 1), 42.0f);
    EXPECT_EQ(t1.data(), nullptr);
    EXPECT_EQ(t1.size(), 0);
}

TEST(TensorTest, OnesAndRand)
{
    Tensor t({5, 5});
    t.ones();
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 1.0f);
    }

    t.rand();
    // At least some diversity
    bool all_same = true;
    float first = t[0];
    for (size_t i = 1; i < t.size(); ++i) {
        if (std::fabs(t[i] - first) > 1e-7f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(TensorTest, BinaryOperators)
{
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(5.0f);
    b.fill(2.0f);

    Tensor sub = a - b;
    Tensor div = a / b;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(sub[i], 3.0f);
        EXPECT_FLOAT_EQ(div[i], 2.5f);
    }

    a -= b;
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a[i], 3.0f);
    }
}

TEST(TensorTest, ScalarOperators)
{
    Tensor a({2, 2});
    a.fill(10.0f);

    Tensor add = a + 5.0f;
    Tensor sub = a - 3.0f;
    Tensor mul = a * 2.0f;
    Tensor div = a / 4.0f;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(add[i], 15.0f);
        EXPECT_FLOAT_EQ(sub[i], 7.0f);
        EXPECT_FLOAT_EQ(mul[i], 20.0f);
        EXPECT_FLOAT_EQ(div[i], 2.5f);
    }
}

TEST(TensorTest, Contiguous)
{
    Tensor t({2, 3});
    t.at(0, 0) = 1;
    t.at(0, 1) = 2;
    t.at(0, 2) = 3;
    t.at(1, 0) = 4;
    t.at(1, 1) = 5;
    t.at(1, 2) = 6;

    // Transpose makes it non-contiguous
    Tensor tr = t.transpose();
    EXPECT_FALSE(tr.is_contiguous());

    Tensor cont = tr.contiguous();
    EXPECT_TRUE(cont.is_contiguous());
    EXPECT_EQ(cont.shape()[0], 3);
    EXPECT_EQ(cont.shape()[1], 2);

    // Verify values in row-major order
    // tr is [[1, 4], [2, 5], [3, 6]]
    EXPECT_FLOAT_EQ(cont.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cont.at(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(cont.at(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(cont.at(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(cont.at(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(cont.at(2, 1), 6.0f);

    // If already contiguous, should return a view
    Tensor already_cont({2, 2});
    Tensor view = already_cont.contiguous();
    EXPECT_EQ(view.data(), already_cont.data());
}

TEST(TensorTest, Expand)
{
    // Expand (3, 1) to (3, 4)
    Tensor t({3, 1});
    t.at(0, 0) = 1;
    t.at(1, 0) = 2;
    t.at(2, 0) = 3;

    Tensor expanded = t.expand({3, 4});
    EXPECT_EQ(expanded.shape()[0], 3);
    EXPECT_EQ(expanded.shape()[1], 4);
    EXPECT_EQ(expanded.strides()[1], 0);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(expanded.at(i, j), static_cast<float>(i + 1));
        }
    }
}

}  // namespace test
}  // namespace serius
