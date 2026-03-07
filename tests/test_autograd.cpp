#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "../src/autograd/engine.h"
#include "../src/autograd/node.h"
#include "../src/autograd/ops.h"
#include "../src/blas/blas.h"
#include "../src/tensor/tensor.h"

using namespace serius;
using namespace serius::autograd;

// Numeric gradient check helper
float compute_numeric_gradient(std::function<float()> func, float &param, float eps = 1e-4f)
{
    float old_val = param;

    param = old_val + eps;
    float f_plus = func();

    param = old_val - eps;
    float f_minus = func();

    param = old_val;
    return (f_plus - f_minus) / (2.0f * eps);
}

TEST(AutogradTest, BasicAdd)
{
    std::vector<std::size_t> shape = {1};
    Tensor data_a(shape);
    Tensor data_b(shape);
    Node *a = new Node(std::move(data_a));
    Node *b = new Node(std::move(data_b));
    a->data[0] = 2.0f;
    b->data[0] = 3.0f;

    Node *c = ops::add(a, b);
    backward(c);

    EXPECT_FLOAT_EQ(c->data[0], 5.0f);
    EXPECT_FLOAT_EQ(a->grad[0], 1.0f);
    EXPECT_FLOAT_EQ(b->grad[0], 1.0f);

    delete a;
    delete b;
    delete c;
}

TEST(AutogradTest, MatMulGrad)
{
    // 2x3 * 3x2 = 2x2
    std::vector<std::size_t> shape_a = {2, 3};
    std::vector<std::size_t> shape_b = {3, 2};
    Tensor ta(shape_a);
    ta.rand();
    Tensor tb(shape_b);
    tb.rand();

    Node *a = new Node(std::move(ta));
    Node *b = new Node(std::move(tb));

    Node *c = ops::matmul(a, b);

    // Loss = sum(C)
    std::vector<std::size_t> shape_loss = {1};
    Tensor loss_data(shape_loss);
    float loss_val_sum = 0.0F;
    for (std::size_t i = 0; i < c->data.size(); ++i) {
        loss_val_sum += c->data[i];
    }
    loss_data[0] = loss_val_sum;
    Node *loss = new Node(std::move(loss_data));
    loss->children = {c};
    loss->backward_fn = [c, loss]() {
        for (std::size_t i = 0; i < c->data.size(); ++i) {
            c->grad[i] += loss->grad[0];
        }
    };

    backward(loss);

    float numeric_grad = compute_numeric_gradient(
        [&]() {
            Tensor res = blas::matmul(a->data, b->data);
            float s_sum = 0.0F;
            for (std::size_t i = 0; i < res.size(); ++i) {
                s_sum += res[i];
            }
            return s_sum;
        },
        a->data[0]);

    EXPECT_NEAR(a->grad[0], numeric_grad, 5e-3);

    delete a;
    delete b;
    delete c;
    delete loss;
}

TEST(AutogradTest, ReLU)
{
    std::vector<std::size_t> shape = {1, 2};
    Tensor data_x(shape);
    data_x[0] = -1.0f;
    data_x[1] = 2.0f;

    Node *x = new Node(std::move(data_x));
    Node *y = ops::relu(x);

    // Manual sum for scalar loss
    std::vector<std::size_t> shape_loss = {1};
    Tensor loss_data(shape_loss);
    loss_data[0] = y->data[0] + y->data[1];
    Node *loss = new Node(std::move(loss_data));
    loss->children = {y};
    loss->backward_fn = [y, loss]() {
        y->grad[0] += loss->grad[0];
        y->grad[1] += loss->grad[0];
    };

    backward(loss);

    EXPECT_FLOAT_EQ(y->data[0], 0.0f);
    EXPECT_FLOAT_EQ(y->data[1], 2.0f);
    EXPECT_FLOAT_EQ(x->grad[0], 0.0f);
    EXPECT_FLOAT_EQ(x->grad[1], 1.0f);

    delete x;
    delete y;
    delete loss;
}

TEST(AutogradTest, Softmax)
{
    std::vector<std::size_t> shape = {1, 3};
    Tensor data_a(shape);
    data_a[0] = 1.0f;
    data_a[1] = 2.0f;
    data_a[2] = 3.0f;

    Node *a = new Node(std::move(data_a));
    Node *s = ops::softmax(a);

    float sum = 0;
    for (std::size_t i = 0; i < s->data.size(); ++i) sum += s->data[i];
    EXPECT_NEAR(sum, 1.0F, 1e-6F);

    EXPECT_NEAR(s->data[0], 0.09003F, 1e-4F);
    EXPECT_NEAR(s->data[1], 0.24473F, 1e-4F);
    EXPECT_NEAR(s->data[2], 0.66524F, 1e-4F);

    delete a;
    delete s;
}

TEST(AutogradTest, Chain)
{
    std::vector<std::size_t> shape_x = {1, 2};
    std::vector<std::size_t> shape_w = {2, 2};
    std::vector<std::size_t> shape_b = {1, 2};
    Tensor data_x(shape_x);
    Tensor data_w(shape_w);
    Tensor data_b(shape_b);

    data_x[0] = 1.0f;
    data_x[1] = 2.0f;
    data_w[0] = 0.5f;
    data_w[1] = -0.5f;
    data_w[2] = 1.0f;
    data_w[3] = 1.0f;
    data_b[0] = 0.1f;
    data_b[1] = 0.2f;

    Node *x = new Node(std::move(data_x), false);
    Node *w = new Node(std::move(data_w));
    Node *b = new Node(std::move(data_b));

    Node *h = ops::add(ops::matmul(x, w), b);
    Node *y = ops::relu(h);

    // Loss = sum(y)
    std::vector<std::size_t> shape_loss = {1};
    Tensor loss_data(shape_loss);
    loss_data[0] = y->data[0] + y->data[1];
    Node *loss = new Node(std::move(loss_data));
    loss->children = {y};
    loss->backward_fn = [y, loss]() {
        y->grad[0] += loss->grad[0];
        y->grad[1] += loss->grad[0];
    };

    backward(loss);

    EXPECT_NEAR(y->data[0], 2.6f, 1e-5);
    EXPECT_NEAR(y->data[1], 1.7f, 1e-5);

    EXPECT_FLOAT_EQ(b->grad[0], 1.0f);
    EXPECT_FLOAT_EQ(b->grad[1], 1.0f);
    EXPECT_FLOAT_EQ(w->grad[0], 1.0f);
    EXPECT_FLOAT_EQ(w->grad[1], 1.0f);
    EXPECT_FLOAT_EQ(w->grad[2], 2.0f);
    EXPECT_FLOAT_EQ(w->grad[3], 2.0f);

    delete x;
    delete w;
    delete b;
    delete h;
    delete y;
    delete loss;
}

TEST(AutogradTest, SharedInput)
{
    std::vector<std::size_t> shape_x = {1, 2};
    std::vector<std::size_t> shape_w = {2, 2};
    Tensor data_x(shape_x);
    Tensor data_w1(shape_w);
    Tensor data_w2(shape_w);

    data_x[0] = 1.0f;
    data_x[1] = 2.0f;
    data_w1.fill(1.0f);
    data_w2.fill(2.0f);

    Node *x = new Node(std::move(data_x));
    Node *w1 = new Node(std::move(data_w1));
    Node *w2 = new Node(std::move(data_w2));

    Node *y1 = ops::matmul(x, w1);
    Node *y2 = ops::matmul(x, w2);
    Node *y_vec = ops::add(y1, y2);

    std::vector<std::size_t> shape_loss = {1};
    Tensor loss_data(shape_loss);
    loss_data[0] = 0;
    for (std::size_t i = 0; i < y_vec->data.size(); ++i) {
        loss_data[0] += y_vec->data[i];
    }
    Node *loss = new Node(std::move(loss_data));
    loss->children = {y_vec};
    loss->backward_fn = [y_vec, loss]() {
        for (std::size_t i = 0; i < y_vec->data.size(); ++i) {
            y_vec->grad[i] += loss->grad[0];
        }
    };

    backward(loss);

    EXPECT_FLOAT_EQ(x->grad[0], 6.0f);
    EXPECT_FLOAT_EQ(x->grad[1], 6.0f);

    delete x;
    delete w1;
    delete w2;
    delete y1;
    delete y2;
    delete y_vec;
    delete loss;
}
