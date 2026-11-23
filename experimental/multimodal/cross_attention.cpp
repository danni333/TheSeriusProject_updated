#include "cross_attention.h"

#include <cmath>
#include <stdexcept>

#include "../autograd/ops.h"
#include "../common/serius_check.h"
#include "../init/weight_init.h"

namespace serius {
namespace multimodal {

namespace {

Node *make_weight(size_t rows, size_t cols, unsigned &seed, const char *name)
{
    Tensor weights({rows, cols});
    init::xavier_uniform(weights, cols, rows, seed++);
    Node *node = new Node(std::move(weights));
    node->name = name;
    return node;
}

Node *split_heads(Node *projected, size_t seq_len, size_t n_heads, size_t d_head)
{
    using namespace core::autograd::ops;
    Node *split = reshape(projected, {seq_len, n_heads, d_head});
    return contiguous(transpose(split, 0, 1));
}

Node *merge_heads(Node *heads, size_t seq_len, size_t d_model)
{
    using namespace core::autograd::ops;
    Node *transposed = contiguous(transpose(heads, 0, 1));
    return reshape(transposed, {seq_len, d_model});
}

Node *attend(Node *queries, Node *keys, Node *values, size_t d_head)
{
    using namespace core::autograd::ops;
    Node *keys_t = transpose(keys, 1, 2);
    Node *scores = matmul(queries, keys_t);
    const float scale = 1.0F / std::sqrt(static_cast<float>(d_head));
    Node *probs = softmax(mul_scalar(scores, scale), -1);
    return matmul(probs, values);
}

}  // namespace

EncoderSelfAttention::EncoderSelfAttention(size_t d_model, size_t n_heads, unsigned seed)
    : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads)
{
    if (d_model == 0 || n_heads == 0 || d_model % n_heads != 0) {
        throw std::invalid_argument("EncoderSelfAttention: d_model must be divisible by n_heads");
    }
    w_q_ = make_weight(d_model, d_model, seed, "enc_w_q");
    w_k_ = make_weight(d_model, d_model, seed, "enc_w_k");
    w_v_ = make_weight(d_model, d_model, seed, "enc_w_v");
    w_o_ = make_weight(d_model, d_model, seed, "enc_w_o");
}

EncoderSelfAttention::~EncoderSelfAttention()
{
    delete w_q_;
    delete w_k_;
    delete w_v_;
    delete w_o_;
}

Node *EncoderSelfAttention::forward(Node *input) const
{
    using namespace core::autograd::ops;
    SERIUS_CHECK(input != nullptr && input->data.ndim() == 2, "EncoderSelfAttention expects (seq, d_model)");
    const size_t seq_len = input->data.shape()[0];
    SERIUS_CHECK(input->data.shape()[1] == d_model_, "EncoderSelfAttention d_model mismatch");

    Node *q_heads = split_heads(matmul(input, w_q_), seq_len, n_heads_, d_head_);
    Node *k_heads = split_heads(matmul(input, w_k_), seq_len, n_heads_, d_head_);
    Node *v_heads = split_heads(matmul(input, w_v_), seq_len, n_heads_, d_head_);
    Node *context = attend(q_heads, k_heads, v_heads, d_head_);
    return matmul(merge_heads(context, seq_len, d_model_), w_o_);
}

CrossAttention::CrossAttention(size_t d_model, size_t n_heads, unsigned seed)
    : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads)
{
    if (d_model == 0 || n_heads == 0 || d_model % n_heads != 0) {
        throw std::invalid_argument("CrossAttention: d_model must be divisible by n_heads");
    }
    w_q_ = make_weight(d_model, d_model, seed, "xattn_w_q");
    w_k_ = make_weight(d_model, d_model, seed, "xattn_w_k");
    w_v_ = make_weight(d_model, d_model, seed, "xattn_w_v");
    w_o_ = make_weight(d_model, d_model, seed, "xattn_w_o");
}

CrossAttention::~CrossAttention()
{
    delete w_q_;
    delete w_k_;
    delete w_v_;
    delete w_o_;
}

Node *CrossAttention::forward(Node *text_hidden, Node *context) const
{
    using namespace core::autograd::ops;
    SERIUS_CHECK(text_hidden != nullptr && context != nullptr, "CrossAttention inputs must be non-null");
    SERIUS_CHECK(text_hidden->data.ndim() == 2 && context->data.ndim() == 2, "CrossAttention expects 2D tensors");
    SERIUS_CHECK(text_hidden->data.shape()[1] == d_model_ && context->data.shape()[1] == d_model_,
                 "CrossAttention d_model mismatch");

    const size_t seq_text = text_hidden->data.shape()[0];
    const size_t seq_ctx = context->data.shape()[0];

    Node *q_heads = split_heads(matmul(text_hidden, w_q_), seq_text, n_heads_, d_head_);
    Node *k_heads = split_heads(matmul(context, w_k_), seq_ctx, n_heads_, d_head_);
    Node *v_heads = split_heads(matmul(context, w_v_), seq_ctx, n_heads_, d_head_);
    Node *attended = attend(q_heads, k_heads, v_heads, d_head_);
    return matmul(merge_heads(attended, seq_text, d_model_), w_o_);
}

}  // namespace multimodal
}  // namespace serius
