#include "llama_model.h"

#include <cmath>

#include "../autograd/ops.h"
#include "../blas/blas.h"

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;
using core::autograd::ops::add;
using core::autograd::ops::matmul;
using core::autograd::ops::mul_scalar;
using core::autograd::ops::reshape;
using core::autograd::ops::rmsnorm;
using core::autograd::ops::rope;
using core::autograd::ops::silu;
using core::autograd::ops::transpose;
using core::autograd::ops::contiguous;
using core::autograd::ops::fused_masked_softmax;

namespace {

Node *make_param(const std::vector<size_t> &shape, const std::string &name)
{
    Tensor t(shape);
    t.zeros();
    Node *n = new Node(std::move(t));
    n->name = name;
    return n;
}

}  // namespace

// --- LLaMABlock ---

LLaMABlock::LLaMABlock(const LLamaConfig &config)
    : hidden_size_(config.hidden_size),
      n_heads_(config.num_attention_heads),
      n_kv_heads_(config.num_key_value_heads),
      head_dim_(config.hidden_size / config.num_attention_heads),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta)
{
    size_t n_q = n_heads_ * head_dim_;
    size_t n_kv = n_kv_heads_ * head_dim_;
    input_layernorm_weight = make_param({hidden_size_}, "input_layernorm");
    q_proj = make_param({hidden_size_, n_q}, "q_proj");
    k_proj = make_param({hidden_size_, n_kv}, "k_proj");
    v_proj = make_param({hidden_size_, n_kv}, "v_proj");
    o_proj = make_param({n_q, hidden_size_}, "o_proj");
    post_attention_layernorm_weight = make_param({hidden_size_}, "post_attention_layernorm");
    gate_proj = make_param({hidden_size_, config.intermediate_size}, "gate_proj");
    up_proj = make_param({hidden_size_, config.intermediate_size}, "up_proj");
    down_proj = make_param({config.intermediate_size, hidden_size_}, "down_proj");
}

LLaMABlock::~LLaMABlock()
{
    delete input_layernorm_weight;
    delete q_proj;
    delete k_proj;
    delete v_proj;
    delete o_proj;
    delete post_attention_layernorm_weight;
    delete gate_proj;
    delete up_proj;
    delete down_proj;
}

Node *LLaMABlock::forward(Node *input, size_t pos_offset) const
{
    using namespace core::autograd::ops;
    Node *normed = rmsnorm(input, input_layernorm_weight, rms_norm_eps_);
    Node *q = matmul(normed, q_proj);
    Node *k = matmul(normed, k_proj);
    Node *v = matmul(normed, v_proj);

    RoPEResult rope_out = rope(q, k, head_dim_, rope_theta_, pos_offset);
    q = rope_out.queries;
    k = rope_out.keys;

    size_t seq_len = input->data.shape()[0];
    Node *q_rs = reshape(q, {seq_len, n_heads_, head_dim_});
    Node *k_rs = reshape(k, {seq_len, n_kv_heads_, head_dim_});
    Node *v_rs = reshape(v, {seq_len, n_kv_heads_, head_dim_});
    q_rs = contiguous(transpose(q_rs, 0, 1));
    k_rs = contiguous(transpose(k_rs, 0, 1));
    v_rs = contiguous(transpose(v_rs, 0, 1));

    if (n_kv_heads_ < n_heads_) {
        size_t rep = n_heads_ / n_kv_heads_;
        Tensor k_expanded({n_heads_, seq_len, head_dim_});
        Tensor v_expanded({n_heads_, seq_len, head_dim_});
        for (size_t h = 0; h < n_heads_; ++h) {
            size_t src = h / rep;
            for (size_t s = 0; s < seq_len; ++s)
                for (size_t d = 0; d < head_dim_; ++d) {
                    k_expanded.at(h, s, d) = k_rs->data.at(src, s, d);
                    v_expanded.at(h, s, d) = v_rs->data.at(src, s, d);
                }
        }
        delete k_rs;
        delete v_rs;
        k_rs = new Node(std::move(k_expanded));
        v_rs = new Node(std::move(v_expanded));
    }

    Node *kt = transpose(k_rs, 1, 2);
    Node *scores = matmul(q_rs, kt);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    Tensor causal_mask({seq_len, seq_len});
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t j = 0; j < seq_len; ++j) causal_mask.at(i, j) = (j > i) ? -1e9f : 0.0f;
    Node *probs = fused_masked_softmax(scores, causal_mask, scale);
    Node *context = matmul(probs, v_rs);
    context = contiguous(transpose(context, 0, 1));
    context = reshape(context, {seq_len, n_heads_ * head_dim_});
    Node *attn_out = matmul(context, o_proj);
    Node *res1 = add(input, attn_out);

    Node *normed2 = rmsnorm(res1, post_attention_layernorm_weight, rms_norm_eps_);
    Node *gate = matmul(normed2, gate_proj);
    Node *up = matmul(normed2, up_proj);
    gate = silu(gate);
    Tensor gate_up_t(gate->data.shape());
    for (size_t i = 0; i < gate->data.size(); ++i) gate_up_t[i] = gate->data[i] * up->data[i];
    Node *gate_up = new Node(std::move(gate_up_t));
    gate_up->requires_grad = false;
    Node *mlp_out = matmul(gate_up, down_proj);
    return add(res1, mlp_out);
}

// --- LLaMAModel ---

LLaMAModel::LLaMAModel(const LLamaConfig &config) : config_(config)
{
    embed_tokens_ = make_param({config.vocab_size, config.hidden_size}, "embed_tokens");
    for (size_t i = 0; i < config.num_hidden_layers; ++i)
        layers_.push_back(new LLaMABlock(config));
    norm_weight_ = make_param({config.hidden_size}, "norm");
    lm_head_ = make_param({config.hidden_size, config.vocab_size}, "lm_head");  // (d_model, vocab) for matmul(h, lm_head)
}

LLaMAModel::~LLaMAModel()
{
    delete embed_tokens_;
    for (auto *b : layers_) delete b;
    delete norm_weight_;
    delete lm_head_;
}

Node *LLaMAModel::forward(Node *token_ids) const
{
    using namespace core::autograd::ops;
    const Tensor &ids = token_ids->data;
    size_t seq_len = ids.size();
    Tensor hidden({seq_len, config_.hidden_size});
    for (size_t i = 0; i < seq_len; ++i) {
        size_t tid = static_cast<size_t>(ids[i]);
        if (tid >= config_.vocab_size) tid = 0;
        for (size_t j = 0; j < config_.hidden_size; ++j)
            hidden.at(i, j) = embed_tokens_->data.at(tid, j);
    }
    Node *h = new Node(std::move(hidden));
    h->requires_grad = false;
    for (size_t layer = 0; layer < layers_.size(); ++layer)
        h = layers_[layer]->forward(h, 0);
    h = rmsnorm(h, norm_weight_, config_.rms_norm_eps);
    return matmul(h, lm_head_);
}

}  // namespace model
}  // namespace serius
