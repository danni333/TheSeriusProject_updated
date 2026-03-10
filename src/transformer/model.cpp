#include "model.h"

#include <cmath>

#include "../autograd/ops.h"

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;

TransformerModel::TransformerModel(const ModelConfig &config) : config_(config)
{
    float scale = 1.0F / std::sqrt(static_cast<float>(config.d_model));

    // Initialize embedding
    Tensor emb_t({config.vocab_size, config.d_model});
    emb_t.rand();
    const float OFFSET = 0.5F;
    const float SCALE_ADJ = 2.0F;
    for (size_t idx = 0; idx < emb_t.size(); ++idx) {
        emb_t[idx] = (emb_t[idx] - OFFSET) * SCALE_ADJ * scale;
    }
    embedding = new Node(std::move(emb_t));
    embedding->name = "embedding";

    // Create transformer blocks
    blocks.reserve(config.n_layers);
    for (size_t layer = 0; layer < config.n_layers; ++layer) {
        blocks.push_back(new TransformerBlock(config.d_model, config.n_heads, config.d_ff));
    }

    // Final layernorm
    final_norm = new LayerNorm(config.d_model);

    // Initialize output projection
    Tensor proj_t({config.d_model, config.vocab_size});
    proj_t.rand();
    for (size_t idx = 0; idx < proj_t.size(); ++idx) {
        proj_t[idx] = (proj_t[idx] - OFFSET) * SCALE_ADJ * scale;
    }
    output_proj = new Node(std::move(proj_t));
    output_proj->name = "output_proj";
}

TransformerModel::~TransformerModel()
{
    delete embedding;
    for (auto *block : blocks) {
        delete block;
    }
    delete final_norm;
    delete output_proj;
}

Node *TransformerModel::embedding_lookup(Node *token_ids) const
{
    const Tensor &ids = token_ids->data;
    bool is_batched = (ids.ndim() == 2);
    size_t batch_size = is_batched ? ids.shape()[0] : 1;
    size_t seq_len = is_batched ? ids.shape()[1] : ids.size();

    Tensor hidden_data(is_batched ? std::vector<size_t>{batch_size, seq_len, config_.d_model}
                                  : std::vector<size_t>{seq_len, config_.d_model});

    EmbeddingDims dims{is_batched, batch_size, seq_len};
    do_copy_embedding(ids, dims, hidden_data);

    Node *hidden = new Node(std::move(hidden_data));
    hidden->name = "embedding_lookup";
    hidden->children = {token_ids, embedding};

    auto lookup_backward = [token_ids, this, dims, hidden]() { this->do_embedding_backward(token_ids, dims, hidden); };
    hidden->backward_fn = std::move(lookup_backward);
    return hidden;
}

void TransformerModel::do_copy_embedding(const Tensor &ids, const EmbeddingDims &dims, Tensor &hidden_data) const
{
    for (size_t batch_idx = 0; batch_idx < dims.batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < dims.seq_len; ++seq_idx) {
            size_t token_id =
                dims.is_batched ? static_cast<size_t>(ids.at(batch_idx, seq_idx)) : static_cast<size_t>(ids[seq_idx]);
            for (size_t dim_idx = 0; dim_idx < config_.d_model; ++dim_idx) {
                float val = embedding->data.at(token_id, dim_idx);
                if (dims.is_batched) {
                    hidden_data.at(batch_idx, seq_idx, dim_idx) = val;
                } else {
                    hidden_data.at(seq_idx, dim_idx) = val;
                }
            }
        }
    }
}

void TransformerModel::do_embedding_backward(Node *token_ids, const EmbeddingDims &dims, Node *hidden) const
{
    if (!embedding->requires_grad) {
        return;
    }
    const Tensor &grad_output = hidden->grad;
    const Tensor &input_ids = token_ids->data;
    for (size_t batch_idx = 0; batch_idx < dims.batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < dims.seq_len; ++seq_idx) {
            size_t token_id = dims.is_batched ? static_cast<size_t>(input_ids.at(batch_idx, seq_idx))
                                              : static_cast<size_t>(input_ids[seq_idx]);
            for (size_t dim_idx = 0; dim_idx < config_.d_model; ++dim_idx) {
                float grad_val =
                    dims.is_batched ? grad_output.at(batch_idx, seq_idx, dim_idx) : grad_output.at(seq_idx, dim_idx);
                embedding->grad.at(token_id, dim_idx) += grad_val;
            }
        }
    }
}

Node *TransformerModel::forward(Node *token_ids, std::vector<PagedKVCache *> *caches) const
{
    using namespace core::autograd::ops;

    // Lookup embeddings
    Node *hidden_state = embedding_lookup(token_ids);
    // Pass through transformer blocks
    for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
        PagedKVCache *current_cache =
            (caches != nullptr && block_idx < caches->size()) ? (*caches)[block_idx] : nullptr;
        hidden_state = blocks[block_idx]->forward_optimized(hidden_state, current_cache);
    }

    // Final layer norm
    hidden_state = final_norm->forward(hidden_state);

    // Project to vocabulary
    return matmul(hidden_state, output_proj);
}

std::vector<Node *> TransformerModel::parameters() const
{
    std::vector<Node *> params;
    params.push_back(embedding);
    for (auto *block : blocks) {
        params.push_back(block->attention.w_q);
        params.push_back(block->attention.w_dkv);
        params.push_back(block->attention.w_uk);
        params.push_back(block->attention.w_uv);
        params.push_back(block->attention.w_o);
        params.push_back(block->norm1.gamma_);
        params.push_back(block->norm1.beta_);
        params.push_back(block->norm2.gamma_);
        params.push_back(block->norm2.beta_);
        params.push_back(block->ffn.w1);
        params.push_back(block->ffn.b1);
        params.push_back(block->ffn.w2);
        params.push_back(block->ffn.b2);
    }
    params.push_back(final_norm->gamma_);
    params.push_back(final_norm->beta_);
    params.push_back(output_proj);
    return params;
}

}  // namespace model
}  // namespace serius
