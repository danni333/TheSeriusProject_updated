#pragma once

#include "../tensor/tensor.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace serius::hub {

/// Nomic-BERT (nomic-embed) sentence embedder — inference only.
///
/// Implements the encoder architecture used by nomic-embed-text-v1.5
/// GGUFs (llama.cpp `nomic-bert`):
///   - WordPiece (BERT uncased) tokenization with [CLS]/[SEP]
///   - token_embd + token-type row 0 + LayerNorm (no position embd; RoPE)
///   - 12 post-norm blocks: RoPE attention (non-causal) -> residual+LN,
///     SwiGLU FFN -> residual+LN
///   - mean pooling over sequence positions + L2 normalization
///
/// All weights are plain float tensors dequantized by io::load_all_tensors.
class NomicBertEmbedder
{
   public:
    struct Config
    {
        int n_layers = 12;
        int n_heads = 12;
        int hidden = 768;
        int ffn = 3072;
        float norm_eps = 1e-12f;
        float rope_theta = 1000.0f;
        int max_len = 2048;
        int cls_id = 101;
        int sep_id = 102;
        int unk_id = 100;
        int pad_id = 0;
    };

    /// Load weights from a tensor map produced by io::load_all_tensors()
    /// (keys like "token_embd.weight", "blk.N.attn_qkv.weight", ...).
    /// Returns false if required tensors are missing.
    bool load(const std::unordered_map<std::string, core::Tensor> &tensors);
    bool load(const std::unordered_map<std::string, core::Tensor> &tensors,
              const Config &cfg);

    /// Vocabulary token strings indexed by token id (from GGUF tokenizer
    /// metadata). Required before tokenize()/embed() produce real ids.
    void set_vocab(const std::vector<std::string> &tokens);

    bool is_loaded() const { return loaded_; }
    int dim() const { return cfg_.hidden; }

    /// Tokenize with WordPiece (uncased) -> ids including [CLS] and [SEP].
    std::vector<int> tokenize(const std::string &text) const;

    /// Full embedding pipeline: tokenize -> forward -> mean pool -> L2 norm.
    /// Returns a `dim()`-sized vector; empty on failure.
    std::vector<float> embed(const std::string &text) const;

    /// Forward pass on pre-tokenized ids (with CLS/SEP already applied is
    /// fine — they are ordinary positions for the encoder).
    /// Returns [seq_len, hidden] row-major hidden states (post last norm).
    std::vector<float> forward_hidden(const std::vector<int> &ids) const;

   private:
    Config cfg_;
    bool loaded_ = false;

    // Flat row-major weights: token_embd [vocab,hidden]; per-layer
    // qkv [3h,h], wo [h,h], up [ffn,h], gate [ffn,h], down [h,ffn];
    // norms as {weight,bias} pairs.
    std::vector<float> tok_embd_;
    std::vector<float> type_embd_row0_;
    std::vector<float> tok_norm_w_, tok_norm_b_;

    struct LayerW
    {
        std::vector<float> qkv, wo, up, gate, down;
        std::vector<float> attn_norm_w, attn_norm_b;
        std::vector<float> out_norm_w, out_norm_b;
    };
    std::vector<LayerW> layers_;

    std::unordered_map<std::string, int> vocab_;

    static bool get_tensor(const std::unordered_map<std::string, core::Tensor> &tensors,
                           const std::string &name, std::vector<float> *out);
    /// Load a 2D weight [out_dim, in_dim] and store it transposed as
    /// [in_dim, out_dim] row-major, ready for a single GEMM forward pass.
    static bool get_tensor_transposed(const std::unordered_map<std::string, core::Tensor> &tensors,
                                      const std::string &name, size_t out_dim, size_t in_dim,
                                      std::vector<float> *out);
};

}  // namespace serius::hub
