#pragma once

#include <memory>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"
#include "patch_embed.h"

namespace serius {
namespace vision {

using core::Tensor;
using core::autograd::Node;

struct VisionConfig
{
    size_t image_size = 224;
    size_t patch_size = 16;
    size_t d_vision = 64;
    size_t n_layers = 1;
    size_t n_heads = 4;
    size_t d_ff = 128;
    size_t d_model = 64;
    unsigned seed = 42;
};

/// CLIP-style Vision Transformer: patches -> encoder -> projection to LLM space.
class VisionEncoder
{
   public:
    explicit VisionEncoder(const VisionConfig &config);
    VisionEncoder();
    ~VisionEncoder();

    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder &operator=(const VisionEncoder &) = delete;

    size_t num_patches() const { return num_patches_; }
    size_t d_model() const { return config_.d_model; }
    const VisionConfig &config() const { return config_; }

    /// image: (3,H,W) already at image_size, or any layout via preprocess_image first.
    /// returns visual tokens (num_patches, d_model)
    Tensor encode(const Tensor &image) const;

    Node *forward(Node *patch_tokens) const;

    PatchEmbed &patch_embed() { return *patch_embed_; }

   private:
    struct Layer;

    VisionConfig config_;
    size_t num_patches_;
    std::unique_ptr<PatchEmbed> patch_embed_;
    Node *pos_embed_;
    Node *proj_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace vision
}  // namespace serius
