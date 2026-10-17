#pragma once

#include <cstddef>

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace vision {

using core::Tensor;
using core::autograd::Node;

/// Resize an image to (channels, out_h, out_w) with bilinear interpolation.
/// Accepts (H, W), (H, W, C) or (C, H, W). Values in [0, 255] are scaled to [0, 1].
Tensor preprocess_image(const Tensor &image, size_t out_h = 224, size_t out_w = 224);

/// ImageNet/CLIP-style per-channel normalize on a (C, H, W) tensor in [0, 1].
void normalize_image(Tensor &image_chw);

/// Flatten non-overlapping patches from a CHW image.
/// image: (3, H, W) with H and W divisible by patch_size
/// returns: (num_patches, patch_size * patch_size * 3)
Tensor extract_patches(const Tensor &image, size_t patch_size);

class PatchEmbed
{
   public:
    PatchEmbed(size_t patch_size, size_t d_vision, unsigned seed = 42);
    ~PatchEmbed();

    PatchEmbed(const PatchEmbed &) = delete;
    PatchEmbed &operator=(const PatchEmbed &) = delete;

    /// Project patches: (num_patches, patch_dim) -> (num_patches, d_vision)
    Node *forward(const Tensor &image_chw) const;

    Tensor extract(const Tensor &image_chw) const { return extract_patches(image_chw, patch_size_); }

    size_t patch_size() const { return patch_size_; }
    size_t d_vision() const { return d_vision_; }
    size_t patch_dim() const { return patch_size_ * patch_size_ * 3; }

    Node *proj() const { return proj_; }

   private:
    size_t patch_size_;
    size_t d_vision_;
    Node *proj_;
};

}  // namespace vision
}  // namespace serius
