#include "patch_embed.h"

#include <algorithm>
#include <cmath>

#include "../autograd/ops.h"
#include "../common/serius_check.h"
#include "../init/weight_init.h"

namespace serius {
namespace vision {

namespace {

constexpr float kClipMean[3] = {0.48145466F, 0.4578275F, 0.40821073F};
constexpr float kClipStd[3] = {0.26862954F, 0.26130258F, 0.27577711F};

Tensor to_chw(const Tensor &image)
{
    if (image.ndim() == 2) {
        const size_t height = image.shape()[0];
        const size_t width = image.shape()[1];
        Tensor chw({3, height, width});
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const float value = image.at(y, x);
                chw.at(0, y, x) = value;
                chw.at(1, y, x) = value;
                chw.at(2, y, x) = value;
            }
        }
        return chw;
    }

    SERIUS_CHECK(image.ndim() == 3, "image must be (H,W), (H,W,C) or (C,H,W)");
    const size_t d0 = image.shape()[0];
    const size_t d1 = image.shape()[1];
    const size_t d2 = image.shape()[2];

    if (d0 == 3 && d1 >= 2 && d2 >= 2) {
        return image.copy();
    }
    if (d2 == 3) {
        Tensor chw({3, d0, d1});
        for (size_t y = 0; y < d0; ++y) {
            for (size_t x = 0; x < d1; ++x) {
                chw.at(0, y, x) = image.at(y, x, 0);
                chw.at(1, y, x) = image.at(y, x, 1);
                chw.at(2, y, x) = image.at(y, x, 2);
            }
        }
        return chw;
    }

    throw serius::Error("unsupported image layout", __FILE__, __LINE__);
}

float sample_bilinear(const Tensor &chw, size_t channel, float fy, float fx)
{
    const size_t height = chw.shape()[1];
    const size_t width = chw.shape()[2];
    fy = std::clamp(fy, 0.0F, static_cast<float>(height - 1));
    fx = std::clamp(fx, 0.0F, static_cast<float>(width - 1));

    const size_t y0 = static_cast<size_t>(fy);
    const size_t x0 = static_cast<size_t>(fx);
    const size_t y1 = std::min(y0 + 1, height - 1);
    const size_t x1 = std::min(x0 + 1, width - 1);
    const float wy = fy - static_cast<float>(y0);
    const float wx = fx - static_cast<float>(x0);

    const float v00 = chw.at(channel, y0, x0);
    const float v01 = chw.at(channel, y0, x1);
    const float v10 = chw.at(channel, y1, x0);
    const float v11 = chw.at(channel, y1, x1);
    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bot = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bot * wy;
}

}  // namespace

Tensor preprocess_image(const Tensor &image, size_t out_h, size_t out_w)
{
    SERIUS_CHECK(out_h > 0 && out_w > 0, "output image size must be positive");
    Tensor chw = to_chw(image);

    float max_abs = 0.0F;
    for (size_t i = 0; i < chw.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(chw[i]));
    }
    if (max_abs > 1.5F) {
        for (size_t i = 0; i < chw.size(); ++i) {
            chw[i] /= 255.0F;
        }
    }

    const size_t in_h = chw.shape()[1];
    const size_t in_w = chw.shape()[2];
    if (in_h == out_h && in_w == out_w) {
        normalize_image(chw);
        return chw;
    }

    Tensor out({3, out_h, out_w});
    const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
    for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < out_h; ++y) {
            const float fy = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
            for (size_t x = 0; x < out_w; ++x) {
                const float fx = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
                out.at(c, y, x) = sample_bilinear(chw, c, fy, fx);
            }
        }
    }
    normalize_image(out);
    return out;
}

void normalize_image(Tensor &image_chw)
{
    SERIUS_CHECK(image_chw.ndim() == 3 && image_chw.shape()[0] == 3, "normalize expects (3,H,W)");
    const size_t height = image_chw.shape()[1];
    const size_t width = image_chw.shape()[2];
    for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                image_chw.at(c, y, x) = (image_chw.at(c, y, x) - kClipMean[c]) / kClipStd[c];
            }
        }
    }
}

Tensor extract_patches(const Tensor &image, size_t patch_size)
{
    SERIUS_CHECK(image.ndim() == 3, "extract_patches expects (C,H,W)");
    SERIUS_CHECK(patch_size > 0, "patch_size must be positive");
    const size_t channels = image.shape()[0];
    const size_t height = image.shape()[1];
    const size_t width = image.shape()[2];
    SERIUS_CHECK(channels == 3, "extract_patches expects 3 channels");
    SERIUS_CHECK(height % patch_size == 0 && width % patch_size == 0, "H and W must be divisible by patch_size");

    const size_t n_h = height / patch_size;
    const size_t n_w = width / patch_size;
    const size_t n_patches = n_h * n_w;
    const size_t patch_dim = patch_size * patch_size * channels;
    Tensor patches({n_patches, patch_dim});

    for (size_t ph = 0; ph < n_h; ++ph) {
        for (size_t pw = 0; pw < n_w; ++pw) {
            const size_t patch_idx = ph * n_w + pw;
            size_t dest = 0;
            for (size_t c = 0; c < channels; ++c) {
                for (size_t i = 0; i < patch_size; ++i) {
                    for (size_t j = 0; j < patch_size; ++j) {
                        patches.at(patch_idx, dest++) = image.at(c, ph * patch_size + i, pw * patch_size + j);
                    }
                }
            }
        }
    }
    return patches;
}

PatchEmbed::PatchEmbed(size_t patch_size, size_t d_vision, unsigned seed)
    : patch_size_(patch_size), d_vision_(d_vision)
{
    SERIUS_CHECK(patch_size > 0 && d_vision > 0, "PatchEmbed dims must be positive");
    Tensor weights({patch_dim(), d_vision});
    init::xavier_uniform(weights, patch_dim(), d_vision, seed);
    proj_ = new Node(std::move(weights));
    proj_->name = "patch_proj";
}

PatchEmbed::~PatchEmbed() { delete proj_; }

Node *PatchEmbed::forward(const Tensor &image_chw) const
{
    Tensor patches = extract_patches(image_chw, patch_size_);
    Node *patch_node = new Node(std::move(patches), false);
    return core::autograd::ops::matmul(patch_node, proj_);
}

}  // namespace vision
}  // namespace serius
