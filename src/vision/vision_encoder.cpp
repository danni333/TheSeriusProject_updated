#include "vision_encoder.h"

#include "../autograd/ops.h"
#include "../common/serius_check.h"
#include "../init/weight_init.h"
#include "../multimodal/cross_attention.h"
#include "../transformer/ffn.h"
#include "../transformer/layernorm.h"

namespace serius {
namespace vision {

struct VisionEncoder::Layer
{
    model::LayerNorm norm1;
    multimodal::EncoderSelfAttention attn;
    model::LayerNorm norm2;
    model::FFN ffn;

    Layer(size_t d_model, size_t n_heads, size_t d_ff, unsigned seed)
        : norm1(d_model), attn(d_model, n_heads, seed), norm2(d_model), ffn(d_model, d_ff, seed + 8)
    {
    }

    Node *forward(Node *input) const
    {
        using namespace core::autograd::ops;
        Node *residual1 = add(input, attn.forward(norm1.forward(input)));
        return add(residual1, ffn.forward(norm2.forward(residual1)));
    }
};

VisionEncoder::VisionEncoder(const VisionConfig &config) : config_(config), pos_embed_(nullptr), proj_(nullptr)
{
    SERIUS_CHECK(config_.image_size > 0 && config_.patch_size > 0, "invalid vision sizes");
    SERIUS_CHECK(config_.image_size % config_.patch_size == 0, "image_size must be divisible by patch_size");
    SERIUS_CHECK(config_.d_vision % config_.n_heads == 0, "d_vision must be divisible by n_heads");

    const size_t n_side = config_.image_size / config_.patch_size;
    num_patches_ = n_side * n_side;

    patch_embed_ = std::make_unique<PatchEmbed>(config_.patch_size, config_.d_vision, config_.seed);

    Tensor pos({num_patches_, config_.d_vision});
    init::normal_init(pos, 0.0F, 0.02F, config_.seed + 1);
    pos_embed_ = new Node(std::move(pos));
    pos_embed_->name = "vision_pos_embed";

    Tensor proj_w({config_.d_vision, config_.d_model});
    init::xavier_uniform(proj_w, config_.d_vision, config_.d_model, config_.seed + 2);
    proj_ = new Node(std::move(proj_w));
    proj_->name = "vision_proj";

    layers_.reserve(config_.n_layers);
    for (size_t i = 0; i < config_.n_layers; ++i) {
        layers_.push_back(std::make_unique<Layer>(config_.d_vision, config_.n_heads, config_.d_ff,
                                                  config_.seed + 16 + static_cast<unsigned>(i) * 32));
    }
}

VisionEncoder::~VisionEncoder()
{
    delete pos_embed_;
    delete proj_;
}

Node *VisionEncoder::forward(Node *patch_tokens) const
{
    using namespace core::autograd::ops;
    SERIUS_CHECK(patch_tokens != nullptr, "patch_tokens is null");
    Node *hidden = add(patch_tokens, pos_embed_);
    for (const auto &layer : layers_) {
        hidden = layer->forward(hidden);
    }
    return matmul(hidden, proj_);
}

Tensor VisionEncoder::encode(const Tensor &image) const
{
    Tensor chw = image;
    if (image.ndim() != 3 || image.shape()[0] != 3 || image.shape()[1] != config_.image_size ||
        image.shape()[2] != config_.image_size) {
        chw = preprocess_image(image, config_.image_size, config_.image_size);
    }
    Node *embedded = patch_embed_->forward(chw);
    Node *out = forward(embedded);
    return out->data.copy();
}

}  // namespace vision
}  // namespace serius
