#include "audio_encoder.h"

#include <algorithm>
#include <cmath>

#include "../autograd/ops.h"
#include "../common/serius_check.h"
#include "../init/weight_init.h"
#include "../multimodal/cross_attention.h"
#include "../transformer/ffn.h"
#include "../transformer/layernorm.h"

namespace serius {
namespace audio {

struct AudioEncoder::Layer
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

AudioEncoder::AudioEncoder(const AudioEncoderConfig &config)
    : config_(config), conv_weight_(nullptr), conv_bias_(nullptr), proj_(nullptr)
{
    SERIUS_CHECK(config_.d_audio > 0 && config_.d_audio % config_.n_heads == 0, "invalid audio encoder dims");
    SERIUS_CHECK(config_.conv_kernel > 0 && config_.conv_stride > 0, "invalid conv hyperparams");

    const size_t n_mels = static_cast<size_t>(config_.mel.n_mels);
    const size_t k = static_cast<size_t>(config_.conv_kernel);
    Tensor conv_w({k, n_mels, config_.d_audio});
    init::he_uniform(conv_w, k * n_mels, config_.seed);
    conv_weight_ = new Node(std::move(conv_w));
    conv_weight_->name = "audio_conv_w";

    Tensor conv_b({config_.d_audio});
    conv_b.zeros();
    conv_bias_ = new Node(std::move(conv_b));
    conv_bias_->name = "audio_conv_b";

    Tensor proj_w({config_.d_audio, config_.d_model});
    init::xavier_uniform(proj_w, config_.d_audio, config_.d_model, config_.seed + 3);
    proj_ = new Node(std::move(proj_w));
    proj_->name = "audio_proj";

    layers_.reserve(config_.n_layers);
    for (size_t i = 0; i < config_.n_layers; ++i) {
        layers_.push_back(std::make_unique<Layer>(config_.d_audio, config_.n_heads, config_.d_ff,
                                                  config_.seed + 20 + static_cast<unsigned>(i) * 32));
    }
}

AudioEncoder::AudioEncoder() : AudioEncoder(AudioEncoderConfig{}) {}

AudioEncoder::~AudioEncoder()
{
    delete conv_weight_;
    delete conv_bias_;
    delete proj_;
}

Tensor AudioEncoder::conv1d(const Tensor &frames_by_mel) const
{
    SERIUS_CHECK(frames_by_mel.ndim() == 2, "conv1d expects (T, n_mels)");
    const size_t time_steps = frames_by_mel.shape()[0];
    const size_t n_mels = frames_by_mel.shape()[1];
    const size_t kernel = conv_weight_->data.shape()[0];
    const size_t stride = static_cast<size_t>(config_.conv_stride);
    const size_t pad = kernel / 2;
    const size_t out_t = (time_steps + 2 * pad - kernel) / stride + 1;
    Tensor out({out_t, config_.d_audio});
    out.zeros();

    for (size_t t = 0; t < out_t; ++t) {
        const int center = static_cast<int>(t * stride) - static_cast<int>(pad);
        for (size_t d = 0; d < config_.d_audio; ++d) {
            float acc = conv_bias_->data[d];
            for (size_t k = 0; k < kernel; ++k) {
                const int src_t = center + static_cast<int>(k);
                if (src_t < 0 || src_t >= static_cast<int>(time_steps)) {
                    continue;
                }
                for (size_t m = 0; m < n_mels; ++m) {
                    acc += frames_by_mel.at(static_cast<size_t>(src_t), m) * conv_weight_->data.at(k, m, d);
                }
            }
            out.at(t, d) = acc;
        }
    }
    return out;
}

Tensor AudioEncoder::encode_mel(const Tensor &mel) const
{
    using namespace core::autograd::ops;
    SERIUS_CHECK(mel.ndim() == 2, "mel must be (n_mels, T)");
    const size_t n_mels = mel.shape()[0];
    const size_t n_frames = mel.shape()[1];
    Tensor frames({n_frames, n_mels});
    for (size_t t = 0; t < n_frames; ++t) {
        for (size_t m = 0; m < n_mels; ++m) {
            frames.at(t, m) = mel.at(m, t);
        }
    }

    Tensor conv_out = conv1d(frames);
    Node *hidden = new Node(std::move(conv_out), false);
    for (const auto &layer : layers_) {
        hidden = layer->forward(hidden);
    }
    Node *projected = matmul(hidden, proj_);
    return projected->data.copy();
}

Tensor AudioEncoder::encode(const std::vector<float> &pcm) const
{
    Tensor mel = compute_mel(pcm, config_.mel);
    return encode_mel(mel);
}

}  // namespace audio
}  // namespace serius
