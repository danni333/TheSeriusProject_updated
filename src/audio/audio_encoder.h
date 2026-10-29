#pragma once

#include <memory>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"
#include "mel.h"

namespace serius {
namespace audio {

using core::Tensor;
using core::autograd::Node;

struct AudioEncoderConfig
{
    MelConfig mel;
    size_t d_audio = 64;
    size_t n_layers = 1;
    size_t n_heads = 4;
    size_t d_ff = 128;
    size_t d_model = 64;
    int conv_kernel = 3;
    int conv_stride = 2;
    unsigned seed = 42;
};

/// Whisper-style encoder: mel spectrogram -> Conv1D downsample -> Transformer -> LLM projection.
class AudioEncoder
{
   public:
    explicit AudioEncoder(const AudioEncoderConfig &config);
    AudioEncoder();
    ~AudioEncoder();

    AudioEncoder(const AudioEncoder &) = delete;
    AudioEncoder &operator=(const AudioEncoder &) = delete;

    size_t d_model() const { return config_.d_model; }
    const AudioEncoderConfig &config() const { return config_; }

    Tensor encode(const std::vector<float> &pcm) const;
    Tensor encode_mel(const Tensor &mel) const;

   private:
    struct Layer;

    Tensor conv1d(const Tensor &frames_by_mel) const;

    AudioEncoderConfig config_;
    Node *conv_weight_;
    Node *conv_bias_;
    Node *proj_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace audio
}  // namespace serius
