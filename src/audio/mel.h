#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace audio {

using core::Tensor;

struct MelConfig
{
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;
};

/// Number of STFT frames for a PCM buffer of n_samples.
size_t n_stft_frames(size_t n_samples, int n_fft, int hop_length);

/// Mel filterbank of shape (n_mels, n_fft/2 + 1).
Tensor create_mel_filterbank(const MelConfig &config = {});

/// Mel spectrogram (n_mels, n_frames) from mono PCM in [-1, 1].
Tensor compute_mel(const std::vector<float> &audio, const MelConfig &config = {});

}  // namespace audio
}  // namespace serius
