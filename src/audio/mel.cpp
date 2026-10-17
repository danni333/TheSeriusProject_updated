#define _USE_MATH_DEFINES
#include "mel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../common/serius_check.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace serius {
namespace audio {

namespace {

float hz_to_mel(float hz) { return 2595.0F * std::log10(1.0F + hz / 700.0F); }

float mel_to_hz(float mel) { return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F); }

}  // namespace

size_t n_stft_frames(size_t n_samples, int n_fft, int hop_length)
{
    SERIUS_CHECK(n_fft > 0 && hop_length > 0, "STFT window/hop must be positive");
    const size_t window = static_cast<size_t>(n_fft);
    const size_t hop = static_cast<size_t>(hop_length);
    if (n_samples < window) {
        return 0;
    }
    return 1 + (n_samples - window) / hop;
}

Tensor create_mel_filterbank(const MelConfig &config)
{
    SERIUS_CHECK(config.n_mels > 0 && config.n_fft > 0 && config.sample_rate > 0, "invalid MelConfig");
    const size_t n_freq = static_cast<size_t>(config.n_fft) / 2 + 1;
    const size_t n_mels = static_cast<size_t>(config.n_mels);
    Tensor filters({n_mels, n_freq});
    filters.zeros();

    const float f_max = static_cast<float>(config.sample_rate) / 2.0F;
    const float mel_min = hz_to_mel(0.0F);
    const float mel_max = hz_to_mel(f_max);

    std::vector<float> mel_points(n_mels + 2);
    for (size_t i = 0; i < mel_points.size(); ++i) {
        const float mel = mel_min + (mel_max - mel_min) * static_cast<float>(i) / static_cast<float>(n_mels + 1);
        const float hz = mel_to_hz(mel);
        const float bin =
            (hz / static_cast<float>(config.sample_rate)) * static_cast<float>(config.n_fft);
        mel_points[i] = bin;
    }

    for (size_t m = 0; m < n_mels; ++m) {
        const float left = mel_points[m];
        const float center = mel_points[m + 1];
        const float right = mel_points[m + 2];
        for (size_t k = 0; k < n_freq; ++k) {
            const float kf = static_cast<float>(k);
            float weight = 0.0F;
            if (kf >= left && kf <= center && center > left) {
                weight = (kf - left) / (center - left);
            } else if (kf >= center && kf <= right && right > center) {
                weight = (right - kf) / (right - center);
            }
            filters.at(m, k) = std::max(0.0F, weight);
        }
    }
    return filters;
}

Tensor compute_mel(const std::vector<float> &audio, const MelConfig &config)
{
    const size_t n_fft = static_cast<size_t>(config.n_fft);
    const size_t hop = static_cast<size_t>(config.hop_length);
    const size_t n_frames = n_stft_frames(audio.size(), config.n_fft, config.hop_length);
    SERIUS_CHECK(n_frames > 0, "audio too short for STFT window");

    const size_t n_freq = n_fft / 2 + 1;
    Tensor power({n_freq, n_frames});

    std::vector<float> window(n_fft);
    for (size_t n = 0; n < n_fft; ++n) {
        window[n] = 0.54F - 0.46F * std::cos(2.0F * static_cast<float>(M_PI) * static_cast<float>(n) /
                                             static_cast<float>(n_fft - 1));
    }

    std::vector<float> frame(n_fft);
    const float two_pi = 2.0F * static_cast<float>(M_PI);
    for (size_t t = 0; t < n_frames; ++t) {
        const size_t offset = t * hop;
        for (size_t n = 0; n < n_fft; ++n) {
            frame[n] = audio[offset + n] * window[n];
        }
        for (size_t k = 0; k < n_freq; ++k) {
            float re = 0.0F;
            float im = 0.0F;
            const float ang_step = -two_pi * static_cast<float>(k) / static_cast<float>(n_fft);
            for (size_t n = 0; n < n_fft; ++n) {
                const float ang = ang_step * static_cast<float>(n);
                re += frame[n] * std::cos(ang);
                im += frame[n] * std::sin(ang);
            }
            power.at(k, t) = re * re + im * im;
        }
    }

    Tensor filters = create_mel_filterbank(config);
    Tensor mel({static_cast<size_t>(config.n_mels), n_frames});
    mel.zeros();
    for (int m = 0; m < config.n_mels; ++m) {
        for (size_t t = 0; t < n_frames; ++t) {
            float acc = 0.0F;
            for (size_t k = 0; k < n_freq; ++k) {
                acc += filters.at(static_cast<size_t>(m), k) * power.at(k, t);
            }
            mel.at(static_cast<size_t>(m), t) = std::log(acc + 1.0e-6F);
        }
    }
    return mel;
}

}  // namespace audio
}  // namespace serius
