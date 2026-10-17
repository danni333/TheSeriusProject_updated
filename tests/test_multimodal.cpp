#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "vision/patch_embed.h"
#include "vision/vision_encoder.h"
#include "audio/mel.h"
#include "audio/audio_encoder.h"
#include "multimodal/cross_attention.h"
#include "multimodal/input_builder.h"
#include "tensor/tensor.h"

using serius::core::Tensor;
using serius::core::autograd::Node;
using serius::vision::PatchEmbed;
using serius::vision::VisionEncoder;
using serius::vision::VisionConfig;
using serius::audio::MelConfig;
using serius::audio::compute_mel;
using serius::audio::AudioEncoder;
using serius::audio::AudioEncoderConfig;
using serius::multimodal::EncoderSelfAttention;
using serius::multimodal::CrossAttention;
using serius::multimodal::MultimodalInputBuilder;
using serius::multimodal::concat_sequence;

TEST(MultimodalFull, PatchEmbedDeterministic)
{
    PatchEmbed pe1(16, 32);
    PatchEmbed pe2(16, 32);
    Tensor img({3, 32, 32});
    img.fill(0.5f);

    Node *o1 = pe1.forward(img);
    Node *o2 = pe2.forward(img);
    ASSERT_NE(o1, nullptr);
    ASSERT_NE(o2, nullptr);
    // Stessi pesi (stesso seed) -> stessi output
    for (size_t i = 0; i < o1->data.size(); ++i) {
        EXPECT_FLOAT_EQ(o1->data[i], o2->data[i]);
    }
    delete o1;
    delete o2;
}

TEST(MultimodalFull, VisionEncoderDeterministic)
{
    VisionConfig cfg;
    cfg.image_size = 32;
    cfg.patch_size = 8;
    cfg.d_vision = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.d_model = 32;

    VisionEncoder enc1(cfg);
    VisionEncoder enc2(cfg);
    Tensor img({3, 32, 32});
    img.fill(0.5f);

    Tensor t1 = enc1.encode(img);
    Tensor t2 = enc2.encode(img);
    ASSERT_EQ(t1.size(), t2.size());
    for (size_t i = 0; i < t1.size(); ++i) {
        EXPECT_FLOAT_EQ(t1[i], t2[i]);
    }
}

TEST(MultimodalFull, AudioEncoderDeterministic)
{
    AudioEncoderConfig cfg;
    cfg.mel.sample_rate = 8000;
    cfg.mel.n_fft = 200;
    cfg.mel.hop_length = 100;
    cfg.mel.n_mels = 40;
    cfg.d_audio = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.d_model = 32;

    AudioEncoder enc1(cfg);
    AudioEncoder enc2(cfg);
    std::vector<float> pcm(8000, 0.01f);

    Tensor a1 = enc1.encode(pcm);
    Tensor a2 = enc2.encode(pcm);
    ASSERT_EQ(a1.size(), a2.size());
    for (size_t i = 0; i < a1.size(); ++i) {
        EXPECT_FLOAT_EQ(a1[i], a2[i]);
    }
}

TEST(MultimodalFull, CrossAttentionOutputRange)
{
    CrossAttention xattn(32, 4);
    Tensor text_hidden({3, 32});
    text_hidden.fill(0.5f);
    Tensor context({7, 32});
    context.fill(0.25f);
    Node text{std::move(text_hidden), false};
    Node ctx{std::move(context), false};
    Node *out = xattn.forward(&text, &ctx);
    ASSERT_NE(out, nullptr);
    for (size_t i = 0; i < out->data.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out->data[i]));
    }
    delete out;
}

TEST(MultimodalFull, InputBuilderEmpty)
{
    // concat_sequence con una sola parte (equivalente a solo-testo).
    Tensor text_embeds({2, 16});
    text_embeds.fill(0.5f);
    Tensor out = concat_sequence({text_embeds});
    EXPECT_EQ(out.shape()[0], 2u);
    EXPECT_EQ(out.shape()[1], 16u);
    EXPECT_FLOAT_EQ(out.at(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(out.at(1, 0), 0.5f);
}
