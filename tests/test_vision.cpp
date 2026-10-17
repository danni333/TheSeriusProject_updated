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
using serius::vision::preprocess_image;
using serius::audio::MelConfig;
using serius::audio::compute_mel;
using serius::audio::AudioEncoder;
using serius::audio::AudioEncoderConfig;
using serius::multimodal::EncoderSelfAttention;
using serius::multimodal::CrossAttention;
using serius::multimodal::MultimodalInputBuilder;
using serius::multimodal::concat_sequence;
using serius::multimodal::concat_sequence;

TEST(Vision, PreprocessImageResizeAndNormalize)
{
    // Immagine quadrata 64x64 a 1 canale, valori in [0,255]
    Tensor img({64, 64});
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = 128.0f;
    }

    Tensor chw = preprocess_image(img, 32, 32);
    ASSERT_EQ(chw.ndim(), 3u);
    ASSERT_EQ(chw.shape()[0], 3u);
    ASSERT_EQ(chw.shape()[1], 32u);
    ASSERT_EQ(chw.shape()[2], 32u);
}

TEST(Vision, PatchEmbedShape)
{
    PatchEmbed pe(16, 64);
    // Immagine 64x64 -> 4x4 = 16 patch, patch_dim = 16*16*3 = 768
    Tensor img({3, 64, 64});
    img.fill(0.5f);

    Node *out = pe.forward(img);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(out->data.ndim(), 2u);
    EXPECT_EQ(out->data.shape()[0], 16u);
    EXPECT_EQ(out->data.shape()[1], 64u);
    delete out;
}

TEST(Vision, PatchExtractCorrectCount)
{
    Tensor img({3, 32, 32});
    img.fill(1.0f);
    Tensor patches = serius::vision::extract_patches(img, 8);
    // 32/8 = 4 per lato -> 16 patch, patch_dim = 8*8*3 = 192
    ASSERT_EQ(patches.ndim(), 2u);
    EXPECT_EQ(patches.shape()[0], 16u);
    EXPECT_EQ(patches.shape()[1], 192u);
}

TEST(Vision, VisionEncoderOutputShape)
{
    VisionConfig cfg;
    cfg.image_size = 64;
    cfg.patch_size = 16;
    cfg.d_vision = 64;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.d_ff = 128;
    cfg.d_model = 64;

    VisionEncoder enc(cfg);
    EXPECT_EQ(enc.num_patches(), 16u);

    Tensor img({3, 64, 64});
    img.fill(0.5f);
    Tensor tokens = enc.encode(img);
    ASSERT_EQ(tokens.ndim(), 2u);
    EXPECT_EQ(tokens.shape()[0], 16u);
    EXPECT_EQ(tokens.shape()[1], 64u);
}

TEST(Vision, VisionEncoderNodeForward)
{
    VisionConfig cfg;
    cfg.image_size = 64;
    cfg.patch_size = 16;
    cfg.d_vision = 64;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.d_ff = 128;
    cfg.d_model = 64;

    VisionEncoder enc(cfg);
    Tensor patch_tokens({16, 64});
    patch_tokens.fill(0.1f);
    Node input{std::move(patch_tokens), false};
    Node *out = enc.forward(&input);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->data.shape()[0], 16u);
    EXPECT_EQ(out->data.shape()[1], 64u);
    delete out;
}

TEST(Audio, MelSpectrogramShape)
{
    MelConfig cfg;
    cfg.sample_rate = 16000;
    cfg.n_fft = 400;
    cfg.hop_length = 160;
    cfg.n_mels = 80;

    // 1 secondo di audio a 16kHz
    std::vector<float> pcm(16000, 0.01f);
    Tensor mel = compute_mel(pcm, cfg);
    ASSERT_EQ(mel.ndim(), 2u);
    EXPECT_EQ(mel.shape()[0], 80u);
    EXPECT_GT(mel.shape()[1], 0u);
}

TEST(Audio, MelFilterbankShape)
{
    MelConfig cfg;
    cfg.n_mels = 40;
    cfg.n_fft = 400;
    cfg.sample_rate = 16000;
    Tensor fb = serius::audio::create_mel_filterbank(cfg);
    ASSERT_EQ(fb.ndim(), 2u);
    EXPECT_EQ(fb.shape()[0], 40u);
    EXPECT_EQ(fb.shape()[1], 201u);
}

TEST(Audio, AudioEncoderOutputShape)
{
    AudioEncoderConfig cfg;
    cfg.mel.sample_rate = 16000;
    cfg.mel.n_fft = 400;
    cfg.mel.hop_length = 160;
    cfg.mel.n_mels = 80;
    cfg.d_audio = 64;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.d_ff = 128;
    cfg.d_model = 64;

    AudioEncoder enc(cfg);
    std::vector<float> pcm(16000, 0.01f);
    Tensor tokens = enc.encode(pcm);
    ASSERT_EQ(tokens.ndim(), 2u);
    EXPECT_EQ(tokens.shape()[1], 64u);
    EXPECT_GT(tokens.shape()[0], 0u);
}

TEST(Multimodal, EncoderSelfAttention)
{
    EncoderSelfAttention attn(64, 4);
    Tensor input({8, 64});
    input.fill(0.1f);
    Node node(std::move(input), false);
    Node *out = attn.forward(&node);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->data.shape()[0], 8u);
    EXPECT_EQ(out->data.shape()[1], 64u);
    delete out;
}

TEST(Multimodal, CrossAttention)
{
    CrossAttention xattn(64, 4);
    Tensor text_hidden({5, 64});
    text_hidden.fill(0.2f);
    Tensor context({16, 64});
    context.fill(0.1f);
    Node text(std::move(text_hidden), false);
    Node ctx(std::move(context), false);
    Node *out = xattn.forward(&text, &ctx);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->data.shape()[0], 5u);
    EXPECT_EQ(out->data.shape()[1], 64u);
    delete out;
}

TEST(Multimodal, ConcatSequence)
{
    Tensor a({3, 4});
    a.fill(1.0f);
    Tensor b({2, 4});
    b.fill(2.0f);
    Tensor c({5, 4});
    c.fill(3.0f);

    Tensor out = concat_sequence({a, b, c});
    ASSERT_EQ(out.ndim(), 2u);
    EXPECT_EQ(out.shape()[0], 10u);
    EXPECT_EQ(out.shape()[1], 4u);
    EXPECT_FLOAT_EQ(out.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(out.at(3, 0), 2.0f);
    EXPECT_FLOAT_EQ(out.at(9, 0), 3.0f);
}

TEST(Multimodal, InputBuilderBuild)
{
    // Simula i token speciali di input_builder e verifica concat_sequence.
    Tensor img_start({1, 64});
    img_start.fill(0.9f);
    Tensor visual({9, 64});
    visual.fill(0.7f);
    Tensor img_end({1, 64});
    img_end.fill(0.9f);
    Tensor text_embeds({4, 64});
    text_embeds.fill(0.5f);

    Tensor out = concat_sequence({img_start, visual, img_end, text_embeds});
    EXPECT_EQ(out.shape()[0], 15u);
    EXPECT_EQ(out.shape()[1], 64u);
    EXPECT_FLOAT_EQ(out.at(0, 0), 0.9f);
    EXPECT_FLOAT_EQ(out.at(1, 0), 0.7f);
    EXPECT_FLOAT_EQ(out.at(10, 0), 0.9f);
    EXPECT_FLOAT_EQ(out.at(11, 0), 0.5f);
    EXPECT_FLOAT_EQ(out.at(14, 0), 0.5f);
}
