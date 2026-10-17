#include "input_builder.h"

#include <cstring>

#include "../common/serius_check.h"
#include "../init/weight_init.h"

namespace serius {
namespace multimodal {

Tensor concat_sequence(const std::vector<Tensor> &parts)
{
    SERIUS_CHECK(!parts.empty(), "concat_sequence requires at least one tensor");
    SERIUS_CHECK(parts[0].ndim() == 2, "concat_sequence expects 2D embeddings");
    const size_t cols = parts[0].shape()[1];
    size_t rows = 0;
    for (const Tensor &part : parts) {
        SERIUS_CHECK(part.ndim() == 2 && part.shape()[1] == cols, "embedding width mismatch");
        rows += part.shape()[0];
    }

    Tensor out({rows, cols});
    size_t row_offset = 0;
    for (const Tensor &part : parts) {
        const Tensor src = part.contiguous();
        for (size_t r = 0; r < src.shape()[0]; ++r) {
            for (size_t c = 0; c < src.shape()[1]; ++c) {
                out.at(row_offset + r, c) = src.at(r, c);
            }
        }
        row_offset += src.shape()[0];
    }
    return out;
}

MultimodalInputBuilder::MultimodalInputBuilder(size_t d_model, unsigned seed) : d_model_(d_model)
{
    SERIUS_CHECK(d_model > 0, "d_model must be positive");
    auto init_token = [&](Tensor &token, unsigned token_seed) {
        token = Tensor({1, d_model_});
        init::normal_init(token, 0.0F, 0.02F, token_seed);
    };
    init_token(img_start_, seed);
    init_token(img_end_, seed + 1);
    init_token(aud_start_, seed + 2);
    init_token(aud_end_, seed + 3);
}

Tensor MultimodalInputBuilder::special_token(const Tensor &embed) const { return embed.copy(); }

Tensor MultimodalInputBuilder::build(const Tensor &text_embeds, const Tensor *visual_tokens,
                                     const Tensor *audio_tokens) const
{
    SERIUS_CHECK(text_embeds.ndim() == 2 && text_embeds.shape()[1] == d_model_,
                 "text_embeds must be (seq, d_model)");

    std::vector<Tensor> parts;
    if (visual_tokens != nullptr) {
        SERIUS_CHECK(visual_tokens->ndim() == 2 && visual_tokens->shape()[1] == d_model_,
                     "visual tokens must be (n, d_model)");
        parts.push_back(special_token(img_start_));
        parts.push_back(*visual_tokens);
        parts.push_back(special_token(img_end_));
    }
    if (audio_tokens != nullptr) {
        SERIUS_CHECK(audio_tokens->ndim() == 2 && audio_tokens->shape()[1] == d_model_,
                     "audio tokens must be (n, d_model)");
        parts.push_back(special_token(aud_start_));
        parts.push_back(*audio_tokens);
        parts.push_back(special_token(aud_end_));
    }
    parts.push_back(text_embeds);
    return concat_sequence(parts);
}

}  // namespace multimodal
}  // namespace serius
