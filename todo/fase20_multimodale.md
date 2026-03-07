# Fase 20 — Multimodalità Nativa (Vision Encoder, Audio, Unified Tokenization)

> Un LLM moderno processa testo, immagini e audio.
> Servono encoder specializzati che producono embeddings
> nello stesso spazio semantico dei token testuali.

---

## 1. Vision Encoder (CLIP-style)

### Architettura: Vision Transformer (ViT)

```
Immagine 224x224
     │
     ▼
Patch Embedding (16x16 patches → 196 token visivi)
     │
     ▼
Positional Encoding (196 posizioni)
     │
     ▼
Transformer Encoder × N layers
     │
     ▼
Projection Linear (d_vision → d_model)
     │
     ▼
196 visual tokens (stessa dimensione dei token testuali)
```

### Patch Embedding

```cpp
Tensor patch_embed(const Tensor& image, int patch_size, int d_vision) {
    // image: (3, H, W) → patches: (num_patches, patch_size²×3)
    int H = image.shape()[1], W = image.shape()[2];
    int n_patches = (H / patch_size) * (W / patch_size);
    Tensor patches({n_patches, patch_size * patch_size * 3});
    // Estrai e flatten ogni patch
    // ...
    // Proiezione lineare
    return matmul(patches, patch_proj);  // (n_patches, d_vision)
}
```

### Projection nello spazio del LLM

```cpp
// Dopo il Vision Transformer:
// visual_tokens: (196, d_vision=768)
// Proietta nello spazio del LLM
Tensor visual_embeds = matmul(visual_tokens, proj);  // (196, d_model=4096)
```

---

## 2. Cross-Attention (Testo guarda Immagine)

### Flusso nel Transformer Block

```
Text tokens (Q)
     │
     ├──▶ Self-Attention (testo ↔ testo)
     │
     ├──▶ Cross-Attention (testo → immagine)
     │         Q = text_hidden
     │         K, V = visual_embeds
     │
     ├──▶ FFN
     │
     ▼
Output
```

### Implementazione

```cpp
Tensor cross_attention(const Tensor& text_hidden,   // Q source
                        const Tensor& visual_embeds, // K, V source
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo) {
    Tensor Q = matmul(text_hidden, Wq);     // (seq_text, d_model)
    Tensor K = matmul(visual_embeds, Wk);   // (196, d_model)
    Tensor V = matmul(visual_embeds, Wv);   // (196, d_model)
    
    // Scaled dot-product (nessuna causal mask qui)
    Tensor scores = matmul(Q, transpose(K)) / sqrtf(d_k);
    Tensor attn = softmax(scores);
    Tensor out = matmul(attn, V);
    return matmul(out, Wo);
}
```

---

## 3. Audio-to-Token (Whisper-style)

### Pipeline

```
Audio WAV
     │
     ▼
Mel Spectrogram (80 bins × T frames)
     │
     ▼
Conv1D layers (downsampling temporale)
     │
     ▼
Transformer Encoder × N
     │
     ▼
Projection (d_audio → d_model)
     │
     ▼
Audio tokens (stessa dimensione dei token testuali)
```

### Mel Spectrogram

```cpp
Tensor compute_mel(const vector<float>& audio, int sr, int n_mels=80) {
    // 1. STFT (Short-Time Fourier Transform)
    auto stft = compute_stft(audio, /*window=*/400, /*hop=*/160);
    // 2. Power spectrum
    auto power = abs_squared(stft);
    // 3. Mel filterbank
    auto mel_filters = create_mel_filterbank(sr, n_mels);
    return matmul(mel_filters, power);  // (n_mels, T)
}
```

---

## 4. Input Unificato

```cpp
// Combina testo + immagine + audio in una singola sequenza
Tensor build_multimodal_input(const string& text,
                                const Tensor* image,
                                const Tensor* audio) {
    vector<Tensor> embeds;
    
    if (image) {
        auto vis_tokens = vision_encoder.encode(*image);
        embeds.push_back(special_token("<IMG>"));
        embeds.push_back(vis_tokens);
        embeds.push_back(special_token("</IMG>"));
    }
    if (audio) {
        auto aud_tokens = audio_encoder.encode(*audio);
        embeds.push_back(special_token("<AUD>"));
        embeds.push_back(aud_tokens);
        embeds.push_back(special_token("</AUD>"));
    }
    
    auto text_tokens = text_encoder.embed(tokenizer.encode(text));
    embeds.push_back(text_tokens);
    
    return concat(embeds, /*dim=*/0);  // sequenza unica
}
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/vision/patch_embed.h/.cpp` | Patch embedding + ViT |
| `src/vision/vision_encoder.h/.cpp` | Vision Transformer completo |
| `src/audio/mel.h/.cpp` | Mel spectrogram |
| `src/audio/audio_encoder.h/.cpp` | Audio encoder (Conv + Transformer) |
| `src/multimodal/cross_attention.h/.cpp` | Cross-Attention layer |
| `src/multimodal/input_builder.h/.cpp` | Combina modalità |
| `tests/test_vision.cpp` | Patch embed, output shape |
| `tests/test_multimodal.cpp` | Forward con testo + immagine |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_patch_embed` | 224×224 → 196 patches corretti |
| `test_vision_shape` | Output: (196, d_model) |
| `test_cross_attn` | Text tokens attendono a visual tokens |
| `test_mel` | Spectrogram shape corretto per audio 1s |
| `test_multimodal_fwd` | Forward con testo+immagine senza crash |
