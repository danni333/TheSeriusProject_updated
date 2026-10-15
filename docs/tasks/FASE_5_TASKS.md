# 🌐 FASE 5 — Multimodale & Long-Context
**Periodo**: Q2-Q3 2027 (Aprile - Luglio) | **Tasks**: 18 | **Completate**: 0

---

## 5.1 Vision

### T5.01 — ViT (Vision Transformer)
- **Priorità**: 🔴 Critica | **Stima**: 5d
- **File**: `src/vision/vit.h`, `src/vision/vit.cpp` (NUOVI)
- **Criteria**: ViT-L/14: image → patches → embedding → transformer layers → features. Support per risoluzioni variabili.

### T5.02 — Image Patch Tokenizer
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/vision/image_tokenizer.h` (NUOVO)
- **Criteria**: Image → patches 14x14 → linear projection → token embeddings. Concat con text embeddings.

### T5.03 — Cross-Modal Attention
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/attention/cross_modal.h` (NUOVO)
- **Criteria**: Text query attende a image key/value. Bidirezionale opzionale.

### T5.04 — Video Frame Sampling
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/vision/video_sampler.h` (NUOVO)
- **Criteria**: Estrai N frame da video a intervalli regolari. Temporal position encoding.

### T5.05 — Multimodal Training
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/train/multimodal_trainer.h` (NUOVO)
- **Criteria**: Training su coppie image-text. Loss: next token prediction con modalità mista.

---

## 5.2 Long-Context

### T5.06 — YaRN RoPE Scaling
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/transformer/positional.cpp` (estendere)
- **Criteria**: YaRN: modifica frequenze RoPE per scaling. Config: target_context_len. Test: modello trainato a 4K funziona a 128K.

### T5.07 — NTK-Aware Interpolation
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/transformer/positional.cpp` (estendere)
- **Criteria**: NTK-aware: scala le frequenze base in base al rapporto di estensione. Alternativa a YaRN.

### T5.08 — Ring Attention (Multi-GPU Context)
- **Priorità**: 🟡 Media | **Stima**: 5d
- **File**: `src/attention/ring_attention.h` (NUOVO)
- **Criteria**: Context distribuito su N GPU. Communication ring per K,V. Total context = N × local_context.

### T5.09 — Needle-in-Haystack Test
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `tests/test_long_context.cpp` (NUOVO)
- **Criteria**: Inserisce fatto noto in posizione random di un context lungo. Chiede il fatto. Target: ≥95% @ 256K.

---

## 5.3 Speculative Decoding

### T5.10 — Draft Model Integration
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/infer/speculative.h` (NUOVO)
- **Criteria**: Carica draft model piccolo (es. 100M). Genera k token candidati. Forward veloce.

### T5.11 — Parallel Verification
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/infer/speculative.h` (estendere)
- **Criteria**: Main model verifica k token in un singolo forward. Accetta prefix corretto, rifiuta dal primo errore.

### T5.12 — Tree-Based Speculative Decoding
- **Priorità**: 🟡 Media | **Stima**: 5d
- **File**: `src/infer/tree_speculative.h` (NUOVO)
- **Criteria**: Draft genera albero di possibilità. Main model verifica più rami. Accetta il ramo più lungo corretto.

### T5.13 — Self-Speculative (Early Exit)
- **Priorità**: 🟢 Bassa | **Stima**: 4d
- **File**: `src/infer/self_speculative.h` (NUOVO)
- **Criteria**: Usa early layers dello stesso modello come draft. No modello separato necessario.

---

## 5.4 Unified Multimodal

### T5.14 — Unified Embedding Space
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/transformer/unified_embed.h` (NUOVO)
- **Criteria**: Text, image, audio tokens tutti nello stesso spazio embedding. Modality embeddings per distinguere.

### T5.15 — Audio Encoder (Whisper-style)
- **Priorità**: 🟢 Bassa | **Stima**: 5d
- **File**: `src/audio/audio_encoder.h` (NUOVO)
- **Criteria**: Mel spectrogram → Conv layers → Transformer → audio embeddings.

### T5.16 — Multimodal Generation
- **Priorità**: 🟢 Bassa | **Stima**: 5d
- **File**: `src/vision/image_gen.h` (NUOVO)
- **Criteria**: Generazione immagini da testo (basic). Diffusion o token-based approach.

### T5.17 — Multimodal Benchmark (VQA)
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/eval/vqa_eval.h` (NUOVO)
- **Criteria**: Visual Question Answering su dataset standard. Accuracy reporting.

### T5.18 — Speculative Decoding Benchmark
- **Priorità**: 🟠 Alta | **Stima**: 1d
- **File**: `tests/bench_speculative.cpp` (NUOVO)
- **Criteria**: Confronto tok/s: no-spec vs draft-spec vs tree-spec vs self-spec. Report con acceptance rate.
