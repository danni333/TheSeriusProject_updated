# Changelog

Cronologia ad alto livello delle modifiche principali a **TheSeriusProject**.

> Nota: questa pagina parte da una versione retroattiva; per ora le date sono indicative.

---

## [Unreleased]

### Added

- **Web UI** (`ui/`): chat browser servita dal server C++ (`--ui`), streaming SSE, impostazioni (temperature, top_p, top_k, max_tokens, system prompt), pannello Models per load/unload a runtime, persistenza `localStorage`, markdown minimale.
- **Multi-modello a runtime**: registry `LoadedModel` con KV cache e tokenizer per modello; `POST /v1/models/load` (path locale o `hf://repo` + `quant`), `DELETE /v1/models/{id}`; load in parallelo all'inference (`load_mutex_` separato da `inference_mutex_`).
- **KV cache persistente** in chat: riuso del prefisso quando il prompt è un'estensione esatta del turno precedente.
- **BOS/EOS reali**: `BPETokenizer::bos_id()/eos_id()` popolati dalle metadata GGUF; stop corretto su `<|im_end|>`/EOS.
- Env: `SERIUS_PREFILL_BATCH=1` (prefill batched opt-in), `SERIUS_KV_FP32=1`, `SERIUS_GGUF_ROPE_PERM=0` (diagnostica), `SERIUS_NO_AVX512`, `SERIUS_NO_QUANT`, `SERIUS_NO_FUSE`, `SERIUS_PROF`, `SERIUS_GPU_TIME`.
- Test: `BatchedVsSequentialPrefill` e `LLaMARealGGUFTest` (parità batched-vs-sequential su GGUF reale, `SERIUS_TEST_GGUF` per cambiare modello); `ServerTest.MultiModelRegistry`.
- Documentazione iniziale:
  - `README.md` arricchito con panoramica, quick start e struttura del codice.
  - Prima versione delle pagine in `docs/`:
    - `getting-started.md`
    - `installation.md`
    - `architecture.md`
    - `api-reference.md` (WIP)
    - `cli-reference.md` (WIP)
    - `configuration.md` (WIP)
    - `supported-models.md` (design)
    - `troubleshooting.md`
    - `contributing.md`
    - `changelog.md`

### Changed

- Migliorata la descrizione del progetto nel `README.md`.

### Fixed

- **Batched prefill degenere**: root cause in `Tensor` — accessor host restituivano dati senza `syncToHost()` dopo matmul CUDA (>1M FLOP) e le scritture CPU non invalidavano la copia GPU. Risolto con sync esplicita + invalidazione. Parità verificata su GGUF reale (logits max_diff ~0.003 a parità di pesi).
- Chat web: prompt ora include tutta la conversazione in ChatML (prima solo l'ultimo messaggio); stop token errato per Qwen (EOS hardcoded); errori SSE mostrati in UI; abort durante lo streaming; messaggi di errore `system` non più inviati al modello.
- Loader GGUF/safetensors: hardening contro input malformati.
- `static_files.cpp`: fix compilazione Windows (`wchar_t` preferred_separator).

---

Versioni future verranno aggiunte man mano che il progetto rilascia milestone significative.

