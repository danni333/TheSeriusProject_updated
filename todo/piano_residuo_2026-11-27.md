# Piano residuo — 2026-11-27

Piano per chiudere tutti gli item rimanenti. Ordine per impatto/dipendenze.
Riferimenti: `report_placeholder_consolidated.md` (quasi tutto RISOLTO —
restano solo Vulkan/Metal, marcati futuri), `todo/roadmap_prodotto_2026-11-23.md`,
`audit_remediation.md` (pending: F4.13/14, F7.53).

Nota: l'utente lavora attivamente su partial offload (`llama_inference_cuda.cpp`,
`llama_inference.cpp`, `llama_gpu_offload.cpp`) e KV cache rolling —
coordinare, non duplicare.

## Fase 0 — Chiusura misure e baseline

| # | Task | Verifica |
|---|------|----------|
| 0.1 | Riverificare gain NTA — **DONE**: idle 2026-11-27, SmolLM2 Q4_0 ~63–76 tok/s, Qwen2.5-0.5B Q4_K_M ~16–19 tok/s; `BENCHMARKS.md` aggiornato | fatto |
| 0.2 | ~~Fix `KVCacheTest.RollingBufferOverwrite`~~ — **PASSA** in ctest (edit utente completati) | fatto |
| 0.3 | ~~`ctest -j8`~~ — **DONE**: 490/491 pass, 7 skip (richiedono modelli), `test_cuda` Not Run = serve `ctest -C Release` sul generator VS (non è un fallimento) | fatto |

## Fase 1 — CPU decode (residuo post NTA/OMP)

| # | Task | Note |
|---|------|------|
| 1.1 | Eventuale tuning prefetch distance (+16→+24/+32 blocchi) A/B | solo se 0.1 conferma margine |
| 1.2 | lm_head top-k: **DECISIONE NO-GO** — il sampler usa top-k/top-p *esatti* sui logits grezzi: servono tutti i vocab logits → GEMV completo inevitabile senza cambiare la distribuzione (parità output). I ~14% sono strutturali | chiuso |
| 1.3 | *(opzionale, grande)* loop stile ggml: thread persistenti, barrier ridotte | unico modo vero verso 0.7x llama.cpp; stimare 2-3 gg |

## Fase 2 — GPU decode

| # | Task | Stato |
|---|------|-------|
| 2.1 | Partial offload: completare (CUDA graph per prefisso, test modello >VRAM) | **utente in corso** |
| 2.2 | `4b`: elegibility CUDA graph — **DONE**: log warn-once del motivo (dtype/embed/lm_head/proiezioni/capture fail) in `llama_inference_cuda.cpp`; estendere quant type coperti resta opzionale | parziale |
| 2.3 | `3`: prefill GPU gap — `prefill_dq_cache` fatta (utente); mancano misure warm / cache FP16 completa | misurare, poi decidere |
| 2.4 | `audit F4.13/14/15`: speculative decoding — **DONE**: verifica target alla posizione corretta, niente early-stop dopo rejection, acceptance_rate su drafted_tokens + regression test | DONE |

## Fase 3 — Server e stabilità

| # | Task | Note |
|---|------|------|
| 3.1 | **Continuous batching HTTP** (roadmap #2): KV pool per request, scheduler decode, SSE isolato | item più grosso; CUDA graph per-slot o eager per slot extra |
| 3.2 | Soak server con `stress_runner` → fix leak/crash | roadmap #6 |
| 3.3 | Parità OpenAI residua (#7): test HTTP live — errori standard, `usage`, `logprobs`, `n>1`, `stop` multipli, penalties, `response_format` | checklist |

## Fase 4 — Packaging e pulizia

| # | Task |
|---|------|
| 4.1 | Versione: **gia' ok** — `project(VERSION 0.1.0)` + `SERIUS_VERSION` in CMake, `cli.h` 0.1.0; fixato `cmd_mcp.cpp` che hardcodava "0.1.0". Restano: changelog, installer Windows, `docs/installation.md` | parziale |
| 4.2 | ~~`.gitignore`~~ — **gia' completo** (build*/, *.obj/*.exe/*.log/*.gguf/models/) | fatto |
| 4.3 | ~~`audit F7.53`~~ — già DONE in audit_remediation | fatto |
| 4.4 | ~~`src/inference/` vs `src/infer/`~~ — **`src/inference/` non esiste piu'`**: `src/infer/` e' canonico, `experimental/inference/` resta nel perimetro sperimentale | fatto |

## Ordine esecuzione

1. **2.2** — log elegibility CUDA graph (self-contained, non tocca il lavoro utente)
2. **0.3 → 0.1** — mappare flaky + riverificare NTA
3. **2.4** — speculative decoding fixes
4. **3.3** — parità API OpenAI (verifiche live)
5. **3.1** — continuous batching (sessione dedicata, item più grosso)
6. **4.x** — packaging/pulizia
7. **1.3** — ggml-style loop solo se il gap CPU resta >0.7x

Fatto: 2026-11-24 — prefetch NTA +67%/+29% decode misurato; OMP=core
fisici +25%; ppo/rlhf/tools/plugin/graph-compiler completati in sessioni
precedenti.
