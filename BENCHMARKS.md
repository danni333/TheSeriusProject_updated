# Benchmark baseline — misurato su hardware reale

Misurazioni eseguite il 2026-11-22 con `serius generate` su:

- **GPU**: NVIDIA GeForce RTX 3060 Laptop (6 GiB)
- **CPU**: Intel Core i9-11900H
- **Build**: `build-ninja`, Release, `SERIUS_USE_CUDA=ON`

## Risultati

| Modello | Backend | Token/s (decode) | Note |
|---------|---------|------------------|------|
| Qwen2.5-0.5B-Instruct Q4_K_M | CUDA + graph | **~84 tok/s** | 16 token in 0.19 s (run breve) |
| Qwen2.5-0.5B-Instruct Q4_K_M | CUDA (eager) | ~44 tok/s | misura precedente, pre-graph |
| Qwen2.5-0.5B-Instruct Q4_K_M | CPU (`--gpu cpu`) | **~2–8 tok/s** | alta varianza termica |
| Qwen2.5-0.5B-Instruct FP16 | CUDA (eager) | **~10 tok/s** | niente pesi quant → graph non eleggibile |
| Qwen2.5-0.5B-Instruct Q4_K_M | CUDA + graph | **147.8 tok/s** | 64 token, misurato post-fix VRAM |
| Qwen2.5-Coder-1.5B Q4_K_M | CUDA + graph | **91.7 tok/s** | 64 token, ~1.94 GiB VRAM — ora entra in GPU |
| Qwen2.5-Coder-1.5B Q4_K_M | CPU (fallback) | **~1.3 tok/s** | solo se `--gpu cpu` forzato o VRAM piena |
| SmolLM2-135M Q4_0 | CUDA + graph | **~120 tok/s** | 16 token in 0.13 s |
| SmolLM2-135M Q4_0 | CUDA (eager, `SERIUS_NO_GRAPH=1`) | ~27 tok/s | stesso modello senza graph |
| SmolLM2-135M Q8_0 | CUDA + graph | **~66 tok/s** | 16 token in 0.24 s |
| Qwen2.5-Coder-1.5B Q4_K_M | parziale 8/28 layer (`--gpu-layers 8`) | **~0.3 tok/s** | corretto ma LENTO — vedi sotto |

Nota: il CUDA graph decode (`llama_forward_single_token_cuda_graph`) è
attivo solo se **tutti** i layer hanno pesi quantizzati packed su GPU
(q/k/v/o/gate/up/down + lm_head). I modelli FP16 ricadono nel path eager
denso, molto più lento (~10 tok/s sul 0.5B).

Comandi usati:

```bash
serius generate -m models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --type qwen -p "Explain what a neural network is in one sentence." \
    --max-tokens 64 --temperature 0 --seed 1 [--gpu cpu]
```

## Confronto con llama.cpp — MISURATO (2026-11-23)

Misura diretta affiancata: stesso modello GGUF, stessa RTX 3060 Laptop,
64 token generati, `temperature=0`. llama.cpp = `llama-server`
(b1-e365e65, CUDA12, `-ngl 99`, endpoint `/completion`); Serius =
`serius generate --gpu auto` sul build `build-ninja` con il fix VRAM.

| Modello | Serius decode | llama.cpp decode | Rapporto |
|---------|---------------|------------------|----------|
| Qwen2.5-0.5B-Instruct Q4_K_M | **147.8 tok/s** | **186.2 tok/s** | ~0.79x (21% più lento) |
| Qwen2.5-Coder-1.5B Q4_K_M | **91.7 tok/s** | **91.3 tok/s** | ~1.0x (parità) |

VRAM occupata da Serius dopo il fix packed-quant-only:

| Modello | Serius GPU footprint | llama.cpp device buffer |
|---------|---------------------|------------------------|
| 0.5B Q4_K_M | ~0.96 GiB | ~374 MiB + compute buffer |
| 1.5B Q4_K_M | ~1.94 GiB | ~935 MiB + compute buffer |

Note:
- Il decode 1.5B è a **parità** con llama.cpp; il 0.5B resta ~20% più lento.
- llama.cpp resta più veloce sul prefill (batch GEMM reale): ~110 tok/s
  sul 0.5B vs prefill Serius token-per-token via CUDA graph. Con
  `SERIUS_PREFILL_BATCH=1` il batched prefill usa i pesi packed
  dequantizzati in scratch FP16 riusabile.
- Misure precedenti ("~2–3x più lento") erano pre-graph/pre-fix; i numeri
  stimati CPU non sono ancora stati riverificati.

## Finding: VRAM stimata eccessiva per modelli quantizzati

Il piano di offload GPU (`src/infer/llama_gpu_offload.cpp`) carica **entrambi**:

- pesi densi FP16 per il prefill cuBLAS, e
- pesi packed quantizzati per il decode GEMV,

più KV cache dimensionata a 16k contesto e ~20% + 512 MiB di margine.

Per Qwen2.5-Coder-1.5B Q4_K_M (~1 GB di pesi) la stima è **~6 GiB**: non entra
in una RTX 3060 da 6 GiB con desktop attivo e il modello ricade in CPU a
~1.3 tok/s — inutilizzabile. llama.cpp gestisce lo stesso modello in ~1–2 GiB.

**Stato dei miglioramenti (2026-11-23):**

1. ~~Copie FP16 dense + packed~~ — **FATTO** (edit utente): l'offload salta le
   copie dense quando tutti i layer hanno pesi packed GPU-resident; la stima
   VRAM è ora dominata dai pesi quantizzati (~1 GB per il 1.5B Q4_K_M).
2. ~~KV cache fissa~~ — **FATTO**: cap iniziale 1024 posizioni/layer con
   crescita on-demand (`ensure_kv_capacity_gpu`); l'eager decode ora supera il
   vecchio cap 4096, il CUDA graph resta a 4096.
3. ~~Partial offload~~ — **FATTO (v1)**: `--gpu-layers N` in `chat`/`generate`.
   I primi N layer girano su GPU in FP32 dual-resident, il resto su CPU;
   decode = prefisso GPU → D2H → suffisso CPU per token. **Limiti noti**:
   prefill sempre per-token (niente batched), path GPU FP32 non ottimizzato
   (allocazioni per layer/proiezione) → ~0.3 tok/s sul 1.5B, più lento del
   CPU pieno; servono scratch persistente e kernel attention FP32.
   `serve` cablato: `--gpu-layers N` → `ServerConfig::gpu_layers` →
   `maybe_offload_to_gpu` prova il partial offload prima del path auto.

## Confronto llama-bench diretto — MISURATO (2026-11-23, post-fix batched prefill)

Misura con `llama-bench` (b12642, win-cuda-13.5, build 5bacb7c86) vs
`serius bench --latency-only` sugli **stessi file GGUF**, stessa RTX 3060
Laptop 6 GiB. Serius = build `build-ninja` Release CUDA.

### Decode (token generation)

| Modello | Serius decode | llama.cpp tg128 | Rapporto |
|---------|---------------|-----------------|----------|
| SmolLM2-135M Q4_0, GPU full-offload | **653 tok/s** | 567.8 tok/s | **1.15x** |
| SmolLM2-135M Q8_0, GPU full-offload | 362 tok/s | 393.1 tok/s | 0.92x |
| Qwen2.5-0.5B Q4_K_M, GPU full-offload | **469.7 tok/s** | 368.7 tok/s | **1.27x** |
| SmolLM2-135M Q4_0, CPU-only | 17.8 tok/s | 67.4 tok/s (ngl 0) | **0.26x** |

### Prefill (prompt processing)

| Modello | Serius prefill | llama.cpp pp512 | Rapporto |
|---------|----------------|-----------------|----------|
| SmolLM2-135M Q4_0, prompt 601 tok | **8667 tok/s** (TTFT 69 ms) | 23701 tok/s | 0.37x |
| Qwen2.5-0.5B Q4_K_M, prompt 601 tok | **5031 tok/s** (TTFT 119 ms) | 18897 tok/s | 0.27x |

### Note

- **Fix batched prefill**: `SERIUS_PREFILL_BATCH` era opt-in perché un bug
  produceva output degenere; i test di parità
  (`BatchedVsSequentialPrefill`, `BatchedVsSequentialRealModel` su Qwen2.5
  Q4_K_M reale, max logit diff 3.2e-3, stesso argmax) verificano che il
  path batched è corretto. Ora è **default ON**; `SERIUS_PREFILL_BATCH=0`
  force il vecchio path per-token. Prima del fix: 383 tok/s prefill
  (63x più lento di llama.cpp); dopo: 8667 tok/s (3.7x più lento).
- Decode GPU è **competitivo o superiore** a llama.cpp sui modelli testati;
  il gap residuo è nel prefill (batched GEMM CUDA di llama.cpp è più
  maturo) e soprattutto nella **CPU** (0.26x: i kernel quantized GEMV CPU
  di llama.cpp sono molto più ottimizzati).
- Misure `serius bench` con prompt lungo includono contesto >600 token;
  llama-bench `tg128` usa contesto più corto — confronto indicative,
  non perfetto.
