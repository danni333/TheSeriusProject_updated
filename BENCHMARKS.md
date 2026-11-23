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
