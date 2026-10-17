# Fase 11+ — Cose mancanti (Flash Attention)

Riferimento: `fase11_flash_attention.md`

---

## 1. Flash Attention

| Elemento | Stato | Nota |
|----------|--------|-----|
| Tiling query/key-value e online softmax | ✅ | Kernel CUDA, output validato contro riferimento stabile |
| Parametri `block_m` / `block_n` configurabili | ✅ | Kernel CUDA usa shared memory dinamica; 0 seleziona il profilo predefinito |
| Benchmark CUDA per confrontare dimensioni dei blocchi | ✅ | `bench_flash_attn`, con CUDA events e lunghezze personalizzabili |
| Kernel CUDA flash_attention | ✅ | flash_attn_cuda.cu |
| Evitare materializzazione N×N in HBM | ✅ | Il kernel processa score incrementali, senza allocare la matrice completa |
| Test numerici forward/backward, causalità e sequenze parziali | ✅ | `tests/test_flash_attn.cpp` |

---

## 2. Avanzato (fase11)

| Elemento | Stato |
|----------|--------|
| FlashAttention-3 (TMA, WGMMA, Hopper) | ❌ | Opzionale, hardware-specific |
| RingAttention (sharding Q,K,V su GPU) | ❌ | Per contesto molto lungo distribuito |

---

**Benchmark:** build Release su NVIDIA RTX 3060 Laptop GPU, batch/head=1, causal=false, 2 warm-up e media di 20 iterazioni:

| Head dim | Sequenza | block_m × block_n | Tempo medio |
|----------|----------|-------------------|-------------|
| 32 | 1024 | 16 × 16 | 0,882 ms |
| 64 | 1024 | 128 × 32 | 1,516 ms |
| 128 | 1024 | 64 × 16 | 2,153 ms |

I default sono scelti minimizzando i tempi misurati su N=1024; le differenze più ridotte possono variare con GPU, build e workload. Per ripetere: compilare con `SERIUS_USE_CUDA=ON` ed eseguire `bench_flash_attn`; passare lunghezze di sequenza come argomenti per personalizzare i casi.

**Rimanente opzionale:** FlashAttention-3 (TMA/WGMMA, richiede GPU Hopper) e RingAttention (richiede un runtime multi-GPU/NCCL integrato).
