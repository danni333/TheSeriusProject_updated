# 🗺️ Roadmap Strategica — 6 Fasi (2026-2028)

> Roadmap operativa dettagliata con tasks, metriche di successo, dipendenze,
> e deliverable concreti per ogni fase.

---

## Timeline Overview

```
2026                          2027                          2028
Q2     Q3     Q4     Q1     Q2     Q3     Q4     Q1     Q2
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│FASE 1│FASE 2│FASE 3│FASE 4│ FASE 5      │  FASE 6       │
│Fonda-│Archi-│Train-│Reason│ Multimodale │  Superamento  │
│menta │tettura│ing  │ing+RL│ Long-Context│               │
└──────┴──────┴──────┴──────┴─────────────┴───────────────┘
```

---

## ⚡ FASE 1: Fondamenta d'Acciaio

**Periodo**: Q2 2026 (Aprile - Giugno)
**Durata**: 3 mesi
**Obiettivo**: Rendere Serius competitivo con llama.cpp per inference su singola macchina

### 1.1 CUDA Backend Production-Grade

**Stato attuale**: Backend CUDA presente ma basico (solo memory management + matmul kernel + fused kernels). Mancano stream management, multi-GPU, e kernels completi.

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| CUDA stream management (async) | Alta | 1 settimana | - |
| Completare kernel: softmax standalone | Alta | 3 giorni | - |
| Kernel: RMSNorm CUDA | Alta | 3 giorni | - |
| Kernel: SwiGLU CUDA | Media | 2 giorni | - |
| Kernel: RoPE CUDA | Media | 2 giorni | - |
| Multi-GPU device selection | Media | 1 settimana | streams |
| GPU memory pool (sub-allocation) | Alta | 1 settimana | - |
| CUDA graph capture per inference | Bassa | 1 settimana | tutti i kernels |

**Deliverable**: `serius chat --device cuda` funzionale con TinyLlama

### 1.2 Flash Attention v2+

**Stato attuale**: Flash Attention CPU + CUDA presenti con causal mask. Mancano varianti.

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Sliding window attention (Mistral-style) | Alta | 1 settimana | - |
| Local + Global interleaved (Gemma-style) | Alta | 1 settimana | sliding window |
| GQA-optimized flash attention | Media | 3 giorni | - |
| Benchmark vs standard attention | Alta | 2 giorni | - |

### 1.3 CPU Optimization

**Stato attuale**: AVX2 e AVX-512 matmul presenti. Manca thread pool.

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Thread pool con work-stealing | Alta | 1 settimana | - |
| Parallelizzare matmul su multi-thread | Alta | 3 giorni | thread pool |
| AMX kernels (per CPU Intel recenti) | Bassa | 1 settimana | - |
| GGUF format importer | Media | 1 settimana | - |
| Benchmark framework | Alta | 3 giorni | - |

### 1.4 Quantizzazione Avanzata

**Stato attuale**: INT4, FP8, GPTQ, AWQ presenti ma non end-to-end integrati.

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Integrare INT4 nei kernel di inference | Alta | 1 settimana | CUDA kernels |
| BitNet 1.58-bit matmul | Alta | 2 settimane | - |
| FP4 emulation | Media | 1 settimana | FP8 |
| Auto-calibration pipeline | Media | 1 settimana | quantization funzionale |

### Metriche Fase 1
| Metrica | Target | Baseline attuale |
|---|---|---|
| TinyLlama tok/s (CPU AVX2) | ≥30 | ~1.4 |
| TinyLlama tok/s (CUDA RTX 3060) | ≥200 | N/A |
| Memoria GPU peak (1.1B model) | ≤3 GB | N/A |
| Build time | ≤2 min | ~1 min |

---

## 🧠 FASE 2: Architettura Next-Gen

**Periodo**: Q3 2026 (Luglio - Settembre)
**Durata**: 3 mesi
**Obiettivo**: Implementare MoE, MLA, Mamba-2, MTP

### 2.1 Mixture-of-Experts

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| `MoERouter`: top-k gating con softmax | Alta | 1 settimana | - |
| `MoELayer`: N experts FFN + routing | Alta | 1 settimana | Router |
| Load balancing auxiliary loss | Alta | 3 giorni | MoELayer |
| Expert capacity limiting + token dropping | Media | 3 giorni | MoELayer |
| `LatentMoE`: compressed routing | Alta | 1 settimana | MoELayer |
| Benchmark: MoE vs dense FFN (throughput + quality) | Alta | 3 giorni | tutto sopra |
| CUDA kernel: MoE dispatch (batched expert compute) | Media | 2 settimane | MoELayer + CUDA |

### 2.2 Multi-head Latent Attention

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| `MLAttention`: down/up projection + attention | Alta | 1 settimana | - |
| `MLACache`: latent KV-cache | Alta | 3 giorni | MLAttention |
| Integrare MLA come opzione in TransformerBlock | Media | 3 giorni | MLAttention |
| Benchmark: KV-cache size MLA vs GQA | Alta | 2 giorni | MLACache |

### 2.3 Mamba-2 / SSM

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Selective scan (CPU, reference) | Alta | 1 settimana | - |
| `Mamba2Block`: conv1d + SSM + gated output | Alta | 1 settimana | scan |
| Selective scan CUDA kernel | Alta | 2 settimane | CPU reference |
| `GatedDeltaNet` block (alternativa) | Media | 1 settimana | - |
| `HybridBlock`: configurable Mamba/Attention ratio | Alta | 1 settimana | Mamba2 + MLA |

### 2.4 Multi-Token Prediction

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| `MTPHead`: N output heads | Media | 3 giorni | - |
| MTP training loss | Media | 3 giorni | MTPHead |
| MTP inference: verify + accept | Media | 1 settimana | MTPHead |

### 2.5 Serius Model Config

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| `SeriusConfig`: unified model configuration | Alta | 3 giorni | - |
| `SeriusBlock`: hybrid block che unisce tutto | Alta | 1 settimana | 2.1-2.4 |
| `SeriusModel`: il modello completo | Alta | 1 settimana | SeriusBlock |
| Weight loading da formati HuggingFace | Media | 1 settimana | SeriusModel |

### Metriche Fase 2
| Metrica | Target |
|---|---|
| MoE routing efficiency | >90% balanced |
| MLA KV-cache riduzione | ≥8x vs GQA |
| Mamba-2 throughput gain | ≥2x vs attention on seq>4K |
| MTP speedup | ≥1.5x vs standard autoregressive |
| Build: tutti i test passano | ✅ |

---

## 🏋️ FASE 3: Training Pipeline di Scala

**Periodo**: Q4 2026 (Ottobre - Dicembre)
**Durata**: 3 mesi
**Obiettivo**: Poter trainare modelli da zero fino a 10B+ parametri

### 3.1 Distributed Training

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| NCCL wrapper per all-reduce | Alta | 1 settimana | CUDA |
| Data parallelism (gradient sync) | Alta | 1 settimana | NCCL |
| Tensor parallelism (split matmul) | Alta | 2 settimane | NCCL |
| Pipeline parallelism (layer split) | Media | 2 settimane | NCCL |
| ZeRO Stage 1 (optimizer state sharding) | Alta | 1 settimana | Data parallel |
| ZeRO Stage 2 (gradient sharding) | Media | 1 settimana | Stage 1 |
| ZeRO Stage 3 (parameter sharding) | Bassa | 2 settimane | Stage 2 |

### 3.2 Mixed Precision Training

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| BF16 matmul per training | Alta | 1 settimana | CUDA |
| Dynamic loss scaling | Alta | 3 giorni | BF16 |
| FP8 forward pass (activations) | Media | 2 settimane | BF16 |
| Gradient checkpointing | Alta | 1 settimana | - |

### 3.3 Data Pipeline

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Streaming dataset loader (memory-mapped) | Alta | 1 settimana | - |
| Multi-worker data loading | Alta | 3 giorni | Loader |
| Data mixing (weighted blending) | Media | 3 giorni | Loader |
| SentencePiece tokenizer training | Media | 1 settimana | - |
| Synthetic data pipeline (template-based) | Bassa | 2 settimane | Tokenizer |

### 3.4 Knowledge Distillation

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Teacher model serving (inference-only) | Alta | 1 settimana | Inference |
| KL divergence loss (logit matching) | Alta | 3 giorni | Teacher |
| Hidden state alignment loss | Media | 1 settimana | Teacher |
| Progressive distillation pipeline | Bassa | 2 settimane | KL + hidden |

### Metriche Fase 3
| Metrica | Target |
|---|---|
| Training throughput vs PyTorch | ≥50% |
| Multi-GPU scaling efficiency | ≥85% (per GPU) |
| Max trainable model size | ≥10B params |
| Data loading throughput | ≥10K samples/s |

---

## 🎯 FASE 4: Reasoning & Reinforcement Learning

**Periodo**: Q1 2027 (Gennaio - Marzo)
**Durata**: 3 mesi
**Obiettivo**: Creare modelli che "pensano" e si migliorano con RL

### 4.1 GRPO Pipeline

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Batch generation (N risposte per prompt) | Alta | 1 settimana | Inference |
| Rule-based reward functions: | | | |
| — Math verificatore (eval expression) | Alta | 1 settimana | - |
| — Code executor (compile + run + compare) | Alta | 2 settimane | Sandbox |
| — Logic consistency checker | Media | 1 settimana | - |
| Group relative scoring | Alta | 3 giorni | Rewards |
| Policy gradient con KL constraint | Alta | 1 settimana | Scoring |
| KL controller (adaptive β) | Media | 3 giorni | Policy gradient |

### 4.2 Thinking Mode

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| `<thinking>` / `</thinking>` special tokens | Alta | 2 giorni | Tokenizer |
| Thinking router (trainable) | Alta | 1 settimana | - |
| Compute budget control | Media | 1 settimana | Router |
| Training con thinking-augmented data | Alta | 2 settimane | Thinking tokens |

### 4.3 Agent Framework

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Tool calling protocol (JSON function calls) | Alta | 1 settimana | - |
| Code execution sandbox (isolato) | Alta | 2 settimane | - |
| Plan-act-verify loop | Alta | 1 settimana | Tool calling |
| Self-reflection (verify own output) | Media | 1 settimana | Tool calling |
| Multi-agent dispatcher | Bassa | 2 settimane | Tool calling |

### 4.4 Evaluation

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| MMLU eval harness | Alta | 1 settimana | Inference |
| HumanEval eval harness | Alta | 1 settimana | Code exec |
| GSM8K / MATH eval | Alta | 3 giorni | Math eval |
| Custom multi-step reasoning benchmark | Media | 1 settimana | - |

### Metriche Fase 4
| Metrica | Target |
|---|---|
| MMLU (5-shot) | ≥65% (con model 10B) |
| HumanEval | ≥50% (con model 10B) |
| GSM8K | ≥70% (con model 10B) |
| Tool calling success rate | ≥90% |

---

## 🌐 FASE 5: Multimodale & Long-Context

**Periodo**: Q2-Q3 2027 (Aprile - Luglio)
**Durata**: 4 mesi
**Obiettivo**: Modelli che vedono, ascoltano, e ragionano su 1M+ token

### 5.1 Vision

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| ViT implementation (ViT-L/14) | Alta | 2 settimane | CUDA |
| Image patching + embedding | Alta | 1 settimana | ViT |
| Cross-modal attention | Alta | 1 settimana | MLA |
| Video frame sampling | Media | 1 settimana | Image |
| Training con image-text pairs | Alta | 2 settimane | tutto sopra |

### 5.2 Long-Context

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| YaRN RoPE scaling | Alta | 1 settimana | RoPE |
| NTK-aware interpolation | Media | 3 giorni | YaRN |
| Ring attention (multi-GPU context) | Media | 2 settimane | Distributed |
| Context length evaluation (Needle in Haystack) | Alta | 3 giorni | - |

### 5.3 Speculative Decoding

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Draft model integration | Alta | 1 settimana | Inference |
| Parallel verification | Alta | 1 settimana | Draft |
| Tree-based speculative | Media | 2 settimane | Parallel verify |
| Self-speculative (early-exit) | Bassa | 2 settimane | - |

### 5.4 Audio (Opzionale)

| Task | Priorità | Stima | Dipendenze |
|---|---|---|---|
| Audio mel-spectrogram encoding | Bassa | 1 settimana | - |
| Whisper-style encoder | Bassa | 2 settimane | Encoding |
| Speech-to-text integration | Bassa | 1 settimana | Encoder |

### Metriche Fase 5
| Metrica | Target |
|---|---|
| Max context length (functional) | ≥1M tokens |
| Needle-in-Haystack accuracy @ 256K | ≥95% |
| Speculative decoding speedup | ≥2x |
| VQA accuracy (basic) | ≥60% |

---

## 🏆 FASE 6: Superamento

**Periodo**: Q4 2027 — Q2 2028 (Agosto 2027 - Giugno 2028)
**Durata**: 6+ mesi
**Obiettivo**: Innovazioni che vanno OLTRE lo stato dell'arte

### 6.1 Architetture Inedite

| Task | Priorità | Stima |
|---|---|---|
| Adaptive Compute per token | Alta | 2 mesi |
| Neural Architecture Search (NAS) per Serius blocks | Media | 2 mesi |
| Continuous learning (aggiornamento senza retrain) | Alta | 2 mesi |
| Memory-augmented reasoning (external memory) | Alta | 1 mese |

### 6.2 Training Rivoluzionario

| Task | Priorità | Stima |
|---|---|---|
| Sub-bit quantization (sotto 1-bit) | Media | 1 mese |
| Sparse training dall'inizio | Alta | 2 mesi |
| Curriculum learning dinamico | Media | 1 mese |
| Self-play RL | Alta | 2 mesi |

### 6.3 Edge AI

| Task | Priorità | Stima |
|---|---|---|
| Serius-Nano (<1B, smartphone) | Alta | 2 mesi |
| NPU runtime optimization | Media | 1 mese |
| On-device LoRA fine-tuning | Media | 1 mese |
| Federated learning | Bassa | 2 mesi |

### 6.4 Self-Evolving

| Task | Priorità | Stima |
|---|---|---|
| Self-improvement data generation | Alta | 2 mesi |
| Automated red-teaming | Media | 1 mese |
| Meta-learning framework | Alta | 2 mesi |

### Metriche Fase 6
| Metrica | Target |
|---|---|
| Serius-Nano su smartphone | ≥5 tok/s on Snapdragon 8 Gen 4 |
| MMLU (model 30B+) | ≥80% |
| HumanEval | ≥75% |
| Self-improvement iteration | ≥3 rounds con degradazione <2% |

---

## Dipendenze tra Fasi

```
FASE 1 ──→ FASE 2 ──→ FASE 3 ──→ FASE 4
  │          │          │          │
  │          │          │          ├──→ FASE 5
  │          │          │          │
  │          │          └──────────┴──→ FASE 6
  │          │
  │          └── FASE 2 richiede CUDA funzionale (da FASE 1)
  │
  └── FASE 1 è prerequisito per tutto
```

**Regola**: Non procedere alla FASE N+1 finché la FASE N non ha superato tutte le metriche minime.
**Eccezione**: Tasks indipendenti (es. BitNet, GGUF importer) possono essere sviluppati in parallelo.
