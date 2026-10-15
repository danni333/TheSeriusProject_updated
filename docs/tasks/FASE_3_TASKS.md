# 🏋️ FASE 3 — Training Pipeline di Scala
**Periodo**: Q4 2026 (Ottobre - Dicembre) | **Tasks**: 22 | **Completate**: 0

---

## 3.1 Distributed Training

### T3.01 — NCCL Wrapper
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/distributed/nccl_wrapper.h` (NUOVO)
- **Criteria**: Wrapper C++ per NCCL: init, all-reduce, broadcast, barrier. Test su 2+ GPU.

### T3.02 — Data Parallelism
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/distributed/data_parallel.h` (NUOVO)
- **Criteria**: Gradient sync via all-reduce dopo backward. Scaling loss per world_size. Test: 2 GPU training loss ≈ 1 GPU.

### T3.03 — Tensor Parallelism
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/distributed/tensor_parallel.h` (NUOVO)
- **Criteria**: Split matmul su N GPU (colonne di W). Riduce memoria per-GPU. Test: modello 10B su 2 GPU.

### T3.04 — Pipeline Parallelism
- **Priorità**: 🟡 Media | **Stima**: 5d
- **File**: `src/distributed/pipeline_parallel.h` (NUOVO)
- **Criteria**: Layer N..M su GPU 0, M+1..L su GPU 1. Micro-batching per riempire pipeline.

### T3.05 — ZeRO Stage 1 (Optimizer Sharding)
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/distributed/zero_optimizer.h` (NUOVO)
- **Criteria**: Partizione stati optimizer tra GPU. Memory saving ≥ Nx per N GPU.

### T3.06 — ZeRO Stage 2-3
- **Priorità**: 🟡 Media | **Stima**: 5d
- **File**: `src/distributed/zero_optimizer.h` (estendere)
- **Criteria**: Stage 2: gradient sharding. Stage 3: parameter sharding con gather on-demand.

---

## 3.2 Mixed Precision Training

### T3.07 — BF16 Training Support
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/train/mixed_precision.h` (NUOVO)
- **Criteria**: Forward in BF16, backward in BF16, master weights in FP32. Dynamic loss scaling.

### T3.08 — FP8 Forward Pass
- **Priorità**: 🟡 Media | **Stima**: 4d
- **File**: `src/train/mixed_precision.h` (estendere)
- **Criteria**: Forward in FP8 (E4M3), backward in BF16. Per-tensor scaling factors.

### T3.09 — Gradient Checkpointing
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/train/gradient_checkpoint.h` (NUOVO)
- **Criteria**: Rilascia activations intermedie, ricalcola durante backward. Memory saving ≥ 2x.

---

## 3.3 Data Pipeline

### T3.10 — Streaming Dataset Loader
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/data/streaming_loader.h` (NUOVO)
- **Criteria**: Memory-mapped file reading. Pre-tokenizzato. Shuffling a blocchi. Throughput ≥ 10K samples/s.

### T3.11 — Multi-Worker Data Loading
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/data/streaming_loader.h` (estendere)
- **Criteria**: N worker thread che pre-caricano batch. Buffer di batch pronti. Zero stall su GPU.

### T3.12 — Data Mixing
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/data/data_mixer.h` (NUOVO)
- **Criteria**: Config: `{code: 0.3, text: 0.5, math: 0.2}`. Sampling proporzionale da N dataset.

### T3.13 — Tokenizer Training
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/tokenizer/bpe_trainer.h` (NUOVO)
- **Criteria**: BPE training da corpus. Output formato vocab.json. SentencePiece-compatible.

### T3.14 — Synthetic Data Generator
- **Priorità**: 🟢 Bassa | **Stima**: 5d
- **File**: `src/data/synthetic_gen.h` (NUOVO)
- **Criteria**: Template-based Q&A, math problems, code snippets. Filtering per qualità.

---

## 3.4 Knowledge Distillation

### T3.15 — Teacher Model Server
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/distill/teacher_server.h` (NUOVO)
- **Criteria**: Serve logits e hidden states da modello teacher. Batched inference.

### T3.16 — KL Divergence Loss
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/distill/kd_loss.h` (NUOVO)
- **Criteria**: `L_kd = KL(student_logits / T || teacher_logits / T)`. T configurabile.

### T3.17 — Hidden State Alignment
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/distill/feature_align.h` (NUOVO)
- **Criteria**: `L_feat = MSE(proj(student_h[layer_i]), teacher_h[layer_j])`. Layer mapping configurabile.

### T3.18 — Progressive Distillation Pipeline
- **Priorità**: 🟢 Bassa | **Stima**: 5d
- **File**: `src/distill/progressive.h` (NUOVO)
- **Criteria**: Pipeline: 200B → 100B → 50B → 10B. Automatizzato con checkpoint intermedi.

---

## 3.5 Training Loop

### T3.19 — Unified Trainer
- **Priorità**: 🔴 Critica | **Stima**: 5d
- **File**: `src/train/trainer.h`, `src/train/trainer.cpp` (NUOVI)
- **Criteria**: Training loop completo: data loading → forward → loss → backward → optimizer step → logging.

### T3.20 — Optimizer (AdamW)
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `src/optim/adamw.h` (NUOVO o estendere esistente)
- **Criteria**: AdamW con weight decay, lr scheduler (cosine, linear warmup).

### T3.21 — Training Logging (WandB-like)
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/train/logger.h` (NUOVO)
- **Criteria**: Log loss, lr, throughput a CSV e stdout. Sampling di testo periodico.

### T3.22 — Training Test End-to-End
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `tests/test_training.cpp` (NUOVO)
- **Criteria**: Train tiny model (1M params) su toy dataset. Verify loss decreases over 100 steps.
