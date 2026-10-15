# 🧠 FASE 2 — Architettura Next-Gen
**Periodo**: Q3 2026 (Luglio - Settembre) | **Tasks**: 24 | **Completate**: 0

---

## 2.1 Mixture-of-Experts (MoE)

### T2.01 — MoE Router
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/moe/router.h`, `src/moe/router.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] `top_k_gating(input, k)`: softmax scores → top-k expert indices + weights
  - [ ] Load balancing loss: auxiliary loss che penalizza distribuzione non uniforme
  - [ ] Router z-loss per stabilità numerica
  - [ ] Test: distribuzione uniforme con input random

### T2.02 — MoE Expert Layer
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/moe/expert_layer.h`, `src/moe/expert_layer.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] N FFN indipendenti avvolti in un layer unificato
  - [ ] Dispatch: token → expert assignment → batch per expert → compute → merge
  - [ ] Weighted sum degli output con i gate scores
  - [ ] Test: MoE con 4 esperti produce output di dimensione corretta

### T2.03 — MoE: Expert Capacity & Token Dropping
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/moe/expert_layer.cpp`
- **Acceptance Criteria**:
  - [ ] Capacity factor configurabile (default 1.25x)
  - [ ] Token overflow → drop o route al secondo esperto
  - [ ] Logging: conteggio token droppati per batch

### T2.04 — LatentMoE (Compressed Routing)
- **Priorità**: 🟠 Alta | **Stima**: 4d
- **File**: `src/moe/latent_moe.h`, `src/moe/latent_moe.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Down-project: d_model → d_latent prima del routing
  - [ ] Expert compute in spazio latente
  - [ ] Up-project: d_latent → d_model dopo merge
  - [ ] Benchmark: throughput LatentMoE vs MoE standard (≥1.5x con stessi n_experts)

### T2.05 — MoE CUDA Kernel: Batched Expert Dispatch
- **Priorità**: 🟡 Media | **Stima**: 5d
- **File**: `src/moe/moe_cuda.cu` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Kernel che raggruppa token per esperto e fa batched matmul
  - [ ] Evita padding eccessivo (dynamic batching)
  - [ ] Benchmark: ≥5x vs dispatch sequenziale su CPU

### T2.06 — MoE Training Integration
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/moe/expert_layer.cpp`, `src/autograd/`
- **Acceptance Criteria**:
  - [ ] Backward pass per router (straight-through estimator per top-k)
  - [ ] Auxiliary loss integrata nella loss totale
  - [ ] Test: train di un tiny MoE model su toy dataset, loss scende

---

## 2.2 Multi-head Latent Attention (MLA)

### T2.07 — MLA Forward Pass
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/attention/mla.h`, `src/attention/mla.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Down-projection: x → c (d_model → d_latent)
  - [ ] Up-projection on-the-fly: c → K, V per ogni head
  - [ ] Standard Q projection
  - [ ] Attention + output projection
  - [ ] Test: output shape corretta, output numericamente ragionevole

### T2.08 — MLA KV-Cache
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `src/attention/mla.h`
- **Acceptance Criteria**:
  - [ ] Cache solo `c` (d_latent) per posizione, non K e V separati
  - [ ] Benchmark: memoria KV-cache MLA vs GQA → ≥4x riduzione
  - [ ] Integrazione con inference loop per generazione incrementale

### T2.09 — MLA: Integrazione in TransformerBlock
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/transformer/block.h`, `src/transformer/serius_block.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Config switch: `attention_type = GQA | MLA`
  - [ ] Forward pass funziona con entrambi i tipi
  - [ ] Test: same model, GQA vs MLA, output ragionevoli per entrambi

---

## 2.3 Mamba-2 / State Space Models

### T2.10 — Selective Scan (CPU Reference)
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/ssm/selective_scan.h`, `src/ssm/selective_scan.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Implementazione sequenziale: `h[t] = A[t]*h[t-1] + B[t]*x[t]`
  - [ ] Input-dependent A, B, C (selective!)
  - [ ] Discretizzazione: A_continuous → A_discrete via ZOH
  - [ ] Test: output corretto per sequenza nota

### T2.11 — Mamba-2 Block
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/ssm/mamba_block.h`, `src/ssm/mamba_block.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Conv1D causale (kernel=4)
  - [ ] SSM block con selective scan
  - [ ] SiLU gating
  - [ ] Input/output projection
  - [ ] Test: Mamba block produce output di shape corretta

### T2.12 — Selective Scan CUDA Kernel
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/ssm/selective_scan_cuda.cu` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Parallel prefix scan su GPU
  - [ ] Shared memory per stato intermedio
  - [ ] Output identico a CPU reference (max diff < 1e-4)
  - [ ] Benchmark: ≥10x vs CPU per seq_len ≥ 1024

### T2.13 — Gated DeltaNet Block
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/ssm/deltanet.h`, `src/ssm/deltanet.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Gated state update: `S = g*S + (1-g)*(k⊗v)`
  - [ ] Output via query: `y = S·q`
  - [ ] Test: output shape e gradients corretti

### T2.14 — Hybrid Block: Configurable Mamba/Attention Ratio
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/ssm/hybrid_layer.h`, `src/transformer/serius_block.h`
- **Acceptance Criteria**:
  - [ ] Config: `layer_types = [Mamba, Mamba, ..., Attention, ...]`
  - [ ] Serius block che dispatcha a Mamba o Attention based su tipo
  - [ ] Test: modello con mix 6:2 (Mamba:Attn) fa forward pass corretto

---

## 2.4 Multi-Token Prediction (MTP)

### T2.15 — MTP Head
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/transformer/mtp_head.h`, `src/transformer/mtp_head.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] N LM heads indipendenti (default N=2)
  - [ ] Head 0: predice token t+1, Head 1: predice token t+2
  - [ ] Output: lista di logits tensors

### T2.16 — MTP Training Loss
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/train/mtp_loss.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] `L = λ1*CE(h0, target[t+1]) + λ2*CE(h1, target[t+2])`
  - [ ] λ configurabili (default: λ1=1.0, λ2=0.3)
  - [ ] Backward pass funzionale

### T2.17 — MTP Inference: Verify & Accept
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/infer/mtp_decoder.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Sample t1 da head0, t2 da head1
  - [ ] Verify: forward con t1, check se output ≈ t2
  - [ ] Accept: se match, emetti entrambi (2 token per 1.x forward)
  - [ ] Fallback: se no match, emetti solo t1
  - [ ] Benchmark: speedup ≥1.3x vs standard autoregressive

---

## 2.5 Serius Model Unificato

### T2.18 — SeriusConfig
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `src/transformer/serius_config.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Struct unificate per tutti i parametri del modello
  - [ ] Layer type per-layer configurabile
  - [ ] Serializzabile in JSON
  - [ ] Presets: "nano", "mini", "base", "pro", "ultra"

### T2.19 — SeriusBlock
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/transformer/serius_block.h`, `src/transformer/serius_block.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Block che switcha tra Mamba/MLA-Local/MLA-Global basato su config
  - [ ] FFN switchabile: Dense / MoE / LatentMoE
  - [ ] RMSNorm + residual connections
  - [ ] Forward pass funzionale per tutti i tipi

### T2.20 — SeriusModel
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/transformer/serius_model.h`, `src/transformer/serius_model.cpp` (NUOVI)
- **Acceptance Criteria**:
  - [ ] Embedding + N SeriusBlocks + LM Head
  - [ ] Config-driven: cambiando config cambia l'architettura
  - [ ] Test: forward pass con config "nano" (12 layers, dense)
  - [ ] Test: forward pass con config "mini" (24 layers, MoE 8 experts)

### T2.21 — Weight Loading per Modelli HF-compatibili
- **Priorità**: 🟠 Alta | **Stima**: 4d
- **File**: `src/io/hf_weight_mapper.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Mapping nomi tensori HuggingFace → nomi Serius
  - [ ] Supporto: Qwen 3 MoE weights
  - [ ] Supporto: Llama 4 Scout/Maverick MoE weights
  - [ ] Test: caricare Qwen3-0.6B e fare inference

### T2.22 — Serius Inference Engine
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/infer/serius_inference.h` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Forward pass inference-only per SeriusModel (no autograd)
  - [ ] KV-cache management per MLA layers
  - [ ] Mamba state management per SSM layers
  - [ ] Benchmark: tok/s su config "nano"

### T2.23 — Test: Serius Model Completo
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `tests/test_serius_model.cpp` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Test forward pass per ogni configurazione (nano, mini, base)
  - [ ] Test che MoE routing è bilanciato
  - [ ] Test che MLA produce KV-cache ridotto
  - [ ] Test che Mamba layers processano sequenze correttamente

### T2.24 — Benchmark: Serius vs LLaMA Architecture
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `tests/bench_serius_vs_llama.cpp` (NUOVO)
- **Acceptance Criteria**:
  - [ ] Stesso n_params, confronta throughput e memoria
  - [ ] Serius (Mamba+MLA+MoE) vs LLaMA (Attention+Dense)
  - [ ] Report con tabella comparativa
