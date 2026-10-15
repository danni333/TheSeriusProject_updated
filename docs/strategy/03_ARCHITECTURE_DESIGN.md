# 📐 Architettura Target — Serius Ultra

> Questo documento definisce l'architettura target finale di Serius e il percorso evolutivo
> dalla codebase attuale all'obiettivo. Include gap analysis, diagrammi, e specifiche tecniche.

---

## 1. Stato Attuale della Codebase (Marzo 2026)

### Moduli Esistenti

```
src/
├── hal/               ← Hardware Abstraction Layer
│   ├── device.h           IBackend interface (DeviceType, DType)
│   ├── cpu_backend.*      CPU: malloc, memcpy (funzionale)
│   ├── cuda_backend.*     CUDA: malloc, memcpy (implementato, richiede SERIUS_USE_CUDA)
│   ├── metal_backend.*    Metal: stub
│   ├── vulkan_backend.*   Vulkan: stub
│   └── fused_kernels.*    CUDA fused ops: LayerNorm+Residual, MaskedSoftmax, BiasGELU
│
├── tensor/            ← Core Tensor Library
│   └── tensor.h           N-dimensional tensor con shape, stride, data
│
├── autograd/          ← Automatic Differentiation
│   ├── node.h             Computation graph node
│   └── ops.h              Differentiable operations
│
├── blas/              ← Linear Algebra
│   ├── blas.h             Interface: matmul, matvec, elementwise ops
│   ├── dispatch.cpp       Runtime dispatch: scalar → AVX2 → AVX512 → CUDA
│   ├── matmul_scalar.cpp  Naive O(n³) reference
│   ├── matmul_avx2.cpp    AVX2 SIMD (8-wide float)
│   ├── matmul_avx512.cpp  AVX-512 SIMD (16-wide float)
│   ├── matmul_cuda.*      CUDA GPU matmul kernel
│   └── elementwise.cpp    Add, sub, mul, div, scale, apply
│
├── attention/         ← Flash Attention
│   ├── flash_attn.h       FlashAttentionConfig (causal, block sizes)
│   ├── flash_attn.cpp     CPU tiled Flash Attention + autograd
│   └── flash_attn_cuda.cu CUDA Flash Attention kernel
│
├── transformer/       ← Model Architecture
│   ├── model.*            Base TransformerModel (embedding, blocks, LM head)
│   ├── block.*            Standard TransformerBlock + optimized variant
│   ├── attention.*        Multi-Head Attention (Q,K,V projections, GQA)
│   ├── ffn.*              Feed-Forward Network (GELU activation, fused variant)
│   ├── layernorm.*        RMSNorm / LayerNorm
│   ├── positional.*       RoPE (Rotary Positional Embedding)
│   ├── kv_cache.*         Flat KV-Cache
│   ├── paged_kv_cache.*   Paged KV-Cache (block allocation)
│   ├── llama_model.*      LLaMA architecture (RMSNorm, GQA, SwiGLU)
│   ├── qwen_model.*       Qwen architecture (biased QKV/O projections)
│   └── config.h           TransformerConfig struct
│
├── infer/             ← Inference Engine
│   ├── llama_inference.*  Inference-only forward pass (no autograd)
│   └── sampler.*          Top-k, top-p, temperature sampling
│
├── quant/             ← Quantization
│   ├── int4.*             INT4 quantization/dequantization
│   ├── fp8.*              FP8 (E4M3/E5M2) emulation
│   ├── gptq.*             GPTQ post-training quantization
│   ├── awq.*              Activation-Aware Weight Quantization
│   ├── conversions.*      Format conversions
│   ├── dequant_matmul.*   Quantized matmul (CPU + CUDA)
│   └── quant_kernels.*    CUDA quantization kernels
│
├── memory/            ← Memory Management
│   ├── memory_pool.*      Pool allocator for tensors
│   └── pool_allocator.h   Templated pool allocator
│
├── train/             ← Training
│   └── checkpoint.*       Checkpoint save/load
│
├── tokenizer/         ← Tokenization
│   └── (BPE tokenizer)
│
├── io/                ← I/O (Safetensors, model loading)
├── distributed/       ← Distributed (placeholder)
├── agent/             ← Agent framework (placeholder)
├── rl/                ← Reinforcement Learning (vuoto!)
├── safety/            ← Safety module (placeholder)
├── server/            ← Server/API (placeholder)
├── hub/               ← Model hub (placeholder)
├── debug/             ← Debug utilities
├── common/            ← Common utilities
├── core/              ← Core utilities
├── model/             ← Model abstractions
├── data/              ← Data loading
└── app/               ← CLI application (main.cpp)
```

---

## 2. Gap Analysis: Attuale vs Target

### Legenda
- 🟢 **Implementato e funzionale**
- 🟡 **Implementato ma incompleto/non testato**
- 🔴 **Mancante**

| Componente | Stato | Dettaglio |
|---|---|---|
| **Tensor core** | 🟢 | N-dim tensor con shape/stride |
| **Autograd** | 🟢 | Computation graph, backward pass |
| **CPU BLAS (scalar)** | 🟢 | Funzionale ma lento |
| **CPU BLAS (AVX2)** | 🟢 | SIMD 8-wide, buone performance |
| **CPU BLAS (AVX-512)** | 🟡 | Implementato, richiede CPU compatibile |
| **CUDA backend** | 🟡 | Memory management OK, matmul e fused kernels presenti |
| **Flash Attention** | 🟡 | CPU + CUDA, manca sliding window |
| **Fused CUDA kernels** | 🟡 | LayerNorm+Residual, Softmax, BiasGELU |
| **LLaMA model** | 🟢 | Completo: RMSNorm, GQA, SwiGLU, RoPE |
| **Qwen model** | 🟢 | Completo: come LLaMA + biased projections |
| **KV-Cache** | 🟢 | Flat + Paged, con scratch buffers |
| **Quantization INT4** | 🟡 | Implementato, non integrato nell'inference |
| **Quantization FP8** | 🟡 | Emulazione, non usato nel training |
| **GPTQ/AWQ** | 🟡 | Presente, non end-to-end testato |
| **Sampler** | 🟢 | Top-k, top-p, temperature |
| **Tokenizer BPE** | 🟢 | Carica vocab.json, merges |
| **Safetensors loader** | 🟢 | Carica modelli HuggingFace |
| **Checkpoint** | 🟢 | Save/load training state |
| **MoE** | 🔴 | Non presente |
| **LatentMoE** | 🔴 | Non presente |
| **MLA** | 🔴 | Non presente |
| **Mamba-2 / SSM** | 🔴 | Non presente |
| **Gated DeltaNet** | 🔴 | Non presente |
| **MTP** | 🔴 | Non presente |
| **Speculative Decoding** | 🔴 | Non presente |
| **BitNet** | 🔴 | Non presente |
| **Distributed training** | 🔴 | Directory vuota |
| **GRPO / RL** | 🔴 | Directory `src/rl/` vuota |
| **Knowledge Distillation** | 🔴 | Non presente |
| **Vision Encoder** | 🔴 | Non presente |
| **Multimodal backbone** | 🔴 | Non presente |
| **Agent framework** | 🔴 | Directory placeholder |
| **Long-context (YaRN)** | 🔴 | RoPE presente ma no scaling |
| **QAT** | 🔴 | Non presente |

---

## 3. Architettura Target: "Serius Ultra"

### 3.1 Overview

```
┌═══════════════════════════════════════════════════════════════┐
║                     SERIUS ULTRA v1.0                         ║
║               ~200B total / ~15B active params                ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐      ║
║  │ Vision      │  │ Audio       │  │ Text Tokenizer  │      ║
║  │ Encoder     │  │ Encoder     │  │ (BPE, 128K vocab)│      ║
║  │ (ViT-L/14) │  │ (Whisper-S) │  │                  │      ║
║  │ 300M params │  │ 200M params │  │                  │      ║
║  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘      ║
║         │                │                   │                ║
║         └───────────┬────┘                   │                ║
║                     ▼                        ▼                ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │           Unified Embedding Space                    │    ║
║  │           d_model = 4096                             │    ║
║  │           Modality embeddings + position encoding     │    ║
║  └───────────────────────┬──────────────────────────────┘    ║
║                          ▼                                    ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │                                                      │    ║
║  │   HYBRID BLOCK STACK (48 layers)                     │    ║
║  │                                                      │    ║
║  │   Layer Type Pattern (repeating every 8 layers):     │    ║
║  │   ┌──────────────────────────────────────────┐       │    ║
║  │   │ L1: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L2: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L3: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L4: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L5: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L6: Mamba-2 + LatentMoE FFN              │       │    ║
║  │   │ L7: MLA Attention (local) + LatentMoE    │       │    ║
║  │   │ L8: MLA Attention (global) + LatentMoE   │       │    ║
║  │   └──────────────────────────────────────────┘       │    ║
║  │                                                      │    ║
║  │   Ratio: 75% Mamba-2 / 12.5% Local Attn / 12.5%     │    ║
║  │          Global Attn                                  │    ║
║  │                                                      │    ║
║  │   Each Mamba-2 block:                                │    ║
║  │   ├─ RMSNorm                                         │    ║
║  │   ├─ Mamba-2 SSM (d_inner=8192, d_state=128)        │    ║
║  │   ├─ RMSNorm                                         │    ║
║  │   └─ LatentMoE FFN (128 experts, top-8, d_latent=1024)│   ║
║  │                                                      │    ║
║  │   Each Attention block:                              │    ║
║  │   ├─ RMSNorm                                         │    ║
║  │   ├─ MLA (d_latent=512, n_heads=32, d_head=128)     │    ║
║  │   ├─ RMSNorm                                         │    ║
║  │   └─ LatentMoE FFN (128 experts, top-8, d_latent=1024)│   ║
║  │                                                      │    ║
║  └───────────────────────┬──────────────────────────────┘    ║
║                          ▼                                    ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │                                                      │    ║
║  │   OUTPUT HEADS                                       │    ║
║  │                                                      │    ║
║  │   ┌────────────────┐  ┌────────────────┐            │    ║
║  │   │ LM Head        │  │ MTP Head       │            │    ║
║  │   │ (next token)   │  │ (token +2, +3) │            │    ║
║  │   └───────┬────────┘  └───────┬────────┘            │    ║
║  │           │                   │                      │    ║
║  │           ▼                   ▼                      │    ║
║  │   ┌────────────────────────────────────┐            │    ║
║  │   │ Thinking Router                    │            │    ║
║  │   │ σ(h · W_think) > threshold        │            │    ║
║  │   │   → YES: emit <thinking> token     │            │    ║
║  │   │   → NO:  direct response           │            │    ║
║  │   └────────────────────────────────────┘            │    ║
║  │                                                      │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  AGENT LAYER                                                 ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ • Tool calling (function_call tokens)                  │  ║
║  │ • Code execution sandbox                               │  ║
║  │ • Web browsing via headless browser                    │  ║
║  │ • File system access                                   │  ║
║  │ • Self-reflection loop (verify own outputs)            │  ║
║  │ • Multi-agent dispatch (specialize sub-agents)         │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║ RUNTIME                                                      ║
║ • CUDA + AVX-512 + NPU backends                             ║
║ • BitNet / NVFP4 / INT4 quantization                        ║
║ • 1M+ token context (Mamba + sparse attention)               ║
║ • Speculative decoding (self-speculative + external draft)   ║
║ • KV-cache: MLA latent (2 KB/token vs 32 KB standard)       ║
╠═══════════════════════════════════════════════════════════════╣
║ TRAINING PIPELINE                                            ║
║ • FP8 mixed-precision pre-training                           ║
║ • GRPO reinforcement learning (no reward model)              ║
║ • Knowledge distillation (teacher → student)                 ║
║ • Synthetic data generation + filtering                      ║
║ • Distributed: Data + Tensor + Pipeline parallelism          ║
╚═══════════════════════════════════════════════════════════════╝
```

### 3.2 Parametri del Modello

| Componente | Parametri |
|---|---|
| d_model | 4096 |
| n_layers | 48 (36 Mamba + 6 Local Attn + 6 Global Attn) |
| n_heads | 32 (per attention layers) |
| d_head | 128 |
| MLA d_latent | 512 (riduzione KV-cache 16x) |
| MoE n_experts | 128 |
| MoE top_k | 8 (attivazione ~6.25%) |
| MoE d_latent | 1024 (LatentMoE compression) |
| MoE d_expert_ff | 2048 (per expert FFN) |
| Mamba d_inner | 8192 |
| Mamba d_state | 128 |
| Vocab size | 128,000 |
| Max context | 1,048,576 (1M tokens) |
| **Total params** | **~200B** |
| **Active params/token** | **~15B** |

### 3.3 Modelli della Famiglia Serius

| Modello | Total | Active | Layers | Target |
|---|---|---|---|---|
| **Serius-Nano** | 1B | 1B (dense) | 12 | Smartphone, IoT, edge |
| **Serius-Mini** | 8B | 2B (MoE) | 24 | Laptop, desktop |
| **Serius-Base** | 30B | 5B (MoE) | 32 | Single GPU |
| **Serius-Pro** | 80B | 10B (MoE) | 40 | Multi-GPU workstation |
| **Serius-Ultra** | 200B | 15B (MoE) | 48 | GPU cluster |

---

## 4. Struttura Directory Target

```
src/
├── hal/                    ← Hardware Abstraction (INVARIATO + esteso)
│   ├── device.h
│   ├── cpu_backend.*
│   ├── cuda_backend.*      ← Estendere con stream management, multi-GPU
│   ├── metal_backend.*     ← Implementare per Apple Silicon
│   ├── npu_backend.*       ← NUOVO: Qualcomm/Apple NPU
│   └── fused_kernels.*     ← Aggiungere: Mamba scan, MoE dispatch
│
├── tensor/                 ← Core Tensor (INVARIATO)
├── autograd/               ← Autograd (INVARIATO)
│
├── blas/                   ← BLAS (INVARIATO + esteso)
│   ├── ...esistente...
│   ├── matmul_bitnet.cpp   ← NUOVO: ternary matmul ({-1,0,+1})
│   └── matmul_fp4.cpp      ← NUOVO: FP4 matmul emulation
│
├── attention/              ← Attention Mechanisms (ESTESO)
│   ├── flash_attn.*        ← Esistente
│   ├── mla.*               ← NUOVO: Multi-head Latent Attention
│   ├── sliding_window.*    ← NUOVO: Sliding window attention
│   ├── sparse_attention.*  ← NUOVO: Configurable sparse patterns
│   └── ring_attention.*    ← NUOVO: Multi-GPU distributed attention
│
├── ssm/                    ← NUOVO: State Space Models
│   ├── mamba_block.*       ← Mamba-2 block
│   ├── selective_scan.*    ← Parallel scan algorithm
│   ├── deltanet.*          ← Gated DeltaNet
│   └── hybrid_layer.*      ← Configurable Mamba/Attn interleaving
│
├── moe/                    ← NUOVO: Mixture of Experts
│   ├── router.*            ← Top-k gating + load balancing
│   ├── expert_layer.*      ← Expert FFN wrapper
│   ├── latent_moe.*        ← LatentMoE (compressed routing)
│   └── expert_parallel.*   ← Multi-GPU expert distribution
│
├── transformer/            ← Model Architecture (ESTESO)
│   ├── ...esistente...
│   ├── serius_model.*      ← NUOVO: Serius Ultra architecture
│   ├── serius_block.*      ← NUOVO: Configurable hybrid block
│   └── mtp_head.*          ← NUOVO: Multi-Token Prediction head
│
├── infer/                  ← Inference (ESTESO)
│   ├── ...esistente...
│   ├── speculative.*       ← NUOVO: Speculative decoding engine
│   ├── draft_model.*       ← NUOVO: Draft model management
│   └── thinking_router.*   ← NUOVO: Think vs respond routing
│
├── quant/                  ← Quantization (ESTESO)
│   ├── ...esistente...
│   ├── bitnet.*            ← NUOVO: 1.58-bit ternary weights
│   ├── fp4.*               ← NUOVO: FP4 (NVFP4 emulation)
│   └── qat.*               ← NUOVO: Quantization-Aware Training
│
├── rl/                     ← NUOVO: Reinforcement Learning
│   ├── grpo.*              ← Group Relative Policy Optimization
│   ├── reward.*            ← Rule-based reward functions
│   ├── rl_trainer.*        ← RL training loop
│   └── kl_controller.*     ← KL divergence control
│
├── distill/                ← NUOVO: Knowledge Distillation
│   ├── teacher_student.*   ← Teacher-student framework
│   ├── logit_distill.*     ← Logit matching loss
│   └── progressive.*       ← Progressive distillation pipeline
│
├── vision/                 ← NUOVO: Vision
│   ├── vit.*               ← Vision Transformer (ViT)
│   ├── image_tokenizer.*   ← Image → token embedding
│   └── video_sampler.*     ← Video frame sampling
│
├── audio/                  ← NUOVO: Audio
│   ├── audio_encoder.*     ← Audio → embedding
│   └── audio_tokenizer.*   ← Audio tokenization
│
├── agent/                  ← NUOVO: Agent Framework
│   ├── tool_calling.*      ← Function/tool call interface
│   ├── code_executor.*     ← Sandboxed code execution
│   ├── web_browser.*       ← Headless browser
│   ├── orchestrator.*      ← Multi-agent dispatch
│   └── reflection.*        ← Self-verification loop
│
├── distributed/            ← NUOVO: Distributed Computing
│   ├── all_reduce.*        ← NCCL-based all-reduce
│   ├── tensor_parallel.*   ← Split matmul across GPUs
│   ├── pipeline_parallel.* ← Layer-wise pipeline
│   └── zero_optimizer.*    ← ZeRO stages 1-3
│
├── train/                  ← Training (ESTESO)
│   ├── checkpoint.*        ← Esistente
│   ├── trainer.*           ← NUOVO: Unified training loop
│   ├── mixed_precision.*   ← NUOVO: FP16/BF16/FP8 training
│   ├── data_loader.*       ← NUOVO: Streaming dataset loader
│   └── curriculum.*        ← NUOVO: Curriculum learning
│
├── safety/                 ← Safety (DA IMPLEMENTARE)
│   ├── constitutional.*    ← Constitutional AI principles
│   ├── red_team.*          ← Automated red-teaming
│   └── content_filter.*    ← Content safety filtering
│
├── server/                 ← Server/API (DA IMPLEMENTARE)
│   ├── http_server.*       ← REST API server
│   ├── streaming.*         ← SSE streaming responses
│   └── openai_compat.*     ← OpenAI-compatible API
│
└── app/                    ← CLI (ESTESO)
    └── main.cpp            ← Aggiungere comandi per tutti i nuovi moduli
```

---

## 5. Flusso Dati Completo (Inference)

```
User Input: "Spiega la teoria della relatività"
    │
    ▼
┌─────────────────┐
│ BPE Tokenizer   │  "Spieg" "a" " la" " teoria" " della" " relativ" "ità"
│ vocab: 128K     │  → [3421, 256, 410, 8923, 1204, 29401, 5621]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Token Embedding  │  [7, 4096]  (7 tokens × d_model)
│ + RoPE Encoding  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Layer 1: Mamba-2 + LatentMoE    │
│   ├─ RMSNorm(x)                │
│   ├─ Mamba-2 SSM(x)            │  ← O(n) lineare, aggiorna state h
│   ├─ residual + RMSNorm        │
│   ├─ LatentMoE:                │
│   │   ├─ compress(x) → z       │  [7, 1024]
│   │   ├─ route(z) → top-8      │  seleziona 8/128 esperti
│   │   ├─ experts(z) → y        │  8 FFN paralleli
│   │   └─ expand(y) → out       │  [7, 4096]
│   └─ residual                   │
├─────────────────────────────────┤
│ Layer 2-6: Mamba-2 + LatentMoE  │  (same pattern)
├─────────────────────────────────┤
│ Layer 7: MLA Local + LatentMoE  │
│   ├─ RMSNorm(x)                │
│   ├─ MLA:                      │
│   │   ├─ Q = x · W_q           │  [7, 32, 128]
│   │   ├─ c = x · W_dkv         │  [7, 512] → KV-cache (SOLO 2 KB/token!)
│   │   ├─ K = c · W_uk           │  [7, 32, 128] (on-the-fly)
│   │   ├─ V = c · W_uv           │  [7, 32, 128] (on-the-fly)
│   │   └─ Sliding Window Attn    │  window=4096 tokens
│   └─ LatentMoE + residual      │
├─────────────────────────────────┤
│ Layer 8: MLA Global + LatentMoE │  (full attention, no window)
├─────────────────────────────────┤
│ ...repeat layers 1-8 × 6...    │  (48 layers totali)
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Output Processing               │
│   ├─ RMSNorm(h_final)          │
│   ├─ LM Head: h → logits       │  [1, 128000] (last token only)
│   ├─ MTP Head: h → logits2      │  [1, 128000] (speculative +2)
│   │                              │
│   ├─ Thinking Router:           │
│   │   score = σ(h · W_think)    │
│   │   if score > 0.7:           │
│   │     emit <thinking> token    │  → il modello "pensa" prima di rispondere
│   │   else:                      │
│   │     direct response          │
│   │                              │
│   └─ Sampler:                   │
│       top-k=40, top-p=0.9       │
│       temperature=0.7           │
│       → next_token              │
└─────────────────────────────────┘
```

---

> **Nota**: Questa architettura è progettata per essere **modulare**.
> Ogni componente (Mamba, MLA, LatentMoE, MTP) può essere abilitato/disabilitato
> indipendentemente, permettendo di creare varianti più piccole (Nano, Mini, Base)
> semplicemente cambiando la config.
