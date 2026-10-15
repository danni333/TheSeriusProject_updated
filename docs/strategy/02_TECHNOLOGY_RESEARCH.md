# 🔬 Ricerca Tecnologica — Tecnologie Chiave per Serius

> Questo documento approfondisce OGNI tecnologia chiave identificata nell'analisi competitiva,
> con dettagli implementativi specifici per il framework Serius in C++.

---

## Parte I — Architetture di Modello

---

### 1. Mixture-of-Experts (MoE)

#### Concetto
Invece di un singolo FFN monolitico per layer, MoE usa **N esperti** (ognuno un FFN completo) e un **router** che seleziona quali esperti attivare per ogni token.

#### Architettura Standard (DeepSeek/Qwen)
```
Input Token x [d_model]
        │
        ▼
┌──────────────┐
│   Router     │  W_gate: [d_model, n_experts]
│  (Linear +   │  scores = softmax(x · W_gate)
│   TopK)      │  top_k = select_top_k(scores, k)
└──────┬───────┘
       │ route to k experts
       ▼
┌────┐ ┌────┐ ┌────┐    ┌────┐
│ E1 │ │ E2 │ │ E3 │ ...│ En │  (N FFN networks, e.g. N=128)
└──┬─┘ └──┬─┘ └──┬─┘    └──┬─┘
   │      │      │          │
   ▼      ▼      ▼          ▼
   y1 × g1 + y2 × g2 + ... (weighted sum)
        │
        ▼
   Output [d_model]
```

#### Parametri Tipici (2026)
- **N esperti**: 64-256 (DeepSeek V3: 256, Qwen 3: 128, Llama 4: 16-128)
- **k attivi**: 2-8 per token (DeepSeek: 8, Mistral: 2)
- **Percentuale attivazione**: 3-15%
- **Shared expert**: 0-1 expert always-on (DeepSeek: 1, Qwen 3: 0)

#### Variante: LatentMoE (Nemotron 3)
```
Input Token x [d_model=4096]
        │
        ▼
┌──────────────────┐
│  Down-project    │  W_down: [d_model, d_latent]  (es. 4096→1024)
│  x_latent = x·W  │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   Router     │  Opera su x_latent: solo [d_latent] dims → 4x meno compute
└──────┬───────┘
       │
       ▼
┌────┐ ┌────┐ ┌────┐
│ E1 │ │ E2 │ ... │ En │  Experts operano su d_latent (4x più piccoli!)
└──┬─┘ └──┬─┘     └──┬─┘
   │      │          │
   ▼      ▼          ▼
   weighted_sum [d_latent]
        │
        ▼
┌──────────────────┐
│  Up-project      │  W_up: [d_latent, d_model]  (1024→4096)
└──────┬───────────┘
       │
       ▼
   Output [d_model]
```

**Vantaggio**: Può attivare **4x più esperti** per lo stesso costo computazionale → specializzazione più fine.

#### Sfide Implementative per Serius
- **Load balancing loss**: aggiungere una loss che penalizza se tutti i token vanno agli stessi esperti
- **Expert parallelism**: distribuire esperti su GPU diverse (all-to-all communication)
- **Token dropping**: se un esperto è sovraccarico, i token in eccesso vengono droppati
- **Auxiliary losses**: balance loss + router z-loss per stabilità numerica

#### File Serius Interessati
- `src/transformer/ffn.h` → base per implementare `MoEFFN`
- `src/transformer/block.cpp` → il block deve switchare tra FFN e MoEFFN
- Nuovo: `src/transformer/moe_router.h`, `src/transformer/moe_layer.h`

---

### 2. Multi-head Latent Attention (MLA)

#### Problema che Risolve
In standard Multi-Head Attention con 32 heads e d_head=128, il KV-cache per un unico token è:
```
KV per token = 2 × n_heads × d_head = 2 × 32 × 128 = 8192 float = 32 KB (FP32)
Per 1M tokens = 32 GB solo di KV-cache!
```

GQA (Grouped Query Attention) migliora usando meno KV heads (es. 8 KV heads per 32 Q heads):
```
KV per token = 2 × n_kv_heads × d_head = 2 × 8 × 128 = 2048 float = 8 KB
Per 1M tokens = 8 GB
```

MLA fa molto meglio:
```
KV per token = 1 × d_latent = 512 float = 2 KB (es. d_latent=512)
Per 1M tokens = 2 GB  ← 16x meno di standard MHA!
```

#### Come Funziona in Dettaglio
```
Input: x [seq, d_model]

# 1. Compress to latent
c = x · W_dkv    # [seq, d_latent], d_latent << n_heads × d_head
                  # QUESTO è ciò che va nel KV-cache

# 2. Reconstruct K, V per ogni head (on-the-fly, no cache)
K = c · W_uk      # [seq, n_heads, d_head]
V = c · W_uv      # [seq, n_heads, d_head]

# 3. Q è standard
Q = x · W_q       # [seq, n_heads, d_head]

# 4. Normal attention
output = softmax(Q·K^T / √d) · V
```

#### Implementazione per Serius
```cpp
// Nuova classe MLA che sostituisce il blocco attention in llama_model
class MultiheadLatentAttention {
    // Parametri
    Tensor W_q;      // [d_model, n_heads * d_head]
    Tensor W_dkv;    // [d_model, d_latent]        ← DOWN projection
    Tensor W_uk;     // [d_latent, n_heads * d_head] ← UP for keys
    Tensor W_uv;     // [d_latent, n_heads * d_head] ← UP for values
    Tensor W_o;      // [n_heads * d_head, d_model]
    
    // KV-cache: solo d_latent per posizione!
    struct MLACache {
        Tensor latents;  // [max_seq, d_latent]
        size_t len = 0;
    };
    
    Tensor forward(Tensor x, MLACache& cache);
};
```

#### File Serius Interessati
- `src/transformer/attention.h` → aggiungere MLA come opzione
- `src/transformer/kv_cache.h` → cache ridotta per MLA
- Nuovo: `src/transformer/mla.h`, `src/transformer/mla.cpp`

---

### 3. State Space Models: Mamba-2

#### Perché Mamba?
L'attention è O(n²) nel seq_len. Per 1M token context:
- Attention: 10¹² operazioni per layer → impraticabile
- Mamba (SSM): O(n) lineare → 10⁶ operazioni per layer

#### Come Funziona Mamba-2
Mamba è basata su State Space Models (SSM) discretizzati:

```
State equation:    h_t = A_t · h_{t-1} + B_t · x_t
Output equation:   y_t = C_t · h_t + D · x_t

Dove A, B, C sono input-dependent (selective!) → la chiave dell'efficienza
```

**Mamba-2 migliora** usando "structured state space duality":
- Riformula l'SSM come una forma speciale di attention lineare
- Permette parallelizzazione efficiente su GPU via **selective scan**
- O(n) in tempo ma sfrutta la banda GPU al massimo

#### Architettura di un Mamba-2 Block
```
Input x [batch, seq, d_model]
        │
        ▼
┌──────────────────────┐
│  Linear projection    │  x → [batch, seq, d_inner * 2]  (split in z, x')
└──────┬───────────────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
┌─────┐  ┌──────────────┐
│ z   │  │ Conv1D(x')   │  Causal 1D convolution (kernel_size=4)
│(gate)│  └──────┬───────┘
│     │         │
│     │         ▼
│     │  ┌──────────────┐
│     │  │ SSM Block     │  A, B, C = f(x'), D = learnable
│     │  │ (selective    │  h_t = A_t·h_{t-1} + B_t·x_t
│     │  │  scan)        │  y_t = C_t·h_t
│     │  └──────┬───────┘
│     │         │
│     │         ▼
│     │    y [batch, seq, d_inner]
│     │         │
└──┬──┘         │
   │            │
   ▼            ▼
  SiLU(z) × y           ← gated output
        │
        ▼
┌──────────────────┐
│  Output proj      │  [d_inner] → [d_model]
└──────────────────┘
```

#### Selective Scan (Algoritmo Core)

L'algoritmo di scan parallelo è il cuore di Mamba:
```
// Sequenziale (reference):
for t in 0..seq_len:
    h[t] = A[t] * h[t-1] + B[t] * x[t]
    y[t] = C[t] * h[t]

// Parallelo (GPU):
// Usa prefix-scan (à la parallel reduction) per calcolare tutti gli h[t] in O(log n)
// Implementato come operazione block-level su GPU con shared memory
```

#### File Serius Necessari
- Nuovo: `src/transformer/mamba_block.h`, `src/transformer/mamba_block.cpp`
- Nuovo: `src/transformer/ssm_scan.h` (selective scan kernel)
- Nuovo: `src/transformer/hybrid_block.h` (configurable Mamba/Attention ratio)
- Modificare: `src/transformer/model.h` per supportare hybrid layer configs

---

### 4. Multi-Token Prediction (MTP)

#### Concetto
Invece di predire solo il prossimo token, il modello predice i prossimi **N token** contemporaneamente.

#### Architettura
```
Hidden States h [seq, d_model]
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  LM Head 1   │  │  LM Head 2   │  (opzionale: Head 3, 4...)
│  (Next token)│  │  (Token +2)  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
  logits_1 [vocab]  logits_2 [vocab]
```

#### Training
```
Loss = λ1 · CrossEntropy(logits_1, target[t+1])
     + λ2 · CrossEntropy(logits_2, target[t+2])
     + ...

Tipicamente: λ1 = 1.0, λ2 = 0.3 (il token principale conta di più)
```

#### Inference (Speculative-like)
```
1. Forward pass → ottieni logits_1, logits_2
2. Sample token_1 da logits_1  (prossimo token)
3. Sample token_2 da logits_2  (token dopo)
4. Verifica: esegui forward con token_1, controlla se logits per posizione +1 ≈ token_2
5. Se sì → accetta entrambi (2 token per 1 forward pass!)
6. Se no → scarta token_2, usa solo token_1 (fallback standard)
```

**Speedup tipico: 1.5-1.8x** (dipende dall'accuracy della head secondaria)

---

### 5. Gated DeltaNet (Alternativa SSM)

Usato da Qwen 3.5 come alternativa a Mamba:

```
Input: x [batch, seq, d]
Gate:  g = σ(x · W_g)           ← soft gating
Delta: δ = x · W_delta           ← update direction
Key:   k = x · W_k
Value: v = x · W_v

State update: S_t = g_t · S_{t-1} + (1 - g_t) · (k_t ⊗ v_t)
Output:       y_t = S_t · q_t

Dove q_t è la query derivata dall'input
```

**Vantaggi vs Mamba:**
- Training più stabile (gating controlla il flusso)
- Non richiede selective scan hardware-optimizzato
- Combina bene con attention layers

---

## Parte II — Tecniche di Training

---

### 6. GRPO (Group Relative Policy Optimization)

#### Problema
Standard RLHF richiede un **reward model** separato, costoso da trainare e mantenere.

#### Soluzione GRPO
```
Per ogni prompt x:
1. Genera N risposte: {y_1, y_2, ..., y_N}  (es. N=16)
2. Valuta con rule-based rewards:
   - Math: esegui la computazione, verifica il risultato
   - Code: compile & run, controlla output vs expected
   - Logic: check che la risposta sia internamente coerente
3. Calcola reward relativo: r̂_i = (r_i - mean(r)) / std(r)
4. Gradient: ∇J = Σ_i r̂_i · ∇log π(y_i|x) - β · KL(π || π_ref)
```

#### Vantaggi
- **Zero reward model** → risparmio di 50% del costo RL
- **Rule-based verification** → reward più accurato per task verificabili
- **KL constraint** → il modello non diverge troppo dalla policy di partenza

---

### 7. Knowledge Distillation

#### Pipeline
```
Teacher Model (es. 200B)
        │
        │ genera logits e hidden states
        ▼
┌──────────────────────────────┐
│  Distillation Losses         │
│                              │
│  L_kd = KL(student_logits || │  ← Logit matching
│          teacher_logits)     │
│                              │
│  L_feat = MSE(student_h,    │  ← Feature matching
│           teacher_h)         │  (opzionale, per specifici layer)
│                              │
│  L_task = CE(student_logits, │  ← Task loss standard
│           ground_truth)      │
│                              │
│  L_total = α·L_kd + β·L_feat│ + γ·L_task
└──────────────────────────────┘
        │
        ▼
Student Model (es. 10B)
```

#### Progressive Distillation
```
200B → 100B → 50B → 20B → 10B
(ogni step trasferisce ~90% della conoscenza del teacher precedente)
```

---

### 8. Mixed Precision & Low-Bit Training

#### Gerarchia delle Precision (2026)
```
FP32 (32-bit) → Old standard, spreco per la maggior parte dei tensori
BF16 (16-bit) → Standard attuale per training
FP16 (16-bit) → Richiede loss scaling, meno stabile di BF16
FP8  (8-bit)  → DeepSeek V3: riduzione 2x vs BF16, qualità mantenuta
NVFP4 (4-bit) → Nemotron: riduzione 4x, richiede calibration
INT4  (4-bit) → Solo per weights, non per activations
BitNet (1.58b)→ Pesi ternari {-1, 0, +1}, no moltiplicazioni!
```

#### BitNet 1.58-bit (Breakthrough per Edge)
```
Standard matmul: y = W · x    (W è FP32, richiede FMA operations)
BitNet matmul:   y = W · x    (W ∈ {-1, 0, +1}, solo addizioni/sottrazioni!)

Implementazione:
for each row w in W:
    y = 0
    for i in 0..d:
        if w[i] == +1:  y += x[i]
        elif w[i] == -1: y -= x[i]
        // w[i] == 0: skip!
    
→ Zero moltiplicazioni, solo add/sub
→ 10x risparmio energetico vs FP32
→ Perfetto per edge/mobile/embedded
```

---

## Parte III — Tecniche di Inference

---

### 9. Speculative Decoding

#### Concetto
```
Draft Model (piccolo, veloce, es. 100M params)
        │
        │ genera k token candidati in batch
        ▼
  [t1, t2, t3, t4, t5]  ← 5 token candidati
        │
        │ verifica in parallelo con un singolo forward del main model
        ▼
Main Model (grande, lento, es. 10B params)
        │
        │ accetta i primi n token che matchano
        ▼
  [t1, t2, t3, ✗]  ← 3 accettati, t4 rifiutato
```

**Risultato**: 3 token al costo di ~1.3 forward passes (invece di 3).

#### Varianti
- **Tree-based**: il draft genera un albero di possibilità, il main model ne verifica più rami
- **Self-speculative**: usa early-exit layers dello stesso modello come draft
- **Medusa heads**: multiple teste aggiuntive che speculano token futuri

### 10. Flash Attention v2

#### Stato in Serius
Serius ha già `src/attention/flash_attn.h` con implementazione CPU e CUDA. Serve estendere con:

#### Estensioni Necessarie
```
Attuale:  causal mask, standard attention
Serve:    
  1. Sliding window attention (Mistral-style)
  2. Local + Global interleaved (Gemma-style)  
  3. Sparse attention patterns (configurable)
  4. Multi-query attention (MQA) / Grouped-query (GQA)
  5. MLA-compatible attention (latent space input)
```

---

## Parte IV — Long-Context Engineering

---

### 11. Context Window Scaling

#### Tecniche Ordinate per Efficacia (2026)
```
1. YaRN (Yet another RoPE extensioN)
   - Modifica le frequenze RoPE per extrapolation
   - Scala da 4K → 128K con fine-tuning breve
   - Serius: modificare src/transformer/positional.cpp

2. Sliding Window + Global Attention (Gemma-style)
   - 75% layers: local window (es. 4096 tokens)
   - 25% layers: full global → recall da tutto il context
   - KV-cache ridotto drasticamente

3. Ring Attention (multi-GPU)
   - Context distribuito su N GPU
   - GPU_i processa tokens [i*chunk, (i+1)*chunk]
   - Communication ring per passare K,V tra GPU
   - Context totale = N × chunk_size

4. Sparse Attention Patterns
   - Non tutti i token attendono a tutti gli altri
   - Pattern: locale + stride + random
   - Riduce O(n²) → O(n·√n) o O(n·log(n))
```

---

## Mappa: Tecnologie → File Serius Impattati

| Tecnologia | File Esistenti | Nuovi File Necessari |
|---|---|---|
| MoE / LatentMoE | `ffn.h`, `block.cpp` | `moe_router.h`, `moe_layer.h`, `latent_moe.h` |
| MLA | `attention.h`, `kv_cache.h` | `mla.h`, `mla.cpp` |
| Mamba-2 | — | `mamba_block.h`, `ssm_scan.h`, `selective_scan.cu` |
| MTP | `llama_inference.h` | `mtp_head.h`, `mtp_decoder.h` |
| Gated DeltaNet | — | `deltanet_block.h` |
| GRPO | `src/rl/` (vuoto) | `grpo.h`, `reward_functions.h`, `rl_trainer.h` |
| Distillation | `src/train/` | `distillation.h`, `teacher_student.h` |
| BitNet | `src/quant/int4.h` | `bitnet.h`, `ternary_matmul.h` |
| Speculative | `src/infer/` | `speculative_decoder.h`, `draft_model.h` |
| Flash Attn v2+ | `flash_attn.h` | Estendere con sliding window, sparse |
| YaRN | `positional.cpp` | Estendere con NTK-aware scaling |

---

> **Priorità Implementativa**: MoE → MLA → Mamba-2 → MTP → GRPO → BitNet → Speculative Decoding
> Questo ordine massimizza l'impatto sul benchmark per sforzo richiesto.
