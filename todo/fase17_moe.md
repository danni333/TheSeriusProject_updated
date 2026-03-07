# Fase 17 — DeepSeekMoE (Shared + Fine-Grained Experts, Aux-Loss-Free Routing)

> Modelli come GPT-4 e Mixtral usano tanti "esperti".
> Il Router decide quali attivare per ogni token,
> permettendo 100B parametri ma solo 2-3B attivi per token.

---

## 1. Architettura MoE

### Sostituzione nel Transformer Block

```
Standard:      Attention → FFN(d_model → 4*d_model → d_model)
MoE (Legacy):  Attention → Router → Top-K Experts (ognuno è un FFN)
DeepSeekMoE:   Attention → Shared Experts (Always ON) + Router → Fine-Grained Routed Experts
```

### Schema

```
                    Input token x
                         │
                    ┌────▼────┐
                    │  Router  │  → gate scores per ogni esperto
                    │ (Linear) │
                    └────┬────┘
                         │
            ┌───────┬────┼────┬───────┐
            ▼       ▼    ▼    ▼       ▼
         [Exp 0] [Exp 1] ... [Exp N-1]   (N esperti, ognuno un FFN)
            │       │              │
            └───┬───┘              │
                │ (solo top-K)     │ (inattivi, costo = 0)
                ▼
         Σ (gate_weight × expert_output)
                │
              Output
```

---

## 2. DeepSeekMoE: Shared Experts e Fine-Grained Routing

### I limiti del MoE Standard

Nel MoE tradizionale gli esperti (es. 8 esperti grandi) catturano conoscenza ridondante e tendono ad inattivarsi ("knowledge collapse").

### La soluzione DeepSeekMoE

1. **Shared Experts**: 1 o più esperti sono **SEMPRE** attivi per ogni token. Imparano la conoscenza "comune" e le feature generali (verbi, punteggiatura, sintassi base).
2. **Fine-Grained Routed Experts**: Invece di avere 8 esperti giganti, si creano es. 64 esperti molto più piccoli e se ne attivano 6-8 per token. Questi imparano concetti altamente specializzati (coding, matematica, una lingua rara) e vengono attivati selettivamente dal router.

```
                    Input token x
                         │
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
[Shared Expert(s)]                       [Router] 
(Sempre Attivo)                             │
      │                  ┌───────┬──────────┼─────────┬───────┐
      │                  ▼       ▼          ▼         ▼       ▼
      │               [Exp 0] [Exp 1] ... [Exp 32] ... [Exp 63] (Fine-Grained)
      │                  │       │                    │
      │                  └───┬───┘                    │
      │                      │ (Top-K attivati)       │
      ▼                      ▼                        ▼
      └──────────────────────┴────────────────────────┘
                             │
            Output = Shared_Out + Σ (Gate_i * Exp_i_Out)
```

---

## 3. Router Layer

### Top-K Gating

```cpp
struct RouterOutput {
    vector<int> expert_indices;    // quali esperti attivare
    vector<float> expert_weights;  // pesi di combinazione
};

RouterOutput route(const Tensor& x, int top_k) {
    // x: (d_model,)  gate: (d_model, n_experts)
    Tensor scores = matmul(x, gate_weight);  // (n_experts,)
    
    // Seleziona top-K esperti
    auto [values, indices] = topk(scores, top_k);
    Tensor weights = softmax(values);  // normalizza
    
    return {indices.to_vector(), weights.to_vector()};
}
```

### Load Balancing Loss

Senza controllo, il router potrebbe mandare tutti i token allo stesso esperto. Si aggiunge una **auxiliary loss**:

$$L_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

dove:

- $f_i$ = frazione di token assegnati all'esperto $i$
- $P_i$ = probabilità media di routing all'esperto $i$
- $\alpha$ = tipicamente 0.01

```cpp
float load_balance_loss(const Tensor& router_probs,
                         const vector<vector<int>>& assignments,
                         int n_experts, float alpha) {
    vector<float> f(n_experts, 0), P(n_experts, 0);
    int total = router_probs.shape()[0];
    for (int i = 0; i < total; i++) {
        for (int e : assignments[i]) f[e] += 1.0f / total;
        for (int e = 0; e < n_experts; e++) P[e] += router_probs(i, e) / total;
    }
    float loss = 0;
    for (int e = 0; e < n_experts; e++) loss += f[e] * P[e];
    return alpha * n_experts * loss;
}
```

---

## 3. Sparse Computation

### Perché è efficiente

| Modello | Parametri totali | Parametri attivi/token | Costo |
|---|---|---|---|
| Dense 7B | 7B | 7B | 100% |
| MoE 47B (8 esperti, top-2) | 47B | ~12B | ~25% |
| Mixtral 8x7B | 47B | ~13B | ~28% |

### Implementazione sparse

```cpp
Tensor moe_forward(const Tensor& x, const vector<Expert>& experts,
                    const Tensor& gate_weight, int top_k) {
    int batch = x.shape()[0];
    Tensor output = Tensor::zeros(x.shape());
    
    for (int i = 0; i < batch; i++) {
        auto [indices, weights] = route(x[i], gate_weight, top_k);
        for (int k = 0; k < top_k; k++) {
            int expert_id = indices[k];
            float weight = weights[k];
            // Solo top-K esperti calcolano
            output[i] += weight * experts[expert_id].forward(x[i]);
        }
    }
    return output;
}
```

### Expert Parallelism (GPU)

Ogni esperto può risiedere su una GPU diversa:

```
GPU 0: Experts 0, 1    GPU 1: Experts 2, 3
GPU 2: Experts 4, 5    GPU 3: Experts 6, 7

Router manda token all'esperto → All-to-All communication
```

---

## 4. Architetture MoE Moderne (2025-2026)

> I modelli più recenti hanno introdotto varianti MoE significative che
> il nostro engine deve supportare.

### Llama 4 — MoE con 128 Esperti

```text
Llama 4 Scout:   109B totali, 17B attivi, 16 esperti, context 10M token
Llama 4 Maverick: 400B totali, 17B attivi, 128 esperti, multimodale nativo

Innovazione: "Early Fusion" — testo, immagini e video nello stesso backbone
→ Il router deve gestire token multimodali (non solo testo)
```

### DeepSeek V3.2 — Sparse Attention + MoE

```text
671B totali, 37B attivi per token

Innovazioni:
1. DeepSeek Sparse Attention (DSA) → efficiency su long-context
2. Multi-Head Latent Attention (MLA) → comprime KV cache 5-10x
3. Agentic routing → il modello decide autonomamente quando usare tool
4. Trained con RL su 1800+ ambienti e 85K istruzioni complesse
```

### Qwen3.5 — Gated Delta Networks + MoE

```text
397B totali, 17B attivi

Innovazioni:
1. Gated Delta Networks → alternativa ai Transformer puri
   - Gate mechanism che filtra aggiornamenti di stato
   - Più efficiente per inference su sequenze lunghe
2. Fine-grained MoE → esperti più piccoli e numerosi
3. Qwen3.5-35B-A3B: solo 3B attivi battono modelli con 22B attivi
4. Context fino a 1M token su varianti Flash/Plus
5. 200+ lingue supportate
```

### Implicazioni per l'Engine

| Feature | Cosa implementare |
| --- | --- |
| **128+ esperti** | Router scalabile, expert parallelism su multi-GPU |
| **Sparse Attention** | Kernel custom per attention selettiva |
| **MLA (Multi-Head Latent)** | Compressione KV cache durante inference |
| **Gated Delta Net** | Nuovo tipo di layer (non FFN standard) |
| **Multimodal routing** | Router che gestisce token di tipo diverso |

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/moe/router.h/.cpp` | Router con Top-K gating |
| `src/moe/expert.h/.cpp` | Singolo esperto (FFN) |
| `src/moe/moe_layer.h/.cpp` | Layer MoE completo |
| `src/moe/load_balance.h/.cpp` | Auxiliary loss per bilanciamento |
| `tests/test_moe.cpp` | Routing, sparsità, load balance |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_router_topk` | Esattamente K esperti attivati per token |
| `test_router_weights` | Pesi sommano a 1 (dopo softmax) |
| `test_load_balance` | Con loss attiva, distribuzione uniforme ±10% |
| `test_sparse_equiv` | top-K=N (tutti) ≡ dense FFN |
| `test_moe_backward` | Gradienti fluiscono solo agli esperti attivati |
