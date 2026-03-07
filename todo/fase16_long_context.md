# Fase 16 — Long-Context Management (YaRN RoPE Scaling, Sliding Window, 1M-10M+)

> Per leggere interi libri o intere codebase, serve un sistema di encoding
> posizionale che extrapoli oltre le lunghezze viste in training e un'attention
> che non faccia esplodere la RAM.
>
> **Riferimenti moderni (2025-2026):**
>
> - Llama 4 Scout: context window fino a **10M token**
> - Qwen3.5 Flash/Plus: **1M token** nativi
> - DeepSeek V3.2: **Sparse Attention** per efficiency su long-context

---

## 1. Rotary Positional Embeddings (RoPE)

### Concetto

Invece di *sommare* vettori posizionali agli embedding (come nel Transformer originale), RoPE *ruota* i vettori Q e K nello spazio complesso. La distanza relativa tra token è codificata nell'angolo di rotazione.

### Formula

Per ogni coppia di dimensioni $(q_{2i}, q_{2i+1})$ alla posizione $m$:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

dove $\theta_i = 10000^{-2i/d}$

### Implementazione

```cpp
void apply_rope(Tensor& q, Tensor& k, int start_pos, int d_head) {
    int seq_len = q.shape()[1];
    for (int pos = 0; pos < seq_len; pos++) {
        int m = start_pos + pos;  // posizione assoluta
        for (int i = 0; i < d_head; i += 2) {
            float theta = powf(10000.0f, -(float)i / d_head);
            float cos_val = cosf(m * theta);
            float sin_val = sinf(m * theta);
            // Ruota Q
            float q0 = q(pos, i), q1 = q(pos, i+1);
            q(pos, i)   = q0 * cos_val - q1 * sin_val;
            q(pos, i+1) = q0 * sin_val + q1 * cos_val;
            // Ruota K (stessa rotazione)
            float k0 = k(pos, i), k1 = k(pos, i+1);
            k(pos, i)   = k0 * cos_val - k1 * sin_val;
            k(pos, i+1) = k0 * sin_val + k1 * cos_val;
        }
    }
}
```

### Estensione del contesto (NTK-aware scaling)

Per estrapolare oltre la lunghezza di training, si modifica la base:

```cpp
// Originale: base = 10000
// Esteso: base = 10000 * (target_len / train_len) ^ (d / (d-2))
float scaled_base = 10000.0f * powf(
    (float)target_ctx / train_ctx, (float)d_head / (d_head - 2));
```

---

## 2. Sliding Window Attention

### Problema

Full attention su 128K token = matrice 128K × 128K = **64 GB** in FP32.

### Soluzione

Ogni token attende solo agli ultimi **W** token (finestra scorrevole):

```
Token:    t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈
Window=3:
t₁ attende a: [t₁]
t₂ attende a: [t₁, t₂]
t₃ attende a: [t₁, t₂, t₃]
t₄ attende a: [t₂, t₃, t₄]        ← finestra scorre
t₅ attende a: [t₃, t₄, t₅]
...
```

### Implementazione (mask)

```cpp
Tensor sliding_window_mask(int seq_len, int window_size) {
    Tensor mask({seq_len, seq_len});
    mask.fill(-INFINITY);
    for (int i = 0; i < seq_len; i++)
        for (int j = max(0, i - window_size + 1); j <= i; j++)
            mask(i, j) = 0.0f;
    return mask;
}
```

### Combinazione con Global Attention (Longformer-style)

Alcuni token speciali (es. `[CLS]`, primo token) hanno attention **globale**:

```
Token:  [CLS]  t₁  t₂  t₃  t₄  t₅
[CLS]:  [G]    [G]  [G]  [G]  [G]  [G]   ← globale
t₁:     [G]    [L]  [-]  [-]  [-]  [-]
t₂:     [G]    [L]  [L]  [-]  [-]  [-]
t₃:     [G]    [-]  [L]  [L]  [-]  [-]
t₄:     [G]    [-]  [-]  [L]  [L]  [-]
t₅:     [G]    [-]  [-]  [-]  [L]  [L]

G = Global, L = Local (window), - = masked
```

---

## 3. Prefix Caching (Automatic Prompt Cache)

> Molte richieste condividono lo stesso prefix (system prompt, RAG context).
> Ricalcolare il KV cache ogni volta è uno spreco enorme.
> Il Prefix Caching salva e riusa i KV cache dei prefix comuni.

### Problema

```text
Richiesta 1: [SYSTEM PROMPT (2000 tok)] + "Qual è la capitale dell'Italia?"
Richiesta 2: [SYSTEM PROMPT (2000 tok)] + "E quella della Francia?"
Richiesta 3: [SYSTEM PROMPT (2000 tok)] + "Quanti abitanti ha Roma?"

→ Il system prompt è identico: 2000 token ricalcolati 3 volte! (spreco)
```

### Soluzione

```text
Cache:  hash("SYSTEM PROMPT...") → KV cache prefill salvato in memoria

Richiesta 1: MISS → calcola KV(system prompt) + KV(domanda) → salva prefix
Richiesta 2: HIT  → riusa KV(system prompt) + calcola solo KV(domanda)  ← 10x più veloce
Richiesta 3: HIT  → riusa KV(system prompt) + calcola solo KV(domanda)
```

### Implementazione

```cpp
namespace serius::cache {

class PrefixCache {
    // Hash dei token prefix → KV cache salvato
    unordered_map<uint64_t, KVCacheSnapshot> cache_;
    size_t max_entries_ = 64;
    size_t max_memory_bytes_;

public:
    // Cerca il prefix più lungo che matcha
    pair<int, KVCacheSnapshot*> find_longest_prefix(const vector<int>& tokens) {
        for (int len = tokens.size(); len > 0; len--) {
            uint64_t hash = hash_tokens(tokens.data(), len);
            auto it = cache_.find(hash);
            if (it != cache_.end())
                return {len, &it->second};
        }
        return {0, nullptr};  // nessun prefix trovato
    }

    // Salva prefix dopo il primo calcolo
    void store(const vector<int>& tokens, int prefix_len,
               const KVCacheSnapshot& kv) {
        uint64_t hash = hash_tokens(tokens.data(), prefix_len);
        if (cache_.size() >= max_entries_)
            evict_lru();  // LRU eviction
        cache_[hash] = kv;
    }
};

} // namespace serius::cache
```

### Impatto Performance

| Scenario | Senza Prefix Cache | Con Prefix Cache |
| --- | --- | --- |
| RAG (system + context) | TTFT: 800ms | TTFT: 80ms (**10x**) |
| Multi-turn chat | TTFT: 400ms | TTFT: 50ms (**8x**) |
| Batch con prompt uguale | N × prefill | 1 prefill + N decode |

### File aggiuntivi

| File | Contenuto |
| --- | --- |
| `src/cache/prefix_cache.h/.cpp` | Prefix cache con LRU eviction |
| `src/cache/kv_snapshot.h/.cpp` | Snapshot e restore di KV cache |
| `tests/test_prefix_cache.cpp` | Hit/miss, eviction, correttezza output |

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/positional/rope.h/.cpp` | RoPE: apply, precompute cos/sin, NTK scaling |
| `src/attention/sliding_window.h/.cpp` | Sliding Window mask + kernel |
| `src/attention/global_local.h/.cpp` | Combinazione globale + locale |
| `tests/test_rope.cpp` | Correttezza rotazione, estensione contesto |
| `tests/test_sliding.cpp` | Mask corretta, output vs full attention |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_rope_relative` | `dot(RoPE(q, pos_m), RoPE(k, pos_n))` dipende solo da `m-n` |
| `test_rope_extension` | NTK scaling: perplexity stabile a 2x lunghezza training |
| `test_sliding_mask` | Solo token nella finestra hanno attenzione ≠ 0 |
| `test_sliding_vs_full` | Per seq < window: output identico a full attention |
| `test_memory` | Sliding window: memoria O(n·w) vs O(n²) |
