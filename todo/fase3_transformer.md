# Fase 3 — Architettura Transformer (MLA, GQA, RoPE — C++ Nativo)

> Assembliamo i pezzi delle Fasi 1 e 2 seguendo
> **"Attention Is All You Need"** (Vaswani et al., 2017).

---

## 1. Multi-head Latent Attention (MLA)

### Il problema dell'MHA

La Multi-Head Attention standard richiede di conservare i vettori Key (K) e Value (V) per ogni token generato nella **KV Cache**, occupando quantità enormi di memoria.

### La soluzione MLA (DeepSeek-V2/V3)

Invece di proiettare X in K e V indipendenti, MLA proietta l'input in un **latent vector** (vettore latente) compresso $c_t$, comune a tutte le teste, usando una low-rank joint compression. Durante il decoding, solo questo vettore latente viene salvato nella cache, riducendo l'ingombro del 90+%.

### Formula core compressa

$$c_t = W_{dKV} \cdot x_t$$
$$k_t, v_t \approx \text{RoPE}(W_{uK} \cdot c_t), W_{uV} \cdot c_t$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Flusso di una singola head

```
Input X ──┬── W_Q ──▶ Q ──┐
           ├── W_K ──▶ K ──┤──▶ scores = Q·Kᵀ / √d_k
           └── W_V ──▶ V ──┘         │
                                      ▼
                              softmax(scores)
                                      │
                              mask (opzionale, per decoder)
                                      │
                                      ▼
                              attn_weights · V = output
```

### Multi-Head (h teste)

```
Per ogni head i = 0..h-1:
    Q_i = X · W_Q_i     (d_model → d_k)
    K_i = X · W_K_i
    V_i = X · W_V_i
    head_i = Attention(Q_i, K_i, V_i)

output = Concat(head_0, ..., head_{h-1}) · W_O
```

| Parametro | Valore tipico (small) | Descrizione |
|---|---|---|
| `d_model` | 512 | Dimensione embedding |
| `n_heads` | 8 | Numero di teste |
| `d_k = d_model/n_heads` | 64 | Dimensione per testa |

### Causal Mask (per decoder/GPT-style)

Per impedire al token `i` di "guardare" i token futuri `j > i`:

```cpp
for (int i = 0; i < seq_len; i++)
    for (int j = i + 1; j < seq_len; j++)
        scores(i, j) = -INFINITY;  // softmax → 0
```

---

## 2. Feed-Forward Network (FFN)

Ogni Transformer block ha una rete feed-forward a due layer:

$$\text{FFN}(x) = \text{GELU}(x \cdot W_1 + b_1) \cdot W_2 + b_2$$

| Parametro | Shape |
|---|---|
| `W_1` | `(d_model, d_ff)` — tipicamente `d_ff = 4 * d_model` |
| `W_2` | `(d_ff, d_model)` |

---

## 3. Layer Normalization

### Formula

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

| Componente | Descrizione |
|---|---|
| $\mu$ | Media lungo l'ultima dimensione |
| $\sigma^2$ | Varianza lungo l'ultima dimensione |
| $\gamma, \beta$ | Parametri apprendibili (scale, shift) |
| $\epsilon$ | `1e-5`, per evitare divisione per zero |

### Posizione

- **Pre-Norm** (GPT-2 style, più stabile): `LayerNorm → Attention → Residual`
- **Post-Norm** (paper originale): `Attention → Residual → LayerNorm`

> [!TIP]
> Pre-Norm è consigliato: rende il training più stabile, specialmente per modelli profondi.

---

## 4. Positional Encoding

Il Transformer non ha nozione di ordine. Aggiungiamo vettori di posizione agli embedding.

### Encoding sinusoidale

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### Implementazione

```cpp
Tensor positional_encoding(int max_seq_len, int d_model) {
    Tensor pe({max_seq_len, d_model});
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / d_model);
            pe(pos, i)     = sinf(pos * freq);
            pe(pos, i + 1) = cosf(pos * freq);
        }
    }
    return pe;  // Pre-calcolato una volta sola
}
```

---

## 5. TransformerBlock — Assemblaggio

```
Input
  │
  ├──▶ LayerNorm ──▶ Multi-Head Attention ──┐
  │                                          │
  └──────────── + (residual) ◀──────────────┘
  │
  ├──▶ LayerNorm ──▶ FFN ──┐
  │                         │
  └──────── + (residual) ◀─┘
  │
Output
```

Il modello completo è uno **stack** di N TransformerBlock:

```
Embedding + Positional Encoding
    → TransformerBlock × N
        → LayerNorm finale
            → Linear (proiezione su vocabolario)
                → Softmax → probabilità token
```

---

## 6. Struttura File

| File | Contenuto |
|---|---|
| `src/transformer/attention.h/.cpp` | Multi-head Latent Attention (MLA) con Joint Compression |
| `src/transformer/ffn.h/.cpp` | Feed-Forward Network con GELU |
| `src/transformer/layernorm.h/.cpp` | Layer Normalization (pre-norm) / RMSNorm |
| `src/transformer/positional.h/.cpp` | RoPE (Rotary Position Embedding) decoupled |
| `src/transformer/block.h/.cpp` | TransformerBlock (MLA + ffn + residual, + Dual-Pipe) |
| `src/transformer/model.h/.cpp` | Stack di blocchi + embedding + output head |
| `tests/test_transformer.cpp` | Unit test per ogni componente |

---

## 7. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_attention_shape` | Output shape corretto: `(batch, seq, d_model)` |
| `test_causal_mask` | Token `i` non vede token `j > i` |
| `test_layernorm` | Output ha media ≈ 0 e varianza ≈ 1 |
| `test_positional` | I vettori PE sono unici per ogni posizione |
| `test_block_forward` | Forward pass completo senza crash |
| `test_block_backward` | Backprop attraverso il blocco: gradienti finiti e non-NaN |
