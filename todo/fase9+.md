# Fase 9+ — Cose mancanti (Stabilità numerica)

Riferimento: `fase9_stabilita.md`

---

## 1. Gradient clipping

| Elemento | Stato | Nota |
|----------|--------|-----|
| clip_grad_norm(params, max_norm) — norma L2 globale | Da verificare in trainer | |
| scale gradienti se total_norm > max_norm | Da verificare | |
| max_norm = 1.0 tipico | Da verificare | |
| Ritorno total_norm per logging | Da verificare | |

---

## 2. Inizializzazione pesi

| Elemento | Stato |
|----------|--------|
| Xavier (Glorot) per linear/tanh | Da verificare (in transformer/embedding o init) |
| He (Kaiming) per ReLU/GELU | Da verificare |
| µP (microparameter) init (citato fase9) | ❌ | Opzionale |
| Inizializzazione in creazione layer (Linear, FFN, Attention) | Da verificare |

---

## 3. Mixed precision e loss scaling

| Elemento | Stato |
|----------|--------|
| Loss scaling per evitare underflow FP16 | Da verificare in optim/amp.h |
| Gradient scaling prima backward, unscale dopo | Da verificare |
| BF16/FP8 mixed precision (fase9) | Parziale (AMP presente) |

---

## 4. Stabilità softmax / LayerNorm

| Elemento | Stato |
|----------|--------|
| Softmax: subtract max prima di exp | Da verificare in autograd/ops e attention |
| LayerNorm: ε per evitare divisione per zero | Da verificare in layernorm |

---

## 5. File e test

| Elemento | Stato |
|----------|--------|
| clip_grad_norm in trainer o modulo dedicato | Da verificare |
| test gradient clipping (norma non supera max_norm) | Da verificare |
| test init (varianza output dopo init) | ❌ | Opzionale |

---

**Priorità:** verificare clip_grad_norm e loss scaling; assicurare Xavier/He nei layer; test clipping.
