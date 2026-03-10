# Fase 3+ — Cose mancanti (Transformer)

Riferimento: `fase3_transformer.md`

---

## 1. Attention

| Elemento | Stato | Nota |
|----------|--------|-----|
| Multi-Head Attention (MHA) classica | ✅ | attention.cpp/h |
| MLA (Multi-head Latent Attention) con joint compression | Da verificare | Fase3 descrive c_t, W_dKV, W_uK, W_uV |
| GQA (grouped query) | Da verificare | num_kv_heads vs num_heads |
| Causal mask (decoder) | Da verificare | scores(i,j)=-inf per j>i |
| Output shape (batch, seq, d_model) | Da verificare | |
| RoPE (positional) | ✅ | positional.cpp/h |

---

## 2. FFN e LayerNorm

| Elemento | Stato |
|----------|--------|
| FFN: GELU(x·W1+b1)·W2+b2 | ✅ ffn.h/cpp |
| LayerNorm (γ, β, ε) pre-norm | ✅ layernorm.h/cpp |
| RMSNorm (alternativa LLaMA) | Da verificare se presente come opzione |
| Residual connections nel block | Da verificare in block.cpp |

---

## 3. Positional encoding

| Elemento | Stato |
|----------|--------|
| Sinusoidale classico (max_seq_len, d_model) | In fase3; progetto usa RoPE |
| RoPE (decoupled) | ✅ positional |

---

## 4. TransformerBlock e Model

| Elemento | Stato |
|----------|--------|
| Block: PreNorm → Attention → Residual → PreNorm → FFN → Residual | ✅ block.cpp/h |
| Stack di N blocchi + embedding + PE + output head | ✅ model.cpp/h |
| Config (hidden_size, num_layers, num_heads, ecc.) | ✅ config.h |
| Dual-Pipe (citato in fase3 block) | Da verificare se implementato |

---

## 5. File e test

| File | Stato |
|------|--------|
| attention, ffn, layernorm, positional, block, model | ✅ |
| test_transformer.cpp | ✅ |
| test_attention_shape, test_causal_mask, test_layernorm, test_positional, test_block_forward/backward | Da verificare in test_transformer |

---

**Priorità:** allineare attention a MLA se richiesto; verificare GQA e causal mask; aggiungere test backward sul block.
