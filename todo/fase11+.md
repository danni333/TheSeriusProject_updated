# Fase 11+ — Cose mancanti (Flash Attention)

Riferimento: `fase11_flash_attention.md`

---

## 1. Flash Attention

| Elemento | Stato | Nota |
|----------|--------|-----|
| Tiling algorithm (Br, Bc blocchi) | Da verificare | attention/flash_attn |
| Online softmax (m, l, O accumulatori) | Da verificare | |
| Kernel CUDA flash_attention | ✅ | flash_attn_cuda.cu |
| Evitare materializzazione N×N in HBM | Da verificare | |
| test_flash_attn.cpp | ✅ | |

---

## 2. Avanzato (fase11)

| Elemento | Stato |
|----------|--------|
| FlashAttention-3 (TMA, WGMMA, Hopper) | ❌ | Opzionale, hardware-specific |
| RingAttention (sharding Q,K,V su GPU) | ❌ | Per contesto molto lungo distribuito |

---

**Priorità:** verificare correttezza numerica flash vs attention standard; ottimizzare blocchi Br/Bc.
