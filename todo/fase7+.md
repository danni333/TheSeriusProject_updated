# Fase 7+ — Cose mancanti (Quantizzazione)

Riferimento: `fase7_quantizzazione.md`

---

## 1. Precisione e conversioni

| Elemento | Stato | Nota |
|----------|--------|-----|
| FP16 / BF16 (float_to_half, half_to_float, bf16) | Da verificare | quant/conversions.h/cpp |
| FP8 E4M3 / E5M2 | Da verificare | Fase7 richiede per Ada/Hopper |
| Block-wise scaling (es. 128 elementi) | Da verificare | |
| quant/int4.cpp, .h | ✅ | |
| quant/dequant_matmul (CUDA + CPU) | ✅ | dequant_matmul.cu/cpp/h |

---

## 2. INT4 e packing

| Elemento | Stato |
|----------|--------|
| Schema scale + zero_point per blocco (32/64/128) | Da verificare in int4 |
| pack_4bit / unpack (due valori per byte) | Da verificare |
| AWQ (activation-aware, salient channels) | ❌ | Non implementato |
| GPTQ (Hessian, calibrazione) | ❌ | Non implementato |
| Dequant on-the-fly in kernel matmul | Da verificare in dequant_matmul |

---

## 3. GGUF / Safetensors

| Elemento | Stato |
|----------|--------|
| Caricamento pesi quantizzati da GGUF | ❌ | Fase7/io: da fare in fase I/O |
| Caricamento Safetensors | ❌ | Idem |
| test_quant.cpp | ✅ |

---

## 4. Mixed precision training

| Elemento | Stato |
|----------|--------|
| Forward FP16/BF16, loss FP32, master weights FP32 | Parziale | amp.h presente |
| Loss scaling (AMP) | Da verificare in optim/amp.h |
| test_amp.cpp | ✅ |

---

**Priorità:** completare INT4 pack/unpack e dequant matmul; opzionale AWQ/GPTQ; FP8 e loader GGUF in fasi successive.
