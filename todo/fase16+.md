# Fase 16+ — Cose mancanti (Long context)

Riferimento: `fase16_long_context.md`

---

## 1. Contesto lungo

| Elemento | Stato | Nota |
|----------|--------|-----|
| RoPE scaling (NTK, YaRN, ecc.) per seq > train length | Da verificare | positional |
| KV cache grande / paged (PagedAttention) | Parziale | paged_kv_cache.cpp/h |
| Attention sparse / sliding window (opzionale) | ❌ | fase36 |
| max_position_embeddings configurabile | Da verificare in config | |

---

## 2. Memoria e performance

| Elemento | Stato |
|----------|--------|
| Eviction policy per KV (FIFO, LRU) | Da verificare in paged_kv |
| Flash Attention per ridurre memoria attention | ✅ fase11 |
| Benchmark contesto lungo (8k, 32k, 128k) | ❌ |

---

**Priorità:** verificare RoPE scaling oltre train length; consolidare paged KV e eviction.
