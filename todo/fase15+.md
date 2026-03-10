# Fase 15+ — Cose mancanti (Checkpoint sharding)

Riferimento: `fase15_checkpoint_sharding.md`

---

## 1. Sharding checkpoint

| Elemento | Stato | Nota |
|----------|--------|-----|
| Salvataggio pesi su più file (es. .ckpt.000, .ckpt.001) | ❌ | |
| Caricamento da shard multipli | ❌ | |
| Metadata (numero shard, map nome → file+offset) | ❌ | |
| Allineamento a ZeRO/FSDP (fase8) | ❌ | |

---

## 2. File

| Elemento | Stato |
|----------|--------|
| Estensione checkpoint.h/cpp per sharding | ❌ |
| Formato .ckpt con indice shard | ❌ |

---

**Priorità:** utile per modelli molto grandi; dopo fase8 se si fa distribuito.
