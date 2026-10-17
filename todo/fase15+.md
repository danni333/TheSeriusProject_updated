# Fase 15+ — Stato Implementazione e Cose Mancanti (Checkpoint sharding & Formati)

Riferimento: `fase15_checkpoint_sharding.md`

---

## 1. Sharding checkpoint & Safetensors

| Elemento | Stato | Nota |
|----------|--------|-----|
| Salvataggio pesi su più shard Safetensors | ✅ | Implementato in `src/io/sharding.cpp/.h` con partizionamento su `max_shard_bytes` |
| Caricamento da shard multipli | ✅ | `load_sharded_safetensors` carica e materializza tutti i tensori dagli shard |
| Indice e Metadata (map nome → shard_file + bytes) | ✅ | Struttura `ShardedSafetensorsIndex` e file `model.safetensors.index.json` |
| Scrittore Safetensors nativo | ✅ | Implementato in `src/io/safetensors_writer.cpp/.h` |
| Lettore GGUF con metadata | ✅ | Implementato in `src/io/gguf_loader.cpp/.h` |
| Allineamento con distributed sharding (ZeRO/FSDP) | [/] | Sharding su disco completo; caricamento per-rank pianificato |

---

## 2. File sorgente presenti (fase 15)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/io/sharding.h/.cpp` | ✅ | Gestione indici sharded safetensors, scrittura su file multipli e lettura |
| `src/io/safetensors_writer.h/.cpp` | ✅ | Serializzazione header JSON e payload binario safetensors conforme |
| `src/io/safetensors.h/.cpp` | ✅ | Parser e loader safetensors zero-copy o in memoria |
| `src/io/gguf_loader.h/.cpp` | ✅ | Parser metadati e tensori GGUF |
| `tests/test_sharding.cpp` | ✅ | Test su partizionamento tensori, scrittura indice JSON e caricamento multi-shard |
| `tests/test_safetensors.cpp` | ✅ | Unit test del formato safetensors |

---

**Priorità rimanenti:** streaming di caricamento selettivo per-rank per distribuzioni multi-GPU molto grandi.
