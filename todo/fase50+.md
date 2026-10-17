# Fase 50+ — Stato Implementazione e Cose Mancanti (Model hub)

Riferimento: `fase50_model_hub.md`

---

## 1. Model Hub & Registry

| Elemento | Stato | Nota |
|----------|--------|-----|
| Registro / catalogo metadati modelli | ✅ | Implementato in `src/hub/metadata_manager.cpp/.h` |
| Server REST Model Hub | ✅ | Implementato in `src/hub/model_hub_server.cpp/.h` con autenticazione e catalogazione |
| Motore di ricerca modelli | ✅ | Implementato in `src/hub/search_engine.cpp/.h` con query per nome, tag e capacità |
| Configurazioni pronte modelli comuni | ✅ | Presenti in `configs/`: `model_1b.json`, `model_7b.json`, `model_moe_1b.json`, `model_moe_demo.json` |
| Download da Hugging Face / URL remoto | [/] | Loader HF safetensors locale presente; client di download HTTP automatico pianificato |
| Formato e versioning modelli | ✅ | Gestito tramite `ModelMetadata` con versioni e checksum SHA256 |

---

## 2. File sorgente presenti (fase 50)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/hub/model_hub_server.h/.cpp` | ✅ | Server HTTP per gestione registry modelli, upload/download e listing |
| `src/hub/metadata_manager.h/.cpp` | ✅ | Persistenza e indicizzazione metadati modelli |
| `src/hub/search_engine.h/.cpp` | ✅ | Ricerca avanzata modelli su parametri, quantizzazioni e task |
| `configs/model_*.json` | ✅ | Template di configurazione per architetture dense e MoE |

---

**Priorità rimanenti:** download automatico di pesi da remote Hub via CLI (`serius import hf://...`).
