# Fase 47+ — Stato Implementazione e Cose Mancanti (CLI)

Riferimento: `fase47_cli.md`

---

## 1. CLI Serius

| Elemento | Stato | Nota |
|----------|--------|-----|
| Eseguibile principale (`serius`) | ✅ | Target CMake `add_executable(serius app/main.cpp)` |
| Comando `serius serve [--port PORT]` | ✅ | Avvia il server API HTTP OpenAI-compatibile su porta configurabile |
| Comando `serius chat [--config PATH] [--type TYPE] [--max-new-tokens N]` | ✅ | Shell interattiva con autoregressive generation loop e KV-caching per LLaMA, Qwen e Transformer |
| Parsing argomenti da riga di comando | ✅ | Supporto opzioni `--port`, `--config`, `--model`, `--type`, `--max-new-tokens`, `--help` |
| Caricamento pesi e vocabolario automatico | ✅ | Risoluzione directory modello, caricamento safetensors e tokenizer BPE |
| Comandi aggiuntivi: `train`, `import`, `convert` | [/] | Training e conversioni GGUF disponibili tramite librerie core; comandi CLI dedicati pianificati |

---

## 2. File sorgente presenti (fase 47)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/app/main.cpp` | ✅ | Entrypoint dell'eseguibile CLI `serius`, dispatch subcomandi `serve` e `chat` |
| `src/CMakeLists.txt` | ✅ | Configurazione di compilazione per l'eseguibile `serius` |

---

**Priorità rimanenti:** aggiunta dei sottocomandi CLI per `train` (SFT/Pretraining) e `convert` (quantizzazione/formati).
