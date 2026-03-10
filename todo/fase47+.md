# Fase 47+ — Cose mancanti (CLI)

Riferimento: `fase47_cli.md`

---

## 1. CLI

| Elemento | Stato | Nota |
|----------|--------|-----|
| Eseguibile principale (es. `serius`) | ❌ | Nessun target CLI in CMake |
| Comandi: chat, serve, import, train, convert | ❌ | |
| Parsing argomenti (getopt, CLI11, ecc.) | ❌ | |
| Config da file + override da CLI | ❌ | |
| docs/cli-reference.md | ❌ | docs in .gitignore |
| Esempio: `./serius chat --model ...` | ❌ | |

---

**Priorità:** alto per usabilità; dopo server e loader modello.
