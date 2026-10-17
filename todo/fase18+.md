# Fase 18+ — Stato Implementazione e Cose Mancanti (Sicurezza)

Riferimento: `fase18_sicurezza.md`

---

## 1. Safety e guardrails

| Elemento | Stato | Nota |
|----------|--------|-----|
| Modulo `src/safety/` | ✅ | Implementato in `src/safety/guardrails.cpp/.h` |
| Blocklist token (ID) | ✅ | Metodi `add_blocklist_token`, `remove_blocklist_token`, `is_blocked_token` |
| Blocklist frasi / filtro risposta | ✅ | `add_blocklist_phrase`, `check_response` per scartare contenuti lesivi |
| Reset filtri | ✅ | Metodo `clear()` per azzeramento dinamico |
| Classificatore tossicità basato su embedding/ML | ❌ | Pianificato come step successivo |
| Grounding / fact-checking (fase 24) | ❌ | Fase 24 |

---

## 2. File sorgente presenti (fase 18)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/safety/guardrails.h/.cpp` | ✅ | Classe `Guardrails` per filtraggio token e testo generato |
| `tests/test_guardrails.cpp` | ✅ | Unit test per blocco token e verifica/rifiuto frasi vietate |

---

**Priorità rimanenti:** hook automatico pre/post generazione nel server HTTP e classificatore di tossicità multi-classe.
