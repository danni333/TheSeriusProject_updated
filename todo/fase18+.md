# Fase 18+ — Cose mancanti (Sicurezza)

Riferimento: `fase18_sicurezza.md`

---

## 1. Safety e guardrails

| Elemento | Stato | Nota |
|----------|--------|-----|
| Modulo safety/ (guardrails, toxicity) | ❌ | src/safety/ assente |
| Filtro contenuto (prompt + risposta) | ❌ | |
| Classificatore tossicità (o integrazione esterna) | ❌ | |
| Blocklist / allowlist token | ❌ | |
| grounding (fase24) | ❌ | |

---

## 2. File previsti

| File | Stato |
|------|--------|
| src/safety/guardrails.h .cpp | ❌ |
| src/safety/toxicity.h .cpp | ❌ |
| Configurazione safety in config | ❌ |
| Test safety | ❌ |

---

**Priorità:** dopo API e inferenza; hook pre/post generazione per filtri.
