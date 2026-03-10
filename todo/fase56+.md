# Fase 56+ — Cose mancanti (E2E tests)

Riferimento: `fase56_e2e_tests.md`

---

## 1. Test end-to-end

| Elemento | Stato | Nota |
|----------|--------|-----|
| E2E: load modello → prompt → risposta verificabile | ❌ | Richiede loader e sampler |
| E2E: train 1 epoch → checkpoint → load | Parziale | Componenti presenti; script E2E da fare |
| E2E: API request → response (con server) | ❌ | |
| Dati e modelli di test (piccoli, in repo o CI) | ❌ | |
| Script e2e_*.sh o test_e2e.cpp | ❌ | |

---

**Priorità:** dopo CLI e loader; test E2E come smoke test release.
