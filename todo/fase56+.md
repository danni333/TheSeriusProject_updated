# Fase 56+ — Cose mancanti (E2E tests)

Riferimento: `fase56_e2e_tests.md`

---

## 1. Test end-to-end

| Elemento | Stato | Nota |
|----------|--------|-----|
| E2E: load modello → prompt → risposta verificabile | ✅ | `E2EServer.FullPipelineChatCompletion` con GGUF reale |
| E2E: train 1 epoch → checkpoint → load | Parziale | Componenti presenti; script E2E da fare |
| E2E: API request → response (con server) | ✅ | `E2EServer.AllEndpoints` + `E2ECli.ServeBinaryLifecycle` (binario reale) |
| E2E: streaming SSE | ✅ | `E2EServer.StreamingSSE` (chunk JSON + [DONE]) |
| Dati e modelli di test (piccoli, in repo o CI) | ✅ | `models/smollm2-q4` (135M Q4_0), skip se assente |
| Script e2e_*.sh o test_e2e.cpp | ✅ | `tests/test_e2e.cpp` — 6 test verdi |
| E2E: import → convert → infer | ❌ | `serius convert`/`generate` non esistono |
| E2E: safety/guardrails | ❌ | non cablati nel path HTTP |

---

**Priorità:** dopo CLI e loader; test E2E come smoke test release.
