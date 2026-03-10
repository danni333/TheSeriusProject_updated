# Fase 12+ — Cose mancanti (Profiling)

Riferimento: `fase12_profiling.md`

---

## 1. Tensor health monitor

| Elemento | Stato | Nota |
|----------|--------|-----|
| TensorStats (mean, std, min, max, has_nan, has_inf, sparsity) | Da verificare | debug/tensor_logger |
| TensorLogger con livelli (SILENT, SUMMARY, FULL, PANIC) | ✅ | tensor_logger.h/cpp |
| compute(tensor, name, step) e log su file | Da verificare | |
| Allarmi (mean/std fuori range, nan/inf) | Da verificare | |
| nan_debugger (trovare primo NaN nel grafo) | ❌ | Da implementare |

---

## 2. Profiling esterno

| Elemento | Stato |
|----------|--------|
| Integrazione NSight Systems/Compute | ❌ | Documentare uso manuale |
| Tracy (frame profiling) | ❌ | Opzionale |
| Roofline analysis (script/strumenti) | ❌ | |

---

## 3. Test

| Elemento | Stato |
|----------|--------|
| Test che inietta NaN e verifica che logger lo segnali | Da aggiungere |

---

**Priorità:** completare TensorLogger con allarmi e opzione PANIC; nan_debugger per backward.
