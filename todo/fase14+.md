# Fase 14+ — Cose mancanti (Speculative decoding)

Riferimento: `fase14_speculative_decoding.md`

---

## 1. Concetto

| Elemento | Stato | Nota |
|----------|--------|-----|
| Modello draft (piccolo) + modello target (grande) | ❌ | |
| Generazione draft K token, verifica con target, accept/reject | ❌ | |
| Integrazione in inference loop | ❌ | |
| src/infer/speculative.h .cpp | ❌ | |

---

## 2. Ottimizzazioni

| Elemento | Stato |
|----------|--------|
| Multi-step speculative (più draft per step) | ❌ |
| Eager decoding / token tree | ❌ |
| Metriche (acceptance rate, latency) | ❌ |

---

**Priorità:** dopo deployment base (fase13); richiede sampler e due modelli in memoria.
