# Fase 14+ — Stato Implementazione e Cose Mancanti (Speculative decoding)

Riferimento: `fase14_speculative_decoding.md`

---

## 1. Concetto & Architettura

| Elemento | Stato | Nota |
|----------|--------|-----|
| Modello draft (piccolo) + modello target (grande) | ✅ | Classe `DraftModel` con `ForwardFn` flessibile in `src/infer/speculative.h/.cpp` |
| Generazione draft $\gamma$ token, verifica con target, accept/reject | ✅ | Algoritmo di speculative sampling implementato in `SpeculativeDecoder::generate()` |
| Campionamento probabilistico o greedy | ✅ | Integrato con `Sampler` e softmax distribution |
| Parametro $\gamma$ (gamma) configurabile | ✅ | `set_gamma(int)` e default configurabile |
| Metriche di efficienza | ✅ | Tracciamento `draft_steps`, `accepted_tokens`, `rejected_tokens`, `acceptance_rate` |
| Integrazione in inference loop | ✅ | Supporto sia callable functional che `DraftModel` |

---

## 2. Ottimizzazioni Avanzate

| Elemento | Stato | Nota |
|----------|--------|-----|
| Multi-step speculative ($\gamma$ passi per iterazione) | ✅ | Supportato nativamente nel ciclo principale del decoder |
| Metriche di acceptance rate e step | ✅ | Struttura `Metrics` esportata per telemetria/profiling |
| Eager decoding / Token tree / Medusa heads | ❌ | Pianificato per estensioni tree-based speculative decoding |

---

## 3. File sorgente presenti (fase 14)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/infer/speculative.h/.cpp` | ✅ | `DraftModel`, `SpeculativeDecoder`, loop speculativo e calcolo metriche |
| `src/infer/draft_model.h` | ✅ | Interfaccia e wrapper draft model |
| `tests/test_speculative.cpp` | ✅ | Test su generazione token aggiuntivi, matching draft/target e metriche |

---

**Priorità rimanenti:** estensioni Medusa heads / tree speculative decoding e benchmark di throughput comparativo.
