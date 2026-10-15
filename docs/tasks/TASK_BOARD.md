# 📋 Serius — Master Task Board

> Tracker centralizzato di tutte le task di sviluppo.
> Ogni fase ha il suo file dettagliato in `docs/tasks/`.

---

## Stato Globale

| Fase | File | Progresso | Status |
|---|---|---|---|
| **Fase 1** — Fondamenta d'Acciaio | [FASE_1_TASKS.md](FASE_1_TASKS.md) | 0/28 | 🔴 Da iniziare |
| **Fase 2** — Architettura Next-Gen | [FASE_2_TASKS.md](FASE_2_TASKS.md) | 0/24 | ⏳ In attesa |
| **Fase 3** — Training Pipeline | [FASE_3_TASKS.md](FASE_3_TASKS.md) | 0/22 | ⏳ In attesa |
| **Fase 4** — Reasoning & RL | [FASE_4_TASKS.md](FASE_4_TASKS.md) | 0/20 | ⏳ In attesa |
| **Fase 5** — Multimodale & Long-Context | [FASE_5_TASKS.md](FASE_5_TASKS.md) | 0/18 | ⏳ In attesa |
| **Fase 6** — Superamento | [FASE_6_TASKS.md](FASE_6_TASKS.md) | 0/16 | ⏳ In attesa |
| **TOTALE** | | **0/128** | |

---

## Priorità Immediate (Prossime 2 Settimane)

- [ ] **T1.01** — Thread pool con work-stealing scheduler
- [ ] **T1.02** — Parallelizzare matmul su multi-thread (OpenMP)
- [ ] **T1.03** — Benchmark harness (tok/s, memoria, latency)
- [ ] **T1.04** — CUDA stream management
- [ ] **T1.05** — Integrare INT4 quantization nell'inference path

---

## Convenzioni

- **ID formato**: `T{fase}.{numero}` (es. T1.01, T2.15)
- **Priorità**: 🔴 Critica | 🟠 Alta | 🟡 Media | 🟢 Bassa
- **Stato**: `[ ]` da fare | `[/]` in corso | `[x]` completata | `[-]` cancellata
- **Stima**: in giorni lavorativi (1d = ~4h di coding focused)
