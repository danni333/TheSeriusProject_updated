# Fase 17+ — Cose mancanti (MoE)

Riferimento: `fase17_moe.md`

---

## 1. Mixture of Experts

| Elemento | Stato | Nota |
|----------|--------|-----|
| Router (softmax su gate, top-k experts) | ❌ | |
| Layer MoE (N experts, routing per token) | ❌ | |
| Expert parallelism (distribuito) | ❌ | fase8 |
| src/model/moe.h .cpp (o transformer/moe) | ❌ | |
| Config moe (num_experts, top_k) in config modello | Da verificare in config.h | |

---

## 2. Varianti

| Elemento | Stato |
|----------|--------|
| DeepSeek-MoE style | ❌ |
| Load balancing loss (auxiliary loss) | ❌ |
| Test MoE forward/backward | ❌ |

---

**Priorità:** dopo transformer base stabile; router + 2–4 expert FFN come MVP.
