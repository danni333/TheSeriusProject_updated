# Fase 36+ — Cose mancanti (Sparse attention)

Riferimento: `fase36_sparse_attention.md`

---

## 1. Sparse attention

| Elemento | Stato | Nota |
|----------|--------|-----|
| Pattern sparse (sliding window, stride, block) | ❌ | |
| Kernel attention che saltano blocchi | ❌ | |
| Configurazione mask sparse in config | ❌ | |
| Riduzione memoria per long context (fase16) | ❌ | |

---

**Priorità:** dopo long context; utile per sequenze molto lunghe.
