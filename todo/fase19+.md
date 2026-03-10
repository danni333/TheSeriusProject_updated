# Fase 19+ — Cose mancanti (LoRA)

Riferimento: `fase19_lora.md`

---

## 1. LoRA (Low-Rank Adaptation)

| Elemento | Stato | Nota |
|----------|--------|-----|
| Layer LoRA: W' = W + A·B (A r×k, B k×d, r piccolo) | ❌ | |
| Solo A,B trainabili; W congelato | ❌ | |
| Inserimento in Linear/Attention (wrapper o sostituzione) | ❌ | |
| src/train/lora.h .cpp | ❌ | |
| Merge LoRA in W per export | ❌ | |
| Salvataggio/caricamento solo adapter | ❌ | |

---

## 2. Training e config

| Elemento | Stato |
|----------|--------|
| Optimizer solo su parametri LoRA | ❌ |
| Config rank, alpha, layer target | ❌ |
| test_lora (convergenza su toy) | ❌ |

---

**Priorità:** alto per finetuning efficiente; dopo fase5 e fase10.
