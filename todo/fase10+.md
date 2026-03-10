# Fase 10+ — Cose mancanti (Finetuning)

Riferimento: `fase10_finetuning.md`

---

## 1. Full finetuning

| Elemento | Stato | Nota |
|----------|--------|-----|
| Training loop su modello pre-addestrato | Parziale | Trainer esistente; caricamento checkpoint da verificare |
| Caricamento pesi da checkpoint/GGUF/Safetensors | ❌ | Loader formato esterno mancante |
| Freeze di layer (es. solo ultimi N) | Da verificare | Flag requires_grad per parametri |
| Learning rate più basso per finetuning | Configurabile via lr_schedule |

---

## 2. LoRA / Adapter (fase 19)

| Elemento | Stato |
|----------|--------|
| LoRA (low-rank adaptation) | ❌ | fase19_lora.md |
| Adapter layers (bottleneck) | ❌ |
| Merge LoRA in base model per export | ❌ |

---

## 3. Dataset e curriculum

| Elemento | Stato |
|----------|--------|
| Dataset instruct (prompt/response) | Da verificare formato in DataLoader |
| Curriculum learning (fase10) | ❌ | Opzionale |
| Valutazione su holdout (perplexity, downstream task) | Da verificare |

---

## 4. File previsti

| File | Stato |
|------|--------|
| src/train/trainer (finetuning = training con load checkpoint) | ✅ trainer esistente |
| Loader pesi (gguf/safetensors) | ❌ |
| src/train/lora.h (fase 19) | ❌ |

---

**Priorità:** loader per checkpoint/GGUF per poter caricare modelli pre-addestrati; opzione freeze layer; LoRA in fase 19.
