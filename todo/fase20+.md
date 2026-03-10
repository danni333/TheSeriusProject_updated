# Fase 20+ — Cose mancanti (Multimodale)

Riferimento: `fase20_multimodale.md`

---

## 1. Input multimodale

| Elemento | Stato | Nota |
|----------|--------|-----|
| Encoder immagini (CLIP/ViT-style) o integrazione | ❌ | |
| Fusion vision + testo (embedding congiunto) | ❌ | |
| Modello vision-language (VL) | ❌ | |
| Formato input (immagini in messaggi API) | ❌ | |

---

## 2. File e pipeline

| Elemento | Stato |
|----------|--------|
| src/model/vision.h .cpp o simile | ❌ |
| Preprocessing immagini (resize, normalize) | ❌ |
| Test multimodale | ❌ |

---

**Priorità:** dopo stack testo stabile; richiede tokenizer e embedding estesi.
