# Supported Models (Design)

Questa pagina descrive i tipi di modelli target per **TheSeriusProject**.

> Nota: l'elenco è una **dichiarazione di intenti** più che un inventario definitivo dei modelli già supportati.

---

## 1. Classi di modelli target

- **LLM decoder‑only** (GPT‑style, Qwen‑style, LLaMA‑style).
- **Modelli istruiti** (Instruct/Chat) per uso conversazionale.
- **Varianti quantizzate** (es. GGUF quantizzato a 4/5/8 bit) per esecuzione su hardware consumer.

---

## 2. Formati previsi

- **Formati nativi del progetto** (checkpoint interno).
- **Formati interoperabili**:
  - GGUF / GPTQ (via strumenti di conversione).
  - Checkpoint Hugging Face (PyTorch/`safetensors`) → conversione a formato interno.

---

## 3. Requisiti hardware indicativi

Esempi di target (puramente indicativi):

- Modelli ~3B parametri, quantizzati a 4 bit:
  - 8–12 GB di RAM / VRAM raccomandati.
- Modelli ~7B parametri, quantizzati a 4 bit:
  - 12–16 GB di RAM / VRAM raccomandati.

L'obiettivo è permettere:

- Esecuzione su CPU con buone ottimizzazioni SIMD.
- Sfruttamento di GPU consumer con 8–12 GB di VRAM.

---

## 4. Roadmap

Future estensioni previste:

- Supporto per modelli con **context window molto lunga**.
- Supporto per modelli **MoE** (Mixture of Experts).
- Supporto per modelli **multimodali** (testo + immagine/audio).

I dettagli concreti verranno aggiunti quando i primi percorsi di import/export modello saranno implementati.

