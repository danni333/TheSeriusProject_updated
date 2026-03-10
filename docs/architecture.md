# Architettura — TheSeriusProject

Panoramica dell’architettura interna per sviluppatori.

---

## 1. Struttura sorgenti

```
src/
├── tensor/          Tensori, shape, strides, view/copy
├── blas/            Matmul, matvec, element-wise, dispatch CPU/CUDA
├── memory/          Memory pool, pool allocator
├── autograd/        Grafo computazionale, Node, engine, ops (matmul, relu, softmax, …)
├── transformer/     Attention, FFN, LayerNorm, RoPE, block, model, KV cache
├── attention/       Flash attention (CPU + CUDA)
├── optim/           AdamW, WSD scheduler, AMP
├── train/           Trainer, checkpoint, freeze_parameters
├── infer/           Sampler (temperature, top_k, top_p)
├── io/              Config loader (JSON → ModelConfig)
├── tokenizer/       BPE, vocab
├── data/            DataLoader, Batch
├── quant/           INT4, conversioni FP16/BF16, dequant matmul
├── hal/             Backend CPU, CUDA, Metal/Vulkan (astrazione device)
├── debug/           TensorLogger (statistiche tensori)
├── server/          Server HTTP (health, /v1/chat/completions, /v1/models)
└── app/             CLI (serius serve, serius chat)
```

---

## 2. Flusso training

1. **DataLoader** → Batch (input_ids, target_ids, padding_mask)
2. **Trainer** → zero_grad, forward (modello), cross_entropy, backward, (AMP unscale), clip_grad_norm, optimizer.step
3. **Optimizer** (AdamW) aggiorna solo i parametri con `requires_grad == true`
4. **freeze_parameters(count)** imposta `requires_grad = false` sui primi `count` parametri (finetuning)

---

## 3. Flusso inferenza (target)

1. Caricamento modello (config da JSON via `io::load_model_config`, pesi da GGUF/safetensors in fase di implementazione)
2. Forward con KV cache (opzionale)
3. **Sampler** sui logits (temperature, top_k, top_p) → token successivo
4. Server espone `/v1/chat/completions`; il sampler sarà usato quando il modello è caricato in memoria

---

## 4. Configurazione modello

- **ModelConfig** (transformer/model.h): vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len
- **io::load_model_config(path)** legge un JSON in stile configs/model_7b.json e mappa i campi su ModelConfig (hidden_size → d_model, num_attention_heads → n_heads, ecc.)

---

## 5. Namespace

- `serius::core` — tensor, blas, memory, autograd
- `serius::model` — transformer, block, attention, ffn, layernorm
- `serius::optim` — AdamW, scheduler, AMP
- `serius::training` — Trainer, checkpoint
- `serius::infer` — Sampler
- `serius::io` — config loader
- `serius::server` — HTTP server
