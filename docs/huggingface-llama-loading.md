# Caricamento modelli LLaMA da Hugging Face

Per scaricare un modello reale da Hugging Face, caricarlo in C++ e ottenere risposte sensate.

## 1. Scaricare il modello (config + tokenizer + pesi)

Dalla root del progetto:

```bash
pip install huggingface_hub
python scripts/download_hf_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama
```

Verranno scaricati in `models/tinyllama`:
- `config.json` (architettura LLaMA)
- `tokenizer.json`, `tokenizer_config.json` (tokenizer)
- `*.safetensors` (pesi, uno o più shard)

Puoi usare lo stesso script per altri modelli LLaMA/LLaMA2 compatibili (es. `meta-llama/Llama-2-7b-hf` se hai accesso).

## 2. Caricare in C++ e fare forward

```cpp
#include "io/hf_llama_loader.h"
#include "transformer/llama_model.h"

// Carica config (stesso config.json scaricato)
serius::model::LLamaConfig config = serius::io::load_llama_config("models/tinyllama/config.json");

// Costruisci modello e carica pesi
serius::model::LLaMAModel model(config);
bool ok = serius::io::load_llama_weights_from_dir("models/tinyllama", &model);

// Forward: token_ids -> logits
serius::core::Tensor token_ids({seq_len});
// ... riempi token_ids con gli id dei token (es. da tokenizer)
serius::core::autograd::Node input_node(std::move(token_ids), false);
serius::core::autograd::Node* logits = model.forward(&input_node);
// logits->data ha shape (seq_len, vocab_size)
```

## 3. Ottenere risposte sensate

- **Tokenizer**: usa il tokenizer scaricato (in Python con `transformers` o in C++ se hai un BPE compatibile) per convertire testo → token ids in input e logits → token id → testo in output.
- **Sampling**: applica il sampler (temperature, top_k, top_p) sui logits dell’ultimo token per scegliere il token successivo; ripeti per generazione autoregressiva.
- **Chat**: per modelli chat (es. TinyLlama-1.1B-Chat), usa il template di prompt del modello (es. `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).

## 4. Codice di spunto in Python

Nella cartella scaricata hai tutto il necessario per confrontarti con il comportamento di riferimento:

- Apri `models/tinyllama` e usa `transformers` per caricare il modello e il tokenizer e generare testo; confronta con l’output del motore C++.
- Esempio minimo Python:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("models/tinyllama")
  tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")
  out = model.generate(**tokenizer("Hello", return_tensors="pt"), max_new_tokens=20)
  print(tokenizer.decode(out[0]))
  ```

## 5. Formato pesi

I file `.safetensors` usano il formato [safetensors](https://huggingface.co/docs/safetensors). Il loader C++ (`io/safetensors_loader.cpp`, `io/hf_llama_loader.cpp`) legge i tensori con i nomi standard LLaMA (es. `model.layers.0.self_attn.q_proj.weight`) e li copia nel modello, trasponendo dove necessario (HF usa `[out_features, in_features]`, il nostro matmul usa `[in, out]`).

## 6. Test

- `tests/test_llama_model.cpp`: forward con config minuscola (senza file HF) e caricamento config da JSON.
- Per testare con pesi reali: scarica un modello con `download_hf_model.py` e in un test (o in un piccolo eseguibile) chiama `load_llama_weights_from_dir` e `model.forward`.
