# Fase 50 — Model Hub & Pipeline di Conversione

> L'utente deve poter scaricare modelli pre-trainati con un comando,
> e convertirli nel formato nativo del progetto.
> Supporto per HuggingFace Hub, download diretto, e conversione offline.

---

## 1. Funzionalità

### Download da HuggingFace Hub

```bash
# Download diretto (senza Python/pip)
serius import --source hf://Qwen/Qwen3.5-9B-Instruct --format gguf --quant q4_k_m

# Download con token privato
serius import --source hf://meta-llama/Llama-4-Scout --token hf_abc123

# Da URL diretto
serius import --source https://example.com/model.gguf --output ./models/
```

### Conversione Formati

```bash
# Safetensors → GGUF
serius convert --input model.safetensors --output model.gguf

# FP16 → INT4 AWQ
serius convert --input model.safetensors --output model-q4.gguf --quant int4-awq

# FP16 → FP8
serius convert --input model.safetensors --output model-fp8.gguf --quant fp8
```

---

## 2. Formati Supportati

| Formato Input | Formato Output | Quantizzazioni |
| --- | --- | --- |
| Safetensors (.safetensors) | GGUF (.gguf) | FP16, FP8, INT8, INT4 |
| GGUF (.gguf) | GGUF (.gguf) | Re-quantizzazione |
| PyTorch (.bin) | GGUF (.gguf) | Tutte |

---

## 3. Registry Modelli Compatibili (Aggiornato Marzo 2026)

### Modelli Dense

| Modello | Parametri | Architettura | Licenza | VRAM Min (Q4) |
| --- | --- | --- | --- | --- |
| **Qwen3.5** (Small) | 0.8B, 2B, 4B, 9B | Gated Delta Net + MoE | Apache 2.0 | 1 / 2 / 3 / 6 GB |
| **Qwen3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B | Dense Transformer | Apache 2.0 | 1-20 GB |
| **Llama 3.x** | 1B, 3B, 8B, 70B | Dense + GQA | Llama 3 | 1.5 / 3 / 6 / 42 GB |
| **Gemma 3** | 1B, 4B, 12B, 27B | Dense + Sliding Window | Gemma | 1 / 3 / 8 / 18 GB |
| **Mistral 3** | 3B, 7B, 14B | Dense multimodale | Apache 2.0 | 2 / 5 / 10 GB |
| **Phi-4** | 3.8B, 5.6B, 14B | Dense + reasoning | MIT | 3 / 4 / 10 GB |
| **DeepSeek-R1** | 7B, 67B | Dense reasoning | MIT | 5 / 42 GB |

### Modelli MoE (Mixture of Experts)

| Modello | Totale | Attivi/token | Esperti | Licenza | VRAM Min (Q4) |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.5** (Flagship) | 397B | 17B | MoE + GDN | Apache 2.0 | ~12 GB |
| **Qwen3.5-35B-A3B** | 35B | 3B | MoE fine-grained | Apache 2.0 | ~3 GB |
| **Qwen3-235B-A22B** | 235B | 22B | MoE | Apache 2.0 | ~15 GB |
| **Llama 4 Scout** | 109B | 17B | 16 esperti | Llama 4 | ~8 GB |
| **Llama 4 Maverick** | 400B | 17B | 128 esperti | Llama 4 | ~12 GB |
| **DeepSeek-V3.2** | 671B | 37B | MoE + Sparse Att. | MIT | ~20 GB |
| **Mixtral / Mistral MoE** | 8x7B | 14B | 8 esperti | Apache 2.0 | ~28 GB |

### Novità Architetturali da Supportare

| Innovazione | Modello | Impatto |
| --- | --- | --- |
| **Gated Delta Networks** | Qwen3.5 | Nuova architettura ibrida (non puro Transformer) |
| **DeepSeek Sparse Attention** | DeepSeek V3.2 | Attention sparsa per long-context efficiente |
| **Early Fusion multimodale** | Llama 4, Qwen3.5 | Testo + immagini + video in un singolo backbone |
| **10M token context** | Llama 4 Scout | Context window 10x più grande dell'attuale max |
| **Thinking Mode** | Qwen3, DeepSeek V3.1+ | Chain-of-thought ragionamento step-by-step |
| **Tool Use nativo** | Qwen3.5, DeepSeek V3.2 | Agentic capabilities built-in nel modello |

> [!IMPORTANT]
> Tutte le licenze sopra sono **compatibili con uso commerciale**.
> Llama 4 ha restrizioni per aziende con >700M utenti attivi mensili.

---

## 4. Implementazione

```cpp
namespace serius::hub {

class ModelDownloader {
public:
    // Download da HuggingFace Hub (API REST, no Python)
    void download_hf(const string& repo_id, const string& output_dir,
                     const string& token = "");

    // Download da URL diretta
    void download_url(const string& url, const string& output_path);

    // Progress callback
    using ProgressFn = function<void(size_t downloaded, size_t total)>;
    void set_progress_callback(ProgressFn fn);

private:
    void download_file(const string& url, const string& path);
    json fetch_model_info(const string& repo_id);
};

class ModelConverter {
public:
    void convert(const string& input_path, const string& output_path,
                 QuantType quant = QuantType::None);
    void safetensors_to_gguf(const string& input, const string& output);
    void requantize(const string& input, const string& output, QuantType quant);
};

} // namespace serius::hub
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/hub/downloader.h/.cpp` | Download HTTP da HuggingFace/URL |
| `src/hub/converter.h/.cpp` | Conversione tra formati |
| `src/hub/model_registry.h` | Lista architetture compatibili |
| `src/hub/progress.h/.cpp` | Progress bar terminale |
| `src/cli/cmd_import.cpp` | Comando CLI `serius import` |
| `src/cli/cmd_convert.cpp` | Comando CLI `serius convert` |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_download_hf` | Download modello piccolo (Qwen-0.5B) da HF |
| `test_convert_safetensors` | Safetensors → GGUF produce file valido |
| `test_quantize_int4` | FP16 → INT4 mantiene qualità (perplexity < +0.5) |
| `test_progress` | Progress bar mostra percentuale corretta |
| `test_model_info` | `serius info` stampa parametri correttamente |
