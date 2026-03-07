# Fase 47 — CLI Completa (Inference, Training, Conversione)

> L'utente deve poter usare il modello da terminale senza scrivere codice.
> Un'unica CLI multi-comando per gestire tutto: chat, inference batch,
> fine-tuning, conversione pesi, benchmarking.

---

## 1. Struttura Comandi

```text
serius chat         → Chat interattiva con il modello
serius generate     → Generazione da prompt (batch o singolo)
serius serve        → Avvia server API HTTP
serius import       → Importa modello da HuggingFace / file locale
serius convert      → Converti pesi (FP16→INT4, safetensors→GGUF)
serius train        → Avvia training / fine-tuning
serius bench        → Esegui benchmark su modello caricato
serius info         → Info su modello (parametri, dimensioni, quant)
serius doctor       → Diagnostica sistema (GPU, RAM, dipendenze)
```

---

## 2. Esempio d'Uso

```bash
# Chat interattiva
serius chat --model ./models/SeriusModel-7B-Q4.gguf --threads 8

# Generazione batch da file
serius generate --model ./models/model.safetensors \
    --input prompts.txt --output results.jsonl --max-tokens 512

# Importa da HuggingFace
serius import --source hf://meta-llama/Llama-3-8B --format gguf --quant q4_k_m

# Conversione locale
serius convert --input model.safetensors --output model-q4.gguf --quant int4-awq

# Avvia server
serius serve --model ./models/model.gguf --port 8080 --gpu auto

# Diagnostica
serius doctor
# Output:
# ✅ GPU: NVIDIA RTX 4090 (24GB VRAM)
# ✅ RAM: 64 GB
# ✅ CUDA: 12.4
# ⚠️ ROCm: non trovato
# ✅ Modelli trovati: 3 in ./models/
```

---

## 3. Modelfile / Template Modelli (Profili Custom)

> Come Ollama: l'utente crea "profili" di modello con system prompt,
> parametri e adapter preconfigurati. Un file di testo = una personalità.

### Formato Modelfile

```text
# models/assistente-italiano.modelfile
FROM models/SeriusModel-7B-Q4.gguf

SYSTEM """
Sei un assistente italiano esperto e cortese. Rispondi sempre in italiano.
Usa un linguaggio chiaro e professionale.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 1024
PARAMETER stop "<|eot_id|>"
PARAMETER stop "###"

# Opzionale: LoRA adapter
ADAPTER lora/italian-v2.bin

# Opzionale: template chat
TEMPLATE """
<|start_header_id|>system<|end_header_id|>
{{.System}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{{.Prompt}}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
```

### CLI

```bash
# Crea un modello da Modelfile
serius create assistente-italiano -f models/assistente-italiano.modelfile

# Usa il modello con il profilo preconfigurato
serius chat --model assistente-italiano
# → Carica automaticamente modello + system prompt + parametri + LoRA

# Lista profili disponibili
serius list
# ├── assistente-italiano    7B Q4    "Assistente in italiano"
# ├── code-helper            3B Q8    "Aiuto programmazione"
# └── base-model             7B Q4    "Modello base senza personalità"

# Copia/condividi profilo
serius export assistente-italiano -o assistente-italiano.modelfile
```

### File aggiuntivi

| File | Contenuto |
| --- | --- |
| `src/cli/modelfile_parser.h/.cpp` | Parser formato Modelfile |
| `src/cli/cmd_create.cpp` | Comando `serius create` |
| `src/cli/cmd_list.cpp` | Comando `serius list` (profili) |

---

## 4. Implementazione

### Parser CLI (senza dipendenze esterne)

```cpp
namespace serius::cli {

struct Command {
    string name;
    string description;
    function<int(int argc, char** argv)> handler;
};

class CLI {
    vector<Command> commands_;
    string version_ = "0.1.0";

public:
    void register_command(Command cmd);
    int run(int argc, char** argv);
    void print_help();
};

} // namespace serius::cli
```

---

## 4. Struttura File

| File | Contenuto |
| --- | --- |
| `src/cli/main.cpp` | Entry point, command dispatch |
| `src/cli/cmd_chat.cpp` | Chat interattiva (readline-style) |
| `src/cli/cmd_generate.cpp` | Generazione batch |
| `src/cli/cmd_serve.cpp` | Avvio server HTTP |
| `src/cli/cmd_import.cpp` | Import modelli HuggingFace |
| `src/cli/cmd_convert.cpp` | Conversione pesi |
| `src/cli/cmd_doctor.cpp` | Diagnostica sistema |
| `src/cli/args_parser.h` | Parser argomenti minimale |

---

## 5. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_cli_help` | `serius --help` stampa tutti i comandi |
| `test_cli_chat` | Chat interattiva carica modello e risponde |
| `test_cli_generate` | Batch generation produce output valido |
| `test_cli_doctor` | Doctor rileva GPU e RAM correttamente |
| `test_cli_import` | Import converte modello HF in GGUF locale |
