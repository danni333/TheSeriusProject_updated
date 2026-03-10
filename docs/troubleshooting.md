# Troubleshooting — TheSeriusProject

Problemi comuni e soluzioni.

---

## Build

### "Could NOT find GTest"

Il progetto usa **FetchContent** per Googletest, non `find_package(GTest)`. Assicurati che in `tests/CMakeLists.txt` non ci sia `find_package(GTest REQUIRED)`. Se compare l’errore, rimuovi quella riga e riconfigura.

### "Cannot find source file: test_ops.cpp"

Alcuni target di test potrebbero riferirsi a file rimossi o rinominati. Controlla l’elenco in `tests/CMakeLists.txt` e rimuovi o aggiorna i riferimenti ai file inesistenti.

### Errori CUDA / nvcc

- Verifica che **CUDA Toolkit 12.x** sia installato e che `nvcc` sia in PATH.
- Su Windows, usa **Developer Command Prompt for VS** con la versione corretta di MSVC.
- Se non usi la GPU, configura senza CUDA: `cmake .. -DSERIUS_USE_CUDA=OFF`.

### Warning / errori di tipo in operazioni tra size_t e int

Nei test o nel codice, conversioni tra `size_t` e `int` possono generare warning. Usa `static_cast` dove il compilatore segnala il problema.

---

## Runtime

### "Tensor::data() used on non-float32 tensor"

Alcuni layer o dati potrebbero essere in FP16/BF16. Il sampler e parte del codice si aspettano Float32. Verifica il `dtype` dei tensori in ingresso (es. logits) o converti a Float32 prima dell’uso.

### Server non si avvia: "Failed to listen on port 8080"

- La porta potrebbe essere già in uso. Usa un’altra porta: `serius serve --port 8081`.
- Su Linux/macOS, le porte &lt; 1024 richiedono spesso permessi di amministratore; usa una porta &gt; 1024.

### Chat: output strano o token sconosciuti

- Il **tokenizer** predefinito ha un vocabolario molto piccolo (PAD, BOS, EOS, UNK + 256 byte). Per risultati sensati carica un tokenizer addestrato con `tokenizer.load(path)` (quando supportato dalla CLI).
- Il **modello** è inizializzato casualmente se non carichi pesi da checkpoint/GGUF. La chat in locale è una demo; per risposte coerenti servono pesi addestrati.

### Loss NaN durante il training

- Riduci il learning rate.
- Verifica **gradient clipping** (es. `grad_clip = 1.0` in `TrainerConfig`).
- Usa **AMP** con loss scaling (es. `use_amp = true` in `TrainerConfig`).
- Controlla i tensori con `debug::find_first_nan()` o il **TensorLogger** per individuare il primo tensore con NaN/Inf.

---

## API / Server

### 404 su /v1/chat/completions

Assicurati di usare **POST** e non GET. Inoltre l’URL deve essere esattamente `/v1/chat/completions` (con la ‘s’ finale).

### Risposta "Serius API is running. Load a model..."

È il messaggio **stub** quando il server non ha un modello caricato in memoria. L’integrazione con un modello reale e il sampler lato server è in sviluppo; la CLI `serius chat` usa già il modello in processo.

---

## Dove trovare altro aiuto

- **Architettura:** `docs/architecture.md`
- **API REST:** `docs/api-reference.md`
- **Quick start:** `docs/getting-started.md`
- **Roadmap e lacune:** file `todo/faseN+.md`
