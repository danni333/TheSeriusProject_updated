# Simple weights format (FP32)

Formato binario per caricare e salvare i pesi di un modello transformer **senza quantizzazione**, adatto a modelli semplici e leggeri (~3–4 GB) per verificare il pipeline e fare benchmark.

## Utilizzo

1. **Configurazione**: carica la config da JSON (come per gli altri modelli):
   ```cpp
   #include "io/config_loader.h"
   #include "io/simple_weights_loader.h"
   #include "transformer/model.h"

   auto config = serius::io::load_model_config("configs/model_7b.json");
   serius::model::TransformerModel model(config);
   bool ok = serius::io::load_simple_weights("path/to/model.weights.bin", &model);
   ```

2. **Salvataggio** (per creare file di test o esportare da training):
   ```cpp
   bool ok = serius::io::save_simple_weights("path/to/model.weights.bin", model);
   ```

Il modello deve essere già costruito con la stessa `ModelConfig` usata quando i pesi sono stati salvati; altrimenti il caricamento fallisce (shape mismatch).

## Formato file

- **4 byte**: magic `SERW` (0x57, 0x52, 0x45, 0x53 in little-endian).
- **Segue**: sequenza di float FP32 in ordine fisso, uguale a `TransformerModel::parameters()`:
  - embedding `(vocab_size, d_model)`
  - per ogni layer: `w_q`, `w_dkv`, `w_uk`, `w_uv`, `w_o`, `norm1.gamma`, `norm1.beta`, `norm2.gamma`, `norm2.beta`, `w1`, `b1`, `w2`, `b2`
  - `final_norm.gamma`, `final_norm.beta`
  - `output_proj` `(d_model, vocab_size)`

Nessun metadato (config, shape) è salvato nel file: la config va letta separatamente da JSON.

## Test

I test in `tests/test_simple_weights_loader.cpp` verificano:
- round-trip save/load e uguaglianza dei pesi
- caricamento da file mancante o magic errato (ritorno `false`)
- forward pass dopo load (pipeline funzionante)
- `load_simple_weights(path, nullptr)` ritorna `false`

Eseguire: dalla cartella `build`, `ctest -R SimpleWeightsLoader --output-on-failure`.
