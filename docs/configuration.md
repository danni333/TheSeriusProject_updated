# Configuration (WIP)

Questa pagina descriverà come configurare **TheSeriusProject** tramite file di configurazione e parametri.

---

## 1. Obiettivi di configurazione

- Separare **codice** e **parametri** (modello, hardware, limiti di risorse).
- Consentire profili multipli (es. `dev`, `benchmark`, `low-mem`).
- Permettere override da CLI e variabili d'ambiente.

---

## 2. Formato previsto

Formato target: file JSON o YAML (esempi bozza in JSON).

Esempio di configurazione (bozza):

```json
{
  "model": {
    "path": "models/Qwen2.5-3B-Instruct-Q4.gguf",
    "dtype": "fp16",
    "quantization": "q4_k_m"
  },
  "runtime": {
    "device": "auto",
    "num_threads": 8,
    "gpu": {
      "enabled": true,
      "memory_limit_gb": 8
    }
  },
  "generation": {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

---

## 3. Ordine di precedenza (design)

1. **Valori di default** compilati nel binario.
2. **File di configurazione** (es. `serius.config.json`).
3. **Variabili d'ambiente** (es. `SERIUS_NUM_THREADS`, `SERIUS_DEVICE`).
4. **Flag CLI** (`--num-threads`, `--device`, ecc.).

---

## 4. Stato attuale

- Il sistema di configurazione è in fase di progettazione e integrazione con il runtime.
- Questa pagina verrà popolata con la lista completa delle chiavi supportate man mano che l'implementazione si stabilizza.

