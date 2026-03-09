# CLI Reference (WIP)

Questa pagina raccoglierà i comandi CLI esposti da **TheSeriusProject**.

> Stato: la CLI completa non è ancora definita; i comandi sotto sono un esempio del design previsto.

---

## 1. Struttura generale

Si prevede un binario principale (es. `serius`) con sottocomandi:

- `serius chat` — avvia una sessione di chat da terminale.
- `serius serve` — espone un server HTTP (API OpenAI‑compat).
- `serius bench` — esegue benchmark preconfigurati.
- `serius import` — importa/convert e modelli da formati esterni (es. Hugging Face, GGUF, ecc.).

---

## 2. Esempi di utilizzo (design target)

### 2.1 Chat locale

```bash
serius chat --model path/to/model.gguf
```

Opzioni tipiche:

- `--max-tokens <N>`
- `--temperature <float>`
- `--top-p <float>`

### 2.2 Server HTTP

```bash
serius serve --model path/to/model.gguf --host 0.0.0.0 --port 8080
```

Questo comando dovrebbe rendere disponibili endpoint compatibili con le API OpenAI (vedi `docs/api-reference.md`).

### 2.3 Benchmark

```bash
serius bench --suite matmul
serius bench --suite memory
```

I risultati verrebbero salvati in file `*_bench.txt` nella root del progetto.

---

## 3. Stato attuale

- Non tutte le funzionalità CLI descritte sono ancora implementate.
- Per ora, l'interazione avviene principalmente tramite test/benchmark C++ in `tests/` e binari di esempio.

Questa pagina verrà aggiornata man mano che la CLI prende forma e viene consolidata.

