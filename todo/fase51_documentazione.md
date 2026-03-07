# Fase 51 — Documentazione Utente & API Reference

> Un progetto senza documentazione non esiste per gli utenti.
> Serve una guida rapida, un reference API completo, e tutorial
> per i casi d'uso principali.

---

## 1. Struttura Documentazione

```text
docs/
├── getting-started.md          ← Guida rapida (5 minuti)
├── installation.md             ← Istruzioni installazione dettagliate
├── cli-reference.md            ← Tutti i comandi CLI
├── api-reference.md            ← Endpoint API OpenAI-compat
├── configuration.md            ← Formato config JSON, parametri
├── supported-models.md         ← Modelli compatibili e requisiti
├── gpu-setup.md                ← Setup NVIDIA / AMD GPU
├── troubleshooting.md          ← Problemi comuni e soluzioni
├── contributing.md             ← Come contribuire al progetto
├── architecture.md             ← Architettura interna (per developers)
└── changelog.md                ← Storico versioni
```

---

## 2. Getting Started (Priorità Massima)

La pagina più importante. L'utente deve passare da zero a chat funzionante in 5 minuti:

```markdown
# Quick Start

## 1. Scarica il binario
    curl -L https://github.com/.../releases/latest -o serius
    chmod +x serius

## 2. Scarica un modello
    ./serius import --source hf://Qwen/Qwen2.5-3B-Instruct --quant q4_k_m

## 3. Chatta!
    ./serius chat --model models/Qwen2.5-3B-Instruct-Q4.gguf

## 4. (Opzionale) Avvia la Web UI
    ./serius serve --model models/Qwen2.5-3B-Instruct-Q4.gguf
    → Apri http://localhost:8080
```

---

## 3. API Reference

Generato automaticamente dagli header C++ (Doxygen-style) + documentazione manuale per endpoint REST.

### Formato

```text
## POST /v1/chat/completions

Genera una risposta basata sulla conversazione fornita.

### Parametri

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| model | string | required | ID del modello |
| messages | array | required | Array di messaggi {role, content} |
| temperature | float | 0.7 | Creatività (0.0 = deterministico) |
| max_tokens | int | 512 | Max token generati |
| stream | bool | false | Abilita streaming SSE |
| top_p | float | 0.9 | Nucleus sampling |
| stop | array | [] | Sequenze di stop |

### Esempio
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"default","messages":[{"role":"user","content":"Ciao!"}]}'
```

---

## 4. Struttura File

| File | Contenuto |
| --- | --- |
| `docs/*.md` | Documentazione in Markdown |
| `scripts/generate_api_docs.py` | Genera API reference da header |
| `README.md` | Overview progetto + link a docs |

---

## 5. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_docs_links` | Tutti i link interni nei .md funzionano |
| `test_quick_start` | Getting Started funziona su macchina pulita |
| `test_api_examples` | Ogni esempio curl nella docs produce risposta valida |
