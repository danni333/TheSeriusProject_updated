# Fase 49 — Chat UI (Web Interface)

> Un modello senza interfaccia non è un prodotto.
> Serve una web UI locale (come Open WebUI / text-generation-webui)
> che si connette al server API della Fase 48.

---

## 1. Funzionalità

| Feature | Descrizione |
| --- | --- |
| **Chat multi-turno** | Storico conversazioni con scroll infinito |
| **Streaming in tempo reale** | Token appaiono uno alla volta (SSE) |
| **Selezione modello** | Dropdown per scegliere tra modelli caricati |
| **Parametri di generazione** | Slider per temperature, top_p, max_tokens |
| **Markdown rendering** | Risposte formattate con code blocks, tabelle |
| **Syntax highlighting** | Code blocks con highlight per linguaggio |
| **Dark/Light mode** | Tema automatico o manuale |
| **Export conversazione** | Download come JSON o Markdown |
| **System prompt editor** | Campo per personalizzare il system prompt |
| **Storico sessioni** | Salvataggio locale (localStorage / SQLite) |

---

## 2. Stack Tecnologico

| Componente | Scelta | Licenza |
| --- | --- | --- |
| **Frontend** | HTML5 + Vanilla JS + CSS | — |
| **Build** | Nessuno (file statici serviti dal server C++) | — |
| **Markdown** | marked.js | MIT ✅ |
| **Syntax Highlight** | highlight.js | BSD-3 ✅ |
| **Icone** | Lucide Icons (SVG) | ISC ✅ |

> La UI è servita direttamente dal server C++ come file statici.
> Nessun Node.js, nessun bundler, nessuna dipendenza di build.

---

## 3. Architettura

```text
Browser (localhost:8080)
    │
    ├── GET /                    → index.html (SPA)
    ├── GET /assets/app.js       → JavaScript
    ├── GET /assets/style.css    → CSS
    │
    └── POST /v1/chat/completions  → API (Fase 48)
         (stream: true → SSE)
```

---

## 4. Struttura File

| File | Contenuto |
| --- | --- |
| `ui/index.html` | Pagina principale (SPA) |
| `ui/assets/app.js` | Logica chat, SSE client, rendering |
| `ui/assets/style.css` | Stile completo (dark/light) |
| `ui/assets/marked.min.js` | Markdown parser (vendored) |
| `ui/assets/highlight.min.js` | Syntax highlighting (vendored) |
| `src/server/static_files.cpp` | Serve file statici da /ui/ |

---

## 5. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_ui_loads` | GET / ritorna HTML valido con status 200 |
| `test_ui_chat` | Invio messaggio → risposta streaming visibile |
| `test_ui_model_switch` | Cambio modello da dropdown funziona |
| `test_ui_export` | Export conversazione produce JSON valido |
| `test_ui_responsive` | Layout corretto su mobile (375px) e desktop |
