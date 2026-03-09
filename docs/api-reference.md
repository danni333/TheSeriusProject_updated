# API Reference (WIP)

Questa pagina documenterà le API esterne di **TheSeriusProject** (es. REST OpenAI‑compat, gRPC, ecc.).

> Stato: in definizione. La struttura seguente è uno scheletro da riempire man mano che l'API si stabilizza.

---

## 1. Principi generali

- Interfaccia ispirata alle API `v1/chat/completions` e simili.
- Focus su:
  - semplicità d'uso,
  - compatibilità con client esistenti,
  - trasparenza dei parametri legati alle performance.

---

## 2. Endpoint esempio — POST `/v1/chat/completions`

**Descrizione**  
Genera una risposta in linguaggio naturale a partire da una conversazione strutturata.

**Request (schema previsto)**:

| Campo       | Tipo    | Default | Descrizione                                      |
|------------|---------|---------|--------------------------------------------------|
| `model`    | string  | —       | ID o percorso del modello da usare (obbligatorio) |
| `messages` | array   | —       | Array di messaggi `{role, content}` (obbligatorio) |
| `temperature` | float | 0.7     | Creatività (0.0 = deterministico)               |
| `max_tokens`  | int   | 512     | Numero massimo di token generati                |
| `top_p`       | float | 0.9     | Nucleus sampling                                |
| `stream`      | bool  | false   | Abilita streaming SSE                           |
| `stop`        | array | []      | Sequenze di stop                                |

**Esempio (bozza)**:

```bash
curl http://localhost:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Ciao!\"}]}"
```

---

## 3. Altri endpoint previsti

- `/v1/models` — elenco dei modelli disponibili.
- `/v1/embeddings` — generazione di embedding.
- `/v1/completions` — completamento testo classico (non chat).

I dettagli verranno aggiunti quando le relative implementazioni saranno stabili.

---

## 4. Generazione automatica della documentazione

L'obiettivo è generare parte di questa pagina a partire dagli header C++ (stile Doxygen) e/o da descrizioni strutturate delle route.

Script previsti:

- `scripts/generate_api_docs.py` (ancora da implementare) per produrre tabelle e sezioni aggiornate automaticamente.

