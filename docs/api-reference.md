# API Reference — TheSeriusProject

Riferimento endpoint REST (compatibili con stile OpenAI) e uso da client.

---

## Base URL

Di default il server è in ascolto su `http://0.0.0.0:8080`. Sostituisci con `http://localhost:8080` se usi la stessa macchina.

---

## Endpoint

### GET /health

Verifica che il server sia attivo (es. load balancer, Kubernetes).

**Risposta:** `200 OK`

```json
{"status":"ok"}
```

---

### GET /metrics

Metriche per monitoring (uptime, numero di richieste).

**Risposta:** `200 OK`

```json
{"uptime_seconds": 123.45, "request_count": 42}
```

---

### GET /v1/models

Elenco modelli disponibili (stub: restituisce un modello `default`).

**Risposta:** `200 OK`

```json
{
  "object": "list",
  "data": [
    {"id": "default", "object": "model"}
  ]
}
```

---

### POST /v1/chat/completions

Genera una risposta a partire da una conversazione. In questa versione restituisce un messaggio stub fino al caricamento del modello reale.

**Body (JSON):**

| Campo       | Tipo   | Default  | Descrizione        |
|------------|--------|----------|--------------------|
| model     | string | required | ID modello         |
| messages  | array  | required | `[{role, content}]`|
| temperature | float | 0.7    | Sampling (non usato nello stub) |
| max_tokens | int   | 512     | Max token (non usato nello stub) |
| stream    | bool   | false    | Streaming (non implementato)    |

**Esempio:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Ciao!"}]}'
```

**Risposta:** `200 OK`

```json
{
  "id": "serius-stub",
  "object": "chat.completion",
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Serius API is running. Load a model to get real completions."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}
}
```

---

## Uso da SDK OpenAI

Con `OPENAI_BASE_URL=http://localhost:8080` puoi puntare i client OpenAI (Python, Node, ecc.) al server Serius; l’endpoint `/v1/chat/completions` è compatibile con il formato richiesto dal SDK.
