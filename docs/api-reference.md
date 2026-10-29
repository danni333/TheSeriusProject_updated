# API Reference

OpenAI-compatible HTTP API reference.

## Base URL

```
http://localhost:8080
```

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen2.5-0.5B-Instruct"
}
```

### `GET /metrics`

Prometheus-format metrics.

**Response:**

```
# TYPE serius_requests_total counter
serius_requests_total 42
# TYPE serius_tokens_generated counter
serius_tokens_generated 1337
```

### `GET /v1/models`

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1709654400,
      "owned_by": "serius"
    }
  ]
}
```

### `GET /v1/models/{id}`

Get model details.

**Response:**

```json
{
  "id": "Qwen2.5-0.5B-Instruct",
  "object": "model",
  "created": 1709654400,
  "owned_by": "serius"
}
```

### `POST /v1/chat/completions`

Generate chat completion.

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stream": false
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | required | Model ID |
| messages | array | required | Array of {role, content} messages |
| temperature | float | 0.7 | Sampling temperature (0.0-2.0) |
| max_tokens | int | 256 | Maximum tokens to generate |
| top_p | float | 0.9 | Nucleus sampling (0.0-1.0) |
| stream | bool | false | Enable SSE streaming |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709654400,
  "model": "Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

**Response (streaming):**

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"!"},"index":0}]}

data: [DONE]
```

### `POST /v1/completions`

Text completion (legacy format).

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "prompt": "The capital of France is",
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false
}
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1709654400,
  "model": "Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "text": " Paris.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 2,
    "total_tokens": 10
  }
}
```

### `POST /v1/embeddings`

Generate embeddings for text.

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "input": ["Hello world", "Test sentence"]
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.4, 0.5, 0.6, ...],
      "index": 1
    }
  ],
  "model": "Qwen2.5-0.5B-Instruct",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## Example: Using curl

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```
