# Fase 55 — Stress Testing & Load Testing

> Un server che funziona con 1 utente non è pronto per la produzione.
> Serve testare i limiti: massimo throughput, stabilità sotto carico,
> leak di memoria, e recovery dopo errori.

---

## 1. Scenari di Test

### 1.1 Ramping Load Test

```text
Scenario: aumenta gradualmente il carico fino al punto di rottura

Utenti concorrenti:  1 → 2 → 4 → 8 → 16 → 32 → 64 → 128
Durata per step:     60 secondi
Prompt:              512 token (fixed)
Max tokens:          256

Metriche tracciate:
  - TPS aggregato (deve salire, poi plateau)
  - P99 latency (deve restare < 10s)
  - Error rate (deve restare 0% fino a saturazione)
  - VRAM usage (non deve superare il limite fisico)
```

### 1.2 Soak Test (durata prolungata)

```text
Scenario: carico costante per ore, cerca memory leak

Durata:              4 ore
Utenti concorrenti:  8 (costanti)
Richieste:           1 ogni 5 secondi per utente 

Verifica:
  - VRAM non cresce nel tempo (leak detection)
  - RAM non cresce nel tempo
  - TPS stabile (no degradazione)
  - Zero crash / zero riavvii
```

### 1.3 Spike Test

```text
Scenario: picco improvviso di traffico

Fase 1 (normal):     4 utenti per 30 sec
Fase 2 (spike):      64 utenti per 60 sec
Fase 3 (recovery):   4 utenti per 30 sec

Verifica:
  - Il server non crasha durante lo spike
  - Le richieste in eccesso ricevono 429 (non timeout)
  - Recovery a TPS normale dopo lo spike
```

### 1.4 Chaos Test

```text
Scenario: errori casuali durante il funzionamento

Test:
  - Input vuoto → risposta di errore, non crash
  - Context length > max → errore graceful
  - Token malformati → errore 400, non segfault
  - Disconnessione client durante streaming → cleanup KV cache
  - OOM simulato → error handling, non abort()
```

---

## 2. Implementazione

```cpp
namespace serius::stress {

struct StressConfig {
    int concurrent_users;
    int duration_seconds;
    int ramp_step_seconds = 60;
    string prompt = "Explain quantum computing in detail...";
    int max_tokens = 256;
    string server_url = "http://localhost:8080";
};

struct StressResult {
    float avg_tps;
    float p50_latency_ms;
    float p99_latency_ms;
    float error_rate;
    size_t total_requests;
    size_t total_errors;
    vector<pair<float, float>> tps_timeline;  // (timestamp, tps)
    vector<pair<float, size_t>> vram_timeline; // (timestamp, bytes)
};

class StressRunner {
public:
    StressResult run_ramp(const StressConfig& cfg);
    StressResult run_soak(const StressConfig& cfg);
    StressResult run_spike(const StressConfig& cfg);
    void run_chaos(const string& server_url);

private:
    // Ogni worker simula un utente
    void worker(const string& url, const string& prompt,
                int max_tokens, atomic<bool>& running,
                StressResult& result, mutex& mtx);
};

} // namespace serius::stress
```

---

## 3. Report Automatico

```text
═══════════════════════════════════════════════════════
  STRESS TEST REPORT — 2026-03-05 02:00:00
═══════════════════════════════════════════════════════

  Model:           SeriusModel-7B-Q4
  GPU:             NVIDIA RTX 4090 (24 GB)
  Test Type:       Ramping Load (1→128 users)
  Duration:        8 min

  ┌──────────┬────────┬──────────┬──────────┬────────┐
  │ Users    │ TPS    │ P50 (ms) │ P99 (ms) │ Errors │
  ├──────────┼────────┼──────────┼──────────┼────────┤
  │ 1        │ 84.2   │ 12       │ 15       │ 0      │
  │ 4        │ 312.5  │ 13       │ 18       │ 0      │
  │ 8        │ 580.1  │ 14       │ 25       │ 0      │
  │ 16       │ 890.3  │ 18       │ 45       │ 0      │
  │ 32       │ 1120.0 │ 29       │ 120      │ 0      │
  │ 64       │ 1180.5 │ 54       │ 850      │ 12     │
  │ 128      │ 1150.2 │ 111      │ 4200     │ 89     │
  └──────────┴────────┴──────────┴──────────┴────────┘

  ✅ Peak throughput: 1180 TPS @ 64 users
  ⚠️ P99 > 1s above 64 users
  ❌ Errors start at 64 concurrent users

  VRAM: stable at 5.2 GB (no leak detected)
  RAM:  stable at 2.1 GB (no leak detected)
═══════════════════════════════════════════════════════
```

---

## 4. CLI

```bash
# Ramping load test
serius stress --type ramp --max-users 64 --duration 300 \
    --model model.gguf --output stress_report.json

# Soak test (4 ore)
serius stress --type soak --users 8 --duration 14400 \
    --model model.gguf --output soak_report.json

# Chaos test
serius stress --type chaos --server http://localhost:8080
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/stress/stress_runner.h/.cpp` | Runner per ramp/soak/spike/chaos |
| `src/stress/worker.h/.cpp` | Worker thread (simula utente) |
| `src/stress/report.h/.cpp` | Report formattato (ASCII table + JSON) |
| `src/stress/leak_detector.h/.cpp` | Trend analysis VRAM/RAM |
| `src/cli/cmd_stress.cpp` | Comando CLI `serius stress` |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_ramp_basic` | TPS sale con utenti, plateau a saturazione |
| `test_soak_no_leak` | 1h di carico → VRAM drift < 1 MB |
| `test_spike_recovery` | TPS torna a baseline dopo spike |
| `test_chaos_no_crash` | Input malformati → errori graceful, no crash |
| `test_report_format` | Report JSON e ASCII table validi |

---

## Stato avanzamento

**Fatto e verificato:**

- `src/stress/`: `StressRunner` (ramp/soak/spike/chaos), `StressWorker`
  (richieste HTTP via cpp-httplib), `report` (ASCII + JSON),
  `leak_detector` (drift/slope VRAM+RAM).
- CLI `serius stress` (`src/cli/cmd_stress.cpp`) con `--type`, `--users`,
  `--duration`, `--output`, `--format json`.
- `tests/test_stress.cpp`: 24 test verdi, inclusi test di integrazione con
  mock server httplib locale (`StressRunnerTest.*`) che coprono i 5 scenari
  del piano di verifica: ramp, soak, spike recovery, chaos, report.

**Esecuzioni reali completate (2026-11-26/27, SmolLM2-135M Q4_0, GPU offload):**

- **Soak 4h** (`stress_results/soak_nightly.json`, orchestrata da
  `scripts/soak_nightly.ps1` + Scheduled Task `SeriusSoakNightly`):
  133.177 richieste, **0 errori**, 8,3M token, 576,8 tok/s medi
  (597→577 tok/s, drift ~3%), p50 328ms / p99 584ms, VRAM costante
  1,65 GB — nessun leak. Il leak detector segnala RAM ma è falso
  positivo (ultimo campione catturato durante lo shutdown del server).
- **Ramp 1→128** (`stress_results/ramp_fixed_1_128.json`): TPS aggregato
  scala fino a ~38 tok/s a 8 utenti; a ≥16 utenti error rate ~100% —
  tetto reale dovuto alla generazione serializzata su `inference_mutex_`
  + timeout client 30s, non starvation HTTP.
- **Chaos**: 3/3 probe passati — ha portato alla scoperta del bug KV cache
  >4096 token (fix `ensure_kv_capacity`) e del context-length guard
  (`prompt_exceeds_context` → HTTP 400).

**Fix infrastrutturali nati dai test:**

- `CPPHTTPLIB_THREAD_POOL_COUNT=64` (era ~7 slot)
- `ServerConfig.health_port` + secondo `httplib::Server` per
  `/health`+`/metrics` (`--health-port`, usato dallo stress via `--health-url`)
- Retry su errore `accept()` in `Server::run()` (connessioni abortite
  sotto carico non sono più fatali)
- `canonical_url()` mappa `localhost`→`127.0.0.1` (localhost risolve a
  ::1 IPv6 e il server bind solo IPv4)
- Bug critico trovato grazie ai test: permutazione RoPE GGUF errata per
  arch=llama (`hf_llama_loader.cpp`) — tutti i GGUF Llama/Mistral
  producevano gibberish; fix verificato contro safetensors.

**Rimasto (opzionale):**

- Spike test e chaos esteso (rate-limit/auth non implementati — Fase 48).
- SSE/streaming non stressato (il worker usa solo richieste non-stream).
