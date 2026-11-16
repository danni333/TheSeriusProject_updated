# Report security scan — 2026-11-16

Scan di sicurezza Devin (effort normal, intera repository) su `danni333/TheSeriusProject_updated`, branch `main` (commit `03260d3`).

- Scan ID: `scan-f9b0a6169bd94871bf22537cc99dbd04`
- Problemi aperti: **7** (3 duplicati unificati dallo scanner)
- Threat model: tutti gli attaccanti considerati in scope (client di rete non autenticati, fornitori di modelli malevoli, plugin/MCP/agent malevoli, utenti registrati dell'hub); nessuna area esclusa né rischio accettato. Per questo tutti i problemi mantengono la severità assegnata dallo scanner.

## Riepilogo

| # | Severità | Problema | Componente | CWE | File principali |
|---|----------|----------|------------|-----|-----------------|
| 1 | Critical | Command injection via repo id `hf://` in `/v1/models/load` non autenticato | Inference Server / Model Hub | CWE-78 | `src/hub/downloader.cpp`, `src/server/server.cpp` |
| 2 | High | Ri-registrare uno username esistente restituisce la sua API key | Model Hub | CWE-287 | `src/hub/model_hub_server.cpp` |
| 3 | High | Ogni utente dell'hub può cancellare i modelli di tutti gli altri | Model Hub | CWE-639 | `src/hub/model_hub_server.cpp` |
| 4 | Medium | Route di load/unload modelli senza autenticazione | Inference Server | CWE-306 | `src/server/server.cpp` |
| 5 | Medium | Nomi file di un repo HuggingFace malevolo finiscono in un comando shell e in un path locale | Model Hub | CWE-78, CWE-22 | `src/hub/downloader.cpp` |
| 6 | Low | `SafetensorsReader` usa `data_offsets` senza bounds check | Model Loaders | CWE-125 | `src/io/safetensors.cpp` |
| 7 | Low | `copy_tensor_to_float` non verifica gli offset contro il buffer del file | Model Loaders | CWE-125 | `src/io/safetensors_loader.cpp` |

## Dettaglio

### 1. [Critical] Command injection via repo id `hf://` in `/v1/models/load` non autenticato

**Problema.** Chiunque raggiunga l'inference server può eseguire comandi shell sull'host con una sola richiesta. L'handler di `POST /v1/models/load` (`src/server/server.cpp:637-664`) toglie il prefisso `hf://` e passa il resto a `resolve_hf_gguf` senza controlli. `list_hf_files` (`src/hub/downloader.cpp:128-140`) inserisce il repo id tra doppi apici in un comando `curl` eseguito con `popen`; lo stesso schema è in `fetch_model_info` e `download_file` (`std::system`), usati anche da `serius import`.

**Esempio.** `{"path":"hf://x\"; id > /tmp/p; echo \""}` chiude l'URL quotato ed esegue `id`.

**Esposizione.** Il server ascolta su `0.0.0.0` di default (`src/common/defaults.h:16`), il Dockerfile avvia `serius --server --port 8080` e `k8s/service.yaml` lo espone con un Service `LoadBalancer` sulla porta 80.

**Fix consigliato.** Non costruire comandi shell: usare un client HTTP (cpp-httplib è già una dipendenza, o libcurl) oppure `execvp`/`posix_spawn` con vettore di argomenti. Validare il repo id con `^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$` e rispondere 400 altrimenti. Aggiungere autenticazione a `/v1/models/load`. Test: `resolve_hf_gguf` con repo id contenenti `"`, `$(touch /tmp/x)` e backtick deve rifiutarli senza creare file.

### 2. [High] Ri-registrare uno username esistente restituisce la sua API key

**Problema.** Account takeover. `register_user` (`src/hub/model_hub_server.cpp:243-262`) cerca un utente con lo stesso username e, se esiste, ne restituisce l'API key. `POST /v1/auth/register` (righe 309-328) non richiede autenticazione e rimanda la chiave al chiamante; `authenticate_user` (righe 264-280) la accetta su tutte le route.

**Impatto.** Con `{"username":"alice"}` un attaccante ottiene la chiave permanente di Alice: può cancellare i suoi modelli, caricarne di avvelenati a suo nome e usare ogni route a cui lei ha accesso. Critico se la porta dell'hub (bind su `0.0.0.0`) è raggiungibile da internet; la repo non contiene manifest di deploy per l'hub.

**Fix consigliato.** Rispondere HTTP 409 se lo username esiste e non restituire mai una chiave esistente. Salvare solo l'hash delle API key e generarle con una sorgente crittografica (`getrandom` o `RAND_bytes`) invece di `mt19937_64`. Test: registrare `alice` due volte; la seconda chiamata deve fallire senza chiave.

### 3. [High] Ogni utente dell'hub può cancellare i modelli di tutti gli altri

**Problema.** Broken object-level authorization. `register_user` assegna a ogni nuovo account `read`, `upload` e `delete` (`src/hub/model_hub_server.cpp:254-258`); `check_permissions` (righe 283-296) restituisce true quando un permesso coincide con l'azione, quindi il controllo owner del delete handler (righe 528-533) non blocca nessuno. Il handler rimuove i file con `fs::remove` e cancella i metadati (righe 535-546). Con lo store utenti vuoto ("open mode") qualsiasi token è trattato come `default_user` con tutti i permessi (righe 273-278, 286-287).

**Impatto.** Chiunque crei un account gratuito può cancellare tutti i modelli dell'hub (gli ID sono pubblici via `GET /v1/models`). Critico se l'hub è esposto su internet.

**Fix consigliato.** Dare ai nuovi account solo `read` e `upload`; riservare `delete` globale ad admin assegnati fuori banda. Rimuovere l'open mode o limitarlo a localhost. Test: due utenti, il secondo deve ricevere 403 cancellando un modello del primo.

### 4. [Medium] Route di load/unload modelli senza autenticazione

**Problema.** `POST /v1/models/load` e `DELETE /v1/models/{id}` (`src/server/server.cpp:637-753`) cambiano lo stato del server senza alcun controllo di autenticazione, e il load accetta qualsiasi path del filesystem, con un errore diverso se il path non esiste (righe 692-701).

**Impatto.** Un utente anonimo può scaricare ripetutamente i modelli (servizio offline), forzare il caricamento di file enormi (esaurimento RAM/VRAM) e scoprire quali file esistono sull'host. È anche ciò che rende raggiungibile anonimamente la command injection del problema 1.

**Fix consigliato.** Richiedere una API key admin (es. bearer token da variabile d'ambiente) su `/v1/models/load` e `DELETE /v1/models/*`. Limitare i path caricabili a una directory modelli configurata dopo la canonicalizzazione, con un unico errore generico. Test: senza chiave entrambe le route rispondono 401; `/etc/passwd` viene rifiutato.

### 5. [Medium] Nomi file di un repo HuggingFace malevolo finiscono in un comando shell e in un path locale

**Problema.** Command injection di secondo ordine e path traversal. `list_hf_files` (`src/hub/downloader.cpp:159`) copia il campo `path` dalla risposta JSON di HuggingFace senza controlli; `resolve_hf_gguf` (righe 225-236) lo usa per `models/<base>/<path>` e per l'URL; `download_file` (righe 68-80) li mette tra doppi apici in un comando eseguito con `std::system`.

**Impatto.** Un operatore che carica un repo malevolo (`serius serve --model hf://mallory/Cool-GGUF`) può subire esecuzione di comandi con i propri privilegi (es. furto del token HuggingFace) o scrittura di file fuori da `models/`. Non verificato se HuggingFace accetti nei nomi file `"`, `$` o componenti `..`.

**Fix consigliato.** Togliere la shell dai download (stesso fix del problema 1). Rifiutare nomi remoti con `..`, path assoluti o caratteri fuori da un set sicuro e verificare con `weakly_canonical` che il path finale sia dentro `models/`. URL-encodare il nome file. Test: JSON con nomi `a"$(id).gguf` e `../../x.gguf` devono essere scartati.

### 6. [Low] `SafetensorsReader` usa `data_offsets` senza bounds check

**Problema.** Out-of-bounds read. `parse_header` (`src/io/safetensors.cpp:60-66`) legge `data_offsets[0]` e `[1]` senza validarli; `get_tensor` e `raw_data` (righe 94-113) li sommano a `data_start_` e confrontano solo `end - start` con la dimensione della shape prima del `memcpy`. Se `end < start` la sottrazione va in wraparound.

**Impatto.** Un file `.safetensors` creato ad arte, aperto dal convertitore GGUF o dall'embedding manager dell'hub, può crashare il processo o copiare memoria esterna al mapping nei tensori. Nessuna route remota consegna file a `SafetensorsReader`.

**Fix consigliato.** In `parse_header` richiedere `start <= end <= file_.size() - (8 + header_size)` per ogni tensore e scrivere il controllo dell'header come `header_size > file_.size() - 8` per evitare wraparound. Test: offset oltre la fine del file → il costruttore lancia eccezione.

### 7. [Low] `copy_tensor_to_float` non verifica gli offset contro il buffer del file

**Problema.** Out-of-bounds read su un percorso di codice separato dal problema 6. `copy_tensor_to_float` (`src/io/safetensors_loader.cpp:145-176`) copia dati usando gli offset del file senza il controllo che invece esiste in `tensor_data` (righe 115-122).

**Impatto.** Un modello safetensors malevolo caricato da una vittima può causare un crash o copiare memoria heap nei pesi di quel processo. `hf://` risolve solo file GGUF e `/v1/models/load` accetta solo path già presenti sull'host, quindi non è sfruttabile da remoto.

**Fix consigliato.** Aggiungere lo stesso controllo di `tensor_data`: rifiutare se `data_start < 0`, `data_end > data_.size()` o `data_start > data_end`, prima di ogni copia. Test: `data_offsets` `[1e12, 1e12+16]` → `copy_tensor_to_float` restituisce false.

## Domande aperte del threat model

- L'inference server è pensato per uso locale, rete interna o pubblico? Il manifest k8s usa un `LoadBalancer`, che sui cluster gestiti ottiene di norma un IP pubblico.
- Il model hub (`serius_hub_server`) viene esposto in rete? La repo non contiene manifest di deploy per l'hub; i problemi 2 e 3 diventano critici se è raggiungibile da internet.
