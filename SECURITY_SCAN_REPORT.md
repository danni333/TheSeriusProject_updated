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

---

# Verifica e remediation — 2026-11-23

Secondo passaggio sul branch `main` dopo le modifiche in `cd1d727` (experimental/ relocation + dependency bumps) e i fix di sicurezza applicati nella stessa sessione. Build Release completa + `ctest`: **475/475 test passati** (un test CUDA richiede `ctest -C Release`; `E2EServer.StreamingSSE` è flaky sotto `-j` parallelo ma passa in isolamento — problema di timing preesistente, non legato ai fix).

## Stato dei 7 problemi originali

| # | Stato | Note |
|---|-------|------|
| 1 | **Risolto** | `src/hub/downloader.cpp` non usa più `popen`/`system`/`curl`: download via `serius::common::run_process` (argv diretto, `CreateProcess`/`posix_spawnp`, niente shell) e repo id validato con `^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`. `src/server/server.cpp` rifiuta repo id `hf://` malformati con 400 prima di chiamare il downloader. Test: `ServerTest.ModelsLoadRejectsMalformedHfRepoId` (passa). |
| 2 | **Risolto** | `register_user` (`src/hub/model_hub_server.cpp`) restituisce 409 se lo username esiste, senza rilasciare la chiave. Le API key sono salvate come hash SHA-256 e generate con CSPRNG (`getrandom`/`/dev/urandom`/`std::random_device`), non più `mt19937_64`. Verificato live: seconda registrazione → `{"error":"username already registered"}` HTTP 409. |
| 3 | **Risolto** | I nuovi account ricevono solo i permessi `read`+`upload`; `delete` globale richiede permesso esplicito (`check_permissions`). Il delete verifica ownership (`uploaded_by`) o permesso `delete`. |
| 4 | **Risolto (opt-in)** | `POST /v1/models/load` e `DELETE /v1/models/{id}` richiedono `Authorization: Bearer $SERIUS_ADMIN_TOKEN` **se la variabile è impostata**; i path di load sono canonicalizzati e, se `SERIUS_MODEL_ROOTS` è impostata, devono ricadere in una delle directory root (403 altrimenti). Vedi rischi residui sotto. |
| 5 | **Risolto** | I nomi file remoti sono validati (niente `..`, path assoluti o caratteri non sicuri), URL-encoded e il path finale è vincolato a `models/`; la scrittura non passa più da shell. |
| 6 | **Risolto** | `src/io/safetensors.cpp`: `header_size` controllato senza overflow, header deve stare dentro il file mappato, ogni `data_offsets` con `start <= end <= dimensione blob`. |
| 7 | **Risolto** | `src/io/safetensors_loader.cpp`: `copy_tensor_to_float` rifiuta offset negativi, oltre `data_.size()` o invertiti prima di calcolare `byte_len`/copiare. |

## Problemi aggiuntivi trovati e corretti in questo passaggio

- **Command injection in `experimental/autoopt`**: `llm_client.cpp` usava `popen("curl ...")` con URL/headerbody utente; `git_manager.cpp` usava `std::system` con messaggio di commit e path del repo; `sandbox.cpp` costruiva comandi `cl`/`g++` via shell con `target_name` non sanitizzato. Tutti migrati a `serius::common::run_process` (nuovo header `src/common/process_utils.h`: argv esplicito, nessuna shell, cattura stdout/stderr su thread separati per evitare deadlock delle pipe) o a `httplib::Client` per le chiamate HTTP.
- **Deadlock potenziale in `run_process`**: la prima versione drenava stdout e stderr in sequenza; un figlio che riempie la pipe di stderr mentre il parent è bloccato su stdout si sarebbe bloccato. Ora entrambi i drain girano su thread dedicati.
- **Docker**: l'immagine esegue ora il binario come utente non-root.
- **Dipendenze FetchContent aggiornate**: nlohmann/json v3.12.0, spdlog v1.17.0, googletest v1.18.0, cpp-httplib v0.67.0, SQLite 3.54.1.

## Terzo passaggio — 2026-11-23 (audit esteso)

Ulteriore sweep su hub server, inference server, loader IO, MCP e SQL. Risultati:

| # | Severità | Problema | Fix |
|---|----------|----------|-----|
| 8 | Medium | **curl senza timeout** in `downloader.cpp`: `http_get`/`http_head`/`download_file` potevano restare bloccati per sempre su connessioni morte, trattenendo thread e request. | Aggiunti `--connect-timeout 10`; `--max-time 30` per le API; `--speed-limit 1024 --speed-time 60` per i download multi-GB (niente cap totale). |
| 9 | Medium | **OOB read in `/v1/embeddings`**: `weights.at(tid, d)` usa indici derivati da `config()` (vocab_size/hidden_size) senza verificarli contro le dimensioni reali del tensore di embedding — `Tensor::at()` non fa bounds check. | Indici ora clampati a `wshape` e il ramo salta se dtype != Float32 o shape non 2D. |
| 10 | Low | **OOB potenziale in `hf_llama_loader.cpp` / `quant_weight.h`**: `take_quant` (permute RoPE) e `repack_quant_for_gpu` assumevano `bytes.size() == out*row_bytes()`; `row_bytes()` tronca per K-quant con `in` non multiplo di 256. | Guardie: skip permute/repack se `rb == 0` o `bytes.size() < out*rb`. |
| 11 | Low | **MCP HTTP endpoint senza autenticazione** (`serius mcp-serve --http`): POST /mcp su `0.0.0.0` era anonimo. | Se `SERIUS_MCP_TOKEN` è impostata, `/mcp` richiede `Authorization: Bearer` (confronto costante); `/health` resta libero. |
| 12 | Low | **`HttpTransport::parse_url`**: `std::stoi` su porta malformata (`host:abc`) lanciava eccezione non gestita. | Convertito in errore "invalid endpoint". |
| 13 | Low | **Registrazione hub senza rate limit**: endpoint anonimo che riscrive tutto `hub_users.json` per ogni chiamata — spam = growth disco + NOTA: aperto solo da loopback in open mode, ma con utenti registrati è esposto a internet. | Rate limit 20 registrazioni/minuto per IP (finestra fissa, cleanup mappa). |
| 14 | Info | **`is_loopback_addr`** non riconosceva `::ffff:127.x.x.x` (IPv4-mapped IPv6). | Normalizzazione del prefisso mappato. |

**Non problemi verificati**: SQL interamente parametrizzato (`sqlite3_bind_*`, LIKE con escape); path traversal su download/delete bloccato dal fatto che `model_id` deve esistere nel DB (ID generati sanitizzati) e `file_name` è forzato a `sanitize_filename` in upload; `config` utente non può iniettare path; static files con canonicalization + check `..`; `Tensor::at` è l'unico uso non protetto trovato (fixato in embeddings); `StdioTransport` non è istanziato da codice raggiungibile.

**Build + test**: Release build completa OK; test mirati (ModelHubServer, ServerTest, Safetensors, MCP, HubDownloader, E2EServer) tutti verdi. `E2EServer.StreamingSSE` resta flaky sotto carico (passa in isolamento, ~60s a freddo) — problema di performance/timing preesistente non legato alla sicurezza.

## Rischi residui (raccomandazioni operative)

1. **`SERIUS_ADMIN_TOKEN` non è obbligatoria**: senza la variabile, `/v1/models/load` e `DELETE /v1/models/{id}` restano anonimi (scelta di retrocompatibilità per l'uso locale). In produzione va **sempre impostata**; valutare di renderla obbligatoria quando il bind non è localhost.
2. **`SERIUS_MODEL_ROOTS` opzionale**: senza di essa qualsiasi path esistente del filesystem è caricabile. In produzione impostare la variabile con le sole directory dei modelli.
3. **Bind `0.0.0.0` di default** (inference server e hub): l'autenticazione mitigata ma non elimina il rischio — esporre solo dietro reverse proxy/auth o bindare a `127.0.0.1`/`SERIUS_HOST` in produzione. Il Service k8s `LoadBalancer` rende il servizio pubblicamente raggiungibile.
4. **`StdioTransport` MCP** (`src/mcp/transport_stdio.cpp`): su POSIX usa `sh -c <command>` perché i comandi MCP stdio sono stringhe di configurazione operatore (es. `npx -y pkg args`); il codice non è ancora istanziato da nessun percorso remoto, ma quando verrà collegato va trattato come configurazione fidata, mai input di rete.
5. **`Sandbox::load_kernel`**: `dlopen`/`LoadLibrary` di path arbitrari è esecuzione di codice per definizione — modulo `experimental/autoopt`, da abilitare solo in ambienti controllati.
6. **Rate limiting**: `/v1/auth/register` è limitato a 20 req/min/IP; le route di autenticazione non hanno un limite sulle *tentate* di login (le chiavi a 128 bit rendono il brute force impraticabile, ma si può aggiungere un limiter anche lì). `SERIUS_MCP_TOKEN` è opzionale come `SERIUS_ADMIN_TOKEN` — da impostare se l'endpoint MCP HTTP è esposto.
7. **Dimensione upload hub**: httplib applica `payload_max_length` di default (100 MB); upload di modelli grandi via hub potrebbero fallire con 413 — limite da alzare esplicitamente se serve, mai rimuoverlo.
