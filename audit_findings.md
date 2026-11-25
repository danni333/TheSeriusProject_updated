# Audit Findings — TheSeriusProject

Legenda severità: **CRITICO** = memory corruption / wrong results silenziosi | **ALTO** = bug reale in path comune | **MEDIO** = bug in edge case o fragilità | **BASSO** = igiene

---

## Fase 1 — Tensor + Memory lifetime

### CRITICO

**F1.1 — `fill()`/`zeros()`/`ones()` scrivono `size_` elementi contigui senza controllare layout né sporcare la GPU**
`src/tensor/tensor.cpp:802-825`
- `fill` fa `std::fill(data_, data_ + size_, ...)` su QUALSIASI tensore, anche non contiguo (slice/permute/expand). Su una view strided sovrascrive elementi fuori dalla view logica → corruzione di dati adiacenti o OOB.
- Su tensore `location_==SYNCED` o `GPU`: chiama `syncToHost()` ma NON aggiorna `location_` → la copia GPU resta "fresh" ma con dati vecchi. I kernel GPU successivi leggono dati stale silenziosamente.

**F1.2 — `syncToHost()`/`syncToDevice()` copiano `size_*element_size()` byte come blocco contiguo**
`src/tensor/tensor.cpp:1188` (h2d), `1238` (d2h)
- Per tensori non contigui (strided view, slice di dim interna, permute) o broadcast (`expand`, stride 0) il memcpy lineare copia regioni sbagliate e, per `expand`, `size_` è il prodotto della shape espansa >> buffer reale → **OOB read/write**.

**F1.3 — `operator+=` / `operator-=` fast path: nessun sync, nessun dirty-tracking**
`src/tensor/tensor.cpp:895-899, 996-1000`
- leggono `data_`/`other.data_` grezzi senza `syncToHost()`: se un operando è GPU-dirty si usano dati host stale.
- scrivono `data_` senza impostare `location_=CPU`: se `location_==GPU`, `syncToDevice()` in futuro salta ("already dirty on GPU") → divergenza permanente host/GPU.

**F1.4 — `operator+ - * /` (Tensor e scalar) fast path leggono `data_` grezzo senza sync**
`src/tensor/tensor.cpp:930-936, 973-979, 1030-1036, 1072-1078, 1097-1102, ...`
- `this->data_` e `other.data_` usati direttamente: dati stale se gli operandi risiedono su GPU.
- Branch broadcast (`(...,D) op (D,)`) usa indicizzazione flat `i*d+j` assumendo contiguità senza verificarla → risultati errati su view non contigue.

**F1.5 — Copie GPU (`memcpy_d2d`, `copy_layout`) assumono contiguità**
`src/tensor/tensor.cpp:179` (copy ctor), `489` (contiguous())
- Stessa classe di F1.2: per tensori GPU strided/espansi copia buffer errati o OOB.

### ALTO

**F1.6 — `alloc_gpu()` marca `location_=GPU` anche quando il buffer GPU è appena allocato e non inizializzato**
`src/tensor/tensor.cpp:1198-1209`
- Se `data_gpu_==nullptr` ma esistevano dati host validi, dopo `alloc_gpu()` `syncToHost()` copia D2H dal buffer non inizializzato → garbage. Richiede che il chiamante sovrascriva tutto su GPU prima di ogni read host — contratto implicito non verificato.

**F1.7 — Costruttori non controllano il ritorno di `malloc`/`pool->allocate` prima di `memset`**
`src/tensor/tensor.cpp:65-66, 86-91, 144-145` (copy ctor)
- `backend->malloc`/`pool_->allocate` possono restituire `nullptr` → `memset(nullptr,...)` crash. In copy ctor `data_` non è mai memset: se `other.data_==nullptr && other.data_gpu_!=nullptr` e GPU backend non disponibile → tensore host con dati non inizializzati marcato CPU.

### MEDIO

**F1.8 — Accessor `at(r,c)` / `at(d0,d1,d2)` / `at(d0..d3)` non verificano `ndim()`**
`src/tensor/tensor.cpp:355-410` — `strides_[1]` ecc. su tensore con meno dimensioni → OOB read del vector strides.

**F1.9 — Nessun bounds check in `compute_offset`, `item`, `set_item`, `operator[]`**
`tensor.cpp:37-44, 412-456` — indici fuori range → accesso memoria arbitrario silenzioso. `at(vector)` controlla solo il numero di indici, non i valori.

**F1.10 — `rand()` usa `static std::mt19937` condiviso e mutato**
`tensor.cpp:843-845` — stato mutabile globale: non thread-safe, e la sequenza è condivisa tra tutti i tensori (potrebbe essere voluto, ma la mutazione non è sincronizzata).

**F1.11 — `Tensor(void*, shape, strides)` non valida `strides.size()==shape.size()`**
`tensor.cpp:110-124` — `compute_offset` legge `strides_[i]` oltre fine vector → UB.

**F1.12 — `set_item`/`at()`/`operator[]` non-const marcano `location_=CPU` anche da `SYNCED`, lasciando la copia GPU stale**
`tensor.cpp:319-321, 336-338, 430-432`
- Corretto come conservative-mark, ma ogni consumer che legge `raw_data_gpu()` senza `syncToDevice()` ottiene dati vecchi. Il protocollo richiede che TUTTI i lettori GPU passino da sync — da verificare in Fase 3/4.

**F1.13 — MemoryPool non è thread-safe; free-list senza coalescing**
`src/memory/memory_pool.cpp`
- `allocate`/`deallocate` su istanze condivise fra thread → race. `get_thread_local()` esiste ma i pool espliciti (passati a `Tensor(shape, pool)`) possono essere condivisi.
- Le free-list non fondono blocchi adiacenti → frammentazione nel tempo.
- Quando `allocate` salta al blocco successivo, la coda del blocco corrente resta "stranded" per sempre (non recuperabile se non via deallocate esplicito).
- `ScopedArena` non salva `peak_used_`/contatori (solo statistiche) e il rewind invalida libere entries — gestito, ma i tensori allocati dentro l'arena e sopravvissuti oltre il suo scope diventano use-after-rewind (contratto implicito).

### BASSO

**F1.14 — Move ctor non resetta `is_pinned_`/`is_gpu_pool_`** (move-assign sì) — innocuo oggi perché `owns_*` vengono azzerati, ma incoerente. `tensor.cpp:193-214`.

**F1.15 — `sum()`/`contiguous()`/operatori lenti sono O(n·ndim) con `item()`/`set_item()`** — corrette ma lente; `item()` chiama `syncToHost()` a ogni elemento (early-return dopo il primo, ok).

**F1.16 — `operator/(float)` usa moltiplicazione per reciproco** — `x * (1/s)` ≠ `x / s` in floating point; discrepanza minima ma reale vs divisione elemento. `tensor.cpp:1163-1167`.

**F1.17 — `MemoryPool::used()`/`peak_used_` aggiornati in modo non atomico e `deallocation_count_` mai esposto** — igiene.

---

## Fase 2 — Autograd

### ALTO

**F2.1 — `cross_entropy` non valida i target: cast diretto a `size_t` + indici OOB**
`src/autograd/ops_basic.cpp:408, 416-418, 445`
- `targets_raw[i]` → `static_cast<size_t>` senza controlli (a differenza di `masked_cross_entropy` che valida). Target negativo o >= vocab → `logits_slice[target_idx]` e `probs_slice[target_idx]` = accesso OOB in forward E backward.
- `ndim` non validato: `ndim==1` → `shape()[1]` OOB; `ndim>3` o `ndim==0` → branch else legge dimensioni sbagliate.
- `total_elements==0` → divisione per zero → loss NaN silenziosa.

**F2.2 — Ops elementwise (`relu`, `gelu`, `silu`, `masked_fill`, `softmax`, `layernorm`, `rmsnorm`, `rope`) assumono tensori contigui**
`ops_basic.cpp:184-186, 219-224, 263-266, 630-633`; `ops_norm.cpp:72-83, 155-162`; `ops_rope.cpp:40-49`
- Usano `data[i]`/`at()` flat su `input_node->data` senza `contiguous()`. Su input non contigui (output di `transpose`/`permute`/`slice`/`expand`) calcolano su elementi sbagliati → forward E backward errati silenziosamente. `mul_scalar` usa `copy()` ed è corretto — inconsistenza.

**F2.3 — `masked_fill` divide per `mask_size` senza controllare che sia >0**
`ops_basic.cpp:631, 653` — mask vuoto → `i % 0` → UB/crash.

**F2.4 — `rope` non valida `d_head` pari né `d_model % d_head == 0`**
`ops_rope.cpp:31-51` — `d_head` dispari → `i_idx+1` strava l'head e corrompe l'elemento adiacente; `d_model % d_head != 0` → scritture oltre l'head nel row (o OOB sull'ultima riga). `d_head==0` → `i_idx += 2` infinita? no, loop `h_idx += d_head` con d_head=0 → **loop infinito**.

### MEDIO

**F2.5 — `rope()` deduce batch/seq solo da `query` e supporta solo ndim 2/3**
`ops_rope.cpp:128-130` — input 4D (B,H,S,D) viene scambiato per [S,D] (seq_len = shape()[0] = B!) → applica RoPE su dimensioni sbagliate. Key shape mai verificata vs query.

**F2.6 — `matmul` complex path (>3D): nessun broadcasting batch dims**
`ops_basic.cpp:19-50` — `out_shape` prende la shape del ndim maggiore e forza `last=last_b`; se i batch dims non combaciano (es. [2,1,M,K] x [3,K,N]) produce shape sbagliata o `reshape` che lancia. `batch_a`/`batch_b` calcolati indipendentemente → `batched_matmul` con batch mismatch.

**F2.7 — `reshape`/`transpose` ops restituiscono Node il cui `data` è una VIEW non-owner del tensore input**
`ops_basic.cpp:583-584, 608`
- `Tensor::reshape/permute` ritornano view (`owns_data_=false`) che punta al buffer del nodo input. Se l'input Node viene distrutto prima dell'output → dangling pointer. La sopravvivenza dipende dall'ordine di `cleanup_graph` — fragilità reale.

**F2.8 — `contiguous()` op ritorna `input_node` stesso quando già contiguo**
`ops_norm.cpp:18-20` — il "nuovo" nodo è in realtà l'input: nessun `forward_fn`, nessun mark nel grafo. Il graph compiler può non poter eseguire il piano se incontrerà un contiguous-identity.

**F2.9 — `backward()` non azzera i gradienti e non controlla `requires_grad` della loss**
`engine.cpp:27-49` — chi chiama deve aver fatto `zero_grad()` su tutti i nodi; `loss_node->grad.ones()` resetta solo la loss. Documentato? comunque fragile. topo_sort ricorsivo → stack overflow su grafi molto profondi.

**F2.10 — `Node::grad` è sempre Float32**
`node.cpp:11,20` — se `data` è Float16/BF16, grad ha dtype diverso dai dati: gli ops scrivono `grad.data()` come float* → ok per float, ma confronti dtype assenti in tutto l'autograd.

### BASSO

**F2.11 — `add` backward riduce solo lungo dim 0** (`ops_basic.cpp:157-159`) — coerente col broadcast limitato di `Tensor::operator+` (solo coda 1D), ma broadcast tipo [1,D]+[B,D] non supportato da nessuna delle due parti.

**F2.12 — matmul backward: branch `grad.ndim() < grad_a.ndim()` è dead code** (`ops_basic.cpp:96-100, 131-135`) — `grad_a` ha sempre la shape di `input_a->data` → mai vero; innocuo ma inganna.

**F2.13 — `fused_masked_softmax`/`layernorm`/`fused_add_layernorm`/`fused_bias_gelu` CUDA: `shape()[3]`/`shape().back()` senza check ndim** (`ops_norm.cpp:296-298, 46, 368`) — input non-4D per softmax o scalare per norm → OOB su `shape()`.

**F2.14 — `gelu` backward approssima la derivata della tanh-approx** — coerente col forward (stessa approssimazione), ok; segnalato solo per completezza.

**F2.15 — Node lifecycle manuale (`new`/`delete` sparsi)** — nessuna ownership nel grafo; ogni leak di Node (es. `target_node` fixato in trainer) resta responsabilità del chiamante. Pattern fragile ma noto.

---

## Fase 3 — BLAS + dispatch + quant

### ALTO

**F3.1 — Tutto il BLAS assume inner stride == 1: view trasposte/permute → risultati silenziosamente errati**
`src/blas/dispatch.cpp:94-96, 205-207, 273-275, 340-342`; kernel `matmul_scalar.cpp:44`, `matmul_avx2.cpp:31,34,108`, `matmul_avx512.cpp:29,48`, `matmul_cuda.cu:268`
- I kernel indicizzano `A[i*lda+k]`, `B[k*ldb+j]`, `C[i*ldc+j]` con j/k contigui: onorano solo il leading stride, non `strides()[1]`. Un tensore `[M,K]` ottenuto da `transpose()` ha `strides={1,M}` → `lda=1` e accessi `A[i+k]` → garbage.
- Il path CUDA di `batched_matmul` si protegge con `contiguous()` (dispatch.cpp:294-301) ma `matmul`, `matvec` e **tutto il path CPU** non verificano nulla.

**F3.2 — Nessuna verifica di dtype coerente tra operandi in `blas::matmul`/`matvec`/`batched_matmul`**
`dispatch.cpp:82-84, 153-154, 218, 263-271`
- CPU controlla solo `A.dtype()`; CUDA passa `A.dtype()` a `matmul_cuda` e tratta B e C allo stesso tipo. A=FP32 + B=FP16 → kernel interprets B come FP32 → garbage silenzioso. `Tensor mat_c` eredità dtype di A.

**F3.3 — `int4::quantize_block` caso degenere (tutti valori uguali) dequantizza a 0 invece che a x_min**
`src/quant/int4.cpp:23-30, 74`
- `x_max==x_min` → scale=1, zero_point=0, packed=0 → `dequantize_block` restituisce `(0-0)*1 = 0` per ogni elemento. Un blocco costante a 0.5 diventa 0.0. Usato da `awq_quantize`/`quantize_tensor`.

### MEDIO

**F3.4 — `matvec` CUDA ignora gli strides di x e y**
`dispatch.cpp:218-219` — passa `ldb=1, ldc=1` hardcoded: view non contigue → garbage su GPU.

**F3.5 — `matvec_q` ritorna silenziosamente senza scrivere y**
`matvec_quant.cpp:1233-1235, 1297-1300` — `!w.valid()` o tipo non supportato (`fn==nullptr`) → y resta non inizializzato, nessun errore.

**F3.6 — `row_bytes()` per Q4_K/Q5_K assume `in % 256 == 0`**
`quant_weight.h:72-73` — se `in` non è multiplo di 256, `row_bytes` è troncato → kernel leggono oltre/fuori righe. `repack_quant_for_gpu` esce silenziosamente nello stesso caso.

**F3.7 — `float_to_half` tronca senza round-to-nearest e flusha tutti i subnormal a 0**
`conversions.cpp:28-36` — errore fino a ~2^-11 relativo + underflow aggressivo. `half_to_float` gestisce le subnormal via loop su `uint32_t exp` che underflowa ma torna corretto per aritmetica modulare — corretto ma fragile.

**F3.8 — GPTQ: propagazione errore solo entro `group_cols`, non su tutte le colonne rimanenti**
`gptq.cpp:194-225` — il feedback `W[:,k] += err*H_inv[j,k]/H_inv[j,j]` si ferma a `col_end` del gruppo → approssimazione di GPTQ (perdita di precisionità, non crash). `L[j*K+j]==0` possibile se damp=0 → divisione per zero silenziosa (NaN propagato).

**F3.9 — `dequant_matmul`/gptq/awq decode: pairing nibble assume K pari**
`dequant_matmul.cpp:20-27`, `gptq.cpp:284-292` — `packed_idx=(j*K+k)/2` con K dispari → allineamento nibble invertito sulle righe dispari.

### BASSO

**F3.10 — `get_cpu_info()` static init non thread-safe** (benigno, scritture idempotiche). `dispatch.cpp:25-26`.

**F3.11 — CPU stream lambda cattura `&A,&B,&C` per riferimento** — richiede che il chiamante mantenga i tensori vivi fino a `sync_stream` (documentato in commento, ma fragile). `dispatch.cpp:177`.

**F3.12 — `fp16_to_fp32_fast` non gestisce subnormal/inf/nan** — documentato come ok per scale ggml, ma è una limitazione silenziosa se riusato altrove. `matvec_quant.cpp:33-42`.

**F3.13 — `element_wise_*`/`scale`/`apply` non supportano GPU ne' broadcasting** — solo CPU float32 contiguo; su GPU-dirty tensor fa syncToHost via data() e scrive host (corretto ma lento); dtype non-Float32 → throw da `data()`.

---

## Fase 4 — Transformer + inference (parziale: transformer/*, sampler, sezione attention/KV di llama_inference.cpp)

### ALTO

**F4.1 — `TransformerModel::embedding_lookup`/`do_embedding_backward` non validano `token_id`**
`src/transformer/model.cpp:82-91, 103-111`
- `ids.at()` → `static_cast<size_t>` → `embedding->data.at(token_id, dim)` senza bound check vs `vocab_size`. Token id fuori range → lettura OOB della embedding matrix in forward e **scrittura OOB in `embedding->grad.at()`** in backward → heap corruption nel training. `LLaMAModel::forward` e `QwenModel::forward` clampano (`tid>=vocab→0`) — comportamento incoerente tra modelli.

**F4.2 — `MultiHeadAttention` con cache non applica causal mask**
`src/transformer/attention.cpp:117-119`
- `cache != nullptr` → `softmax(mul_scalar(scores, scale))` senza mask. OK per decode step-by-step (1 token), ma se il cache path viene usato con `seq_len>1` (prefill su cache esistente) ogni query attende posizioni FUTURE → output errato silenzioso.

**F4.3 — `KVCache` rolling: `get_keys/get_values` ritornano slice `[0:len]` ma i dati più vecchi non sono in testa dopo il wrap**
`src/transformer/kv_cache.cpp:31-46, 68-82`
- In rolling mode le posizioni sono `(current_len+i)%max_len`: dopo il wrap l'ordine cronologico nel buffer circolare non parte da 0 → la slice restituisce K/V in ordine sbagliato → attention su chiavi errate.

**F4.4 — `LLaMABlock::forward`/`QwenBlock::forward` cancellano `key_rs`/`value_rs` mentre sono ancora figli nel grafo autograd**
`src/transformer/llama_model.cpp:122-125`, `src/transformer/qwen_model.cpp:126-129`
- `delete key_rs` dopo averli usati in `transpose` — i `backward_fn` delle ops a valle catturano il puntatore `input_node` → use-after-free se `backward()` viene mai eseguito su questo grafo (es. test di training). In inferenza pura non morde, ma è una bomba per chi usa `forward()` in autograd.

### MEDIO

**F4.5 — `PagedKVCache::reorder` usa `beam_indices.size()` come batch size**
`paged_kv_cache.cpp:93-105` — se `beam_indices.size() != keys_->shape()[0]` scrive righe sbagliate o legge OOB (`src` validato contro la dimensione sbagliata).

**F4.6 — `PagedKVCache` pool GPU: `set_item` marca `location_=CPU` e lascia la copia GPU stale**
`paged_kv_cache.cpp:109-112` — se `k_pool_`/`v_pool_` sono device tensors, ogni write va su host e la GPU non viene aggiornata; `gather_*` usa `item()` (host) quindi coerente, ma un consumatore GPU del pool leggerebbe dati vecchi.

**F4.7 — `rope()` autograd: `d_head==0` → loop infinito; `d_model%d_head!=0` o d_head dispari → scritture oltre head**
`ops_rope.cpp:31-51` — già segnalato F2.4; qui confermato che i modelli LLaMA/Qwen lo usano davvero. `model::apply_rope` (positional.cpp) valida correttamente — **due implementazioni RoPE divergenti**.

**F4.8 — `TransformerBlock::forward` (non-optimized) e `forward_optimized` divergono nel grafo**
`block.cpp:34-73` — forward_optimized usa fused_add_layernorm + fused_bias_gelu: stesso risultato matematico ma backward diverso; `forward()` resta per MoE path (moe_layer_->forward senza fused). Coerenza ok, ma attenzione: `fused_add_layernorm` CPU fa fallback `layernorm(add(...))` creando layernorm node con name "layernorm" — il grafo perde il nome fused (cosmetico).

**F4.9 — `LLaMAModel::forward`/`QwenModel::forward` creano causal_mask `[S,S]` a ogni layer, ogni forward**
`llama_model.cpp:131-136`, `qwen_model.cpp:135-140` — allocazione + fill O(S²) per layer per token batch: con S=4096 e 32 layer = 512M scritture per forward. Cache mancante → lentissimo.

**F4.10 — `Sampler::sample` — NaN nei logits non gestiti**
`sampler.cpp:55-57, 82-94` — `fkey(NaN)` produce chiave altissima → NaN può essere selezionato come token in greedy e domina la distribuzione in sampling. `max_element` con NaN restituisce la prima occorrenza NaN.

**F4.11 — `sample_gpu_candidates`/`sample` assumono logits contigui `[rows, vocab]`**
`sampler.cpp:28-36, 286-291` — `ptr += (seq_len-1)*vocab` e `lgp = gp + row*vs` ignorano strides; logits da view trasposta → riga sbagliata.

**F4.12 — KV cache inference: `key_buf.data() + i*local_kv_dim` assume layout `[seq, dim]` contiguo**
`llama_inference.cpp:669-670` — se il producer del key_buf non è contiguo/permuta → scramble silenzioso. Anche `ensure_kv_capacity` copia fp32 con `memcpy` del numero di elementi — ok perché k è [cap,dim] contiguo.

### NOTE / da verificare

- `llama_inference.cpp` (righe 879-1362: `llama_forward_inference`, `llama_forward_single_token`, compute_logits) e `llama_inference_cuda.cpp` (1358 righe), `speculative.cpp`, `llama_gpu_offload.cpp` **non ancora letti** — copertura parziale, da completare.
- `compute_head_attention`: `npos = pos_offset + qi + 1` corretto per causale incrementale; `scores` mascherate oltre npos con -1e9 — ok. `stack_scores[1024]` fallback heap — ok.
- `initialize_kv_cache` alloca ~2× (fp32+fp16) × layers × DEFAULT_MAX_SEQ_LEN × kv_dim — potenzialmente centinaia di MB; `ensure_kv_capacity` raddoppia — growth ok.

### Aggiunte dopo speculative.cpp + coda llama_inference.cpp

**F4.13 — `SpeculativeDecoder::generate`: verifica target alla posizione SBAGLIATA**
`src/infer/speculative.cpp:171-178`
- `prefix = sequence + draft_tokens[0..i]` (include il draft token verificato), poi `logits_to_last_row` prende la riga che predice il token SUCCESSIVO. `target_prob = p_target(· | …, x̂_i)` valutata su `x̂_i` — ma la distribuzione corretta per verificare x̂_i è `p_target(· | …, x_{i-1})` (riga precedente). Acceptance ratio `target_prob/draft_prob` usa posizioni diverse → la correzione non produce la distribuzione del target model (speculative sampling non più esatto).

**F4.14 — `SpeculativeDecoder::generate`: al primo rifiuto la generazione SI FERMA**
`speculative.cpp:212-219` — `rejected=true; break` esce dal for interno, poi `if (rejected) break;` esce dal **while** → `generate` ritorna dopo il primo draft rifiutato, senza generare i `max_tokens` richiesti. Speculative decoding standard continua dopo la rejection (resample + nuovo round). Output troncato silenziosamente.

**F4.15 — `acceptance_rate` metric errata**
`speculative.cpp:236-238` — `accepted_tokens / draft_steps` invece di `accepted / (draft_steps * gamma)` → rate sovrastimato.

**F4.16 — `logits_to_last_row` assume logits 2D `[seq,vocab]`**
`speculative.cpp:62-65` — per logits 3D batched `[B,S,V]` prende `shape[0]` come seq e `shape.back()` come vocab → riga errata. Nessun check.

**F4.17 — `llama_forward_inference` batched: `pos_offset` hardcoded a 0**
`llama_inference.cpp:1079` — `llama_attention_layer_cpu(..., seq_len, 0, ...)` — se `llama_forward_inference` viene chiamata per append su cache esistente (prompt chunks), le posizioni RoPE sono sempre 0..seq_len-1 e la cache parte da `current_len` per i K/V ma non per le query → mismatch posizioni. Verificare che il caller non usi mai questa funzione con cache non vuota; altrimenti bug ALTO.

**F4.18 — `preprocess_embeddings` in PP non-zeroth rank**: `hidden_states` non inizializzato se `pp_rank != 0` (single-rank path ok; PP path usa execute_forward con input nullptr) — ok, ma `run_transformer_layers` fa `memcpy(hidden_states.data(), input_act,...)` solo se input_act != data — su rank intermedi input_arriva via pipeline — verificato ok.

### Copertura Fase 4 — file ancora non letti
- `llama_inference.cpp` righe 1-598 (helper: rms_norm, proiezioni QKV, rope calls, attention_layer inizio)
- `llama_inference_cuda.cpp` (1358 righe) — interamente
- `llama_gpu_offload.cpp` (496) — interamente
- `transformer/block_optimized.cpp`, `moe*` files, header `attention.h`/`kv_cache.h`/`paged_kv_cache.h` (check `cached_mask_` mutable, `KVCacheLayer` fields)
- `infer/` altri file se presenti

### Aggiunte — llama_inference.cpp (CPU), llama_inference_cuda.cpp, llama_gpu_offload.cpp, block_optimized.cpp, header

**F4.19 — CUDA batched prefill IGNORA la KV cache esistente nell'attention**
`llama_inference_cuda.cpp:426-483`
- K/V vengono scritti in cache (r. 426-434) ma `kT_buf`/`v_hs_buf` sono costruiti SOLO dal chunk corrente: `scores = q @ kT` copre `seq_len × seq_len` e la mask è causale interna. Se `llama_forward_cuda` viene chiamata con cache non vuota (continuation/prefill a chunk), la history non viene mai attesa → output silenziosamente errato. In più `hal::rope(..., 0, ...)` (r. 414-415) hardcoda pos_offset=0 → posizioni RoPE sbagliate su cache non vuota. Stessa classe di problema di F4.17 (CPU batched, pos_offset=0).

**F4.20 — `fused_prefill_weights`: side-effect distruttivo + cache non thread-safe**
`llama_inference_cuda.cpp:145-220`
- La funzione **distrugge i pesi densi del modello** (`blk->q_proj->data = Tensor()`, r. 205-209) come effetto collaterale di una "cache". `static std::unordered_map` con `cache.size()>2 → clear()`: se il modello A viene evacuato mentre è in uso, alla chiamata successiva `ok_shape` fallisce (densi già distrutti) → `built==0` → return nullptr → fallback su `matmul` con tensori **vuoti** → crash o garbage. Eviction anche `cache.erase(&model)` a r. 165 prima di sapere se la rebuild riesce. ALTO.

**F4.21 — Stato decode globale non thread-safe**
`llama_inference_cuda.cpp:116-129, 641-682, 1036-1040`
- `decode_mask_for` (static p/cap), `graph_ctx()` (un solo GraphDecodeCtx globale), `graph_disabled()` (una sola chiave), `static int *tid_gpu` — tutti senza mutex. Due richieste decode concorrenti (server multi-client) corrompono stato/graph o rompono il tensor `tid_gpu` condiviso. MEDIO (verificare se il server serializza).

**F4.22 — `partial_layers_registry`: chiave = indirizzo modello, mai ripulita**
`llama_gpu_offload.cpp:427-440`
- Se un `LLaMAModel` viene distrutto e un altro allocato allo stesso indirizzo, `partial_gpu_layers()` restituisce il vecchio N → dispatch su path parziale per un modello senza layer GPU → crash/garbage. `KVCache::instance_id` risolve lo stesso problema per le cache — qui manca equivalente.

**F4.23 — Full offload: `move_param` sostituisce i pesi FP32 con FP16 GPU**
`llama_gpu_offload.cpp:323-346`
- `p->data = p->data.cast(Float16); set_device(CUDA)` distrugge la copia host FP32 (i "CPU fallback" citati nei commenti esistono solo per i QuantWeight, non per embed/norm/bias/proiezioni dense). Qualunque path CPU invocato dopo l'offload (cpu_prefill_batch, call non dispatchate) legge dtype/location errati.

**F4.24 — Decode CUDA eager: `use_quant` per-layer all-or-nothing + `proj` alloca `acc` a ogni chiamata**
`llama_inference_cuda.cpp:1119-1131, 1137-1139`
- Se solo alcuni layer hanno tutti e 4 i QuantWeight GPU-resident, quel layer usa `proj` → `matmul(x_gpu, w_cpu_fp32)` → mismatch silenzioso. E `acc` Tensor+alloc_gpu dentro la lambda → ~6 allocazioni device per layer per token (il commento dice "reused scratch" ma non lo è).

**F4.25 — Decode `cache==nullptr` con pos>0 → reshape esplode**
`llama_inference_cuda.cpp:1240-1248` — `key_buf` {1,kv_dim} reshaped a {pos+1, n_kv, head_dim}: per pos>0 size mismatch. Caller contract implicito (cache sempre passato dal server) ma non verificato.

**F4.26 — `block_optimized.cpp` è codice morto con definizione duplicata**
`src/transformer/block_optimized.cpp` — non in `src/CMakeLists.txt`; ridefinirebbe `forward_optimized` (gia' in block.cpp:53) e per giunta non compilerebbe (`ffn` e' unique_ptr, `ffn.forward_optimized` e' sintassi invalida; ignora MoE). Se qualcuno lo aggiunge alla build → link error + versione senza MoE.

**F4.27 — Conversioni FP16 scalar: troncamento invece di round-to-nearest**
`llama_inference.cpp:574-588` — `f32_to_f16_scalar` tronca la mantissa (`m>>13`) mentre il path SIMD usa `_mm256_cvtps_ph` round-to-nearest → la KV cache fp16 ha precisione diversa a seconda del path (SIMD vs tail scalare, F16C vs no). Anche `f16_to_f32_scalar` subnormal path (r. 597+) non verificato.

**F4.28 — `apply_rope_inplace`: `head_dim==0` → loop infinito; head_dim dispari → ultimo elemento mai ruotato**
`llama_inference.cpp:269-356` — `head_offset += head_dim` con head_dim=0 non avanza mai; half_dim=floor(head_dim/2) scarta il centro per dim dispari. Config validation a monte probabilmente lo impedisce, ma niente check locale.

**F4.29 — Tre implementazioni RoPE da mantenere allineate**
`autograd/ops_rope.cpp`, `transformer/positional.cpp::apply_rope`, `llama_inference.cpp::apply_rope_inplace` + `hal::rope` CUDA — convenzioni di pairing (half-split vs interleaved) e formula freq devono combaciare con i pesi GGUF. Nessun test di parita' registrato → rischio di divergenza silenziosa train/inference.

**F4.30 — `softmax_rows_inplace`/`compute_head_attention`: edge cases**
`llama_inference.cpp:388-440` — `row[0]` letto senza check (total_len==0 → OOB); NaN propagato (max resta, exp(NaN)=NaN → inv_sum NaN → riga NaN).

**F4.31 — Warmup offload: `gpu_view` A e C sullo stesso buffer**
`llama_gpu_offload.cpp:377-388` — `f16_view` restituisce view sullo stesso `buf` per input e output dei GEMM warmup → calcoli garbage (irrilevanti, e' warmup) ma se un kernel non tollera aliasing input/output potrebbe dare errori silenziosi.

**F4.32 — `llama_forward_single_token_cuda_graph` ritorna vista non-owning su buffer ctx**
`llama_inference_cuda.cpp:1003-1006` — documentato ("consume immediately") ma il tipo Tensor non lo segnala: un caller che salva il risultato legge i logits del token SUCCESSIVO. Hazard API.

---

## Fase 5 — Server + CLI + concorrenza (parziale: server.cpp, server_generation.cpp, batcher.cpp, cmd_chat, cmd_serve, stress core)

### ALTO

**F5.1 — `generate_transformer`/`stream_transformer`: leak del grafo autograd a ogni token**
`src/server/server_generation.cpp:459-460, 522-523`
- `model.forward(&input_node)` ritorna `logits_node` che non viene mai `delete`-ato. Ogni token generato perde l'intero grafo (decine di Node per layer). Con `max_tokens` fino a 32768 → leak di centinaia di MB/GB. Anche se il grafo non viene più traversato, la memoria non torna mai.

**F5.2 — `finish_reason` sempre "stop", mai "length"**
`server.cpp:1021`, `server_generation.cpp:424-431`
- Quando la generazione finisce per `max_tokens` (non per EOS), la API risponde comunque `finish_reason: "stop"` — client che si basano su "length" (OpenAI semantics) non lo vedranno mai. Stessa cosa nello stream (stop_chunk emesso sempre).

**F5.3 — `/v1/embeddings` ritorna vettori ZERO su modelli GPU-offloaded**
`server.cpp:1216-1239`
- Richiede `embed_node->data.dtype() == Float32`; dopo `offload_llama_model_to_gpu` `embed_tokens` è Float16 su GPU → la condizione fallisce → `emb_vector` resta `{0,...,0}` per ogni input. Silenzioso, nessun warning.

### MEDIO

**F5.4 — KV cache condivisa tra request: prefix-reuse dipende da `cached_token_ids` aggiornato in OGNI path di uscita**
`server_generation.cpp:208-225, 337-339, 436-438`
- `lm->kv_cache` e `lm->cached_token_ids` sono una coppia sola per modello. Il reuse del prefisso (common_prefix_len → truncate_kv_cache) è corretto SOLO se `kv_ids` rispecchia sempre il contenuto della cache. Se un futuro path (o una eccezione a metà generazione) lascia la cache con più/meno token di `kv_ids` → context bleed tra utenti o attensione a K/V stantii. Oggi i path coperti sono coerenti, ma il contratto è fragile e non verificato.

**F5.5 — `ContinuousBatcher::iteration` tiene `mutex_` durante `forward_batch` e durante le callback**
`batcher.cpp:16-70` — la forward pass (lenta) blocca `enqueue`/`active_slots`; `slot.callback` sotto lock → deadlock se la callback tocca il batcher; `next_tokens[i]` senza check di size → OOB se forward_batch ritorna meno elementi. (Inoltre max 16 token hardcoded — sembra codice demo.)

**F5.6 — Chat CLI multi-turno perde la storia**
`cmd_chat.cpp:197-202` — ogni `run_turn` costruisce il prompt solo con system+user corrente; le risposte precedenti non entrano mai nel contesto → "chat" senza memoria. Se voluto, ok; altrimenti bug di prodotto.

**F5.7 — `token_logprob`/`gpu_sample` in cmd_chat forzano sync host ogni token**
`cmd_chat.cpp:298, 358` — `token_logprob` chiama `logits.data()` (syncToHost) su logits GPU-residenti → annulla il beneficial async del decode path; `pending_lp` calcolato anche quando `SERIUS_EXPERIMENTAL_GROUNDING` non è definito (l'uso è dentro ifdef ma il calcolo no) → spreco.

**F5.8 — `is_stop_token` dipende dal decode di UN token**
`server_generation.cpp:178-181` — `tok_str == "<|im_end|>"` su decode di singolo token: se il tokenizer produce "<|im_end|>" spezzato in più token BPE, il check non scatta → risposta non terminata correttamente. Con Qwen il check funziona perché <|im_end|> è un token unico, ma è fragile.

**F5.9 — `model_path_allowed` case-sensitive su Windows**
`server.cpp:155-188` — `weakly_canonical` + compare di stringhe: su Windows i path sono case-insensitive ma il compare è case-sensitive → falsi 403 per root con case diverso.

**F5.10 — Double-submit `/v1/models/load` stesso name → non 409**
`server.cpp:786-798` — il check 409 avviene prima di `load_model`; due request concorrenti con lo stesso name esplicito superano entrambi il check e `pick_model_name` deduplica con suffisso "-2" invece di fallire → nome diverso da quello richiesto, nessun errore.

### BASSO

- **F5.11** — `metrics_->on_token_generated(0.0f)` e `on_request_end` con durata 0 per embeddings → metriche tok/s e latency inutilizzabili (`server_generation.cpp:322,384,479,539`, `server.cpp:1248`).
- **F5.12** — `catch(...)` muto nel flusso SSE (`server.cpp:979`) e in `sample_server_memory`/worker — errori ingoiati senza log.
- **F5.13** — `calculate_drift` = last−first (non slope) in leak_detector — "drift rate" riportato è in realtà drift totale: se la memoria sale e scende il rate orario è fuorviante (`leak_detector.cpp:107-115`).
- **F5.14** — `Server::run` re-bind loop: `bind_to_port` una seconda volta sullo stesso `httplib::Server` dopo `listen_after_bind` fallito — httplib tiene stato interno (`is_bound_`); il retry potrebbe non funzionare come previsto (`server.cpp:1320-1332`).
- **F5.15** — `generate_llama`/`stream_llama`: `logits.shape().back()` su `Tensor({0})` (rank PP non-ultimo) → vocab_size=0 → sampler su tensore vuoto → next_id=0 silenzioso.
- **F5.16** — `prefill_llama` chiama `llama_forward_single_token` per OGNI token del prompt (commento dice "batched prefill disabilitato") — O(prompt) launch CUDA per prefills lunghi; mitigato dal decode graph solo se logit path lo consente.

### Copertura Fase 5 — file NON ancora letti
`cmd_doctor.cpp` (620), `args_parser.cpp`, `cmd_{generate,bench,think,mcp,convert,eval,import,inspect}.cpp`, `metrics.*`, `static_files.cpp`, `llama_loader.cpp`, `report.cpp`, `batcher.h`, `server.h`, `download_progress.*`, `hf` downloader/search.

### Aggiunte finali Fase 5 (args_parser, llama_loader, static_files, metrics, cmd_doctor/think/eval, report)

**F5.17 — `load_llama_weights_from_*` falliti → modello con pesi CASUALI servito senza errore**
`src/server/server.cpp:256, 437-440`, `src/cli/llama_loader.cpp:176-189`
- In `load_llama_gguf` e `load_llama_dir` il ritorno del weight loader è ignorato (`(void)weights_loaded` / nemmeno controllato). Se il loader fallisce (file troncato, tensori mancanti), il modello resta con inizializzazione random e `/v1/models/load` risponde **success** → il server genera testo meaningless credendo di servire un modello reale. Nel CLI c'è almeno un warning su stderr, nel server no.

**F5.18 — `ArgsParser`: flag ripetuti → vale il PRIMO, non l'ultimo**
`src/cli/args_parser.cpp:94-111` — `--model a --model b` restituisce "a"; convenzione CLI standard è last-wins. Anche `has()` controlla `"-" + short_name` che `parse()` non scrive mai in positionals_ (dead check).

**F5.19 — `static_files` traversal check case-sensitive su Windows**
`src/server/static_files.cpp:78-79` — `target.string().compare(0, root_str.size(), root_str)` è case-sensitive; su Windows un path con case diverso nel prefisso darebbe 404 spurio (sola disponibilità, non sicurezza). Anche `filename.find("..")` rifiuta nomi legittimi con ".." (es. "foo..bar") — over-blocking ma sicuro.

**F5.20 — `InferenceMetrics::on_request_end` può guidare `active_requests_` in negativo**
`src/server/metrics.cpp:14-17` — se un handler lancia prima di `on_request_end` (le eccezioni dentro i try sono gestite, ma errori fuori try es. `find_model` su mutex) il gauge scende sotto zero; Prometheus espone il valore negativo.

**F5.21 — `available_ram_bytes` su macOS ritorna sempre 0**
`cmd_doctor.cpp:180-182` — `#elif defined(__APPLE__)` vuoto → doctor riporta "could not query" su Mac.

### Note di copertura
- Letti integralmente: server.cpp, server_generation.cpp, batcher.cpp, static_files.cpp, metrics.cpp, args_parser.cpp, llama_loader.cpp, cmd_chat.cpp, cmd_serve.cpp, cmd_doctor.cpp, cmd_think.cpp, cmd_eval.cpp, stress/{runner,worker,report,leak_detector}.cpp.
- Non letti integralmente (wrapper sottili, grep pulito da TODO/catch/system): cmd_{generate,bench,think→letto,convert,import,inspect,mcp,stress}.cpp, cli.cpp, serius_cli.cpp, download_progress.h, header vari.

---

## FASE 6 — Tokenizer + IO + GGUF

### Tokenizer (src/tokenizer/bpe.cpp)

**F6.1 — `load()` JSON non imposta mai `bos_id_`/`eos_id_`** — ALTO
`bpe.cpp:309-417`, `bpe.h:61-62`
- Solo `load_from_gguf()` valorizza bos/eos dal file. Il path JSON (`tokenizer.json` / `vocab.json` / `added_tokens`) lascia i default `BOS_ID=1`, `EOS_ID=2`. Per Qwen (bos=151643, eos=`<|im_end|>`~151645) caricato via tokenizer.json, `generate` inserisce il token id 1 (un token normale) come BOS e non riconosce mai il vero EOS → generazione senza stop e prompt malformato. `added_tokens` ha i flag `special`/`content` ma non estrae bos/eos.

**F6.2 — `encode()` per gpt2_byte_level_/sentencepiece usa longest-match greedy, non BPE vero** — MEDIO
`bpe.cpp:91-129, 172-174`
- Per i due stili più comuni la funzione fa greedy longest-match (max 32 char) sul vocabolario e **salta `merge_tokens`** → la tabella `merges` caricata da tokenizer.json/GGUF non viene mai usata. La tokenizzazione greedy non è equivalente all'ordine di merge BPE → token id diversi dal reference HF tokenizer → degrado qualità silenzioso.

**F6.3 — `load()`: detection formato fragile + eccezioni non gestite** — MEDIO
`bpe.cpp:330-381`
- Qualunque JSON object senza chiave `"model"` viene trattato come vocab.json (`it.value().get<TokenId>()` lancia su valori non numerici → eccezione propaga fuori da `load()` → crash del chiamante). Le merges `[a,b,c]` usano `merge_item[0]` con conversione implicita — idem. `try/catch` copre solo il parse iniziale.

**F6.4 — `merge_tokens` usa come priorità il nuovo id più basso** — BASSO
`bpe.cpp:184-196`
- Assume che ordine id == rank di merge; per vocab caricato da file esterno l'ordine dei merge è dato dal file, non dagli id.

**F6.5 — `decode()` gpt2: char non mappato → UTF-8 potenzialmente invalido; special token non rimossi** — BASSO
`bpe.cpp:218-249, 252-267`
- Se una sequenza UTF-8 del token non è in `inv_map` viene appesa grezza; i token speciali (`<|im_end|>`) decodificano come testo letterale nel output.

### IO — safetensors

**F6.6 — `SafetensorsFile::tensor_data` F16 ritorna buffer `thread_local`** — MEDIO
`safetensors_loader.cpp:140-144`
- Il puntatore ritornato è invalidato dalla prossima chiamata a `tensor_data` (stesso thread) — footgun per i caller che tengono il puntatore. `copy_tensor_to_float` è la API sicura; `tensor_data` andrebbe deprecata o documentata come tale.

**F6.7 — `SafetensorsReader::get_tensor` F16 subnormal: `exponent` può andare in underflow** — BASSO
`safetensors.cpp:145-157`
- `exponent` è `uint32_t`; nel loop di normalizzazione può diventare negativo → `exponent + 127 - 15` wrappa → float errato per subnormali estremi. `safetensors_loader.cpp` usa `std::ldexp` (corretto) — implementazioni duplicate e divergenti.

**F6.8 — `SafetensorsWriter::add_tensor` / `simple_weights_loader` scrivono `raw_data()`/`data()` senza sync/contiguous check** — MEDIO
`safetensors.cpp:207-216`, `simple_weights_loader.cpp:29-37`
- Se il Tensor è su GPU o è una view non contigua, `raw_data()` restituisce il puntatore device / layout strided → file corrotto silenziosamente. Nessun `syncToHost()`/`contiguous()`/dtype check (un tensor Float16 viene scritto come se fosse float32 in `save_simple_weights`).

**F6.9 — `load_safetensors` carica l'intero data blob in RAM (no mmap)** — BASSO
`safetensors_loader.cpp:104-113`
- `out->data_.resize(data_size)` → per un modello da 70GB servono 70GB di RAM; esiste già `MappedFile`/`SafetensorsReader` mmap-based ma sono due implementazioni parallee divergenti (una valida bounds-check, l'altra no).

**F6.10 — `read_le<T>` assume host little-endian** — BASSO
`safetensors_loader.cpp:17-22`
- `memcpy` grezzo dell'header size; su big-endian (raro ma es. some ARM) header_size sbagliato.

### IO — mmap / config / sharding

**F6.11 — `MappedFile::is_open()` ritorna true per oggetto default-constructed** — BASSO
`mmap.h:150`
- `data_==nullptr && size_==0` → `is_open()` true anche se `open()` non è mai stato chiamato o è fallito. Su Windows un file vuoto fallisce `MapViewOfFile` (POSIX lo gestisce) → eccezione su file 0-byte.

**F6.12 — `parse_config` non legge `rope_theta` (e altri campi HF)** — MEDIO
`config_loader.cpp:13-92`
- `ModelConfig` ha `rope_theta` (scritto da `gguf_export`) ma il parser JSON non lo legge → modelli con theta != 10000 (es. alcune varianti) usano il default → posizioni RoPE sbagliate. Idem `tie_word_embeddings`, `partial_rotary_factors` ecc. non presenti.

**F6.13 — `load_sharded_safetensors` si aspetta formato `{"tensors": {name:{file}}}` ma l'indice HF usa `weight_map`** — ALTO
`sharding.cpp:26-40` + `gguf_writer.cpp:308-317`
- L'indice ufficiale HF (`model.safetensors.index.json`) ha `"weight_map": {tensor_name: "file.safetensors"}` — `index.at("tensors")` lancia → return false → **modelli HF sharded non caricabili** nonostante `load_hf_tensors` cerchi esplicitamente `*.safetensors.index.json`. Incompatibilità silenziosa: il formato "serius-sharded" scritto da `write_sharded_safetensors*` è proprietario e diverso da HF.

### IO — GGUF

**F6.14 — GGUF: `bos/eos_token_id` negativo non validato → OOB a catena con F4.1** — MEDIO
`gguf_loader.cpp:831-876, 1808-1822`, `bpe.cpp:431-438`
- `read_scalar_value` per int8 fa cast a `uint64_t` (negativo → huge) poi `static_cast<int32_t>` → può restare negativo. In `load_from_gguf` il check è solo `>= data.tokens.size()` — un id negativo passa → `bos_id_`/`eos_id_` negativi → `generate` inserisce token id <0 → `embedding_lookup` OOB (F4.1). File GGUF corrotti/malformati bastano.

**F6.15 — `data_base` usa allineamento 32 hardcoded** — MEDIO
`gguf_loader.cpp:1252-1256`
- GGUF v3 dichiara `general.alignment` nel KV (default 32 ma può variare) — il loader non lo legge → file con alignment diverso vengono parsati con offset sbagliati → pesi corrotti o errori.

**F6.16 — Dequant K-quant: `nb = n/256` floor → coda scartata** — BASSO
`gguf_loader.cpp:174-280` (tutti i `dequantize_q*_k`)
- Se `numel` non è multiplo di 256, l'ultimo superblocco parziale viene ignorato → quegli elementi restano 0. La validazione bounds calcola `payload` con lo stesso floor → nessun errore, solo zeri silenziosi.

**F6.17 — `gguf_writer`: `general.file_type=1` ma i tensori sono scritti F16** — MEDIO
`gguf_writer.cpp:656, 708-709`
- `file_type` GGUF: 0=F32, 1=Q4_0-ish. Con `use_f32=false` scrive F16 ma dichiara file_type 1 → llama.cpp/altri tool si aspettano Q4_0 → **file illeggibile o interpretato male** da tool esterni. La propria `load_all_tensors` ignora file_type (guarda dtype per-tensor) quindi il round-trip interno funziona — bug di interop.

**F6.18 — `gguf_writer`: loop a due passate identiche** — BASSO
`gguf_writer.cpp:717-734`
- `for pass<2` ricalcola `infos_buf`/`data_base` con valori invarianti — codice morto, nessun effetto (commento dice "fixed point" ma non serve).

**F6.19 — `quantize_fp32_to_q40`: scala `d = amax/7` asimmetrica** — BASSO
`hf_llama_loader.cpp:293`
- Q4_0 ggml usa `d = amax/(-8)` (intervallo simmetrico -8..7); `/7` restringe il range per i negativi → slight quality loss sulla head quantizzata al volo.

### IO — hf_llama_loader

**F6.20 — `load_one_file_into_model` ignora tutti i ritorni di `copy_into_param`** — ALTO
`hf_llama_loader.cpp:200-224, 335-363`
- A differenza del path GGUF (che ha `missing_required` e rifiuta il caricamento), il path safetensors **non controlla se i tensori esistono**: se un file manca `mlp.down_proj` o ha shape errata, `copy_into_param` ritorna false e il parametro resta a zero → modello parzialmente vuoto caricato "con successo". Stessa classe di bug di F5.17.

**F6.21 — `copy_into_param` accede `shape[0]`/`shape[1]` senza verificare `shape.size()==2`** — MEDIO
`hf_llama_loader.cpp:114-116`
- Con `should_transpose=true` e un tensore 1-D il cui numel coincide per caso con `expected_size`, `metadata.shape[1]` è accesso OOB su `std::vector` → UB.

**F6.22 — `load_llama_weights_from_dir` carica OGNI `.gguf` nella directory** — MEDIO
`hf_llama_loader.cpp:724-742`
- Non fa `break` dopo il primo successo: ogni GGUF trovato sovrascrive i pesi del precedente; una directory con più GGUF (o un mmproj) produce un modello "frankenstein". In più un GGUF che fallisce a meta-load ha già copiato parte dei tensori → stato parziale.

### Data

**F6.23 — `DataLoader` non mescola mai i dati** — BASSO
`dataloader.cpp:16-64`
- `next_batch` scorre i token in ordine sequenziale; nessuno shuffle → bias di addestramento. `padding_mask` valorizzata solo per input, `target_ids` a PAD_ID sui bordi — la CE deve mascherare internamente o i pad token generano loss spuria.

**F6.24 — `minhash`/`dedup`: documenti vuoti hanno signature tutta `UINT32_MAX`** — BASSO
`minhash.cpp:44-47`, `dedup.cpp`
- Due testi senza shingles producono jaccard=1.0 → dedup considerati duplicati (forse voluto); `minhash_jaccard` ritorna 0 se le signature hanno size diverse (silenzioso).

---

## FASE 7 — Moduli restanti

### Train + Optim

**F7.1 — `load_checkpoint`: `total_size` dal file non validato vs dimensione parametro** — CRITICO
`checkpoint.cpp:85-104`
- `shape`/`total_size` letti dal checkpoint e usati direttamente in `is.read(p->data.data(), total_size*4)` senza verificare che `total_size == p->data.size()`. Un checkpoint corrotto o con shape diverso → **lettura oltre il buffer del tensor (heap overflow)**. Anche `state.m/v` stessa dimensione assunta. Nessun check di `is.good()` dopo le read → load "riuscito" con dati a metà.

**F7.2 — `save_checkpoint`: `optimizer.state().at(p)` lancia se parametro non in stato** — MEDIO
`checkpoint.cpp:47-50`
- Se un parametro è stato aggiunto al modello dopo la costruzione di AdamW (o `freeze_parameters` cambia requires_grad — no, state esiste comunque) `.at()` throw → save fallisce a metà file (header già scritto → file troncato illeggibile).

**F7.3 — `Trainer::cleanup_graph` usa `model_params_set_` snapshot del costruttore** — BASSO
`trainer.cpp:19-31`
- Se i parametri del modello cambiano dopo (es. nuovo layer, weight tying), i nuovi node non-param vengono cancellati ma i vecchi param rimossi restano leak; viceversa un param "nuovo" verrebbe cancellato → UAF.

**F7.4 — `AdamW::step` assume `grad` popolato e stessa size di `param`** — MEDIO
`adamw.cpp:34,51-52`
- Se `requires_grad=true` ma il backward non ha scritto `grad` (path senza ops, o grad mai accumulato), `grad_tensor[elem_idx]` legge buffer vuoto → OOB/garbage. Nessun check `grad.size()==param.size()`.

**F7.5 — `AdamW` weight decay è L2-nel-gradiente, non AdamW disaccoppiato** — BASSO
`adamw.cpp:64-65`
- `param -= lr*(m_hat/(sqrt(v_hat)+eps) + wd*param)` = weight decay dentro il gradiente (come L2), non il decay disaccoppiato di AdamW → comportamento diverso dallo standard, specialmente con grad clipping.

**F7.6 — `WSDScheduler`: `warmup_steps_=0` o `decay_steps_=0` edge** — BASSO
`lr_schedule.cpp:16-29`
- Con warmup=0 il primo step ha già LR max (ok); con decay_steps_=0 la fase decay è irraggiungibile e si passa direttamente a 0.1*max — comportamento plausibile ma `decay_progress` divide per decay_steps_ solo quando step<...+decay → se decay>0 ok. Nessun crash, solo semantica da verificare.

### MoE + LoRA

**F7.7 — `LoRALinear::forward` usa `transpose()` views in matmul** — ALTO
`lora_linear.cpp:46,55-56`
- `matmul(x, W_.data.transpose())`, `A_->data.transpose()`, `B_->data.transpose()` producono view non contigue → se `blas::matmul` assume `stride[1]==1` (finding F3.1) il risultato è silenziosamente sbagliato. Dipende dalla stessa root cause ma rende LoRA inutilizzabile di fatto.

**F7.8 — `LoRALinear::unmerge` sottrae `B*A` ricalcolato, non il delta applicato** — MEDIO
`lora_linear.cpp:85-100`
- Se A/B vengono aggiornati (training) dopo `merge()`, `unmerge()` sottrae un delta diverso da quello sommato → W resta corrotta permanentemente. Standard richiede merge solo a fine training o disabilitare grad durante merge.

**F7.9 — `LoRAAdapter::load` non valida dimensioni e disallinea lo stream** — MEDIO
`lora_adapter.cpp:139-151`
- `a_bytes`/`b_bytes` letti dal file usati direttamente in `in.read` senza check vs `A().data.size()*4` → file malformato = heap overflow. Inoltre se un layer nel file non esiste in `layers_`, il `continue` salta la read → **tutti i layer successivi leggono dati shiftati** → adapter corrotto silenziosamente. `A_offset`/`B_offset` scritti in save ma ignorati in load.

**F7.10 — `LoRAAdapter::save` scrive `data()` raw senza sync/contiguous check** — MEDIO
`lora_adapter.cpp:115-120`
- Stessa classe di F6.8: tensor GPU o view → byte garbage nel file.

**F7.11 — `Router::forward` produce Node senza children e con math opinabile** — BASSO
`router.cpp:75-95`
- Ritorna `Node(out, false)` senza children/backward_fn → il grafo si ferma lì (silenziosamente nessun grad). La formula `out[i] += w*x[i]*gate_w[i%d_model, expert]` non corrisponde a una router-standard; probabilmente codice morto/alternativo.

**F7.12 — `MoELayer::forward` espone `route_weights->data[k]` come gate weight ma gradient path assume softmax sui selezionati** — BASSO
`moe_layer.cpp:122-143`, `router.cpp:97-147`
- Coerente internamente ma `selected_weights` fa softmax solo sugli expert selezionati (non top-k masked full softmax) → il bilanciamento del carico non ha il gradiente standard MoE; `load_balance` loss non è presente nel layer.

### MCP

**F7.13 — Audit log scrive `args.dump()` in chiaro: segreti non scrubbati** — ALTO
`mcp/server.cpp:78-80`, `mcp/scrubber.cpp`
- `Scrubber` esiste ma non viene applicato né agli args mandati al tool handler né al log di audit → `api_key`/`token` passati come argomenti finiscono in chiaro nel file di audit e in stdout. Il guard (`guard.cpp`) logga args grezzi per risk>=MODERATE.

**F7.14 — `MCPGuard::tool_risk_` è `static` condiviso, ctor lo resetta, nessuna sincronizzazione** — MEDIO
`guard.cpp:19-24,56-59`
- Map condivisa tra istanze; `set_risk` concorrenza con `risk_level` → data race. Anche il reset nel ctor azzera customizzazioni fatte da un'altra istanza.

**F7.15 — `handle_message` non supporta batch JSON-RPC (array)** — BASSO
`server.cpp:87-124`
- Una request array (valida in JSON-RPC) ritorna "Invalid JSON-RPC request".

### Distributed

**F7.16 — `ProcessGroup` condivide `arrived_`/`departed_`/`expected_count_` tra TUTTI i collectives e `barrier()`** — ALTO
`local_comm.cpp:31-321`
- Nessun tag di tipo: se un rank esegue `all_reduce` mentre un altro esegue `broadcast` (o `barrier`), i contatori si mescolano → count-mismatch throw, deadlock o **risultati corrotti silenziosi** se i count coincidono. L'API non impedisce l'uso concorrente/misto sullo stesso group.

**F7.17 — `LocalCommunicator::send/recv` sono no-op silenziosi se `group_==nullptr`** — MEDIO
`local_comm.cpp:388-400`
- Il default ctor (world_size=1) ha `group_=nullptr` → `send()`/`recv()` non fanno nulla e non segnalano errore → pipeline che usa send/recv su world_size=1 perde i dati silenziosamente.

**F7.18 — `Communicator::create(Local, world_size>1)` ritorna communicator single-rank** — MEDIO
`local_comm.cpp:425-434`
- Ignora `world_size` e ritorna `LocalCommunicator()` default → chi chiede 4 rank ottiene rank 0 world 1 → compute distribuito silenziosamente errato.

**F7.19 — `scatter_input` tronca `in_features/tp_size` senza check di divisibilità** — BASSO
`tensor_parallel.cpp:86-94`
- Se `in_features % tp_size != 0`, le feature resto vengono ignorate silenziosamente.

### Attention + HAL + Common

**F7.20 — `dynamic_sparse_attention` assume Q/K/V float32 2D contigui senza check** — MEDIO
`sparse_attention.cpp:41-57`
- `Q.shape()[1]` su tensor 1-D → OOB; `Q.data()` raw senza verificare dtype/contiguity/device → view o GPU tensor producono garbage silenzioso. `partial_sort` con `scores.begin()+min(top_k,size)` ok, ma se `scores` vuoto (i=0? no, j<=i sempre ≥1) ok.

**F7.21 — `generate_block_mask`: `seq_len % block_size != 0` → `block_size=1` silenzioso** — BASSO
`sparsity_mask.cpp:12-15`
- Invece di arrotondare o fallire, forza block_size=1 → maschera `seq_len x seq_len` potenzialmente enorme (memoria) per input non multipli.

**F7.22 — `CpuStream::sync()` / async memcpy: nessuna protezione su lifetime di src/dst** — MEDIO
`cpu_backend.cpp:187-201, 240-265`
- `memcpy_*_async` cattura solo ptr+size: se il caller libera `src` prima che lo stream esegua → UAF. `cpu_stream_enqueue` check-then-cast TOCTOU: stream distrutto tra check e enqueue → UAF. È il contratto async standard ma non documentato/enforced.

**F7.23 — `get_backend(CPU)` singleton mai distrutto** — BASSO
`cpu_backend.cpp:270-277`
- `new CpuBackend()` leak intenzionale (singleton) — ok, ma `CpuStream` registry `g_cpu_streams` cresciuto? no, erase in destroy. Pulito.

**F7.24 — `ThreadPool::parallel_for` cattura `&func` per riferimento** — BASSO
`thread_pool.h:160,205`
- Corretto perché `fut.get()` join prima del return, ma se `func` lancia eccezione la future la propaga — ok. `Func&&` universale: se chiamato con lambda temporanea che cattura per riferimento variabili locali dello scope chiamante → hang se caller return prima di get? no, get() prima del return → sicuro.

### Hub downloader

**F7.25 — `resolve_hf_gguf`: file esistente con `pick->size==0` considerato "complete"** — MEDIO
`downloader.cpp:405-409`
- Se l'API non riporta la size (o fallisce listing), un file parziale su disco viene accettato come completo → modello troncato caricato.

**F7.26 — `download_hf` fallback lista file "voluta" senza verificarne l'esistenza reale** — BASSO
`downloader.cpp:232-238`
- Se `list_hf_files` fallisce (rete), prova nomi standard: `download_file` fallirà per i mancanti ma `downloaded_any` può essere true con solo config.json → "success" con modello incompleto.

**F7.27 — `download_file` non verifica che il download completato sia integro** — BASSO
`downloader.cpp:185-212`
- curl `-C -` resume + `--retry 3`: se il server non supporta range e il file esistente è parziale, curl può corrompere; nessun check dimensione finale vs content-length.

### Safety

**F7.28 — `TokenToxicityClassifier::score` division by zero se `hidden_dim`/`embedding_dim` = 0** — BASSO
`toxicity.cpp:52,75`
- `token_sum / config_.embedding_dim` e `hidden /= config_.hidden_dim` senza check → Config con dim 0 → NaN/inf o FPE.

**F7.29 — `ToxicityClassifier::proto_cache_` mutable in const method, nessuna sincronizzazione** — MEDIO
`red_team.cpp:86-94`
- `prototype_embedding` è const ma muta `proto_cache_` (unordered_map) → data race se `score()` chiamato da thread multipli (es. server).

**F7.30 — `TokenToxicityClassifier` hidden layer usa solo la riga 0 della weight matrix** — BASSO
`toxicity.cpp:70-78`
- `hidden_weight_.at(0,i)` itera su hidden_dim ma sempre riga 0 → hidden layer degenere (embedding_mean scalare × vettore); `output_weight_.at(0,0)` solo elemento. Classifier toy, ma i pesi "addestrati" altrove non avrebbero effetto.

**F7.31 — `Sanitizer::sanitize` tronca PRIMA di applicare i pattern** — BASSO
`sanitizer.cpp:41-48`
- Se `max_response_length` taglia a metà un pattern dannoso (es. "DR"+"OP TABLE"), il regex non matcha → l'iniezione supera il sanitizer.

### Optim graph_compiler + AMP + Plugin

**F7.32 — `ExecutionPlan::compile()` alloca `workspace_` mai usato** — ALTO
`graph_compiler.cpp:483-515`, `execute()` 37-79
- `workspace_size_` = somma di TUTTI gli output (o peak liveness) e `workspace_` viene mallocato, ma `execute()` non lo referenzia mai — i tensor sono allocati da `values[]`/Tensor. Per un modello grande è una malloc potenzialmente di GB mai usata → spreco memoria / OOM.

**F7.33 — `AMP::unscale_and_update` su overflow azzera solo dopo `break` al primo parametro overflow** — BASSO
`amp.h:73-104`
- Corretto perché `zero_grad()` del passo successivo pulisce tutto, ma se il chiamante non azzera (uso API diretta) i grad restano scaled. `item(i)`/`set_item` per elemento → O(n) sync potenziali.

**F7.34 — `PluginManager::load_plugin` esegue `LoadLibrary`+`initialize` sotto `plugins_mutex_`** — MEDIO
`plugin_manager.cpp:40-128`
- Codice arbitrario del plugin gira con il lock del manager → se il plugin chiama back `PluginManager` (o tarda) deadlock/blocco di tutte le operazioni.

**F7.35 — Sandbox "resource limits" via `setrlimit` colpisce l'INTERO processo** — ALTO
`plugin_manager.cpp:147-168`
- `setrlimit(RLIMIT_AS, 512MB)` e `RLIMIT_NOFILE` si applicano al processo server, non al plugin → caricare un plugin su POSIX abbassa il limite di memoria FD di tutto il server → DoS auto-inflicted. `network_access=false` non è enforced da nessuna parte.

**F7.36 — `get_process_cpu_usage` usa static condivisi, chiamato da thread di monitor multipli** — MEDIO
`plugin_manager.cpp:249-286`
- `last_process_time`/`last_cpu_time` static non atomici → data race tra monitor thread di plugin diversi → CPU% errato.

**F7.37 — `PluginFactory::create_plugin` leak del library handle** — BASSO
`plugin_manager.cpp:474-504`
- Se `create_plugin_func` esiste, la libreria resta caricata per sempre (handle mai conservato) → impossibile unload/reload di plugin creati via factory.

**F7.38 — `extract_metadata` inventa id/version da filename** — BASSO
`plugin_manager.cpp:518-526`
- Due DLL con stesso stem in directory diverse collidono su `metadata.id` → "already loaded" errato.

### MCP transports + Distributed bootstrap

**F7.39 — `StdioTransport::send_request` loop `fgets` senza timeout** — MEDIO
`transport_stdio.cpp:172-184`
- Se il subprocess non risponde (hang), `fgets` blocca per sempre → la richiesta MCP pinna il thread chiamante indefinitamente. Nessun timeout né meccanismo di cancel.

**F7.40 — `StdioTransport`: risposta JSON >64KB spezzata e persa** — MEDIO
`transport_stdio.cpp:172-173`
- `char buf[65536]` + `fgets`: una riga più lunga viene spezzata in chunk; ogni chunk fallisce `json::parse` → `continue` → la risposta vera non arriva mai → errore "closed stdout" solo a EOF. Tool che ritornano output grandi (es. file listing) rompono il transport.

**F7.41 — `HttpTransport` accetta response con `id` diverso senza segnalarlo** — MEDIO
`transport_http.cpp:99-103`
- `if (resp.value("id", -1) != id && !resp.contains("id"))` → fix id solo se MANCANTE; se il server risponde con un id diverso (errato), la response viene ritornata come se fosse quella richiesta → risposta sbagliata silenziosa.

**F7.42 — `spawn()` POSIX non chiude fd ereditati nel child (no CLOEXEC)** — BASSO
`transport_stdio.cpp:114-144`
- `pipe()` senza `O_CLOEXEC`/`fcntl` → il subprocess eredita TUTTE le pipe/fd del parent → leak di fd e possibili corse sui capi pipe.

**F7.43 — `env_int` rifiuta `SERIUS_LOCAL_RANK=-1` esplicito** — BASSO
`bootstrap.cpp:39-55`
- `parsed < 0` → throw; il default è -1 ma settarlo esplicitamente a -1 (logico per "non impostato") fallisce.

### Hub server

**F7.44 — Download endpoint `content_length` ignora Range/offset scorretti** — BASSO
`model_hub_server.cpp:627-639`
- `set_content_provider(fsize,...)` ma il lambda ignora `length` ai fini del bounds check: `file.read(buffer.data(), length)` con `length` dal client — se `offset+length > fsize`, `gcount` ritorna meno ed è gestito, ma `std::vector<char> buffer(length)` alloca `length` bytes non limitato → un client può chiedere range enorme → allocazione enorme. httplib di solito clampma, ma nessun check esplicito.

**F7.45 — `calculate_checksum` carica l'intero file in RAM** — BASSO
`model_hub_server.cpp:846-855`
- `ss << file.rdbuf()` → stringa grande quanto il modello → per file multi-GB OOM; usata solo in path non-HTTP (`upload_model` usa bytes già in RAM dal multipart — anch'esso interamente in memoria).

**F7.46 — Upload multipart tiene `model_file.content` interamente in RAM** — MEDIO
`model_hub_server.cpp:529-570`
- `req.form.get_file("model").content` è la request body intera in memoria + `ofstream` write + `metadata_.save` → upload di modelli grandi = RAM esplosa. Nessun streaming.

**F7.47 — `validate_model_file` ritorna true per QUALUNQUE file ≥4 byte** — BASSO
`model_hub_server.cpp:872-882`
- Legge 4 byte e ritorna `gcount()==4` — non verifica affatto magic GGUF/safetensors nonostante il commento.

**F7.48 — `resolve_user_id`/`user_permissions`/`load_users` rileggono e parsano JSON ad OGNI request** — BASSO
`model_hub_server.cpp:275-334`
- Ogni chiamata auth = open+parse+scan di `hub_users.json` → O(users) per request, no cache → DoS facile su file grande.

**F7.49 — `register_user` race: check-then-write senza lock** — MEDIO
`model_hub_server.cpp:336-362`
- Due register concorrenti stesso username: entrambi passano il check → entrambi scrivono (ultimo vince) → uno dei due riceve api_key che non è nel file → account "fantasma". `save_users` non è atomico (write diretto, no tmp+rename) → crash a metà = file JSON corrotto.

**F7.50 — Rate limit `/v1/auth/register` conta PER (bucket,ip) ma qualunque utente può bypassare con IP spoofati** — BASSO
`model_hub_server.cpp:412-434`
- `remote_addr` da httplib è il peer TCP reale (non spoofabile via header) — ok, ma dietro proxy/NAT tutti condividono il bucket.

### Distributed comms + HAL CUDA + activation hooks

**F7.51 — `MPICommunicator`: `size_t count` castato a `int` senza check** — MEDIO
`mpi_comm.cpp:64-110`
- `static_cast<int>(count)` in all_reduce/broadcast/all_gather/reduce_scatter/send/recv: count > INT_MAX → overflow → count negativo a MPI → UB. Nessuna validazione.

**F7.52 — `CudaBackend::get_cuda_backend` singleton non thread-safe** — MEDIO
`cuda_backend.cu:193-201`
- `g_cuda_backend` check-then-new senza atomic/mutex → due thread che chiamano insieme → double allocation, ptr perso (leak) o peggio.

**F7.53 — `CudaBackend::free` su puntatori pinned/managed fa `cudaFree` su memoria host** — ALTO
`cuda_backend.cu:86-110`, `malloc_pinned` 78-84, `malloc_managed` 70-76
- `malloc_pinned`/`malloc_managed` NON registrano in `alloc_sizes_`; se `free()` viene chiamato su tali ptr → `alloc_sizes_.find` fallisce → `cudaFree(ptr)` su memoria `cudaMallocHost`/`cudaMallocManaged` → errore CUDA silenzioso (return ignorato) o UB. E il fallback `cudaFree` non è CHECK_CUDA → errori inghiottiti.

**F7.54 — `activation_hooks` globali non atomici, scritti mentre worker OpenMP li leggono** — MEDIO
`activation_hook.cpp:9-31`
- `set_activation_hooks`/`clear_activation_hooks` assegnano la struct senza sync; hook chiamati da thread OpenMP → data race su function pointer → crash potenziale.

**F7.55 — `NCCLCommunicator::to_nccl_op` ritorna `ncclSum` per op invalidi** — BASSO
`nccl_comm.cpp:151-164`
- `ReduceOp` enum coperto, ma nessun default esplicito → op sconosciuto silenziosamente diventa Sum.

### Distributed pipeline/TP + GPU offload registry

**F7.56 — `TensorParallelLinear::scatter_input` tronca `in_features/tp_size` senza check di divisibilità** — MEDIO
`tensor_parallel.cpp:86-94`
- Se `in_features % tp_size != 0`, `local_in` arrotonda per difetto → offset errato, feature perse, niente errore.

**F7.57 — `TensorParallelLinear::local_output_buffer_` membro condiviso, non thread-safe** — MEDIO
`tensor_parallel.cpp:18-60`, header
- `column_parallel_forward` usa `local_output_buffer_` come scratch: due chiamate concorrenti sullo stesso oggetto → race. Anche `gathered`/`Y` assumono layout rank-major del comm — accoppiamento implicito.

**F7.58 — `PipelineExecutor::execute`/`execute_1f1b` memcpy da `input_data` senza null-check sul primo stage** — MEDIO
`pipeline.cpp:42-44, 151-153`
- `execute_forward` controlla `input_data != nullptr`, `execute` e `execute_1f1b` no → `memcpy` da nullptr = crash se chiamato con input nullo (es. sola backward o generazione).

**F7.59 — `execute_1f1b` warmup_bias: `(n_stages-1) - my_stage` presuppone rank==stage** — BASSO
`pipeline.cpp:144`
- Se `comm_->rank()` non coincide con l'ordine di stage atteso (es. world_size diverso), il calcolo warmup è sbagliato → deadlock o race nei send/recv.

**F7.60 — `partial_layers_registry` keyed by address: mai rimossa, dangling se modello distrutto** — MEDIO
`llama_gpu_offload.cpp:427-431, 480`
- `unordered_map<const LLaMAModel*, size_t>` static: se il modello viene distrutto e un nuovo modello allocato allo stesso indirizzo → `partial_gpu_layers` ritorna layer count di un modello diverso → offload parziale silenziosamente sbagliato. Stesso pattern F4.21.

**F7.61 — `offload_llama_model_partial` `set_device(CUDA)` senza verificare `is_available` per ogni layer** — BASSO
`llama_gpu_offload.cpp:456-477`
- Controlla backend una volta; se `set_device` fallisce a metà (OOM) → stati parziali non ripristinati, niente rollback.

### Aggiunte Fase 7 — seconda passata (hub search, safety keyword, agent parser, nomic_bert, rag)

**F7.56 — `SearchEngine::semantic_search`/`find_similar_models` ordinano per `sim` grezzo ma `r.score` è blend** — MEDIO
`search_engine.cpp:163-175, 205-215`
- `scored.emplace_back(sim, r)` e sort su `a.first` (= cosine) mentre `r.score = 0.7*sim + 0.3*jaccard` → l'ordine restituito non corrisponde allo score esposto nel risultato; due modelli con score dichiarato diverso possono apparire invertiti.

**F7.57 — Keyword matching per sostanza in safety: falsi positivi da substring** — MEDIO
`red_team.cpp:70-84` (`lexical_score`), `bias_scorer.cpp:21-44` (`SentimentScorer`), `guardrails.cpp:31-49`
- `low.find(kw)`: "kill" ⊂ "skill", "hate" ⊂ "whatever", "great" ⊂ "greater", "bad" ⊂ "badge" → falsi hit di toxicity/sentiment/blocchi. Serve word-boundary matching o tokenizzazione.

**F7.58 — `TokenToxicityClassifier` ha pesi casuali non addestrati: score ~rumore** — MEDIO
`toxicity.cpp:15-31`
- `embedding_init`/`xavier_uniform` + bias 0 → sigmoid di proiezione random ≈ 0.5 per ogni input; `is_toxic` contro threshold è arbitrario. Ulteriore limite già in F7.30 (solo riga 0 usata). È un classificatore che "mente" se integrato in un path reale — va documentato come non-addestrato o rimosso.

**F7.59 — `ActionParser` estrae `args` fermandosi al primo `}`** — MEDIO
`action_parser.cpp:45-52`
- JSON annidato `{"a":{"b":1}}` viene troncato → `json::parse` fallisce in `react.cpp` → args={} → tool eseguito con argomenti vuoti silenziosamente. Serve brace-matching o parse del blocco intero.

**F7.60 — `NomicBertEmbedder::load` non verifica `type_embd_row0_.size() >= H`** — BASSO
`nomic_bert.cpp:118-123, 256`
- Se `token_types.weight` ha meno di H float (GGUF compatto o dtype strano), `type_embd_row0_[d]` in `forward_hidden` è OOB. Anche `tok_norm_w_/b_` non sono dimension-checked.

**F7.61 — `LocalCommunicator::{all_reduce,broadcast,all_gather}(Tensor&)` scrivono `data()` senza dirty-marking** — MEDIO
`local_comm.cpp:342-377`
- `syncToHost()` + modifica `data()` grezza: se il tensor ha copia GPU, `location_` non viene marcato dirty → `syncToDevice()` successivo può saltare → GPU stale. Stessa classe dei finding F1.3/F1.12. NCCL path marca correttamente (`mark_dirty(GPU)`).

**F7.62 — `Expert::forward(const Tensor&)` leak dei nodi intermedi FFN** — MEDIO
`expert.cpp:24-37`
- `ffn_->forward(&in)` crea una catena di Node; viene cancellato solo l'ultimo (`delete out`) → tutti gli intermedi leakano a ogni chiamata (path inference/eager). Se il caller usa l'overload Tensor in loop, memoria cresce.

**F7.63 — `semantic_search`/`find_similar_models` ri-embeddano ogni documento a ogni query** — BASSO
`search_engine.cpp:161,195-204`
- `compute_document_embedding` per ogni modello per ogni ricerca: O(N modelli) forward BERT per query — nessuna cache embedding lato SearchEngine (l'embedding manager potrebbe cachare internamente, da verificare).

**F7.64 — `StdioTransport` POSIX: `pipe()` senza CLOEXEC sulle pipe della richiesta** — BASSO
`transport_stdio.cpp:114-144`
- I `pipe()` non impostano `FD_CLOEXEC`/`O_CLOEXEC`: fd di communicazione ereditati dal child beyond exec (peggio se spawn a sua volta spawn). Windows path usa handle inherit corretto.

**F7.65 — `StressRunner::run_chaos` probe "long context" conta come PASS qualunque risposta** — BASSO
`stress_runner.cpp:209-217`
- `expect_client_error=false` → `ok = got_response`: un 500 del server passa il check. Se l'intento è "graceful", andrebbe accettato solo 200/4xx, non 5xx.

**F7.66 — `sample_server_memory` ingoia eccezioni JSON in silenzio** — BASSO
`stress_runner.cpp:250-257`
- `catch(...)` vuoto → metriche ram/vram a 0 senza segnalazione; un endpoint /health che ritorna JSON rotto è invisibile.

### Checkpoint + PagedKV + CLI loader

**F7.62 — `load_checkpoint` nessuna validazione di `name_len`/`ndim`/`shape`/`version`; `total_size` dal file non confrontato con `p->data.size()`** — ALTO
`checkpoint.cpp:79-107`
- File corrotto o di modello diverso: `name_len` negativo → `std::string(name_len,' ')` con size_t enorme → bad_alloc; `ndim` negativo → `vector<size_t>(neg)` enorme; `total_size *= shape[d]` overflow; `is.read(p->data.data(), total_size*4)` scrive OOB se `total_size > p->data.size()` → heap corruption. `version` letto e ignorato. `is.read` results mai controllati → file troncato = garbage silenzioso.

**F7.63 — `save_checkpoint` scrive `p->data.data()` senza `syncToHost()`** — MEDIO
`checkpoint.cpp:45`
- Se i parametri sono GPU-resident, `data()` ritorna ptr host stale/vuoto → checkpoint con pesi garbage. Nessun dtype check (solo float).

**F7.64 — `PagedKVCache::update` non valida shape di `value` vs `key`; `gather_*`/`update` usano `item()`/`set_item` → rottura se pool su GPU** — MEDIO
`paged_kv_cache.cpp:62-171`
- `value.item({0,head,i,dim})` con shape diversa da key → OOB read silenzioso. `item`/`set_item` per elemento sono lentissimi (O(n) sync) e non funzionano correttamente se `k_pool_`/`v_pool_` sono su GPU (ctor accetta `location=GPU` e fa syncToDevice, ma gather/update non sincronizzano).

**F7.65 — `BlockManager::free` accetta block_id duplicato/esterno senza check** — MEDIO
`paged_kv_cache.cpp:38-41`
- `free_blocks_.push(block_id)` senza validazione: double-free o id mai allocato → stesso physical block dato a due sequence → KV corruption. `reset()` chiama free per ogni entry — se `update` ha lanciato a metà, `block_table_` e `free_blocks_` possono divergere.

**F7.66 — `load_llama_cli_model`: fallimento caricamento pesi = solo warning, prosegue con random init** — MEDIO
`llama_loader.cpp:176-189`
- Stessa classe di F5.17/F6.20: `weights_loaded` flag esiste ma il modello viene comunque restituito e usato. `config.vocab_size` fallback a 32000 se tokenizer vuoto → mismatch silenzioso.

**F7.67 — Directory GGUF: primo `.gguf` alfabetico caricato come tokenizer/config, indipendentemente dal modello scelto** — BASSO
`llama_loader.cpp:94-101, 131-138`
- Se la directory ha più GGUF (es. modelli quantizzati diversi), tokenizer e config vengono estratti dal primo trovato, pesi da `load_llama_weights_from_dir` — possibili mismatch arch/tokens.

### CUDA attention kernels

**F7.68 — `flash_attn_fwd_kernel`: K/V head_offset usa `gridDim.y` (num_heads) per K/V — niente GQA/MQA** — MEDIO
`flash_attn_cuda.cu:42-43`, dispatch 292
- `kv_head_offset = (batch*gridDim.y + head_id)*seq_len_k*HEAD_DIM` assume K/V hanno stesso num_heads di Q. Con grouped/multi-query attention (K/V heads < Q heads) gli offset sono sbagliati → attention su memoria errata. Host non valida `num_kv_heads`.

**F7.69 — `flash_attn` backward: `dS_ij`/`dq`/`dk` scaling duty-cycle da verificare numericamente; `s_LSE` non inizializzato per righe Q fuori range** — MEDIO
`flash_attn_cuda.cu:174-181, 212-262`
- `s_LSE[tid]` scritto solo se `q_idx < seq_len_q` ma il thread con `q_idx >= seq_len_q` non lo usa — ok; HOWEVER `s_Q`/`s_dO` per `tid` corrispondono a `q_idx = br*BLOCK_M + tid` — se `seq_len_q` non è multiplo di BLOCK_M, le righe parziali sono gestite. Verificare con test numerico: la formula `dS = p*(dP - D)` standard richiede `D = rowsum(dO*O)` e missing `*softmax_scale` placement — potenziale grad errato.

**F7.70 — `block_sparse_attention_cuda`: `cudaMalloc`/`cudaMemcpy`/`cudaFree` senza error check; `block_mask` mai `syncToHost`** — MEDIO
`block_sparse.cu:187-237`
- `cudaMalloc(&d_block_mask,...)` fallito → ptr null → `cudaMemcpy` fallisce silenzioso → kernel legge null → crash. `block_mask.item(i)` assume tensor CPU: se passato GPU tensor, `item()` non sincronizza → garbage mask. `cudaGetLastError` controllato solo DOPO `cudaFree`.

**F7.71 — `max_smem` static cache non tiene conto di cambio device** — BASSO
`block_sparse.cu:205-212`
- `static int max_smem` calcolato una volta per il device corrente; in setup multi-GPU con device diversi → limite sbagliato.

### fused_kernels.cu host-side

**F7.72 — `topk_logits_f32`/`gpu_topk_raw_candidates`/`d_meta`: static device buffers non thread-safe, mai liberati** — MEDIO
`fused_kernels.cu:1356-1376, 1560-1591, 2083-2088`
- `static unsigned *d_gmax` ecc. controllati da `if (ptr==nullptr)` + realloc `if (d_cap < cap)` senza mutex → due thread concorrenti (o stream diversi) → race su cudaMalloc/cudaFree → double-free o use-after-free. Mitigato se tutti i caller passano da `inference_mutex_`, ma niente lo garantisce. `coop_grid` static calcolato una volta — cambio device → valore stale.

**F7.73 — `gemv_quant_f16`: qtype non 3/4/5 → nessun kernel lanciato, `y_f32` resta non inizializzato** — ALTO
`fused_kernels.cu:1650-1672`
- `if (qtype==3)...else if(4)...else if(5)` senza else: tipo quant non supportato ritorna silenziosamente con output garbage. Caller non può distinguere "non supportato" da "computed".

**F7.74 — realloc path `cudaFree` senza CHECK e `d_cap` update dopo leak su eccezione** — BASSO
`fused_kernels.cu:1366-1376, 1573-1586`
- `cudaFree(d_idx)` return ignorato; se `cudaMalloc` fallisce dopo il free, `d_cap` non aggiornato ma ptr pendenti persi.

### nomic_bert + quant CUDA + trainer/static/metrics

**F7.75 — `NomicBertEmbedder::tokenize` WordPiece: match che finisce esattamente a fine parola con `start>0` marcato `failed` → UNK spurio** — MEDIO
`nomic_bert.cpp:206-228`
- `best` è inizializzato a `word.size()` e usato come sentinel "nessun match", ma un match legittimo può anch'esso finire a `word.size()`. Il check `best == word.size() && vocab_.find(word.substr(start)) == end` cerca la sotto-stringa SENZA prefisso `##` → per `start>0` fallisce sempre → `failed=true` → token UNK al posto degli pezzi corretti.

**F7.76 — `forward_hidden`: `cfg_.unk_id` non validato contro `vocab` → OOB read su `tok_embd_`** — MEDIO
`nomic_bert.cpp:248-254`
- `id = (ids[t]>=0 && ids[t]<vocab) ? ids[t] : cfg_.unk_id` — se `unk_id >= vocab` (config errata) → `tok_embd_.data() + id*H` OOB.

**F7.77 — `get_tensor`/`linear_gemm` assumono tensor CPU Float32 senza `syncToHost`/dtype check** — MEDIO
`nomic_bert.cpp:77-88, 36-46`
- `memcpy(out->data(), it->second.data(), n*4)` — se il tensor è GPU-resident o non-Float32 → garbage silenzioso. `C.syncToHost()` dopo matmul aiuta solo se C è view host non-owning — comportamento dipendente da Tensor view semantics.

**F7.78 — `dequant_matmul_cuda`: nessun check errori CUDA, nessuna validazione input** — MEDIO
`dequant_matmul.cu:87-96`
- `cudaDeviceSynchronize()` return ignorato, `cudaGetLastError` mai chiamato → launch failure silenzioso. K dispari → packing byte errato. M/N/K<=0 o block_size==0 → grid calcolato male.

**F7.79 — `quant_kernels.cu`: subnormal FP8 hand-rolled da verificare numericamente; nessun error check CUDA** — BASSO
`quant_kernels.cu:18-112, 173-220`
- `float_to_fp8_e4m3/e5m2_dev` subnormal path (`biased_exp<=0`) usa `mant*8`/`mant*4` — verificare contro spec FP8. `cudaDeviceSynchronize` ignorato in tutte le API.

**F7.80 — `Trainer::train`: eccezioni a metà step leakano grafo + `target_node`** — BASSO
`trainer.cpp:46-103`
- Se `forward`/`backward`/`step` lancia, `cleanup_graph`/`delete target_node` mai eseguiti → leak. `loss_node->data[0]` stampa solo primo elemento.

**F7.81 — `static_files` path traversal: protezione ok, ma `read_file` carica asset interi in RAM** — BASSO
`static_files.cpp:11-93`
- `weakly_canonical` + prefix check corretto; `contents << file.rdbuf()` ok per asset piccoli.

### SFT + PrefixCache

**F7.82 — `SFTTrainer::train_step`: `loss_mask`/`target`/`input` come float, nessuna upper-bound su token id** — MEDIO
`sft_trainer.cpp:65-90`, `sft_loader.cpp:24-43`
- I token id finiscono in `Tensor` float e passano a `model_.forward` → `embedding_lookup` senza validazione (F4.1): un id >= vocab_size dal tokenizer → OOB. `loss_mask[i] = i+1 >= prompt_ids.size()` maschera correttamente il prompt ma include EOS nel loss (probabilmente voluto).

**F7.83 — `PrefixCache::hash_tokens` FNV-1a a 64 bit: collisioni = KV snapshot di prompt sbagliato** — MEDIO
`prefix_cache.cpp:73-82`
- Solo hash, nessuna verifica dei tokens: due prefissi diversi con stesso hash → `find_longest_prefix` ritorna snapshot KV di un'altra sequenza → **output silenziosamente errato**. Birthday bound su 2^64 è ok ma adversarial/crafted input può forzare collisioni; almeno verificare `tokens[0..len)` contro una copia.

**F7.84 — `PrefixCache::store`: `memory_usage_` può diventare negativo se entry aggiornata ha size diversa** — BASSO
`prefix_cache.cpp:37-45`
- Aggiornamento: sottrae size vecchio snapshot, aggiunge nuovo — ok; ma se `kv.keys.size()` cambia dimensionalità (altro modello/UI) il conto resta coerente. `while` eviction: se `kv` da solo > max_memory_bytes e cache vuota → `break` → inserisce comunque superando il limite.

### Stability + init

**F7.85 — `EMAWeights::update/apply/restore` scrivono/leggono `data.data()` senza `syncToHost`** — ALTO
`ema.cpp:19-75`
- Se i parametri sono GPU-resident, `data()` punta al buffer host stale → shadow weights = garbage, `apply()` scrive nel buffer host senza `mark_dirty` → GPU mantiene pesi vecchi → **silenzioso**. `clip_grad_norm`/`loss_scaler` fanno `syncToHost` correttamente — EMA no.

**F7.86 — `DynamicLossScaler::unscale_grads` non azzera i gradienti su overflow** — BASSO
`loss_scaler.cpp:28-51`
- Ritorna false ma lascia grad con NaN/Inf scalati; il chiamante deve azzerare manualmente (diverso da `AMP::unscale_and_update` che zeroa). Infatti `update(true)` resetta solo il contatore.

**F7.87 — `weight_init::*` scrivono `w.data()` senza sync/dtype check** — BASSO
`weight_init.cpp` tutto
- Chiamato tipicamente prima dell'offload GPU (ok), ma se riusato su tensor GPU/dtype diverso → scrittura su buffer errato. `fan_in+fan_out==0` → division by zero in xavier.

### MoE layer + LoRA adapter + tools

**F7.88 — `MoELayer::forward` backward_fn cattura `token_outputs`/`out` per valore: `out` self-reference grezza** — MEDIO
`moe_layer.cpp:120-144`
- `out->backward_fn` cattura `out` (il nodo stesso) e `token_outputs` (Node* grezzi): se il grafo viene cancellato prima di `backward()` o il MoELayer distrutto → dangling. In più `token_node->backward_fn` cattura `token_node` stesso — stesso pattern.

**F7.89 — `MoELayer::forward` scala quadratic in token_count: copia per-token + FFN call per token** — BASSO
`moe_layer.cpp:71-118`
- Per ogni token: new Node + router.route + N_expert forward separati → batch [T,d] processato come T forward singoli → lentissimo e T nodes extra nel grafo.

**F7.90 — `LoRAAdapter::load` scrive in `A().data.data()`/`B().data.data()` con `a_bytes`/`b_bytes` dal file senza bound check** — ALTO
`lora_adapter.cpp:139-151`
- `a_bytes` da JSON usato direttamente in `in.read` → se > dimensione tensor → heap overflow. `header_len` non validato → alloc enorme. `in.read` return mai controllato. Se `layers_` è vuoto (load prima di add_layer) tutti gli entry sono skip → file "caricato" senza effetto.

**F7.91 — `ToolRegistry::execute`/`validate_schema`: `tools_` non sincronizzato** — MEDIO
`registry.cpp:140-247`
- `register`/`unregister`/`execute` senza mutex → race se tool registrati a runtime mentre richieste in volo. `validate_schema` `properties.contains(it.key())` — `it.key()` su json iterator ok.

**F7.92 — `FunctionCallParser::extract_json` regex greedy: JSON nested >2 livelli non matcha** — BASSO
`parser.cpp:11-19`
- Pattern `\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}` supporta max 2 livelli di nesting; tool call con args annidati profondi → parse fallisce silenziosamente.

### Hub main + search embedding + bench

**F7.93 — `signal_handler` chiama `exit(0)` da signal handler: leak/destructors saltati, race con worker attivi** — MEDIO
`hub_main.cpp:28-34`
- `exit()` in signal handler è async-signal-unsafe; thread httplib attivi vengono killati di netto → potenziali write a metà su `hub_users.json`/`models_metadata.json`.

**F7.94 — `EmbeddingManager::store_embedding_matrix` `dim_major` guess ambiguo** — BASSO
`search_engine.cpp:~640-680`
- Se `rows==cols` o shape ambigua, `dim_major`/`vocab` indovinati → embedding matrix trasposta silenziosamente → similarity search su features sbagliate. Hash fallback (FNV) ha collisioni frequenti: `h % kDim` a 10 bit segnati — qualità search bassa ma non crash.

**F7.95 — `bench/memory.cpp`: `new LLaMAModel` + `delete` manuale, eccezioni tra le due → leak** — BASSO
`bench/memory.cpp:~50-90`
- Uso di raw new/delete invece di unique_ptr; se un'eccezione viene lanciata tra alloc e delete → leak (processo bench, basso impatto).

**F7.96 — `run_process` (process_utils.h): nessun timeout, `WaitForSingleObject(INFINITE)`/`waitpid` bloccante** — MEDIO
`process_utils.h:135-217`
- Usato da `hub::downloader` per curl: un download hanging blocca il chiamante per sempre. Stesso problema di F7.39.

## FASE 7b — experimental/ (71 file compilati, ~7.3k righe)

### autoopt

**F7b.1 — `Sandbox` esegue codice generato da LLM in-process via dlopen, senza sandboxing reale** — ALTO (design)
`experimental/autoopt/sandbox.cpp:319-356`, `refactorer.cpp:100-123`
- `compile()` compila C++ generato dall'LLM e `load_kernel` lo `dlopen`a nella stessa processo con `RTLD_NOW` → codice arbitrario generato da un modello esterno gira con tutti i privilegi del server. Il nome "Sandbox" è fuorviante: nessun isolamento (nessun container/seccomp/rlimit). Se l'LLM endpoint è compromesso o prompt-injection, è RCE. Documentare come "solo uso locale/trusted".

**F7b.2 — `create_temp_dir` usa `random_device`+mt19937 con solo 4 cifre → path prevedibile** — MEDIO
`sandbox.cpp:85-96`
- `uniform_int_distribution(1000,9999)` → ~9000 possibilità; su sistemi multi-utente o con symlink pre-creati → temp dir squatting. `std::random_device` seed + PRNG non crittografico.

**F7b.3 — `Sandbox::verify`/`benchmark` assumono Tensor CPU Float32, nessun dtype/location check** — MEDIO
`sandbox.cpp:247-265`, `load_kernel` 348-355
- `ref_out[i]`/`cand_out[i]` via `operator[]` senza `syncToHost`; `load_kernel`'s KernelFn chiama `fn(in_c.data(), ...)` con `const float*` ma non verifica `in_c.dtype()==Float32` → tensor F16 reinterpretato come float → kernel generato legge/scrive buffer con dtype sbagliato.

**F7b.4 — `HotSwapManager::rollback` non fa `dlclose` della libreria della versione rimossa** — MEDIO
`hot_swap.cpp:120-130`, `hot_swap_from_library` 89-118
- `loaded_libraries_` accumula handle per ogni swap; rollback pop la fn ma la .so resta mappata per sempre → leak di memoria + codice non più referenziato ma eseguibile. Chiave `name+"_v"+size` collide dopo rollback+reswap → overwrite → handle perso.

**F7b.5 — `optimize_hotspot` compila `candidate_source` ma benchmarka `candidate_kernel` passato dal caller** — MEDIO
`refactorer.cpp:31-39`
- La compilazione è solo un gate; il kernel effettivamente hot-swappato è `candidate_kernel` (funzione C++ del chiamante), non la libreria compilata da `candidate_source` → se i due divergono, si verifica una cosa e si installa un'altra. `optimize_with_llm` usa correttamente `load_kernel`.

**F7b.6 — `commit_optimization` chiamato sempre con `dry_run=true`** — BASSO
`refactorer.cpp:70`
- Il commit git reale non avviene mai; `commit_history_` cresce solo in RAM → feature incompleta.

**F7b.7 — `llm_client::split_url` `std::atoi` su port e scarto userinfo silenzioso** — BASSO
`llm_client.cpp:30-41`
- `port = atoi(...)`: "abc"→0→reject ok; "80abc"→80 accepted. Userinfo `user:pass@host` viene stripato e le credenziali perse (forse voluto). Nessun timeout su connect=10s/read=120s — ok.

**F7b.8 — `extract_code_from_response` ritorna la response INTERA se non trova code fence** — BASSO
`llm_client.cpp:82-106`
- Prosa senza ```cpp → il testo intero va in `compile()` → fallisce la compilazione (wasted cycle) o peggio: se la prosa contiene ``` mascherate, estrae blocco sbagliato.

### agent

**F7b.9 — `ReActEngine::run` esegue tool del registry con args JSON del modello, nessun allowlist/validation oltre schema** — MEDIO
`react.cpp:53-60`, `action_parser.cpp`
- L'output del LLM decide quale tool chiamare e con quali argomenti; `tool_registry->execute` valida solo schema JSON, non semantica → prompt injection nel modello = invocazione arbitraria dei tool registrati. `ActionParser::parse` estrae JSON con `find('{')`/`find('}')` → args annidati si rompono (primo `}` chiude presto) → `args_json` troncato → parse fail o args parziali.

**F7b.10 — `Supervisor::plan_tasks` parsing JSON "a mano" con find/quotes** — MEDIO
`supervisor.cpp:43-76`
- Estrae "agent"/"prompt" cercando virgolette nel testo grezzo: JSON con escape `\"`, valori con `:`/`;` o nesting → parse errato → task con agent_name errato → "not found" o peggio match su nome sbagliato. `pos = q2` dopo ogni match: se `q2==npos` → loop infinito? No: `pos` avanzato ma `plan.find("\"agent\"", pos)` con pos beyond size → npos → fine. Ma se `q1==npos`, `q2` resta npos e `pos=q2`=npos → fine.

### federated

**F7b.11 — `add_dp_noise` usa `std::mt19937 gen(42)` fisso → rumore DP deterministico e prevedibile** — ALTO
`dp_noise.cpp:28-35`
- Seed fisso 42 per OGNI chiamata → lo stesso delta produce lo stesso noise → la "privacy" è zero: un avversario che conosce il delta può sottrarre esattamente il rumore. DP richiede CSPRNG o almeno seed casuale. `normal_distribution` su mt19937 non è crittograficamente sicuro comunque.

**F7b.12 — `compute_epsilon` formula RL accountant semplificata, non verificata** — BASSO
`dp_noise.cpp:40-54`
- Implementazione custom di advanced composition: `eps_single = sqrt(2 ln(1.25/δ))/σ` poi composition — plausibile ma non è il bound standard (manca δ composition corretta); solo da trattare come stima euristica.

**F7b.13 — `FederatedAggregator::aggregate` assume tutti i delta stessa shape, `operator+` silenzioso su mismatch** — BASSO
`aggregator.cpp:19-28`
- `sum + it->second` con shape diversi → comportamento di Tensor::operator+ non verificato qui (potrebbe throw o broadcast errato).

### quantum

**F7b.14 — `QuantumState` `1<<n_qubits` overflow: n_qubits>=32 → size_t ok su 64-bit ma `size_t(1)<<n` per n>=64 UB; n>26 → >64M amp → OOM** — MEDIO
`state.cpp:77-82`
- Nessun bound check su n_qubits: `QuantumState(40)` → `1<<40` amp = ~16TB alloc → crash/OOM. `apply_gate`/`apply_cnot`/`measure` iterano `total_dim` — con n grande hang. Solo euristico.

**F7b.15 — `measure` collapse: `norm_factor` clamp 1e-8 maschera stati non normalizzati** — BASSO
`state.cpp:168-181`
- Se `prob_1`≈0 o 1 (stato quasi deterministico), divide per sqrt(1e-8)=1e-4 → amplifica errori FP; outcome `(rand < 1-prob_1)` con prob_1 leggermente >1 (errore numerico) → `1-prob_1` negativo → outcome sempre 1.

**F7b.16 — `apply_gate`/`apply_cnot`/`apply_cz` mutano `amplitudes_` in-place durante iterazione: gate ok (coppie disgiunte), CNOT swap ok; ma nessuna normalizzazione post-gate** — BASSO
`state.cpp:92-144`
- Gate unitarie preservano norma in teoria; accumulo FP drift non corretto (nessun renormalize). `apply_cz` con control==target: mask ha bit singolo, `(i&mask)==mask` → flip segno per tutti i basis con quel bit → equivale a Z su quel qubit, non a identità — comportamento ambiguo.

### rlhf

**F7b.17 — `kl_divergence`/`ppo_loss` usano `min(size)` silenzioso su vettori mismatched** — MEDIO
`ppo.cpp:17,88`
- Se `log_probs_new.size() != log_probs_old.size()` → troncamento silenzioso → KL/ratio calcolati su subset → reward/gradienti errati senza errore. Stessa classe dei finding "silent truncation" della Fase 7.

**F7b.18 — `ppo_clipped_objective` ritorna `min(ratio*A, clip*A)` ma NON tiene il max con `clip*A` nel caso A<0** — ALTO (correttezza PPO)
`ppo.cpp:77-81`
- PPO standard: `min(ratio*A, clip(ratio)*A)` vale solo se si prende poi `-min(...)` come loss; ma la formula corretta del surrogate è `min(ratio*A, clipped_ratio*A)` SOLO quando si massimizza — con `std::fmin` e poi `-total` in `ppo_loss`, per A negativo il `fmin` sceglie il più negativo → incentiva allontanare ratio dal clip → **segno invertito per advantage negativi**. Verificare: con A<0, ratio>1+eps → clipped*A = (1+eps)*A < ratio*A? No: A<0 → (1+eps)*A < ratio*A → fmin sceglie clipped*A = più negativo → loss -negative = positivo... va ricalcolato, ma il punto è che `fmin` non è `min` del PPO standard che usa `std::min(ratio*A, clipped*A)` per massimizzare — in loss negato è `max`. Da verificare con test numerico.

**F7b.19 — `kl_divergence` somma `p*(logp_new - logp_ref)` dove p=exp(logp_new) non normalizzato** — BASSO
`ppo.cpp:19-20`
- Se `log_probs_new` non sono log-prob normalizzate (somma p≠1) → KL errato. Assume input già normalizzato, non documentato.

### rag

**F7b.20 — `VectorDBClient` è un finto "client": tutto in-RAM, nessuna persistenza/rete, `Config` inutilizzata** — BASSO (design)
`vector_client.cpp`
- `search` O(N) cosine su tutti i documenti, nessun indice; `documents_` mai serializzato → restart = perdita. Il nome "client" implica servizio esterno che non esiste.

**F7b.21 — `VectorDBClient` nessuna thread-safety: `search`/`upsert`/`remove` concorrenti → data race** — MEDIO
`vector_client.cpp:50-139`
- `documents_` vettore modificato/letto senza mutex; se usato da più thread (es. server RAG) → crash.

### inference (sperimentale — diverso da src/infer)

**F7b.22 — `generate()`/`beam_search()` leakano `logits_node` a ogni step E peggio: `input_node` è stack-allocated** — ALTO
`experimental/inference/generate.cpp:34-37,104-105`
- `model.forward(&input_node)` riceve puntatore a Node su stack: se forward registra `input_node` come child nel grafo, `logits_node` leakato trattiene un puntatore dangling → UAF latente se mai si chiama backward. `logits_node` mai `delete` → leak grafo intero per token (stessa classe di F5.1). `beam_search`: ×beam_width leak per step.

**F7b.23 — `beam_search` `partial_sort(begin, begin+beam_width)` senza verificare `beam_width <= vocab_size`** — ALTO
`generate.cpp:126-129`
- Se `beam_width > vocab_size` → iteratore oltre `end()` → UB/crash. `beam_width` è parametro libero, nessun clamp.

**F7b.24 — beam score senza length-normalization: beam finiti (EOS) competono con score cumulativo minore** — MEDIO
`generate.cpp:131-148`
- `new_beam.score = beam.score + lp_data[idx]` accumula logprob senza dividere per lunghezza → i beam "finished" vengono penalizzati vs quelli più lunghi → tendenza a scegliere sequenze infinite. Comportamento standard ma da documentare.

**F7b.25 — `generate()` assume logits contigui (seq,vocab) senza check: `data() + (seq_len-1)*vocab_size`** — MEDIO
`generate.cpp:41-50`
- Se `logits_tensor` è 1-D o non contiguo o shape diversa → offset errato → logits sbagliati silenziosamente. Nessuna verifica `ndim==2`.

### rlhf (seguito)

**F7b.26 — `RewardModel::score` crasha su `tokens` vuoto: `at(seq_len-1)` con seq_len=0** — MEDIO
`reward_model.cpp:28,139`
- `tokens.shape()[0]==0` → `forward_hidden` crea Tensor({0,hidden}) → `hidden->data.at(SIZE_MAX, d)` → OOB. Nessun check early-return.

**F7b.27 — `free_graph` elimina nodi intermedi mentre i param Node possono avere `children` che li referenziano** — ALTO
`reward_model.cpp:62-70`
- I param sono esclusi dalla delete, ma se un param Node ha `children`/`backward_fn` che puntano a intermedi eliminati → dangling pointer → UAF al prossimo backward (stessa classe di F4.4). `topological_sort(root)` copre solo i discendenti del root, non i link dai param.

**F7b.28 — `generate_with_logprobs`: `last_row = data + (nrows-1)*vocab_size` assume logits (rows,vocab) contigui** — MEDIO
`reward_model.cpp:243-244`
- Se `llama_forward_single_token` ritorna shape {vocab} 1-D → `nrows=vocab_size` → offset `(vocab-1)*vocab` → OOB read massiccio. Verificare shape reale del ritorno.

**F7b.29 — `PPOTrainer::train_step` non fa MAI backward né aggiorna pesi: `all_log_probs_new == all_log_probs_old` → ratio=1, KL=0** — ALTO
`ppo_trainer.cpp:77-78,105-122`
- `all_log_probs_old.push_back(gen.log_probs)` copia esattamente new → `ppo_loss` con ratio=exp(0)=1 → loss = -advantage costante, nessun gradiente possibile; e non c'è nemmeno chiamata a backward/optimizer. L'intero "PPO training" è un no-op matematico: ritorna un numero che non aggiorna nulla. `reward_head` è `rand()` e il transformer nel RewardModel ha pesi casuali scollegati dal policy model.

### quantum/circuit

**F7b.30 — `parameter_shift_gradient` `shifted.resize(P)` silenzioso: params più corti → zeri fantasma, più lunghi → param ignorati** — BASSO
`circuit.cpp:49-52`
- `num_params()` deve coincidere esattamente con i param consumati da `execute()` (n_layers*n_qubits); se il caller passa un vettore diverso → gradienti calcolati su param che non esistono o param reali saltati. Nessuna validazione `params.size()==P`.

### agent/consensus

**F7b.31 — `DelphiConsensus::cosine_sim` usa `embed_cache_` mutabile in metodo const, cache illimitata** — MEDIO
`consensus.cpp:70-86`
- Ogni stringa response vista in eterno nella cache (grow-only, mai invalidata) → leak in debate loop. `embed_fn_` può essere nullptr? Controllato da `if(embed_fn_)` ok.

**F7b.32 — `debate()` nessun null-check su `agents_[i]`** — BASSO
`consensus.cpp:144`
- `register` di nullptr o agent distrutto → crash. Anche `majority_vote` clustering greedy order-dependent: similarità non transitiva → cluster arbitrari.

---
## Aggiornamento remediation (2026-11-24)

Fix applicati: F7b.11 (DP seed), F7b.22+F7b.23 (generate/beam_search graph leak + beam_width clamp), F7b.26 (reward_model empty-tokens guard), F4.1 (embedding_lookup token validation → throw). F7b.18 verificato come non-bug (formula PPO corretta). F7b.29 = limitazione di design del modulo RLHF didattico. Fix manuali utente: downloader path traversal, GGUF nested-KV depth cap, server payload/embeddings cap, CUDA prefill DQ cache. Tracker: `audit_remediation.md`. Verifica: ninja build ok, 717/719 test (2 flaky passati al rerun), 6 skip modelli reali.

## Aggiornamento remediation (2026-11-25)

Fix applicati in questa sessione:

- **F1.1** `Tensor::fill` rispetta il layout (no flat-fill su view strided/broadcast) e marca `location_=CPU` se esiste copia GPU.
- **F1.2** `syncToHost`/`syncToDevice`: memcpy piatto solo se `is_contiguous()`; altrimenti copia element-wise via `compute_offset` (fixa OOB su `expand`/slice/permute).
- **F1.3/F1.4** `operator+= -= + - * /` (Tensor e scalar): `syncToHost()` sugli operandi prima del fast path e dirty-marking CPU dopo le scritture in-place.
- **F1.5** Copy ctor: `memcpy_d2d` solo se `other.is_contiguous()`; fix anche del copy host element-wise che scriveva flat invece che per indici.
- **F1.16** `operator/(float)` ora fa divisione elemento-e-levamento (non `x*(1/s)`).
- **F7.1** `load_checkpoint`: validazione magic/version/num_params/name_len/ndim/shape (no overflow, no zero-dim), `total_size` deve coincidere con `p->data.size()`, check `is_contiguous()`, size di `state.m/v`, controlli stream dopo ogni read, dirty-mark CPU su param GPU.
- **F7.52** `get_cuda_backend`: singleton via function-local static (magic statics, thread-safe).
- **F7.53** `CudaBackend::free`: i puntatori `malloc_pinned` sono tracciati in `pinned_ptrs_` e liberati con `cudaFreeHost` (la managed memory era già corretta via `cudaFree`).
- **F7.8** `LoRALinear::unmerge` sottrae il delta esatto salvato in `merged_delta_` da `merge()` (no ricomputo B*A).
- **F7.9** `LoRAAdapter::load`: header_len bounded, offsets A/B rispettati, size validate vs tensori attesi, controlli stream, dirty-mark CPU.
- **F7.7** verificato non-bug: `blas::dispatch` materializza già `.contiguous()` sugli operandi (solo costo di copia).
- **F7.56** `semantic_search`/`find_similar_models`: ordinamento ora per blended score esposto (era cosine grezzo).
- **F7.57** `red_team`/`bias_scorer`: keyword matching con word-boundary ("kill" non matcha più "skill", "great" non matcha "greater"). Guardrails blocklist lasciato substring di proposito.
- **F7.59** `ActionParser`: estrazione `args` a brace bilanciate con gestione stringhe/escape (JSON annidato ok).
- **F7.60** `NomicBertEmbedder::load`: check dimensioni `type_embd_row0_`, norm vettori e norm per layer (no OOB su GGUF compatto).
- **F7.61** `LocalCommunicator::{all_reduce,broadcast,all_gather}` Tensor overloads: dirty-mark CPU dopo collectives su host.
- **F7.62** `Expert::forward(Tensor)`: cleanup grafo autograd completo via `topological_sort` (solo params FFN + input conservati) — prima leak di tutti i nodi intermedi.
- **F7.65** `run_chaos`: probe "long context" non conta più HTTP 500 come PASS.
- **F7.16** `MPICommunicator`: `mpi_count()` rifiuta count > INT32_MAX prima del cast a `int`.
- **F7.54** `activation_hooks`: `std::atomic<ActivationHooks>` snapshot (fix race OpenMP writer/reader); `activation_hooks()` ritorna per valore.

Da verificare/fix successivi: F7.46 (upload RAM), F7.49 (registration race), F7.35 (setrlimit), F7.63 (embed cache), F7b residui.

### Verifiche extra (2026-11-24, remediation)

- **F6.17 NON-BUG**: `general.file_type=1` in GGUF = `MOSTLY_F16` nell'enum ufficiale (0=F32, 1=F16, 2=Q4_0...). Coerente coi tensori F16 scritti. L'audit l'aveva interpretato come Q4_0 — errato.
- **F7.1 già fixato**: `load_checkpoint` ha validazione completa (magic, version, bounds, overflow, size match, contiguity, stream state, GPU dirty).

## Aggiornamento remediation (2026-11-25, secondo batch)

- **F7.49** `register_user`: `users_mutex_` serializza check-then-write; `save_users` ora atomico (tmp + rename, con fallback remove+rename su Windows).
- **F7.35** `setup_resource_limits`: `setrlimit` POSIX ora opt-in via `SERIUS_PLUGIN_SETRLIMIT=1` (era process-wide → DoS self-inflicted); warn esplicito di default.
- **F7.44** download `content_provider`: clamp di `length` a `fsize-offset` (no allocazione enorme da range request).
- **F7.47** `validate_model_file`: verifica reale magic — "GGUF" oppure header safetensors (uint64 len + '{').
- **F7.63** `EmbeddingManager::compute_embedding`: cache bounded (8192, clear-on-full, mutex) in `compute_embedding_uncached` wrapper; invalidata in `unload_model`.
- **F7.63-alt** (`save_checkpoint` syncToHost): verificato non-bug — `Tensor::data()` chiama già `syncToHost()`.
- **F7.46** upload multipart in RAM: limitazione intrinseca di httplib (body già bufferizzata); fix reale richiede streaming upload — documentato, non fixato.

Verifica: test_hub_server 2/2, test_hub_search 4/4, test_hub_downloader 5/5.
