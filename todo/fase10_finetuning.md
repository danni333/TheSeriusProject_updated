# Fase 10 — Fine-Tuning e Allineamento (SFT + DPO + SimPO)

> Il modello pre-addestrato sa solo "completare il testo".
> Per renderlo un assistente, serve lo strato di istruzioni (SFT)
> e strategie di generazione intelligenti (sampling).

---

## 1. Supervised Fine-Tuning (SFT)

### Obiettivo

Addestrare il modello su coppie **"Istruzione - Risposta"** così che impari a seguire comandi anziché completare testo arbitrario.

### Formato dei dati

```json
{
    "instruction": "Traduci in inglese:",
    "input": "Il gatto e' sul tavolo",
    "output": "The cat is on the table"
}
```

### Formattazione del prompt

```
[SYSTEM] Sei un assistente utile.
[USER] Traduci in inglese: Il gatto e' sul tavolo
[ASSISTANT] The cat is on the table [EOS]
```

### Loss masking

La loss viene calcolata **solo sulla risposta**, non sull'istruzione:

```
Token:  [SYS] Sei un ... [USR] Traduci ... [ASST] The cat ...  [EOS]
Mask:    0    0   0       0     0           1    1   1     1
Loss:    -    -   -       -     -         cross_entropy su questi ↑
```

### SFT DataLoader

```cpp
struct SFTSample {
    vector<int> input_ids;     // prompt completo tokenizzato
    vector<int> target_ids;    // shifted di 1
    vector<bool> loss_mask;    // true solo per i token di risposta
};

class SFTDataLoader {
    vector<SFTSample> samples;
    BPETokenizer& tokenizer;

    SFTSample prepare(const json& example) {
        string prompt = format_prompt(
            example["instruction"],
            example["input"],
            example["output"]
        );
        auto ids = tokenizer.encode(prompt);
        // Trova dove inizia la risposta
        int resp_start = find_response_start(ids);
        // Crea mask
        vector<bool> mask(ids.size(), false);
        for (int i = resp_start; i < ids.size(); i++)
            mask[i] = true;
        return {ids, shift(ids, 1), mask};
    }
};
```

---

## 2. Sampling Strategies

### Temperature Scaling

Controlla la "creatività" del modello modificando la distribuzione di probabilità.

```
logits_scaled = logits / temperature

temperature = 0.1  → molto deterministico (picco su un token)
temperature = 1.0  → distribuzione originale
temperature = 2.0  → molto casuale (distribuzione piatta)
```

### Top-K Sampling

Considera solo i **K token più probabili**, redistribuisce la probabilità tra loro.

```cpp
vector<int> top_k_sample(const Tensor& logits, int k, float temp) {
    // 1. Applica temperature
    Tensor scaled = logits / temp;
    // 2. Trova i top-K indici
    auto top_indices = argsort_descending(scaled);
    top_indices.resize(k);
    // 3. Prendi i logits corrispondenti
    Tensor top_logits = gather(scaled, top_indices);
    // 4. Softmax solo su K valori
    Tensor probs = softmax(top_logits);
    // 5. Campiona da questa distribuzione
    return categorical_sample(probs, top_indices);
}
```

### Top-P (Nucleus) Sampling

Prende il set minimo di token la cui probabilità cumulativa supera **P**.

```cpp
int top_p_sample(const Tensor& logits, float p, float temp) {
    Tensor scaled = logits / temp;
    Tensor probs = softmax(scaled);
    // 1. Ordina per probabilità decrescente
    auto sorted = argsort_descending(probs);
    // 2. Calcola probabilità cumulativa
    float cumsum = 0;
    int cutoff = 0;
    for (int i = 0; i < sorted.size(); i++) {
        cumsum += probs[sorted[i]];
        if (cumsum >= p) { cutoff = i + 1; break; }
    }
    // 3. Ritaglia e ri-normalizza
    auto nucleus = vector(sorted.begin(), sorted.begin() + cutoff);
    Tensor nucleus_probs = gather_and_renormalize(probs, nucleus);
    // 4. Campiona
    return categorical_sample(nucleus_probs, nucleus);
}
```

| Parametro | Tipico | Effetto |
|---|---|---|
| `temperature` | 0.7 | Bilanciato creatività/coerenza |
| `top_k` | 40 | Evita token rarissimi |
| `top_p` | 0.9 | Nucleus: adattivo al contesto |

### Beam Search

Mantiene **B sequenze candidate** (beam) e ad ogni step espande le più probabili.

```
Step 0: beam = ["Il"]
Step 1: beam = [("Il gatto", -1.2), ("Il cane", -1.5), ("Il sole", -2.0)]
Step 2: per ogni beam, genera top-K continuazioni
        raccogli tutti i candidati, tieni i top-B
        beam = [("Il gatto è", -2.1), ("Il gatto mangia", -2.3), ...]
...
Fino a [EOS] o max_length
```

```cpp
struct Beam {
    vector<int> tokens;
    float score;         // log-probability cumulativa
    bool finished;
};

vector<Beam> beam_search(Model& model, vector<int> prompt,
                          int beam_width, int max_len) {
    vector<Beam> beams = {{prompt, 0.0f, false}};
    for (int step = 0; step < max_len; step++) {
        vector<Beam> candidates;
        for (auto& beam : beams) {
            if (beam.finished) { candidates.push_back(beam); continue; }
            Tensor logits = model.forward(beam.tokens);
            Tensor log_probs = log_softmax(logits.last());
            auto top = topk(log_probs, beam_width);
            for (auto [idx, lp] : top) {
                Beam new_beam = beam;
                new_beam.tokens.push_back(idx);
                new_beam.score += lp;
                new_beam.finished = (idx == EOS_TOKEN);
                candidates.push_back(new_beam);
            }
        }
        // Tieni i top beam_width per score
        sort(candidates, by_score_desc);
        beams = vector(candidates.begin(),
                       candidates.begin() + beam_width);
        if (all_finished(beams)) break;
    }
    return beams;
}
```

> [!NOTE]
> **Beam Search** è deterministico e trova la sequenza più probabile,
> ma tende a generare testo ripetitivo. Combinarlo con Top-P o
> length penalty per risultati migliori.

---

## 3. Repetition Penalty

Per evitare che il modello ripeta le stesse frasi:

```cpp
void apply_repetition_penalty(Tensor& logits,
                                const vector<int>& past_tokens,
                                float penalty) {  // tipicamente 1.1-1.3
    for (int token : past_tokens) {
        if (logits[token] > 0)
            logits[token] /= penalty;
        else
            logits[token] *= penalty;
    }
}
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/finetune/sft_loader.h/.cpp` | SFT DataLoader con loss masking |
| `src/finetune/sft_trainer.h/.cpp` | Training loop SFT |
| `src/inference/sampler.h/.cpp` | Top-K, Top-P, Temperature, Repetition Penalty |
| `src/inference/beam_search.h/.cpp` | Beam Search con length penalty |
| `src/inference/generate.h/.cpp` | Loop di generazione completo |
| `tests/test_sampling.cpp` | Unit test per ogni strategia |
| `tests/test_sft.cpp` | Test SFT: loss mask, formato prompt |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_temperature` | T=0.01 → quasi deterministico; T=2.0 → quasi uniforme |
| `test_topk` | Solo K token hanno probabilità > 0 |
| `test_topp` | Somma prob dei token selezionati >= P |
| `test_beam_search` | Output ottimale vs greedy su toy vocabulary |
| `test_repetition` | Con penalty, token già usati hanno prob ridotta |
| `test_sft_mask` | Loss calcolata solo su token di risposta |
| `test_sft_convergence` | 50 step SFT su 10 coppie: loss decresce |
| `test_generate_eos` | Generazione si ferma a [EOS] |
