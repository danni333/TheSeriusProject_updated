# Getting Started — TheSeriusProject

Guida rapida per passare da zero a server API e chat in pochi minuti.

---

## 1. Requisiti

- **Compilatore C++20** (MSVC 2022, GCC 12+, Clang 16+)
- **CMake 3.25+**
- **CUDA Toolkit 12.x** (opzionale, per GPU NVIDIA)

---

## 2. Compilazione

```bash
# Clone (se non l'hai già)
git clone https://github.com/<user>/TheSeriusProject.git
cd TheSeriusProject

# Configura e build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Con CUDA:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DSERIUS_USE_CUDA=ON

cmake --build . --config Release
```

L'eseguibile CLI si trova in `build/serius` (o `build/Release/serius.exe` su Windows).

---

## 3. Avviare il server API

```bash
# Dalla cartella build
./serius serve --port 8080
# oppure: ./serius serve
```

Il server è in ascolto su `http://0.0.0.0:8080`.

- **Health:** `curl http://localhost:8080/health`
- **Modelli:** `curl http://localhost:8080/v1/models`
- **Chat (stub):**  
  `curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Ciao!\"}]}"`

---

## 4. Comando Chat (in sviluppo)

```bash
./serius chat --model path/to/model.gguf
```

Il caricamento modelli (GGUF/safetensors) e l’inferenza reale sono in fase di implementazione. Per ora il comando stampa un messaggio informativo.

---

## 5. Eseguire i test

```bash
cd build
ctest --output-on-failure
```

---

## 6. Configurazione modelli

I file di configurazione JSON per i modelli sono in `configs/`:

- `configs/model_7b.json` — esempio 7B
- `configs/model_1b.json` — esempio 1B
- `configs/training_config.json` — parametri di training

---

## Prossimi passi

- **Installazione dettagliata:** `docs/installation.md`
- **Riferimento API:** `docs/api-reference.md`
- **Architettura:** `docs/architecture.md`
