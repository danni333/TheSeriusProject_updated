# Fase 57 — Voice Interface (STT + TTS Locale)

> Un modello che parla e ascolta è un prodotto completamente diverso.
> Speech-to-Text e Text-to-Speech locali, senza cloud, senza API esterne.

---

## 1. Architettura

```text
🎤 Microfono → STT Engine → Testo → LLM → Testo → TTS Engine → 🔊 Speaker
                (Whisper.cpp)              (Serius)              (Piper/Bark)
```

---

## 2. Speech-to-Text (STT)

### Whisper.cpp (locale, no Python)

| Feature | Dettaglio |
| --- | --- |
| **Modello** | OpenAI Whisper (tiny → large-v3) |
| **Implementazione** | whisper.cpp (MIT ✅, C/C++ puro) |
| **Lingue** | 99+ lingue, auto-detect |
| **Latenza** | ~1s per 10s di audio (GPU), ~3s (CPU) |
| **Formati** | WAV, MP3, FLAC (via libsndfile) |

```cpp
namespace serius::voice {

class STTEngine {
    whisper_context* ctx_;

public:
    bool load(const string& model_path);  // whisper-base.bin, whisper-large-v3.bin

    string transcribe(const vector<float>& audio_pcm, int sample_rate = 16000);

    // Streaming: trascrivi in tempo reale chunk per chunk
    void transcribe_stream(AudioStream& stream, function<void(string)> on_text);
};

} // namespace serius::voice
```

---

## 3. Text-to-Speech (TTS)

### Piper TTS (locale, no cloud)

| Feature | Dettaglio |
| --- | --- |
| **Modello** | Piper (VITS-based) |
| **Licenza** | MIT ✅ |
| **Lingue** | 30+ lingue, voci maschili/femminili |
| **Latenza** | ~200ms per frase (CPU) |
| **Output** | WAV PCM 22050 Hz |

```cpp
class TTSEngine {
    piper::PiperConfig config_;
    piper::Voice voice_;

public:
    bool load(const string& model_path, const string& voice_name);

    vector<float> synthesize(const string& text);  // PCM float

    // Streaming: genera audio frase per frase
    void synthesize_stream(const string& text, function<void(vector<float>)> on_audio);
};
```

---

## 4. Integrazione CLI e API

```bash
# Chat vocale locale
serius voice --model model.gguf --stt whisper-base --tts piper-it

# API endpoint
POST /v1/audio/transcriptions   → STT (compatibile OpenAI)
POST /v1/audio/speech           → TTS (compatibile OpenAI)
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/voice/stt_engine.h/.cpp` | Wrapper Whisper.cpp |
| `src/voice/tts_engine.h/.cpp` | Wrapper Piper TTS |
| `src/voice/audio_stream.h/.cpp` | Cattura audio da microfono |
| `src/voice/voice_chat.h/.cpp` | Pipeline STT → LLM → TTS |
| `src/cli/cmd_voice.cpp` | Comando `serius voice` |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_stt_basic` | Audio noto → trascrizione corretta |
| `test_tts_basic` | Testo → audio WAV valido (non silenzio) |
| `test_voice_roundtrip` | Parla → trascrivi → rispondi → sintetizza |
| `test_stt_streaming` | Stream audio real-time → testo incrementale |
