# Fase 53 — Plugin System & Estensibilità

> Un prodotto maturo permette a terzi di estendere le funzionalità
> senza modificare il core. Un sistema di plugin leggero per aggiungere
> tool, backend, formati di modello e pipeline personalizzate.

---

## 1. Architettura Plugin

```text
serius (core)
  │
  ├── Built-in plugins (compilati nel binario)
  │   ├── plugin_gguf        → Loader GGUF
  │   ├── plugin_safetensors → Loader Safetensors
  │   ├── plugin_cuda        → Backend GPU NVIDIA
  │   └── plugin_mcp         → Client MCP
  │
  └── External plugins (shared libraries, caricati a runtime)
      ├── plugin_rocm.so     → Backend GPU AMD
      ├── plugin_whisper.so  → Speech-to-text
      └── plugin_tts.so      → Text-to-speech
```

---

## 2. Interfaccia Plugin

```cpp
namespace serius::plugin {

// Ogni plugin implementa questa interfaccia
class IPlugin {
public:
    virtual ~IPlugin() = default;
    virtual string name() const = 0;
    virtual string version() const = 0;
    virtual PluginType type() const = 0;  // Backend, Loader, Tool, Pipeline

    virtual bool init(const json& config) = 0;
    virtual void shutdown() = 0;
};

// Specializzazioni
class IBackendPlugin : public IPlugin {
public:
    virtual Tensor alloc(Shape shape, DType dtype) = 0;
    virtual void matmul(const Tensor& A, const Tensor& B, Tensor& C) = 0;
};

class IToolPlugin : public IPlugin {
public:
    virtual string tool_name() const = 0;
    virtual json schema() const = 0;      // JSON Schema del tool
    virtual json execute(const json& args) = 0;
};

class ILoaderPlugin : public IPlugin {
public:
    virtual bool can_load(const string& path) const = 0;
    virtual Model load(const string& path) = 0;
};

// Macro per registrare un plugin
#define SERIUS_REGISTER_PLUGIN(PluginClass) \
    extern "C" serius::plugin::IPlugin* create_plugin() { \
        return new PluginClass(); \
    }

} // namespace serius::plugin
```

---

## 3. Plugin Manager

```cpp
class PluginManager {
    vector<unique_ptr<IPlugin>> plugins_;
    map<string, void*> handles_;  // dlopen handles

public:
    // Carica plugin da directory
    void load_plugins(const string& plugin_dir) {
        for (auto& entry : fs::directory_iterator(plugin_dir)) {
            if (entry.path().extension() == ".so" ||
                entry.path().extension() == ".dll") {
                load_shared(entry.path());
            }
        }
    }

    // Trova plugin per tipo
    template<typename T>
    vector<T*> get_plugins() const;

    // Trova plugin per nome
    IPlugin* find(const string& name) const;
};
```

---

## 4. Struttura File

| File | Contenuto |
| --- | --- |
| `src/plugin/plugin.h` | Interfaccia IPlugin e specializzazioni |
| `src/plugin/plugin_manager.h/.cpp` | Caricamento e gestione plugin |
| `src/plugin/builtin/` | Plugin built-in (GGUF, CUDA, etc.) |
| `plugins/` | Directory per plugin esterni |
| `docs/plugin-development.md` | Guida per sviluppatori plugin |

---

## 5. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_plugin_load` | Plugin .so caricato correttamente a runtime |
| `test_plugin_tool` | Tool plugin registrato e invocabile |
| `test_plugin_backend` | Backend plugin usato per matmul |
| `test_plugin_unload` | Shutdown e dlclose senza leak |
