# Fase 52 — Installer & Packaging Multi-Piattaforma

> L'utente finale non compila da sorgente.
> Serve un binario statico per ogni piattaforma, con installer
> o package manager, pronto all'uso in un click.

---

## 1. Target di Release

| Piattaforma | Formato | GPU Support |
| --- | --- | --- |
| **Linux x86_64** | AppImage + .tar.gz | CUDA + ROCm |
| **Linux ARM64** | .tar.gz | CPU only |
| **Windows x64** | .exe installer + .zip portable | CUDA |
| **macOS ARM64** | .dmg + Homebrew | Metal (Apple Silicon) |
| **Docker** | Immagine multi-stage (Fase 26) | NVIDIA Container Toolkit |

---

## 2. Build Statico

```cmake
# Link statico per massima portabilità
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

# CUDA: link statico delle librerie runtime
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart static")
```

---

## 3. Release Automation (CI/CD)

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  build-linux:
    runs-on: ubuntu-22.04
    container: nvidia/cuda:12.2.0-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      - run: cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
      - run: cmake --build build -j$(nproc)
      - run: tar czf serius-linux-x64-cuda.tar.gz -C build serius
      - uses: softprops/action-gh-release@v2
        with:
          files: serius-linux-x64-cuda.tar.gz

  build-windows:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - uses: Jimver/cuda-toolkit@v0.2.14
        with: { cuda: '12.2.0' }
      - run: cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
      - run: cmake --build build --config Release
      - run: Compress-Archive build/Release/serius.exe serius-win-x64-cuda.zip
      - uses: softprops/action-gh-release@v2
        with:
          files: serius-win-x64-cuda.zip

  build-macos:
    runs-on: macos-14  # M1/M2/M3
    steps:
      - uses: actions/checkout@v4
      - run: cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
      - run: cmake --build build -j$(sysctl -n hw.ncpu)
      - run: tar czf serius-macos-arm64-metal.tar.gz -C build serius
      - uses: softprops/action-gh-release@v2
        with:
          files: serius-macos-arm64-metal.tar.gz
```

---

## 4. One-Line Install

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash

# Windows (PowerShell)
iwr -useb https://raw.githubusercontent.com/.../install.ps1 | iex

# Homebrew (macOS)
brew install serius

# Lo script:
# 1. Rileva OS e architettura
# 2. Scarica il binario corretto dall'ultima release
# 3. Lo mette in /usr/local/bin (o equivalente)
# 4. Verifica con: serius --version
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `.github/workflows/release.yml` | CI/CD per build multi-piattaforma |
| `scripts/install.sh` | Installer one-line Linux/macOS |
| `scripts/install.ps1` | Installer one-line Windows |
| `cmake/StaticLink.cmake` | Config per link statico |
| `packaging/AppImage.cmake` | Config AppImage Linux |
| `packaging/windows/installer.nsi` | NSIS installer Windows |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_static_binary` | Binary non ha dipendenze dinamiche (ldd) |
| `test_install_script` | install.sh funziona su Ubuntu pulito |
| `test_windows_portable` | .zip funziona senza installazione |
| `test_macos_metal` | Binary macOS rileva GPU Apple Silicon |
| `test_version` | `serius --version` stampa versione corretta |
