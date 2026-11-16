#!/bin/bash

# Serius One-Line Installer for Linux/macOS

set -e

VERSION="${VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
REPO="danni333/TheSeriusProject_updated"

echo "Installing Serius LLM Engine..."

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        if [ "$ARCH" = "x86_64" ]; then
            FILE="serius-linux-x64"
        elif [ "$ARCH" = "aarch64" ]; then
            FILE="serius-linux-arm64"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
        ;;
    Darwin)
        if [ "$ARCH" = "arm64" ]; then
            FILE="serius-macos-arm64"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Download URL
if [ "$VERSION" = "latest" ]; then
    DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${FILE}.tar.gz"
else
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${FILE}.tar.gz"
fi

echo "Downloading from: $DOWNLOAD_URL"

# Create temp directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download
curl -fsSL "$DOWNLOAD_URL" -o serius.tar.gz

# Extract
tar xzf serius.tar.gz

# Install binary and web UI (server resolves ui/ next to the executable)
if [ -d "$INSTALL_DIR" ] && [ -w "$INSTALL_DIR" ]; then
    cp serius "$INSTALL_DIR/"
    [ -d ui ] && cp -r ui "$INSTALL_DIR/ui"
else
    sudo mkdir -p "$INSTALL_DIR"
    sudo cp serius "$INSTALL_DIR/"
    [ -d ui ] && sudo cp -r ui "$INSTALL_DIR/ui"
fi

# Make executable
chmod +x "$INSTALL_DIR/serius" 2>/dev/null || sudo chmod +x "$INSTALL_DIR/serius"

# Cleanup
cd /
rm -rf "$TMP_DIR"

# Verify
if command -v serius &> /dev/null; then
    echo "Serius installed successfully!"
    serius --version
else
    echo "Installation completed but serius not in PATH"
    echo "Add $INSTALL_DIR to your PATH or use: $INSTALL_DIR/serius"
fi
