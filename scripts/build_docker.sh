#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-theseriusproject:latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

echo "=== Building Docker image: ${IMAGE_NAME} ==="
docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

echo "=== Build complete ==="
echo "Image: ${IMAGE_NAME}"
echo "Run locally: docker run --gpus all -p 8080:8080 -v \$(pwd)/models:/models ${IMAGE_NAME}"
