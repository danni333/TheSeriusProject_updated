#!/usr/bin/env python3
"""
TheSeriusProject - Download model weights (placeholder).
Usage: python scripts/download_model.py [--source URL] [--output dir]
Supported sources: Hugging Face (hf://org/model), URL to .gguf/.safetensors.
Requires: huggingface_hub (optional), requests.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model for TheSeriusProject")
    parser.add_argument("--source", default="", help="Source: hf://org/model or URL")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--quant", default="", help="Quantization: q4_k_m, q8_0, none")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if not args.source:
        print("No --source given. Example: --source hf://Qwen/Qwen2.5-3B-Instruct")
        print("Creating placeholder: models/.gitkeep")
        with open(os.path.join(args.output, ".gitkeep"), "w") as f:
            f.write("")
        return 0

    if args.source.startswith("hf://"):
        ref = args.source[5:].strip("/")
        print(f"Would download from Hugging Face: {ref}")
        try:
            from huggingface_hub import hf_hub_download, snapshot_download

            print("huggingface_hub available; use snapshot_download or hf_hub_download in code.")
        except ImportError:
            print("Install: pip install huggingface_hub")
        return 0

    print(f"Would download from URL: {args.source}")
    print("Implement HTTP download or use wget/curl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
