#!/usr/bin/env python3
"""
Download a full Hugging Face LLaMA-style model (config, tokenizer, safetensors)
for use with TheSeriusProject C++ loader and as Python reference.

Usage:
  python scripts/download_hf_model.py [--model NAME] [--output DIR]
  python scripts/download_hf_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama

Requires: pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face LLaMA model (config, tokenizer, safetensors) for Serius"
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--output",
        default="models/hf_llama",
        help="Output directory (config.json, tokenizer files, *.safetensors)",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Skip downloading safetensors (config + tokenizer only)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading {args.model} -> {args.output}")

    allow_patterns = ["config.json", "tokenizer.json", "tokenizer_config.json", "*.model", "*.vocab"]
    if not args.no_safetensors:
        allow_patterns.append("*.safetensors")
        allow_patterns.append("*.safetensors.index.json")

    try:
        path = snapshot_download(
            repo_id=args.model,
            local_dir=args.output,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print(f"Done. Model files in: {path}")
    print("C++: use io::load_llama_config(path + '/config.json') and io::load_llama_weights_from_dir(path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
