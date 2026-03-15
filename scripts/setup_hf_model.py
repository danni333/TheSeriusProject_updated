#!/usr/bin/env python3
"""
TheSeriusProject - Download HF model and convert tokenizer
Usage: python scripts/setup_hf_model.py [HF_REPO] [OUTPUT_DIR]
Example: python scripts/setup_hf_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/tinyllama
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def convert_tokenizer(hf_tokenizer_path: str, output_path: str):
    """Convert HuggingFace tokenizer.json to Serius BPETokenizer format.
    
    Serius format:
    {
      "vocab": [ {"t": [byte_array], "i": token_id}, ... ],
      "merges": [ [id_a, id_b, id_ab], ... ]
    }
    
    The C++ BPETokenizer::load() clears all existing vocab/merges and rebuilds
    from the JSON, so we must provide ALL tokens including special & byte tokens.
    """
    print(f"Converting HF tokenizer {hf_tokenizer_path} to Serius format...")
    with open(hf_tokenizer_path, "r", encoding="utf-8") as f:
        hf_tok = json.load(f)

    hf_vocab = hf_tok.get("model", {}).get("vocab", {})
    hf_merges = hf_tok.get("model", {}).get("merges", [])
    
    print(f"  HF vocab size: {len(hf_vocab)}, merges: {len(hf_merges)}")
    
    # Build the output vocab array.
    # For each token string from HF, we encode it to UTF-8 bytes.
    # Special handling for SentencePiece byte tokens like <0x00>..<0xFF>:
    #   these represent single raw bytes and should be stored as that single byte.
    
    byte_token_re = re.compile(r'^<0x([0-9A-Fa-f]{2})>$')
    
    out_vocab = []
    for token_str, token_id in hf_vocab.items():
        m = byte_token_re.match(token_str)
        if m:
            # SentencePiece byte fallback token -> store as the single byte
            byte_val = int(m.group(1), 16)
            token_bytes = [byte_val]
        else:
            # Regular token -> UTF-8 encode
            # LLaMA/SentencePiece uses U+2581 (▁) for leading space
            token_bytes = list(token_str.encode('utf-8'))
        
        out_vocab.append({"t": token_bytes, "i": token_id})

    # Also include added_tokens from the tokenizer (special tokens like <s>, </s>, etc.)
    added_tokens = hf_tok.get("added_tokens", [])
    existing_ids = {entry["i"] for entry in out_vocab}
    for at in added_tokens:
        tid = at.get("id")
        content = at.get("content", "")
        if tid is not None and tid not in existing_ids:
            token_bytes = list(content.encode('utf-8'))
            out_vocab.append({"t": token_bytes, "i": tid})
            existing_ids.add(tid)

    # Convert merges: "token_a token_b" -> [id_a, id_b, id_ab]
    out_merges = []
    for merge_str in hf_merges:
        parts = merge_str.split(" ", 1)
        if len(parts) == 2:
            a, b = parts
            id_a = hf_vocab.get(a)
            id_b = hf_vocab.get(b)
            id_ab = hf_vocab.get(a + b)
            if id_a is not None and id_b is not None and id_ab is not None:
                out_merges.append([id_a, id_b, id_ab])

    out_data = {"vocab": out_vocab, "merges": out_merges}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f)  # No indent to keep file smaller
    
    print(f"  Saved {len(out_vocab)} tokens and {len(out_merges)} merges to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HF model and convert tokenizer for TheSeriusProject"
    )
    parser.add_argument("repo", help="HuggingFace repo (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading model {args.repo} into {args.output}...")
    local_dir = snapshot_download(
        repo_id=args.repo,
        local_dir=args.output,
        allow_patterns=["*.json", "*.safetensors"],
        ignore_patterns=["*pytorch_model.bin", "*.msgpack", "*.h5"]
    )

    tokenizer_json = os.path.join(local_dir, "tokenizer.json")
    vocab_json = os.path.join(local_dir, "vocab.json")
    
    if os.path.exists(tokenizer_json):
        convert_tokenizer(tokenizer_json, vocab_json)
    else:
        print("Warning: No tokenizer.json found.")

    print("\n--- Done ---")
    print(f"Run: serius chat --type llama --config {args.output}/config.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
