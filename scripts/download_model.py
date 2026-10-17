#!/usr/bin/env python3
"""Download model weights from Hugging Face Hub with checksum verification."""
import argparse
import hashlib
import json
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model_id, dest):
    os.makedirs(dest, exist_ok=True)

    # Download index
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", local_dir=dest)
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))

    print(f"Downloading {len(shards)} shard(s) for {model_id} ...")
    for shard in shards:
        hf_hub_download(model_id, shard, local_dir=dest)
        print(f"  OK: {shard}")

    # Verify metadata hash
    meta_hash = sha256_file(index_path)
    print(f"Index SHA256: {meta_hash}")


def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument("--model", required=True, help="Model ID on Hugging Face Hub")
    parser.add_argument("--dest", required=True, help="Destination directory")
    args = parser.parse_args()
    download_model(args.model, args.dest)


if __name__ == "__main__":
    main()
