#!/usr/bin/env python3
"""
TheSeriusProject - Generate reference tensors (e.g. for matmul) with NumPy.
Usage: python scripts/generate_reference.py [--out test_data]
Writes binary files (A.bin, B.bin, C.bin) for test_data/ used by test_blas.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reference data for tests")
    parser.add_argument("--out", default="test_data", help="Output directory")
    parser.add_argument("--shape", default="64,128,128,64", help="M,K for A(M,K), K,N for B(K,N), M,N for C")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("NumPy required: pip install numpy")
        return 1

    os.makedirs(args.out, exist_ok=True)
    parts = [int(x) for x in args.shape.replace(",", " ").split()]
    if len(parts) < 4:
        M, K, N = 64, 128, 64
    else:
        M, K, N = parts[0], parts[1], parts[3]

    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    C = A @ B

    A.tofile(os.path.join(args.out, "A.bin"))
    B.tofile(os.path.join(args.out, "B.bin"))
    C.tofile(os.path.join(args.out, "C.bin"))
    print(f"Wrote A({M}x{K}), B({K}x{N}), C({M}x{N}) to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
