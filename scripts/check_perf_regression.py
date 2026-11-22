#!/usr/bin/env python3
"""CI performance gate: fail if metrics regress beyond a threshold.

Usage:
    python scripts/check_perf_regression.py \
        --baseline perf/baseline.json --current pr.json [--max-regression 5]

`--current` is a `serius bench -f json` report. The baseline is a flat JSON
object of metric -> value, e.g.:
    {"decode_tps": 183.0, "prefill_tps": 2100.0, "ttft_ms": 187.0}

For *_tps / *_per_sec metrics a drop is a regression; for *_ms / *_bytes
metrics an increase is a regression.
"""
import argparse
import json
import sys


def flatten(data):
    """Extract comparable scalar metrics from a serius bench JSON report."""
    lat = data.get("latency", {})
    thr = data.get("throughput", {})
    mem = data.get("memory", {})
    return {
        "decode_tps": lat.get("decode_tps"),
        "prefill_tps": lat.get("prefill_tps"),
        "ttft_ms": lat.get("ttft_ms"),
        "requests_per_sec": thr.get("requests_per_sec"),
        "total_tokens_per_sec": thr.get("total_tokens_per_sec"),
        "p99_latency_ms": thr.get("p99_latency_ms"),
        "vram_peak_bytes": mem.get("vram_peak_bytes"),
        "ram_peak_bytes": mem.get("ram_peak_bytes"),
    }


def is_lower_better(metric):
    return metric.endswith("_ms") or metric.endswith("_bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--max-regression", type=float, default=5.0,
                    help="max allowed regression in percent (default 5)")
    args = ap.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.current) as f:
        current = flatten(json.load(f))

    threshold = args.max_regression / 100.0
    failures = []
    checked = 0
    for metric, base_val in baseline.items():
        cur_val = current.get(metric)
        if cur_val is None or base_val in (None, 0):
            continue
        checked += 1
        if is_lower_better(metric):
            regression = (cur_val - base_val) / base_val
        else:
            regression = (base_val - cur_val) / base_val
        if regression > threshold:
            failures.append(
                f"{metric}: {base_val} -> {cur_val} "
                f"({regression * 100:+.1f}% regression, limit {args.max_regression}%)"
            )

    if failures:
        print("PERFORMANCE REGRESSIONS DETECTED:")
        for f_ in failures:
            print(f"  FAIL {f_}")
        sys.exit(1)

    print(f"OK: {checked} metrics within {args.max_regression}% of baseline")


if __name__ == "__main__":
    main()
