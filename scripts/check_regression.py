#!/usr/bin/env python3
"""Regression check: compare current eval scores against baseline."""
import argparse
import json
import sys


def load_scores(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Check for eval regressions")
    parser.add_argument("--baseline", required=True, help="Baseline scores JSON")
    parser.add_argument("--current", required=True, help="Current scores JSON")
    parser.add_argument("--margin", type=float, default=0.02, help="Regression margin (default 2%%)")
    args = parser.parse_args()

    baseline = load_scores(args.baseline)
    current = load_scores(args.current)

    regressions = []
    for bench, base_score in baseline.items():
        cur_score = current.get(bench)
        if cur_score is None:
            regressions.append(f"{bench}: missing in current")
            continue
        if cur_score < base_score - args.margin:
            regressions.append(
                f"{bench}: {base_score:.1%} -> {cur_score:.1%} (drop > {args.margin:.0%})"
            )

    if regressions:
        print("REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  ❌ {r}")
        sys.exit(1)

    print("✅ No regressions detected")


if __name__ == "__main__":
    main()
