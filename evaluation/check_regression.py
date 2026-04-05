"""CI regression check — exit 0 if no regression vs baseline, exit 1 otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
CURRENT_PATH = RESULTS_DIR / "evaluation_results.json"

# Metrics to check and their maximum allowed regression (absolute)
REGRESSION_THRESHOLDS = {
    "ndcg@5": 0.01,
    "ndcg@10": 0.01,
    "mrr@10": 0.015,
    "recall@10": 0.01,
}

VARIANT = "full"


def main() -> int:
    if not BASELINE_PATH.exists():
        print(f"No baseline found at {BASELINE_PATH}")
        print("Run the evaluation and save a baseline first:")
        print("  python -m evaluation.evaluate")
        print(f"  cp {CURRENT_PATH} {BASELINE_PATH}")
        return 0  # not an error, just no baseline yet

    if not CURRENT_PATH.exists():
        print(f"No current results at {CURRENT_PATH}")
        print("Run: python -m evaluation.evaluate")
        return 1

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(CURRENT_PATH) as f:
        current = json.load(f)

    if VARIANT not in baseline or VARIANT not in current:
        print(f"Variant '{VARIANT}' not found in results")
        return 1

    b_summary = baseline[VARIANT]["summary"]
    c_summary = current[VARIANT]["summary"]

    regressions = []
    print(f"Comparing '{VARIANT}' variant against baseline:")
    print(f"{'Metric':<15} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>10}")
    print("-" * 60)

    for metric, threshold in REGRESSION_THRESHOLDS.items():
        b_val = b_summary.get(metric, 0)
        c_val = c_summary.get(metric, 0)
        delta = c_val - b_val
        regressed = delta < -threshold
        status = "REGRESSED" if regressed else ("improved" if delta > 0.001 else "ok")

        print(f"{metric:<15} {b_val:>10.4f} {c_val:>10.4f} {delta:>+10.4f} {status:>10}")

        if regressed:
            regressions.append((metric, b_val, c_val, delta))

    print()
    if regressions:
        print(f"FAILED: {len(regressions)} metric(s) regressed beyond threshold")
        for metric, b_val, c_val, delta in regressions:
            print(f"  {metric}: {b_val:.4f} -> {c_val:.4f} ({delta:+.4f}, threshold: {REGRESSION_THRESHOLDS[metric]})")
        return 1

    print("PASSED: no regressions detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
