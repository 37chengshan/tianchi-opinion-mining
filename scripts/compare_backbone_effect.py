#!/usr/bin/env python3
"""Compare two grid screening runs on the signal that is version-stable.

Calibrated aggregate OOF is NOT comparable across runs produced by different
decoder revisions, because the score scale itself changes.  Fold-level monitor
F1 uses fixed 0.12/0.10 thresholds inside every run, so it isolates the
backbone (or any other single changed factor) even when the decoder moved.

Read-only: consumes ``oof_report.json`` files and prints a table plus the
per-fold deltas.  Used to read the TAPT verdict against a same-decoder control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def load_folds(run_dir: Path) -> tuple[list[dict[str, float]], dict[str, object]]:
    report_path = run_dir / "oof_report.json"
    if not report_path.exists():
        raise SystemExit(f"missing {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    folds = [
        {
            "fold": int(item["fold"]),
            "monitor_f1": float(item["monitor"]["f1"]),
            "monitor_precision": float(item["monitor"]["precision"]),
            "monitor_recall": float(item["monitor"]["recall"]),
        }
        for item in payload["fold_reports"]
    ]
    return folds, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="run directory of the control group")
    parser.add_argument("--candidate", type=Path, required=True, help="run directory of the changed group")
    parser.add_argument("--label-baseline", default="baseline")
    parser.add_argument("--label-candidate", default="candidate")
    args = parser.parse_args()

    base_folds, base_payload = load_folds(args.baseline)
    cand_folds, cand_payload = load_folds(args.candidate)
    by_fold_base = {row["fold"]: row for row in base_folds}
    by_fold_cand = {row["fold"]: row for row in cand_folds}
    shared = sorted(set(by_fold_base) & set(by_fold_cand))
    if not shared:
        raise SystemExit("the two runs share no folds")

    print(f"baseline  : {args.label_baseline}")
    print(f"candidate : {args.label_candidate}")
    print()
    print(f"  {'fold':>4}  {'baseline F1':>12}  {'candidate F1':>13}  {'delta':>9}")
    deltas = []
    for fold in shared:
        base = by_fold_base[fold]["monitor_f1"]
        cand = by_fold_cand[fold]["monitor_f1"]
        deltas.append(cand - base)
        print(f"  {fold:>4}  {base:>12.4f}  {cand:>13.4f}  {cand - base:>+9.4f}")

    mean_delta = sum(deltas) / len(deltas)
    base_mean = sum(by_fold_base[f]["monitor_f1"] for f in shared) / len(shared)
    cand_mean = sum(by_fold_cand[f]["monitor_f1"] for f in shared) / len(shared)
    wins = sum(1 for value in deltas if value > 0)
    print()
    print(f"  fold-mean monitor F1: {base_mean:.4f} -> {cand_mean:.4f}  (delta {mean_delta:+.4f})")
    print(f"  folds improved: {wins}/{len(deltas)}")
    print()
    print(f"  basline calibrated OOF : {float(base_payload['metrics']['f1']):.4f}  (decoder-dependent, informational)")
    print(f"  candidate calibrated OOF: {float(cand_payload['metrics']['f1']):.4f}  (decoder-dependent, informational)")
    print()
    if mean_delta >= 0.005 and wins == len(deltas):
        print("verdict: EFFECTIVE - every fold improved and the fold-mean gain clears the 0.005 gate.")
    elif mean_delta > 0:
        print("verdict: INCONCLUSIVE - mean gain is positive but not consistent across every fold.")
    else:
        print("verdict: NOT EFFECTIVE - fold-mean gain does not support keeping the change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
