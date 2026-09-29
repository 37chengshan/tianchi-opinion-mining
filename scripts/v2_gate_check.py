#!/usr/bin/env python3
"""Decide whether a Grid V2 OOF run clears the locked plan's promotion gates.

Read-only.  Consumes one or more OOF candidate maps plus gold labels, measures
the factors that determine strict quadruple F1, and evaluates the gates from
``docs/superpowers/plans/2026-09-28-locked-final-training-plan.md`` plus the
process indicators from ``docs/GRID_V2_FRAMEWORK_DESIGN.md``.

    F1 = 2*c*r*G / (k*r*G/p + G)

Exit code is 0 when every gate passes, 1 when any gate fails, and 2 on a usage
problem.  It never trains and never writes to the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data
from opinion_mining.metrics import strict_f1
from opinion_mining.pipeline import select_threshold, threshold_predictions

# Baseline the V2 run must beat.  V1 WWM screening, fold-exclusive 3-fold OOF.
V1_F1 = 0.6252


def measure(candidates_path: Path, reviews: Path, labels: Path) -> dict[str, float]:
    rows = load_train_data(reviews, labels)
    gold = {row.id: set(row.labels) for row in rows}
    candidates = load_candidates(candidates_path)
    calibration = select_threshold(candidates, gold, separate_implicit=True)
    predictions = threshold_predictions(
        candidates, float(calibration["threshold"]), float(calibration["implicit_threshold"])
    )
    score = strict_f1(gold, predictions)

    emitted: set[tuple[int, str, str]] = set()
    gold_pairs: set[tuple[int, str, str]] = set()
    for rid, quads in predictions.items():
        for quad in quads:
            emitted.add((rid, quad.aspect, quad.opinion))
    for rid, labels_row in gold.items():
        for quad in labels_row:
            gold_pairs.add((rid, quad.aspect, quad.opinion))
    hit_pairs = emitted & gold_pairs

    emitted_n = len(emitted)
    hit_n = len(hit_pairs)
    pair_precision = hit_n / emitted_n if emitted_n else 0.0
    pair_recall = hit_n / len(gold_pairs) if gold_pairs else 0.0
    class_accuracy = score.correct / hit_n if hit_n else 0.0
    classes_per_pair = score.predicted / emitted_n if emitted_n else 0.0

    per_review = sorted(len(values) for values in predictions.values())
    median_emitted = per_review[len(per_review) // 2] if per_review else 0

    # High-confidence pair precision on the RAW candidate map.  This is the
    # metric quoted in the framework design (0.569 at V1); unlike the emitted
    # count it stays discriminative after threshold calibration, because the
    # failure mode is wrong pairs rather than too many pairs.
    gold_pair_keys = {(rid, quad.aspect, quad.opinion) for rid, labels_row in gold.items() for quad in labels_row}
    by_cell: dict[tuple[int, str, str], float] = {}
    for rid, items in candidates.items():
        for item in items:
            key = (rid, item.quadruple.aspect, item.quadruple.opinion)
            if item.score > by_cell.get(key, -1.0):
                by_cell[key] = float(item.score)
    high_conf = [key for key, value in by_cell.items() if value >= 0.8]
    high_conf_hits = sum(1 for key in high_conf if key in gold_pair_keys)
    high_conf_precision = high_conf_hits / len(high_conf) if high_conf else 0.0
    review_pair_counts = sorted(
        len({(item.quadruple.aspect, item.quadruple.opinion) for item in items}) for items in candidates.values()
    )
    median_raw_pairs = review_pair_counts[len(review_pair_counts) // 2] if review_pair_counts else 0

    return {
        "f1": score.f1,
        "precision": score.precision,
        "recall": score.recall,
        "tp": float(score.correct),
        "predicted": float(score.predicted),
        "gold": float(score.gold),
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "class_accuracy": class_accuracy,
        "classes_per_pair": classes_per_pair,
        "median_predicted_per_review": float(median_emitted),
        "high_conf_pair_precision": high_conf_precision,
        "high_conf_pair_support": float(len(high_conf)),
        "median_raw_pairs_per_review": float(median_raw_pairs),
        "threshold": float(calibration["threshold"]),
        "implicit_threshold": float(calibration["implicit_threshold"]),
    }


def evaluate(measured: dict[str, float], baseline: float) -> list[dict[str, object]]:
    delta = measured["f1"] - baseline
    # Plan gate: delta >= 0.015, OR delta >= 0.005 together with oracle/implicit
    # improvement.  The conditional half needs oracle evidence, so it is
    # reported as unmet-and-pending rather than silently auto-passed.
    gates = [
        {"name": "strict F1 delta >= 0.015", "value": delta, "target": 0.015, "pass": delta >= 0.015},
        {"name": "strict F1 delta >= 0.005 (conditional)", "value": delta, "target": 0.005, "pass": delta >= 0.005},
        {"name": "pair precision >= 0.75", "value": measured["pair_precision"], "target": 0.75, "pass": measured["pair_precision"] >= 0.75},
        {"name": "pair recall >= 0.80", "value": measured["pair_recall"], "target": 0.80, "pass": measured["pair_recall"] >= 0.80},
        {"name": "class accuracy >= 0.95", "value": measured["class_accuracy"], "target": 0.95, "pass": measured["class_accuracy"] >= 0.95},
        {"name": "high-conf pair precision >= 0.75", "value": measured["high_conf_pair_precision"], "target": 0.75, "pass": measured["high_conf_pair_precision"] >= 0.75},
    ]
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Grid V2 OOF candidate map")
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--baseline", type=float, default=V1_F1, help="strict F1 the run must beat")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    measured = measure(args.candidates, args.reviews, args.labels)
    gates = evaluate(measured, args.baseline)
    overall = all(bool(gate["pass"]) for gate in gates)
    primary = bool(gates[0]["pass"])
    conditional = bool(gates[1]["pass"]) and not primary

    payload = {
        "candidates": str(args.candidates),
        "baseline_f1": args.baseline,
        "measured": measured,
        "gates": gates,
        "promote_to_task4": primary,
        "conditional_review_required": conditional,
        "all_gates_pass": overall,
    }

    print(f"candidates : {args.candidates}")
    print(f"baseline   : strict F1 {args.baseline:.4f} (V1 WWM screening)")
    print()
    print("measured factors")
    for key in ("f1", "precision", "recall", "pair_precision", "pair_recall", "class_accuracy", "classes_per_pair", "high_conf_pair_precision"):
        print(f"  {key:24s} {measured[key]:.4f}")
    print(f"  {'hi-conf pair support':24s} {measured['high_conf_pair_support']:.0f}")
    print(f"  {'median raw pairs/review':24s} {measured['median_raw_pairs_per_review']:.0f}")
    print()
    print("gates")
    for gate in gates:
        mark = "PASS" if gate["pass"] else "FAIL"
        print(f"  [{mark}] {gate['name']:38s} value {gate['value']:+.4f} vs {gate['target']:+.4f}")
    print()
    if primary:
        print("verdict: PROMOTE - strict F1 gain clears the 0.015 gate; Task 4 is unlocked.")
    elif conditional:
        print("verdict: CONDITIONAL - gain is 0.005-0.015; oracle/implicit evidence must be supplied before promotion.")
    else:
        print("verdict: NOT PROMOTED - gain below 0.005; per the locked plan only one V2.1 patch is allowed next.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written: {args.output}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
