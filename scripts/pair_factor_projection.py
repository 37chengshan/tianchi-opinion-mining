#!/usr/bin/env python3
"""Closed-form strict-F1 projection from pair precision, recall, class accuracy.

Read-only.  Establishes the identity used by docs/GRID_V2_FRAMEWORK_DESIGN.md:

    F1 = 2 * c * r * G / (k * r * G / p + G)

with c = class accuracy among correctly paired predictions, r = pair recall,
p = pair precision, k = emitted classes per emitted pair, G = gold count.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data
from opinion_mining.metrics import strict_f1
from opinion_mining.pipeline import select_threshold, threshold_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    args = parser.parse_args()

    rows = load_train_data(args.reviews, args.labels)
    gold = {row.id: set(row.labels) for row in rows}
    candidates = load_candidates(args.candidates)
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
    gold_pair_n = len(gold_pairs)
    if not emitted_n or not gold_pair_n:
        raise SystemExit("candidate map produced no usable predictions")

    pair_precision = hit_n / emitted_n
    pair_recall = hit_n / gold_pair_n
    class_accuracy = score.correct / hit_n
    classes_per_pair = score.predicted / emitted_n
    gold_n = score.gold

    print("measured factors")
    print(f"  emitted pairs          {emitted_n}")
    print(f"  gold pairs hit         {hit_n}")
    print(f"  gold pairs             {gold_pair_n}")
    print(f"  pair precision (p)     {pair_precision:.4f}")
    print(f"  pair recall    (r)     {pair_recall:.4f}")
    print(f"  class accuracy (c)     {class_accuracy:.4f}")
    print(f"  classes per pair (k)   {classes_per_pair:.4f}")
    print(f"  observed strict F1     {score.f1:.4f}")
    print()

    def project(precision: float, recall: float) -> float:
        denominator = classes_per_pair * recall * gold_n / precision + gold_n
        return 2 * class_accuracy * recall * gold_n / denominator

    print(f"  closed form at measured values: {project(pair_precision, pair_recall):.4f}")
    print()
    print("projection (c and k held at measured values)")
    print(f"  {'pair prec':>10} {'pair recall':>12} {'strict F1':>11}")
    for precision in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for recall in (0.80, 0.85, 0.90):
            print(f"  {precision:>10.2f} {recall:>12.2f} {project(precision, recall):>11.4f}")


if __name__ == "__main__":
    main()
