#!/usr/bin/env python3
"""Project strict quadruple F1 from pair precision, pair recall, class accuracy.

Read-only analysis helper for the Grid V2 framework design.  It reads one OOF
candidate map plus gold labels, measures the three factors that determine
strict F1, and prints the projection table used in
``docs/GRID_V2_FRAMEWORK_DESIGN.md``.  It never trains anything.
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


def measure(candidates_path: Path, reviews: Path, labels: Path) -> dict[str, float]:
    rows = load_train_data(reviews, labels)
    gold = {row.id: set(row.labels) for row in rows}
    candidates = load_candidates(candidates_path)
    calibration = select_threshold(candidates, gold, separate_implicit=True)
    predictions = threshold_predictions(
        candidates, float(calibration["threshold"]), float(calibration["implicit_threshold"])
    )
    score = strict_f1(gold, predictions)

    predicted_pairs: set[tuple[int, str, str]] = set()
    gold_pairs: set[tuple[int, str, str]] = set()
    for rid, labels_row in gold.items():
        for quad in labels_row:
            gold_pairs.add((rid, quad.aspect, quad.opinion))
    hit_pairs: set[tuple[int, str, str]] = set()
    for rid, quads in predictions.items():
        gold_row = gold.get(rid, set())
        for quad in quads:
            predicted_pairs.add((rid, quad.aspect, quad.opinion))
            if quad in gold_row:
                hit_pairs.add((rid, quad.aspect, quad.opinion))
    correct_pairs = predicted_pairs & gold_pairs

    class_accuracy = score.correct / len(correct_pairs) if correct_pairs else 0.0
    pair_precision = len(correct_pairs) / len(predicted_pairs) if predicted_pairs else 0.0
    pair_recall = len(hit_pairs) / len(gold_pairs) if gold_pairs else 0.0
    return {
        "f1": score.f1,
        "precision": score.precision,
        "recall": score.recall,
        "tp": float(score.correct),
        "predicted": float(score.predicted),
        "gold": float(score.gold),
        "class_accuracy": class_accuracy,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "threshold": float(calibration["threshold"]),
        "implicit_threshold": float(calibration["implicit_threshold"]),
    }


def project(class_accuracy: float, pair_recall: float) -> list[tuple[float, float]]:
    """Predict strict F1 for a range of pair precisions (exact F1 identity)."""
    predicted_correct = class_accuracy  # placeholder to keep the signature explicit
    del predicted_correct
    rows: list[tuple[float, float]] = []
    for precision in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        # Strict F1 = 2*TP/(P+G).  With quad-recall fixed, TP = pair_recall*G,
        # and the emitted count follows from pair precision at fixed TP.
        recall = pair_recall * class_accuracy
        precision_quad = precision * class_accuracy
        f1 = 2 * precision_quad * recall / (precision_quad + recall) if precision_quad + recall else 0.0
        rows.append((precision, f1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    args = parser.parse_args()

    measured = measure(args.candidates, args.reviews, args.labels)
    print("measured factors")
    for key, value in measured.items():
        print(f"  {key:18s} {value:.4f}")
    reconstructed = 2 * measured["pair_precision"] * measured["pair_recall"] / (
        measured["pair_precision"] + measured["pair_recall"]
    )
    print(f"  {'pair-only F1 bound':18s} {reconstructed:.4f}  (lower bound; ignores class accuracy)")
    print()
    print("projection (class accuracy and pair recall held at measured values)")
    for precision, f1 in project(measured["class_accuracy"], measured["pair_recall"]):
        marker = "  <-- 0.75 target" if 0.74 <= f1 < 0.76 else ""
        print(f"  pair-precision {precision:.2f} -> strict F1 {f1:.4f}{marker}")


if __name__ == "__main__":
    main()
