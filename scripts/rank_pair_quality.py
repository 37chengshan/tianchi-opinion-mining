#!/usr/bin/env python3
"""Threshold-free pair quality for a grid candidate map.

The grid decoder's score is a product of a span score, a validity sigmoid and
a relation score, so its scale changes whenever the decoder is revised.  Raw
thresholds therefore do not transfer between runs, and calibrated F1 is not
comparable across decoder revisions either.

This script measures the things that DO transfer:

* pair precision at fixed top-K cells per review (rank-based, scale-free);
* pair recall at the same K, which bounds what any threshold could reach;
* the recall ceiling of the full candidate map, which bounds the whole run.

It reads OOF candidate maps plus gold labels and never trains.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--top-k", default="1,2,3,5,10")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    rows = load_train_data(args.reviews, args.labels)
    gold = {row.id: set(row.labels) for row in rows}
    gold_cells = {(rid, quad.aspect, quad.opinion) for rid, labels_row in gold.items() for quad in labels_row}
    candidates = load_candidates(args.candidates)

    label = args.label or str(args.candidates)
    print(f"map: {label}")
    print(f"  reviews {len(candidates)}  gold pairs {len(gold_cells)}")
    print()

    # Best-scoring cell per (review, aspect, opinion) keeps the ranking faithful
    # to what a decoder would emit for that pair.
    best_per_cell: dict[int, dict[tuple[str, str], float]] = {}
    for rid, items in candidates.items():
        cells: dict[tuple[str, str], float] = {}
        for item in items:
            key = (item.quadruple.aspect, item.quadruple.opinion)
            if item.score > cells.get(key, float("-inf")):
                cells[key] = float(item.score)
        best_per_cell[rid] = cells

    ranked: dict[int, list[tuple[str, str]]] = {
        rid: [key for key, _ in sorted(cells.items(), key=lambda kv: -kv[1])]
        for rid, cells in best_per_cell.items()
    }

    ks = [int(value) for value in args.top_k.split(",") if value.strip()]
    print(f"  {'K/review':>8}  {'precision':>10}  {'recall':>8}  {'F1 bound':>9}  emitted")
    for k in ks:
        hits = 0
        emitted = 0
        for rid, keys in ranked.items():
            for key in keys[:k]:
                emitted += 1
                if (rid, key[0], key[1]) in gold_cells:
                    hits += 1
        precision = hits / emitted if emitted else 0.0
        recall = hits / len(gold_cells) if gold_cells else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  {k:>8}  {precision:>10.4f}  {recall:>8.4f}  {f1:>9.4f}  {emitted}")

    full_hits = sum(
        1 for rid, cells in best_per_cell.items() for key in cells if (rid, key[0], key[1]) in gold_cells
    )
    full_emitted = sum(len(cells) for cells in best_per_cell.values())
    print()
    print(
        f"  full-map pair ceiling: {full_hits}/{len(gold_cells)} = "
        f"{full_hits / len(gold_cells) if gold_cells else 0.0:.4f} recall over {full_emitted} pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
