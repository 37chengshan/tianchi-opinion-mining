#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data


def main() -> None:
    train = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    candidates = load_candidates(ROOT / "artifacts/experiments/neural_5fold_confirm/oof_candidates.json")
    gold = {row.id: set(row.labels) for row in train}

    total = sum(len(v) for v in gold.values())
    hit = 0
    candidate_total = 0
    implicit_total = implicit_hit = 0
    explicit_total = explicit_hit = 0
    by_category = Counter()
    by_category_hit = Counter()
    by_polarity = Counter()
    by_polarity_hit = Counter()
    miss_reason = Counter()

    for rid, gold_set in gold.items():
        cand_set = {item.quadruple for item in candidates.get(rid, [])}
        candidate_total += len(cand_set)
        hit += len(gold_set & cand_set)
        for q in gold_set:
            by_category[q.category] += 1
            by_polarity[q.polarity] += 1
            is_hit = q in cand_set
            if is_hit:
                by_category_hit[q.category] += 1
                by_polarity_hit[q.polarity] += 1
            if q.aspect == "_":
                implicit_total += 1
                implicit_hit += int(is_hit)
            else:
                explicit_total += 1
                explicit_hit += int(is_hit)
            if is_hit:
                continue
            same_opinion = [c for c in cand_set if c.opinion == q.opinion]
            same_pair = [c for c in same_opinion if c.aspect == q.aspect]
            same_category = [c for c in same_pair if c.category == q.category]
            if not same_opinion:
                miss_reason["opinion_span_missing"] += 1
            elif not same_pair:
                miss_reason["aspect_or_pair_missing"] += 1
            elif not same_category:
                miss_reason["category_missing"] += 1
            else:
                miss_reason["polarity_missing"] += 1

    oracle_recall = hit / total if total else 0.0
    oracle_f1_if_perfect_filter = 2 * oracle_recall / (1 + oracle_recall) if oracle_recall else 0.0
    payload = {
        "gold": total,
        "candidate_quadruples": candidate_total,
        "candidate_per_review": candidate_total / len(gold),
        "oracle_hit": hit,
        "oracle_recall": oracle_recall,
        "oracle_f1_if_perfect_filter": oracle_f1_if_perfect_filter,
        "implicit": {"gold": implicit_total, "hit": implicit_hit, "recall": implicit_hit / implicit_total if implicit_total else 0.0},
        "explicit": {"gold": explicit_total, "hit": explicit_hit, "recall": explicit_hit / explicit_total if explicit_total else 0.0},
        "miss_reason": dict(miss_reason),
        "category_oracle_recall": {k: by_category_hit[k] / by_category[k] for k in sorted(by_category)},
        "polarity_oracle_recall": {k: by_polarity_hit[k] / by_polarity[k] for k in sorted(by_polarity)},
    }
    out = ROOT / "artifacts/reports/oof_ceiling_audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
