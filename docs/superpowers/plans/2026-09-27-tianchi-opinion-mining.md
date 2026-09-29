# Tianchi Opinion Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local pipeline that evaluates strict quadruple F1, trains a high-precision statistical/retrieval baseline, and emits a validated Tianchi `Result.csv`.

**Architecture:** Python package with separate data, metrics, candidate model, OOF pipeline, and submission modules. Model thresholds are selected from out-of-fold predictions; the final model refits on all labeled reviews and predicts test reviews.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-tianchi-opinion-mining-design.md`

## Global Constraints

- Output must have no header and UTF-8 without BOM.
- Every test id must appear at least once, ids sorted ascending.
- Non-underscore AspectTerm and OpinionTerm must be exact substrings of the review.
- Category must be one of the 13 official categories; Polarity one of 正面/中性/负面.
- Local scoring is strict micro F1 over deduplicated quadruples.
- OOF folds must not leak validation labels into model statistics.

## Review Focus

- Duplicate predictions must not inflate P or S.
- Empty prediction ids must become exactly one all-underscore submission row.
- CSV quoting must preserve commas/newlines inside review-independent output fields.
- A candidate with a non-source substring must be rejected.
- Threshold search must include the boundary that predicts nothing and not crash on zero denominators.

---

### Task 1: Core data and strict metric

**Files:**
- Create: `src/opinion_mining/__init__.py`
- Create: `src/opinion_mining/data.py`
- Create: `src/opinion_mining/metrics.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces `Quadruple`, `ReviewExample`, `load_train_data()`, `load_test_reviews()`, `strict_f1()`.

- [ ] Write failing tests for strict matching, deduplication, implicit `_`, and grouped ids.
- [ ] Run `pytest tests/test_metrics.py tests/test_data.py -q` and confirm RED.
- [ ] Implement minimal data structures/readers and metric.
- [ ] Run the same tests and confirm GREEN.

### Task 2: Submission writer and validator

**Files:**
- Create: `src/opinion_mining/submission.py`
- Create: `tests/test_submission.py`

**Interfaces:**
- Consumes `Quadruple` and review ids/text.
- Produces `write_submission(path, reviews, predictions)` and `validate_submission(path, reviews)`.

- [ ] Write failing tests for no header/BOM, sorted full id coverage, underscore fallback, category/polarity validity, substring validity, and duplicate removal.
- [ ] Run `pytest tests/test_submission.py -q` and confirm RED.
- [ ] Implement writer/validator.
- [ ] Run test and full suite; confirm GREEN.

### Task 3: Statistical/retrieval candidate model

**Files:**
- Create: `src/opinion_mining/baseline.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Produces `Candidate(quadruple, score, sources)` and `StatisticalOpinionMiner.fit(examples)`, `.predict_candidates(text)`.
- Uses character n-gram TF-IDF retrieval plus phrase purity statistics.

- [ ] Write failing tests for stable opinion mapping, aspect-opinion match, implicit aspect, rejection of absent substrings, candidate merging, and retrieval transfer.
- [ ] Run `pytest tests/test_baseline.py -q` and confirm RED.
- [ ] Implement minimal candidate generation and scoring.
- [ ] Run test and full suite; confirm GREEN.

### Task 4: OOF and threshold selection

**Files:**
- Create: `src/opinion_mining/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces `run_oof(examples, n_splits, seed)`, `select_threshold(oof_candidates, gold)`, `fit_predict(train, test, threshold)`.

- [ ] Write failing tests proving fold isolation and threshold maximization on a toy dataset.
- [ ] Run `pytest tests/test_pipeline.py -q` and confirm RED.
- [ ] Implement KFold OOF, candidate collection, threshold grid, final fit/predict.
- [ ] Run test and full suite; confirm GREEN.

### Task 5: CLI and reproducible run report

**Files:**
- Create: `scripts/run_baseline.py`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `docs/runbook.md`

**Interfaces:**
- CLI accepts extracted train/test directories or zip paths, output path, folds and seed.
- Writes `Result.csv` and adjacent `run_report.json`.

- [ ] Add CLI smoke test or import-level test before implementation.
- [ ] Confirm RED.
- [ ] Implement zip extraction/loading, OOF, final fit/predict, submission validation and report serialization.
- [ ] Run full suite and CLI against fixture data; confirm GREEN.

### Task 6: Run on official uploaded data and validate artifact

**Files:**
- Generate: `Result.csv`
- Generate: `run_report.json`

**Interfaces:**
- Consumes the two official zip files.
- Produces the actual competition submission.

- [ ] Run OOF on the official train set and record strict F1/threshold.
- [ ] Refit on all official training examples and predict all 2237 test ids.
- [ ] Validate file structure, encoding, field domains, id coverage, order, duplicates and substring constraints.
- [ ] Save final report with counts and hashes.

### Task 7: Final verification and handoff

**Files:**
- Update: `README.md`

- [ ] Run full pytest suite fresh.
- [ ] Run submission validator fresh on final `Result.csv`.
- [ ] Review source tree and document exact reproduction command.
- [ ] Report any remaining limitation: local OOF is evidence, hidden leaderboard score is unknown until Tianchi submission.
