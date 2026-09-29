# Agent Status

## Current P0 integration (primary Codex)

- **Status:** P0 implementation in progress; the previous RBT3 seed-only run was stopped and its artifacts were preserved.
- **Touched:** `src/opinion_mining/data.py`, `src/opinion_mining/neural.py`, `src/opinion_mining/folds.py`, `src/opinion_mining/pipeline.py`, `src/opinion_mining/resource_guard.py`, `src/opinion_mining/dashboard_runtime.py`, `dashboard/index.html`, `scripts/autopilot.py`, `scripts/resume_after_strict.py`, `scripts/calibrate_group_thresholds.py`, `scripts/resource_monitor.py`, and focused `tests/` additions.
- **Verified:** official offsets on 6,633 raw labels have 0 text-range mismatches; 172 `Opinion="_"` labels remain in the training representation; fixed folds are shared by OOF and neural CV; LOFO calibration is reported separately from the in-sample reference; `PYTHONPATH=src python3 -m pytest -q` passes 16 tests; Python compilation passes for `src/` and `scripts/`.
- **Observed:** previous global group calibration was 0.7326 in-sample; the new shrinkage/LOFO report gives 0.7173 LOFO and 0.7267 full-OOF-after-shrink, so the former number is not treated as strict evidence.
- **Resource changes:** 12GB memory budget, 10GB soft and 12GB hard tracked footprint; process-tree RSS, MPS allocated/driver counters, available memory, pressure free percentage, compressor and swap are recorded. Dashboard MPS rendering now reads the actual fields.
- **Remaining:** review the four new parallel agent outputs; integrate the grid module and MLX/manifest/test deliverables only after scope and evidence review; run a clean RBT3 regression, then MacBERT/WWM 3-fold screening.

## 2026-09-28 continuation

- **WWM V2 code 3-fold:** completed without OOM; original decoder strict F1 `0.5764523358`, so it was not promoted.
- **WWM V2.1 offline re-decode:** reused the saved checkpoints and switched the relation score to the anchor grid logits. Full OOF strict F1 is `0.6271228980`; fold-exclusive cross-fit is `0.6174375767`.
- **Verifier ensemble:** WWM V1 predictions filtered by the V2.1 same `(Aspect, Opinion)` pair verifier gives cross-fit strict F1 `0.6337410991` versus WWM V1 `0.6136957` in the same diagnostic. Candidate artifacts and SHA256 are stored under `artifacts/experiments/ensemble_wwm_v1_v21_pairfilter_3fold/` and `artifacts/submissions/candidates/`.
- **Submission queue:** WWM V1 anchor and the verifier ensemble are validated and marked `pending` in `artifacts/submissions/manifest.csv`; no online score is recorded yet because the in-app browser file chooser did not open through the available automation surface.
- **Qwen preflight:** `scripts/mlx_qwen_smoke.py` reports `queue`: `mlx`/`mlx_lm` are absent, the 4-bit model is not cached, and available memory is about 4 GiB. No Qwen process was started.
- **MacBERT V2.1 offline re-decode:** the same grid score source gives full-OOF F1 `0.6359712230` and fold-exclusive cross-fit F1 `0.6210804098`; the validated 2,237-ID candidate has 4,209 rows, 3,865 predicted quadruples, and SHA256 `7dbb0afb56b13474cfa60f72c31873d0f75d5a7b899fc6f0c63202e08c16681d`.
- **Agent routing constraint:** future delegated agents may use only Luna or DeepSeek; no new agent was dispatched when the available route could not guarantee that model selection.

## 2026-09-28 Anchor Task 1

- **Status:** Anchor RBT3 v0 frozen; exact five-fold replay passed. No new training started.
- **Touched:** `scripts/reproduce_anchor.py`, `tests/test_anchor_reproduction.py`, `artifacts/champions/anchor_rbt3_v0/manifest.json`, `artifacts/champions/anchor_rbt3_v0/Result.csv`, `artifacts/champions/anchor_rbt3_v0/test_candidates_merged.json`, `dashboard/archive.json`.
- **Verified:** checkpoint configs for folds 1–5 load with meta tensors; archived test candidate maps cover the same 2,237 IDs; thresholds explicit=0.727 and implicit=0.503; regenerated 5,043-row CSV has 4,930 predicted quadruples and 113 empty IDs; SHA256 exactly matches `f9eac4b6d4ea769e13ea62cac9d3fa9364a70ddf9a0297c8332cc7a80d40d43f`.
- **Scores:** local OOF F1 `0.7208976157082748`; leaderboard F1 `0.7162020541`; kept as separate fields in the manifest and dashboard archive.
- **Checks:** `PYTHONPATH=src python3 -m pytest -q tests/test_anchor_reproduction.py tests/test_neural_targets.py` → 6 passed; `PYTHONPATH=src python3 -m pytest -q` → 86 passed; replay CLI → exact SHA and expected row counts.
- **Next gate:** Task 2 fixed-fold evaluator; do not start ablations until its evaluator contract is implemented and verified.

## 2026-09-28 Anchor Task 2

- **Status:** Fixed-fold evaluator implemented; canonical `artifacts/reports/fold_assignments_seed42.json` is reusable through `fixed_splits(..., assignment_path=...)` and `evaluate_anchor_variant(config, folds)`.
- **Touched:** `src/opinion_mining/folds.py`, `src/opinion_mining/anchor_eval.py`, `tests/test_anchor_eval.py`.
- **Evidence:** `artifacts/experiments/anchor_same_fold_baseline/report.json` records strict F1 `0.7208976157082748`, TP `4626`, FP `1576`, FN `2006`; OOF candidates are saved beside it; report includes state/category metrics, candidate count, config hash, and baseline delta schema.
- **Checks:** `PYTHONPATH=src python3 -m pytest -q tests/test_anchor_eval.py` → 5 passed; full suite → 91 passed. Leakage test rejects candidate ids overlapping `train_ids`.
- **Next gate:** Task 3 B1 official-offsets isolated ablation; no B2/B3 or architecture changes are included in the evaluator commit.

## 2026-09-28 Anchor Task 3 — B1 official offsets

- **Status:** Three-fold isolated training completed; only `use_official_offsets=True` changed relative to the historical RBT3 setup.
- **Evidence:** `artifacts/experiments/anchor_b1_offsets_3fold/report.json`; strict F1 `0.7140305924`, P `0.7360332960`, R `0.6933051870`, TP/FP/FN `4598/1649/2034`; baseline `0.6874480466`, delta `+0.0265825459`, keep gate passed.
- **Checks:** focused neural/P0 tests `13 passed`; full suite `93 passed`. Target exact-span mapping audit is recorded transparently as target mapping scope; both modes had 0 mapping FN/FP on 6,633 spans.

## 2026-09-29 Anchor Task 4 — B2 implicit Opinion

- **Status:** Explicit CLS sentinel for `Opinion="_"` is now gated by `use_implicit_opinion_sentinel`; Aspect implicit behavior remains unchanged. Three-fold isolated training completed with official offsets disabled and B2 sentinel enabled.
- **Evidence:** `artifacts/experiments/anchor_b2_implicit_o_3fold/report.json`; strict F1 `0.7115776281`, P `0.7404629932`, R `0.6848612786`, TP/FP/FN `4542/1592/2090`; baseline delta `+0.0241295815`; implicit-O recall `0.3546511628` vs baseline `0.0`, implicit-O F1 `0.4250871080`; keep gate passed.
- **Checks:** B2/target/audit focused tests `16 passed`; full suite `96 passed`; all 172 implicit-O gold spans enter B2 targets and decoder emits `_`, never `[CLS]`.
- **Next gate:** Task 5 B3 multi-relation isolated ablation; do not combine B1/B2 until B3 is measured independently.

## 2026-09-29 Anchor Task 5 — B3 multi-relation

- **Status:** Shared-span multi-relation preservation is explicit through `preserve_multi_relation`; B3 decoder expands relation top-k while NMS keeps distinct Category/Polarity labels and suppresses only redundant full relations. Three-fold isolated training completed with B1/B2 flags disabled.
- **Evidence:** `artifacts/experiments/anchor_b3_multirelation_3fold/report.json`; strict F1 `0.7063734147`, P `0.7552351528`, R `0.6634499397`, TP/FP/FN `4400/1426/2232`; baseline `0.6874480466`, delta `+0.0189253681`, global keep gate passed.
- **Checks:** B3/shared-target focused tests `15 passed`; full suite `98 passed`. Multi-label shared-span targets preserve both relation classes and decoder retains at least three distinct labels for one span pair under B3.
- **Decision:** B1 remains the strongest isolated RBT3 ablation (`0.7140305924`); B2 is retained as a candidate fix for implicit-O recall; B3 is not yet composed. Next gate is Task 6 deterministic composition using only evidence-backed subset selection.

## 2026-09-29 Anchor Task 6 — anchor_rbt3_v1

- **Status:** Deterministic B1+B2+B3 composition runner completed fixed 3-fold training with no architecture, loss, or threshold-search change.
- **Evidence:** `artifacts/experiments/anchor_rbt3_v1_3fold/report.json`; strict F1 `0.7140305924`, P `0.7360332960`, R `0.6933051870`; best isolated B1 F1 `0.7140305924`; delta `0.0`; composition gate passed. Report embeds the isolated six-bucket/error payloads and explicit fix list/config hash.
- **Decision:** Combined result ties B1 and is the strongest RBT3-compatible anchor; keep `anchor_rbt3_v1` as the RBT3 control for the next architecture stage. No online submission was made.
- **Checks:** Anchor config/runner focused tests `4 passed`; full suite `100 passed`. HF background auto-conversion emitted non-fatal 403 thread warnings; local model loads and all three folds completed.
- **Next gate:** Task 7 OpinioNet-style one-stage challenger, first with RBT3 control before any stronger encoder.
