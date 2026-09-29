#!/usr/bin/env bash
# Final ensemble for the 2026-09-29 late-night run (submit before local midnight).
#
# Members:
#   - anchor_wwm_long_5fold : WWM 5-fold, 10 epochs, lr 1.2e-5   (strongest expected)
#   - anchor_wwm_long_3fold : WWM 3-fold,  8 epochs, lr 1.2e-5
#   - anchor_wwm_5fold      : WWM 5-fold,  4 epochs, lr 2e-5
#   - anchor_macbert_5fold  : MacBERT 5-fold, 4 epochs          (diversity, helped online)
#
# Two weight configurations only, to keep the critical path short (~25 min):
#   A) 1 : 0.7 : 0.4 : 0.3
#   B) 1 : 1.0 : 0.5 : 0.4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NAME="${1:-ensemble_4src_final}"
PYTHONPATH=src python3 scripts/ensemble_submit.py \
  --run artifacts/experiments/anchor_wwm_long_5fold \
  --run artifacts/experiments/anchor_wwm_long_3fold \
  --run artifacts/experiments/anchor_wwm_5fold \
  --run artifacts/experiments/anchor_macbert_5fold \
  --n-splits 5 \
  --name "$NAME" \
  --weight-configs "1,0.7,0.4,0.3|1,1,0.5,0.4"
