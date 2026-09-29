#!/usr/bin/env bash
# Build the upload bundle for cloud GPU training (no secrets included).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M)"
OUT_DIR="$ROOT/artifacts/cloud"
BUNDLE="$OUT_DIR/cloud_bundle_${STAMP}.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/opinion_mining_cloud"
cd "$STAGE/opinion_mining_cloud"

# --- data (public competition text only) ---
mkdir -p llm_data
cp -R "$ROOT/artifacts/llm_data/fold1_3fold" llm_data/ 2>/dev/null || true
cp -R "$ROOT/artifacts/llm_data/full_train" llm_data/ 2>/dev/null || true
mkdir -p artifacts/reports
cp "$ROOT/artifacts/reports/fold_assignments_seed42.json" artifacts/reports/ 2>/dev/null || true

# --- code needed for training + scoring (pure python, no torch import at rest) ---
mkdir -p src/opinion_mining scripts run_cloud_lora.yaml docs
for module in __init__.py data.py submission.py llm_parse.py llm_prompt.py metrics.py folds.py; do
  [ -f "$ROOT/src/opinion_mining/$module" ] && cp "$ROOT/src/opinion_mining/$module" src/opinion_mining/
done
for script in prepare_llamafactory_data.py infer_cloud.py score_returned_predictions.py; do
  cp "$ROOT/scripts/$script" scripts/
done
cp "$ROOT/run_cloud_lora.yaml" .
cp "$ROOT/docs/cloud-training-plan-2026-09-29.md" docs/README_CLOUD.md

cat > README_FIRST.md <<'EOF'
# 云端执行顺序

1) 数据转换（LLaMA-Factory sharegpt）
   PYTHONPATH=src python3 scripts/prepare_llamafactory_data.py --data-dir llm_data/fold1_3fold --out-dir lf_data
   # 把 lf_data/dataset_info.json 与 LLaMA-Factory 的 data/dataset_info.json 合并

2) 训练（配置已对齐公开 0.75~0.78 配方：lr 1e-4 / rank16 / alpha32 / dropout0.05 / cutoff1024 / batch4xaccum8 / 3 epochs / bf16）
   llamafactory-cli train run_cloud_lora.yaml
   # 或改成 Qwen/Qwen3-4B-Instruct 得到 0.75 档；Qwen/Qwen2.5-32B-Instruct 得到 0.81 档

3) 推理 fold-1 的 1077 条
   PYTHONPATH=src python3 scripts/infer_cloud.py --model saves/acos_qwen25_7b_lora --data llm_data/fold1_3fold/test.jsonl --out predictions_fold1.jsonl

4) 回传 predictions_fold1.jsonl（+ 可选 adapter 目录）

5) 本地评分（严格口径，与 BERT 路线可比）
   PYTHONPATH=src python3 scripts/score_returned_predictions.py --predictions predictions_fold1.jsonl --n-splits 3 --valid-fold 1 --candidates-out artifacts/experiments/llm_cloud_fold1/candidates.json
EOF

tar -czf "$BUNDLE" -C "$STAGE" opinion_mining_cloud
echo "bundle: $BUNDLE"
du -sh "$BUNDLE"
