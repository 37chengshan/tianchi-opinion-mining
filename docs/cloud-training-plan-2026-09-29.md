# 云端并行训练方案（2026-09-29）

本地现状与瓶颈：WWM 5 折本地无泄漏 **0.7543**、线上 **0.7583**；本地 4-bit QLoRA 的 Qwen3-4B 只有 **0.6417**（fold-1），说明"LLM 路线在本地被量化与算力压住了"。同赛题公开方案在云端用 bf16 LoRA 达到：Qwen3-4B **0.75**、Qwen2.5-7B **0.78**、Qwen2.5-32B **0.81**。

→ 云端最高性价比任务是：**同一份数据、同一份 fold 划分下，用 bf16 LoRA 训练 Qwen2.5-7B（可升级 32B）**，本地则继续跑 BERT 集成与大 backbone。

## 1. 训练协议（必须与本地一致，才能严格比较）

- 数据：`artifacts/llm_data/fold1_3fold/`（train 2152 / valid 1077 / test 1077），划分来自 `artifacts/reports/fold_assignments_seed42.json` 的 3 折 fold-1。
- 监督目标：严格 JSON（`{"quadruples":[{aspect,opinion,category,polarity}]}`），**保留 implicit-O**（opinion 为 "_"）。
- 评测：把预测写成 `id -> 四元组`，用本地 `scripts/predict_llm_lora.py` 的同一 strict 评估口径打分（fold-1 对照：BERT WWM **0.7442**）。
- 晋级：fold-1 ≥ 0.72 → 用 `artifacts/llm_data/full_train/`（3229 行）重训 → 预测测试集 2237 条 → 回传 CSV。

## 2. 机器与时间/成本（估算）

| 方案 | 显存 | 训练时长 | 参考价 | 预期 fold-1 |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B LoRA bf16, rank16, cutoff 1024, batch 4×accum 8, 3 epochs | ≥24GB（4090/A10） | 2–4 h | ¥2–4/h | 0.72–0.78 |
| Qwen3-4B LoRA bf16（同配方） | ≥16GB | 1.5–2.5 h | 同上 | 0.70–0.75 |
| Qwen2.5-32B LoRA（对齐 0.81 方案） | 2×24GB 或 1×80GB | 4–8 h | ¥8–20/h | 0.75–0.81 |
| 备选：`hfl/chinese-roberta-wwm-ext-large` 5 折（我们的 BERT 路线） | ≥16GB | 1.5–2 h（比本地快 4–6×） | ¥2–4/h | 本地等价 +0.01~0.02 |

供应商候选：阿里云 PAI-DSW（与赛题同生态）、AutoDL/AutoDL 类 GPU 租赁、硅基流动（公开方案用的就是它的 LoRA 服务）、gcloud（本机已装 gcloud CLI）。

## 3. 上传包

```bash
bash scripts/make_cloud_bundle.sh          # 生成 artifacts/cloud/cloud_bundle_<date>.tar.gz
```

包内含：`llm_data/`（train/valid/test + full_train）、`fold_assignments_seed42.json`、`llm_prompt.py`（prompt 契约）、`predict_score.py`（严格评分）、`run_cloud_lora.yaml`（LLaMA-Factory 配置）、`README_CLOUD.md`（本文件）。

## 4. 云端执行（LLaMA-Factory 路径，对齐公开 0.75–0.81 配方）

```bash
# 1) 环境
pip install "llamafactory[torch,metrics]"   # 或 git clone LLaMA-Factory && pip install -e .
# 2) 数据注册：把 llm_data 转成 sharegpt/alpaca 格式（脚本已含转换）
python tools/prepare_llamafactory_data.py --data-dir llm_data/fold1_3fold --out-dir lf_data
# 3) 训练（等价配置见 run_cloud_lora.yaml：lr 1e-4、cosine+warmup25、rank16/alpha32/dropout0.05、cutoff1024、batch4×accum8、3 epochs、bf16）
llamafactory-cli train run_cloud_lora.yaml
# 4) 推理 fold-1 的 1077 条，导出 predictions.jsonl（id + quadruples）
python tools/infer_cloud.py --model <merged_or_adapter> --data lf_data/test.jsonl --out predictions_fold1.jsonl
```

替代路径（不装 LLaMA-Factory）：`peft` + `transformers` 的常规 LoRA 微调，超参同上；推理用 `transformers.generate` + 相同 prompt。

## 5. 回传与本地集成（严格可比）

回传三样：`predictions_fold1.jsonl`、`run_config.json`（超参+数据指纹）、`adapter/`（可选）。

本地打分与融合：

```bash
PYTHONPATH=src python3 scripts/score_returned_predictions.py --predictions predictions_fold1.jsonl --fold 1   # strict F1
PYTHONPATH=src python3 scripts/ensemble_offline.py --source "wwm=artifacts/experiments/anchor_wwm_5fold/oof_candidates.json" --source "llm=<llm候选json>" --n-splits 5
```

若 LLM fold-1 ≥ 0.73：本地已有 `scripts/ensemble_submit.py` 可直接做"BERT + LLM"同档次融合出提交；若 LLM 单独 ≥0.76，也可单独提交（更符合公开方案 0.78–0.81 的路线）。

## 6. 并行分工（今晚起）

| 位置 | 任务 | 产出 |
| --- | --- | --- |
| 本机 GPU | MacBERT 5 折（19:30 完成）→ WWM+MacBERT 融合提交 | 预计线上 0.760–0.765 |
| 本机 GPU | WWM 长训练 8 epochs（~20:40 完成）验证欠训练 | 若提升，替换主模型 |
| 云端 GPU（新） | Qwen2.5-7B bf16 LoRA（fold-1 协议） | fold-1 strict F1；≥0.72 即全量重训并预测测试集 |
| 云端 GPU（可选） | `roberta-wwm-ext-large` 5 折 / TAPT | 直接替换本地最强 BERT，与本地模型融合 |

## 7. 风险与约束

- 云端数据只上传公开赛题数据与自建脚本，不含密钥；`run_cloud_lora.yaml` 中不放任何 token。
- 必须使用同一 fold 划分与同一 strict 口径，否则无法与本地 0.7543 / 线上 0.7583 比较。
- 不做阈值暴搜：LLM 输出直接取集合，融合阈值一律用本地 fold-safe 交叉拟合。
- 提交仍遵循"仅提交离线已证明的候选"，一次只改一个变量。
