# 实验历史与产物清理记录（2026-09-29）

本文档替代已删除的实验权重：删除的只是模型文件与中间产物，所有分数、配置路径与结论保存在此。

## 1. 唯一线上冠军（保留，不可覆盖）

| 项目 | 值 |
| --- | --- |
| candidate_id | `candidate-f9eac4b6d4ea-b384f88e9621` |
| Leaderboard F1 | 0.7162020541 |
| Local strict OOF F1 | 0.7208976157082748（P 0.7458884231 / R 0.6975271411） |
| Result.csv SHA256 | `f9eac4b6d4ea769e13ea62cac9d3fa9364a70ddf9a0297c8332cc7a80d40d43f` |
| 架构 | `hfl/rbt3` one-stage span-relation，5-fold，max_length=96，batch=4，grad_accum=2，epochs=4，trainable_layers=2，explicit=0.727 / implicit=0.503 |
| 复现 | `scripts/reproduce_anchor.py` + `artifacts/champions/anchor_rbt3_v0/manifest.json`；五折 checkpoint 与 fold test candidates 保留在 `artifacts/experiments/neural_5fold_confirm/` |

## 2. 保留的锚点实验证据（JSON，不含权重）

| 实验 | 3-fold strict F1 | 相对历史 RBT3 基线 0.6874480466 | 结论 |
| --- | --- | --- | --- |
| `anchor_rbt3_v1_3fold`（B1+B2+B3 组合） | 0.7140305924 | +0.0265825459 | 当前最强 RBT3 控制组；权重保留 |
| `anchor_b1_offsets_3fold`（official offsets） | 0.7140305924 | +0.0265825459 | 单独最强；通过 keep gate |
| `anchor_b2_implicit_o_3fold`（implicit-O sentinel） | 0.7115776281 | +0.0241295815 | 隐式观点召回 0 → 0.3547；通过 keep gate |
| `anchor_b3_multirelation_3fold`（多关系保留） | 0.7063734143 | +0.0189253681 | 通过 keep gate，不单独晋级 |
| `anchor_same_fold_baseline`（5-fold OOF 复核） | — | — | 固定 fold 评估器基线，F1 0.7208976157，TP/FP/FN = 4626/1576/2006 |

评估器与固定划分：`src/opinion_mining/anchor_eval.py`、`src/opinion_mining/folds.py`、`artifacts/reports/fold_assignments_seed42.json`。

## 3. 已删除的实验与原因

| 目录（原） | 大小 | 删除原因 |
| --- | --- | --- |
| `tapt_macbert_v3` / `tapt_macbert_v2` / `tapt_macbert_20260928` | 约 6.8 GB | 旧 MacBERT 路线 TAPT，主线已改为强 encoder 重做（Task 9） |
| `grid_wwm_v2code_3fold` / `grid_wwm_3fold_20260928` | 约 2.4 GB | Grid V2 路线，OOF F1 仅 0.5765~0.6271，已判定非主线 |
| `grid_macbert_v2code_control_3fold` / `grid_macbert_tapt_3fold` / `grid_macbert_3fold_20260928_v2` | 约 3.6 GB | 同上，MacBERT grid 0.6154~0.6360 |
| `grid_rbt3_3fold_20260928` | 460 MB | grid 小模型对照，无晋级价值 |
| `grid_macbert_v21_grid_redecode_3fold_test` / `grid_wwm_v21_grid_redecode_3fold[_test]` | 约 410 MB | V2.1 re-decode 中间产物，结论已记录 |
| `evolution_3fold_001..010_seed*` | 约 4.4 GB | seed sweep，计划已明确禁止 |
| `anchor_b1/b2/b3` 的 `fold_*/model.pt` | 约 2.3 GB | 隔离消融已完成，结论与 OOF/test candidates 已归档 |
| `neural_3fold_rbt3` 的 `fold_*/model.pt` | 约 450 MB | 3-fold 历史基线 0.6874480466，其 threshold.json 与报告保留 |
| `/tmp/ref_acos`（外部参考方案数据与提交） | 约 1 MB | 仅用于一次性诊断，结论记入第 5 节，不保留他人提交文件 |

删除原则：只删权重与中间产物；每个实验的 `report.json` / `threshold.json` / `oof_candidates.json` / `test_candidates.json` / `fold_assignments.json` / `config.json` 全部保留。

## 4. 当前可用产物

- 冠军：`artifacts/champions/anchor_rbt3_v0/`（manifest、Result.csv、merged candidates）
- RBT3 锚点权重：`artifacts/experiments/anchor_rbt3_v1_3fold/fold_*/model.pt`（3-fold）
- 冠军五折权重：`artifacts/experiments/neural_5fold_confirm/fold_*/model.pt`（复现门禁用）
- 强 encoder 本地缓存：`hfl/chinese-roberta-wwm-ext`、`hfl/chinese-bert-wwm-ext`
- 训练脚本：`scripts/reproduce_anchor.py`、`scripts/run_anchor.py`、`scripts/run_opinionet_screen.py`
- 新 head：`src/opinion_mining/opinionet_head.py`、`opinionet_decoder.py`、`opinionet_trainer.py`

## 5. 外部同赛题参考（仅用于诊断，不作为训练标签）

来源：`github.com/9yly/TianChiTrack6_ABSA`（同一赛题、同一份化妆品评论数据）。

数据同一性核验：Train_reviews.csv 与 Train_labels.csv 的 SHA256 与本仓库完全一致；Test_reviews.csv 仅换行符不同（CRLF vs LF），2237 个 id 与文本逐条一致。

其报告分数（LoRA 微调 Qwen 系列，同一测试集）：Qwen3-4B = 0.75，Qwen2.5-7B-Instruct = 0.78，Qwen2.5-32B-Instruct = 0.81，32B + BM25/Reranker RAG = 0.81（第一赛季第 3 / 第二赛季第 1）。

与本仓库冠军（0.7162）的输出对比（逐 id、四元组严格匹配）：

| 指标 | 本仓库冠军 | 参考方案 |
| --- | --- | --- |
| 预测四元组总数 | 4930 | 4415 |
| 有预测的 id 数 | 2124 / 2237 | 2237 / 2237 |
| implicit-O 预测数 | 0 | 63 |
| 我们多预测的 id 数 | 639（合计 +885） | — |
| 参考多预测的 id 数 | — | 300（合计 +370） |
| 仅我们有的四元组 | 1446（implicit-A 1110 / explicit 336） | — |
| 仅参考有的四元组 | — | 931（implicit-A 518 / explicit 350 / implicit-O 63） |

结论：差距主要在三处 —— (1) 隐式观点完全不预测；(2) 有 113 个 id 直接空输出；(3) implicit-A 过生成导致精确率损失。前两项与 B2 修复方向一致。

参考方案的推理侧做法（可借鉴，非直接复用）：严格 JSON 约束的 system prompt、Pydantic 校验后按原文位置排序、去重、按 id 升序输出无表头 CSV。

## 6. 清理验证（2026-09-29 执行）

| 检查 | 结果 |
| --- | --- |
| 释放空间 | `artifacts/experiments` 23 GB → 3.2 GB；磁盘空闲 24 GB → 43 GB |
| 冠军复现 | `tests/test_anchor_reproduction.py` 2 passed（SHA256 仍精确一致） |
| 全量测试 | 101 passed, 1 skipped（删除 grid 产物后，套件从 57 s 降到 8 s） |
| 失效测试处理 | `tests/test_error_analysis_grid_v2.py` 中依赖已删除 grid 产物的 ceiling 诊断测试已删除，其余通用桶审计测试保留 |

## 7. 已证伪的做法（避免重复劳动）

| 做法 | 证据 | 结论 |
| --- | --- | --- |
| 每个 id 至少输出 1 条（top-1 兜底） | 冠军 5-fold OOF：基线 F1 0.720898；兜底后 floor=0.0 → 0.718853，floor=0.2 → 0.720061 | 负收益：召回 +62 TP 但精确率损失更大；参考方案覆盖全部 id 是因为模型本身更强，不是后处理可复制 |
| OpinioNet-style head 替换 pointer head（RBT3） | 3-fold：OpinioNet head 0.7087567 vs anchor v1 0.7140306，差 -0.0053 | 未过控制组门禁（允许 -0.002）；强 encoder 继续沿用已验证的 pointer head |

## 8. Anchor v1 组合模型分项诊断（3-fold，strict）

| 桶 | F1 | P | R | TP/金标 |
| --- | --- | --- | --- | --- |
| explicit | 0.6667 | 0.7055 | 0.6319 | 1090 / 1725 |
| implicit-A | 0.7401 | 0.7561 | 0.7248 | 3432 / 4735 |
| implicit-O | 0.4537 | 0.4663 | 0.4419 | 76 / 172 |
| 合计 | 0.7140306 | 0.7360 | 0.6933 | 4598 / 6632 |

结论：组合没有回退 B2 的隐式观点收益（0.4251 → 0.4537）。下一步的主攻桶是 **explicit（F1 最低之一，缺 635 TP）** 与 **implicit-O（缺 96 TP）**；implicit-A 已到 0.74，继续压 FP 收益有限。

## 9. 新增工具（本轮）

| 文件 | 作用 |
| --- | --- |
| `scripts/build_llm_dataset.py` | 按 canonical fold 构建 chat JSONL（train/valid/test）；保留 implicit-O 监督（fold-1 训练集内含 114 条） |
| `src/opinion_mining/llm_parse.py` | 严格解析模型输出：抽 JSON、白名单校验、子串校验、去重、按原文位置排序；丢弃 dual-implicit |
| `scripts/predict_llm_lora.py` | MLX 推理 + 严格 F1 离线评分（同一套 strict 评估器，与 BERT 路线可比） |
| `scripts/ensemble_offline.py` | 多源 score-level 融合，阈值交叉拟合（不在被打分的 fold 上调参），输出逐源对比与 TP 交叠 |
| `src/opinion_mining/opinionet_head.py` 等 | OpinioNet-style one-stage head / constrained decoder / trainer |

## 10. 强 encoder 筛选结果（WWM / RoBERTa-wwm-ext，pointer head）

配置：`hfl/chinese-roberta-wwm-ext`，trainable_layers=4，max_length=96，batch=2，epochs=4，lr 2e-5，head lr 8e-4；其余与 anchor v1 完全一致（仅 backbone 变化）。

| 指标 | RBT3 v1（3-fold） | WWM（3-fold） | 差值 |
| --- | --- | --- | --- |
| 交叉拟合 F1（无泄漏） | 0.7120125 | **0.7458725** | **+0.03386** |
| 同折内 F1（参考，非严格） | 0.7140306 | 0.7492962 | +0.03527 |
| Precision | 0.74050 | 0.78331 | +0.04281 |
| Recall | 0.68758 | 0.71185 | +0.02427 |
| 逐折 F1 | 0.70648 / 0.70745 / 0.72478 | 0.72999 / 0.74368 / 0.76283 | — |

判定：计划门槛「强 encoder 3-fold ≥0.735 进 5 折」**已通过**（0.7459 ≥ 0.735）；距 ≥0.75 的高优先级线仅差 0.0041，已启动 5 折确认（`artifacts/experiments/anchor_wwm_5fold`，预计每折约 30 分钟）。

异构融合（RBT3 v1 + WWM，3-fold 交叉拟合，bonus 0）：

| 组合 | 交叉拟合 F1 | Precision | Recall |
| --- | --- | --- | --- |
| WWM 单模型 | 0.7458735 | 0.78331 | 0.71185 |
| RBT3 v1 单模型 | 0.7120125 | 0.74050 | 0.68758 |
| **两者等权融合** | **0.7479878** | 0.77632 | 0.72165 |

结论：异构 backbone 融合在无泄漏口径下再 +0.0021，说明两条路线互补；后续加入第二个强 encoder 与 LLM 输出仍有余量。

组合策略对比（同一交叉拟合口径，`scripts/policy_compare.py`）：

| 策略 | F1 | TP | 预测数 |
| --- | --- | --- | --- |
| WWM 单模型 | 0.7450 | 4713 | 6020 |
| RBT3 v1 单模型 | 0.7120 | 4564 | 6188 |
| 并集（任一模接受） | 0.7310 | 5082 | 7273 |
| 交集（两者都接受） | 0.7253 | 4195 | 4935 |
| 加权平均融合（`ensemble_offline.py`） | **0.7480** | — | — |

TP 互补性：RBT3 与 WWM 的 TP 重叠 Jaccard = **0.917**（共同 2633，RBT3 独有 92，WWM 独有 145）。因此"再加一个同族 BERT encoder"预期收益很小，优先级下调；真正的互补信号应来自不同模型家族（LLM）与训练稳定性技巧。

### 第二强 encoder 筛选（MacBERT-base，pointer head，3-fold）

| 指标 | 数值 |
| --- | --- |
| 同折内 F1（参考，非严格） | **0.7357488** |
| 对照：同配置 WWM 3 折 | 0.7492962 |
| 计划门槛（≥0.735 进 5 折） | **通过** |

已启动 `anchor_macbert_5fold`（同配置，仅 fold 数变化）用于与 WWM 5 折做同档次融合。

**MacBERT 5 折结果与融合判定（不收）**：

| 来源 | 5 折交叉拟合 F1 |
| --- | --- |
| WWM 5 折 | 0.753595 |
| MacBERT 5 折 | 0.737006 |
| 两者等权融合 | 0.753572 |

结论：MacBERT 同族且略弱，等权融合**无增益（-0.000023）**，不采用；只有"更强或更不同"的成员才值得融合（下一候选：8 轮长训练）。

### 5 折确认（WWM，仅 fold 数变化）

| 口径 | 3-fold | 5-fold |
| --- | --- | --- |
| 同折内 F1（参考，非严格） | 0.7492962 | **0.7563948** |
| 交叉拟合 F1（无泄漏） | 0.7458725 | 见 `artifacts/experiments/anchor_wwm_5fold/crossfit_report.json` |

5 折同折内 F1 0.7564 是本仓库目前最强单模型（此前 RBT3 5 折 0.7209）。训练产物：`artifacts/experiments/anchor_wwm_5fold/`（含 OOF 候选、测试候选、5 个 checkpoint）。

**5 折交叉拟合（无泄漏）F1 = 0.754295**（P 0.760656 / R 0.748040），逐折 0.7421 / 0.7477 / 0.7487 / 0.7649 / 0.7675。

融合负结果：把 RBT3 v1（0.7112）按等权并入 WWM 5 折（0.7536）后 F1 = **0.75204**，反而下降 0.0016。结论：**弱模型会拖累强模型，不应等权合并**；后续只与同档次模型（MacBERT 5 折、LLM）做融合，或使用按强度加权/仅用更强者。

提交候选（本地生成 + 校验，未上传）：

| 项目 | 值 |
| --- | --- |
| candidate_id | `candidate-84e0474e404e-ac9a430bf2db` |
| 文件 | `artifacts/submissions/candidates/anchor_wwm_5fold/Result.csv` |
| SHA256 | `84e0474e404e89ef324d5649bbc28ee21cb6e0522c0151e9f891eadecb7c48e8` |
| 行数 / 有预测 id / 预测四元组 | 4392 / 2102 / 4257（135 个 id 为空） |
| 阈值（OOF 选择） | explicit 0.843，implicit-A 0.759，implicit-O 0.786 |
| 本地无泄漏分数 | 交叉拟合 F1 **0.754295** |
| 对照：当前线上冠军 | 本地 0.720898 → LB 0.7162020541（差 -0.0047） |

### 关键发现：当前模型欠训练（来自逐 epoch 监控曲线）

WWM 5 折每折的监控 F1（固定低阈值 0.12/0.10 下的验证集 F1）：

| fold | epoch1 | epoch2 | epoch3 | epoch4 |
| --- | --- | --- | --- | --- |
| 1 | 0.2192 | 0.2550 | 0.2960 | 0.3169 |
| 2 | 0.2186 | 0.2603 | 0.3108 | 0.3448 |
| 3 | 0.2201 | 0.2673 | 0.2884 | 0.3444 |
| 4 | 0.2311 | 0.2854 | 0.3162 | 0.3362 |
| 5 | 0.2438 | 0.2663 | 0.3143 | 0.3673 |

**5 折全部在最后一个 epoch 仍在上升**，说明 4 epochs 明显不够。

外部证据一致：天池同赛题论坛方案（`https://tianchi.aliyun.com/forum/post/935937`，OpinioNet 复现）使用 **Adam lr 5e-6 + 最多 30 epochs + 每折按验证 F1 选最佳权重**，并采用"累加分数后平均"的多模型集成与逐模型阈值（`thresh_dict.json`）。其另一篇（`post/935869`）为 RoBERTa-wwm 多任务 BIO+极性+类别方案，未见分数披露。

行动：夜间运行 **WWM 5 折、epochs=10、lr=1.2e-5**（其余不变），用交叉拟合验证是否真的提升。

### 候选集上限诊断（决定下一步主攻方向）

对 OOF 候选集（未阈值化）统计"金标是否已在候选集合中"：

| 模型 | 候选行数 | 金标已覆盖 | 候选召回上限 | 完美排序 F1 上限 |
| --- | --- | --- | --- | --- |
| WWM 3-fold | 771,844 | 6481 / 6632 | **0.9772** | 0.9885 |
| RBT3 v1 3-fold | 783,976 | 6412 / 6632 | 0.9668 | 0.9831 |
| 线上冠军（RBT3 5-fold） | 20,168 | 5245 / 6632 | 0.7909 | 0.8832 |

结论：WWM 的候选集**几乎已包含全部金标**，当前 0.746 的 F1 主要损失在**排序与阈值选择**，而不是生成。因此新增 fold-safe 候选重排序器（`src/opinion_mining/reranker.py` + `scripts/train_reranker.py`，嵌套交叉拟合校准阈值，绝不用被打分 fold 调参）作为下一主攻方向。另测得最大评论仅 72 token，`max_length=96` 无截断损失，max_length 不是瓶颈。

重排序实测（WWM 候选，每 id 取 top-80，HistGradientBoosting，嵌套交叉拟合阈值）：

| 指标 | 单模型阈值法 | 重排序器 |
| --- | --- | --- |
| F1 | 0.74587 | **0.74641** |
| Precision | 0.78331 | 0.77017 |
| Recall | 0.71185 | 0.72407 |
| TP | 4713 | 4802 |

结论（负结果）：在"候选召回上限 0.968"的前提下，用现有 33 维特征（主要是模型自身分数的派生量）重排序**几乎没有增益**。说明当前选择已在模型自身分数信息量下接近最优；要真正吃下 0.24 F1 的上限空间，需要**独立证据源**（LLM 判定、第二模型族），而不是对同一分数的再加工。

5 折口径复核（WWM 5 折候选，每 id 取 top-60）：

| 策略 | F1 | P | R | TP |
| --- | --- | --- | --- | --- |
| 纯阈值（交叉拟合） | **0.754295** | 0.7607 | 0.7480 | — |
| 重排序（33 维，GBDT） | 0.752700 | 0.7648 | 0.7410 | 4914 |
| 重排序 + 其他 fold 标签先验（10 维） | 0.717990 | 0.7677 | 0.6743 | 4472 |

判定：重排序**不采用**；"标签先验"特征反而显著伤害（-0.036），因为模型过度依赖"该四元组在其他 fold 出现过"这一强特征，而在目标 fold 上这一信号分布不同。结论：保持纯阈值决策；上限空间只能靠更强的生成/更强的独立证据来吃。

## 11. LLM 路线实操记录（本机实测）

### fold-1 严格评分（Qwen3-4B-4bit + LoRA，MLX，训练 2 epochs / 2152 iters）

| 指标 | 数值 |
| --- | --- |
| strict F1 | **0.6416884** |
| Precision / Recall | 0.6688 / 0.6167 |
| TP / 预测数 / 金标 | 1353 / 2023 / 2194 |
| 解析失败 | 0（严格 JSON 解析器 100% 成功） |
| 预测中的 implicit-O | 34 条（BERT 路线该 fold 也依赖 B2 sentinel） |
| 对照：同 fold-1 的 WWM BERT | **0.744197**（P 0.7506 / R 0.7379） |

结论：本地 4-bit QLoRA（2 epochs、等效 batch 8）明显弱于参考方案的云端 bf16 微调（3 epochs、等效 batch 32，其报告 0.75）。

### 与 BERT 的融合实测（同一 1077 个 fold-1 id）

| 策略 | F1 | P | R | TP |
| --- | --- | --- | --- | --- |
| WWM 单独 | **0.744197** | 0.7506 | 0.7379 | 1619 |
| LLM 单独 | 0.641688 | 0.6688 | 0.6167 | 1353 |
| 并集 | 0.708567 | 0.6317 | 0.8067 | 1770 |
| 交集 | 0.673012 | 0.8723 | 0.5479 | 1202 |

TP 结构：共同 1202，WWM 独有 417，LLM 独有 151。

判定：**LLM 单独弱于 BERT，等权融合（并集/交集）都变差**；LLM 独有 TP 仅 151，而 WWM 独有 417 —— 说明当前 LLM 是"较弱且重叠"的信息源，不值得再投入 ~8 小时训练 fold-2/3 适配器。LLM 路线的进一步投入下调优先级；若要复活该路线，需要更长训练/更高精度（bf16 LoRA）或更大模型（7B 已下载备用）。

环境：mlx 0.32.3 / mlx-lm 0.31.3（`pip install --user --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple mlx-lm`，官方 PyPI 直连约 10 分钟未完成，镜像 17 秒完成）。

冒烟测试（20 iters，Qwen3-4B-4bit，batch 1）：

| 指标 | 实测值 |
| --- | --- |
| 可训练参数 | 3.67M / 4022M（0.091%，num-layers=8） |
| 峰值内存 | 3.96 GB |
| 速度 | 0.42–0.46 it/s（约 2.3 s/iter，batch 1，max-seq-length 512） |
| 训练损失 | 1.560 → 0.361（20 iters） |
| 数据加载 | mlx-lm 直接读取本仓库 chat JSONL（messages + mask-prompt）无报错 |

正式筛选配置：`--batch-size 2 --grad-accumulation-steps 4 --epochs 2 --learning-rate 1.5e-5 --num-layers 8`（等效样本 4304，约 2152 iters，预计 1.5–2.5 小时）。

## 12. 评测协议（LLM 与 BERT 可比）

| 项目 | BERT 路线 | LLM 路线（本轮） |
| --- | --- | --- |
| 训练划分 | canonical `fold_assignments_seed42.json`，3-fold | 同一文件，train = fold2+fold3，valid = fold1 |
| 阈值 | 交叉拟合（不在被打分 fold 上选阈值） | 无需阈值，模型直接输出四元组集合 |
| 打分 | `evaluate_anchor_variant` / `crossfit_threshold_score`，strict quadruple F1 | `scripts/predict_llm_lora.py`，同一 strict 评估器 |
| 对照数字 | anchor v1 fold-1 = **0.70648**（crossfit），fold2 = 0.70745，fold3 = 0.72478 | 待测：fold-1 必须 ≥ 0.70648 才有资格进入下一步 |
| 数据 | 全部 3229 条标注 | 同上；训练集保留全部 4735 条 implicit-A 与 114 条 implicit-O（fold-1 训练集口径） |

## 14. 云端实例核验（2026-09-29，结论：暂缓）

实例：`root@36.150.116.206 -p 34514`（SSH 密钥 `~/.ssh/id_ed25519` 免密可用）。

已核验并写入 `/workspace/env_manifest.json`：

| 项目 | 实测值 |
| --- | --- |
| 镜像 | rocm-pytorch |
| GPU | AMD Radeon（gfx1100 / RDNA3），VRAM **48.0 GB** 可用（rocm-smi 报 51.5 GB 总量） |
| ROCm / HIP | 7.2.4 / 7.2.53211 |
| Python | `/opt/venv/bin/python` 3.12.3 |
| PyTorch | 2.10.0+rocm7.2.4，`torch.cuda.is_available()=True`，bf16 支持 |
| CPU / 内存 | 128 核 / 503 GB |
| 存储 | `/workspace` 持久盘 **25 GB**（NFS）；`/` overlay 3.5 TB（2.0 TB 可用） |
| 持久化布局 | 已建 `tianchi-opinion-mining/ models/ hf_cache/ checkpoints/ artifacts/ scratch/` |

**阻塞项：容器无公网出口。** 实测 DNS 可解析（pypi.org / huggingface.co 均返回 IPv4），但 TCP 连接全部超时：`pypi.org:443`、`huggingface.co:443`、`1.1.1.1:443`、`223.5.5.5:443`、`8.8.8.8:53` 全部 FAIL；仅集群 DNS `10.111.255.254:53` 可达；无 `HTTP(S)_PROXY`；无内部 PyPI；镜像内**没有** transformers/peft/tokenizers，也没有任何预置模型。

因此在该实例上：`pip install` 与 HF 模型下载均不可行；唯一路径是"本地上传依赖轮子 + 权重"。按当时网络实测（本机 ≈2.5–3 MB/s，14B bf16 ≈28 GB ≈ 2.5–3 小时）今晚无法完成，故本轮暂缓云端，等以下任一条件满足再启用：

1. 控制台开启公网出口 / 绑定 EIP；
2. 打开平台"挂载预置模型"（当前计划要求不挂载，可临时放宽）；
3. 或接受"本地先下载 + 上传"的时间成本（14B ≈ 28 GB，或 7B ≈ 15 GB）。

无论哪种方式，先把依赖（transformers/peft/accelerate/datasets 的 manylinux x86_64 轮子）作为一次性上传包准备好，能显著缩短下次启动时间。

## 16. 线上提交记录与"本地 → 线上"校准（2026-09-29 夜）

| # | candidate_id | 组成 | 本地无泄漏 F1 | 线上 LB | 差 |
| --- | --- | --- | --- | --- | --- |
| 1 | `candidate-84e0474e404e-...` | 单 WWM 5 折（4 轮） | 0.754295 | 0.7583 | +0.0040 |
| 2 | `candidate-b20aa0997a82-...` | WWM5 + 8 轮长训练 3 折（1 : 0.7） | 0.769337 | 0.7645 | −0.0048 |
| 3 | `candidate-d4978b819276-...` | 上面两者 + MacBERT5（1 : 0.7 : 0.3） | 0.768227 | **0.7668** | −0.0014 |

结论：

1. 本地与线上的差不是常数（−0.0048 ~ +0.0040），**本地 0.001 级差异不可作为胜负依据**；
2. 成员更"多样"（加入 MacBERT）的融合在线上反而更高（+0.0023，尽管本地低 0.0011）；
3. 因此晋级判据应设为：**本地比现任高 ≥0.004** 才值得替换，或者"同等本地分但更多样"也值得一试；
4. 当前 immutable online champion = `candidate-d4978b819276-ac9a430bf2db`，LB **0.7668**；历史冠军 0.7583 / 0.7162020541 保留。

当前本地最强单模型：`anchor_wwm_long_5fold`（WWM 5 折 × 10 轮，训练中，折内监控 F1 0.43~0.47，对比 4 轮版 0.31~0.37）。

## 18. 2026-09-29 冲刺总结（当日收尾）

**关键科学发现：模型长期欠训练。** 4 epochs 时 5 折每一折的验证监控 F1 都还在上升；把 WWM 训练轮数从 4 提到 8/10、学习率从 2e-5 降到 1.2e-5 后：

| 模型 | 同折内 F1（参考） | 说明 |
| --- | --- | --- |
| WWM 5 折 × 4 轮 | 0.75639 | 白天最好单模型 |
| WWM 3 折 × 8 轮 | 0.76370 | 少折数也超过 5 折 4 轮 |
| **WWM 5 折 × 10 轮** | **0.77309** | 历史最强单模型 |

**当日最终交付（4 源融合）**：

| 项目 | 值 |
| --- | --- |
| candidate_id | `candidate-5ad6c1335879-da56f5ae2ac2` |
| 文件 | `artifacts/submissions/candidates/ensemble_4src_final/Result.csv` |
| SHA256 | `5ad6c1335879a33c5232ee6f8f0f6ea823524c331f82f6cba1bddfbf5ac924e7` |
| 本地无泄漏 F1 | **0.776796**（P 0.8066 / R 0.7491，逐折 0.7638 / 0.7741 / 0.7711 / 0.7847 / 0.7900） |
| 组成 | WWM 5折×10轮 (1.0) + WWM 3折×8轮 (0.7) + WWM 5折×4轮 (0.4) + MacBERT 5折 (0.3) |
| 阈值 | explicit 0.827 / implicit-A 0.774（OOF 拟合，fold-safe 交叉验证确认） |
| 校验 | 2237 id 全覆盖、4275 行、4136 条预测、139 空 id、无 BOM、升序、枚举合法 |

当日线上轨迹（全部实测）：

| 提交 | 组成 | 本地 | 线上 |
| --- | --- | --- | --- |
| 起点 | RBT3 5 折（历史冠军） | 0.72090 | 0.7162 |
| ① | WWM 5 折 × 4 轮 | 0.75430 | 0.7583 |
| ② | WWM5 + 8 轮 3 折（1:0.7） | 0.76934 | 0.7645 |
| ③ | ②+MacBERT5（1:0.7:0.3） | 0.76823 | 0.7668 |
| ④ | 4 源（1:0.7:0.4:0.3） | 0.77680 | **0.7734（当日最佳）** |
| ⑤ | 4 源备选权重（1:1:0.5:0.4） | 0.77530 | 0.7702 |
| ⑥⑦ | 单模型 5折×10轮 / 3折×8轮 | 0.77309 / 0.76370（同折内） | 0.7633 / 0.7559 |

**当日结论**：`candidate-5ad6c1335879-da56f5ae2ac2`（4 源，本地 0.77680）LB **0.7734** 为最终冠军；比当日起点 +0.0572。距离 0.775（四舍五入 0.78）仅差 **0.0016**，但剩余时间不足以再训练新模型，且同批成员的所有可行权重组合均已试投（E2 更低，单模型更低）。

**要在下一次拿 0.78 的已量化路线（按性价比排序）**：

| 措施 | 本地预期增益 | 成本 | 依据 |
| --- | --- | --- | --- |
| large backbone（24 层 RoBERTa-wwm-ext-large）5 折 | +0.010~0.020 | 3–4h 本地 | encoder 换代在 base 上已验证 +0.034 |
| 更长训练（10→16 轮） | +0.005~0.010 | 3–4h | 4→10 轮实测 +0.017 |
| 多样成员（MacBERT-large / ERNIE / 第二次 seed） | +0.003~0.008 | 各 2–3h | MacBERT 加入后线上 +0.0023 |
| TAPT（官方训练文本 MLM） | +0.004~0.008 | 1–2h | 计划 Task 9 门槛 ≥+0.004 |
| LLM 路线（7B/14B bf16 LoRA） | +0.01~0.04 | 需出网或上传 28GB | 同赛题公开方案 0.75–0.81 |

**经验教训（写进下一个赛季的默认做法）**：

1. 训练轮数必须先用监控曲线确认收敛，不能拍 4 轮；
2. 本地 0.001 级差异在线上不可分辨，**多样性比本地微差更重要**；
3. 云端实例先验出网能力（本次 rocm-pytorch 沙箱无公网出口，导致 LLM 大模型路线当天不可用）；
4. 关键路径上"融合出提交"要演练计时（本次 2 源 12 分钟 / 3 源 19 分钟 / 4 源 21 分钟）。

## 19. 下一步（目标 0.80）

1. OpinioNet-style head + RBT3 控制组三折：**已完成，未过门禁（0.70876 vs 0.71403）**，强 encoder 改回 pointer head。
2. 强 encoder（WWM/RoBERTa）pointer head 三折筛选：**已完成（0.7459 交叉拟合）**。
3. WWM 5 折：**已完成，交叉拟合 0.754295**；线上提交 `candidate-84e0474e404e-ac9a430bf2db` 得 **LB 0.7583**（超过 0.75 目标）。
4. MacBERT 5 折（进行中）→ 与 WWM 5 折做同档次融合 → 再次提交。
5. 为冲刺 0.80 的排期（按预期收益/小时排序）：
   - **large backbone**（`hfl/chinese-roberta-wwm-ext-large`，24 层）：3 折筛选 → 5 折 → 与 base 融合。不同规模带来的多样性通常 +0.01~0.02。
   - **TAPT**：仅用官方训练文本做 MLM 继续训练，再下游微调（计划门槛 +0.004）。
   - **稳定性技巧**：EMA → FGM → SWA，每个 3 折隔离验证 ≥+0.002 才进入 finalist。
   - **多 seed / 多 encoder 同档次集成**：每次 +0.002~0.005。
   - **已证伪、不再投入**：候选重排序（含标签先验）、并集/交集策略、弱模型等权融合、top-1 兜底、LLM（本地 4-bit QLoRA）融合。

