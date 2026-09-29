# Relation Grid 结构性问题分析（结论：暂不启动第二轮）

日期：2026-09-28
状态：只做诊断与方案设计，未启动任何训练
数据来源：`artifacts/experiments/grid_wwm_3fold_20260928/oof_candidates.json`（V1 解码）、`artifacts/reports/oof_wide_wwm.json`（宽解码重跑）、`artifacts/reports/error_audit_v1.json`

---

## 1. 问题确认：关系头没有使用 span 表示

用户的判断经代码核对成立，且比预想更彻底。

训练路径（`grid_trainer.py:189`）：

```python
outputs = {"relation": self.relation_head.grid_logits(hidden), ...}
```

`grid_logits` 的实际构造（`grid_model.py:514-543`）：

```python
aspect  = [implicit_vec, hidden[:, 1:]]     # 每个位置的“span 表示”就是该位置的单个 token 向量
opinion = [implicit_vec, hidden[:, 1:]]
logits  = bilinear(hidden[:, a_start], hidden[:, o_start]) + ...
```

因此关系分数是 `f(hidden[a_start], hidden[o_start])`，即**两个起点 token 的交互**。span 的 end token 从不参与，span 内部的 pooling 从不参与。

对照：同一个 head 里存在真正做 span pooling 的 `pair_logits`（用 cumulative-sum 对 `[start, end]` 求平均，`grid_model.py:478-492`），但它在 grid 训练和解码路径中**从未被调用**；`pair_relation_loss` 有定义、被导出，同样从未进入训练。即“正确的实现写好了，但没有接线”。

直接后果：

- `电池` 与 `电池续航` 在关系头看来完全等价（都取起点 token）。
- span 边界错误不产生关系头惩罚，边界信号只能靠 `boundary_loss` 单独承担。
- 解码时 `constrained_topk_pair_decode` 用 `relation_probabilities[a_anchor, o_anchor]` 取分，与训练时的单 token 语义一致，因此训练与解码没有错配 —— 但两者共享同一个缺陷。

## 2. 问题确认：没有“无关系”建模

39 类全部是多标签 sigmoid，没有任何 background / no-relation 单元。某对 (a, o) 只要起点分数偏高，就必然被发射，且发射时**无条件取 top-2 类**（`relation_top_k=2`），下限只有 `min_relation_score=0.05`。

量化（V1 OOF，53,259 个候选）：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| pair 与 class 都正确 | 5,828 | 10.9% |
| pair 正确但 class 错 | 2,489 | 4.7% |
| **pair 完全错误** | **44,942** | **84.4%** |

即每 10 个候选里只有 1 个是正确的四元组，接近 9 个是错误 pair。每个 review 中位发射 11 个（V1）/ 20 个（宽解码）候选，而 gold 平均只有 2.05 个。

## 3. 问题确认：负样本监督几乎不存在

`sparse_grid_bce_loss` 的负采样预算：

```python
budget = ceil(num_positive * negative_ratio)   # negative_ratio=3.0
budget = min(budget, max_negative=4096, num_valid_negatives)
```

按实际数据：每个 review 约 2.05 个正例，L=128 时网格有 16,384 个 cell。batch=2 时正例约 4 个，负采样预算约 **12 个 cell**，而可用负 cell 约 32,000 个。也就是说 **99.96% 的负 cell 在整个训练中从未被惩罚过**。

这解释了为什么分数严重不可信：

pair 级精确率随分数阈值的变化（V1，按 max-score 取 cell）：

| 阈值 | 命中/发射 | 精确率 |
|---|---:|---:|
| ≥0.5 | 5450/12494 | 0.436 |
| ≥0.8 | 4634/8143 | 0.569 |
| ≥0.9 | 2241/3648 | 0.614 |
| ≥0.99 | 226/310 | 0.729 |

即使到 0.99 也只有 0.73。说明该分数**不是“是否构成关系”的概率**，只是一个未校准的排序信号。同时 31.6% 的 cell 最高分 ≥0.5 —— 模型对近三分之一的位置对都给出了高置信输出。

## 4. 关键量化：上限分解

用已有 OOF 候选做 oracle 分解（不训练、不调参）：

| 分解 | V1 | 宽解码 |
|---|---:|---:|
| 当前校准后 strict F1 | 0.6252 | 0.6059 |
| 每 cell 只保留 top-1 类 | 0.6261 | 0.6066 |
| **UB-A：完美 pair 过滤 + 保留模型自己的类预测** | **0.8860** | **0.9012** |
| UB-C：完美答案 | 1.0000 | 1.0000 |

UB-A 的含义：如果能把错误 pair 全部滤掉、但类别仍用模型自己预测，F1 就能到 **0.886 / 0.901**，已经越过 0.85 冲刺线。

UB-A 中 P=0.9264、R=0.8489 分别对应：

- **pair 正确时，类别正确率约 92.6%** —— 类别头工作良好。
- **85% 的 gold pair 至少被发射过一次** —— 召回不是主要矛盾。

结论：**瓶颈 100% 在“哪些 pair 是真关系”的判别上，不在类别分类、不在召回、不在阈值。**

## 5. 为什么“调阈值”救不了

固定阈值扫描（V1，top-1 per cell）：

| 阈值 | P | R | F1 |
|---:|---:|---:|---:|
| 0.5 | 0.4159 | 0.7835 | 0.5433 |
| 0.7 | 0.5058 | 0.7381 | 0.6003 |
| 0.8 | 0.5516 | 0.6773 | 0.6081 |
| 0.9 | 0.5984 | 0.3292 | 0.4247 |

最优单阈值只到 0.6081，低于已校准的 0.6252。**不存在一个能同时保住 P 和 R 的阈值** —— 这正是“未建模 no-relation”的典型症状：分数分布本身没有把正负分开。

同理，宽解码把 oracle 从 0.875 抬到 0.920，但 strict F1 从 0.6252 降到 0.6059，因为多出来的候选全是未受监督的负 cell。

## 6. 优化框架（按证据排序，尚未执行）

### P0 —— 让关系头看到完整 span

- 训练与解码统一改走 span-aware 路径：对候选 pair 用 `pair_logits`（已实现、已验证的 pooled 版本），或在网格上加 start/end 双线性项 `S(a_s,a_e,o_s,o_e)`。
- 代价：训练时需要对“候选 pair”而非“全网格”打分，因此需要先做候选采样再打分（这也顺带解决了 P1 的负采样问题）。
- 预期收益来源：UB-A 显示 pair 判对后 F1 可达 0.886，而当前边界信息完全没进入关系头。

### P0 —— 显式建模 no-relation

- 方案 A：额外一个 background logit，作为 pair 有效性的门控（`is_relation`），解码时先过门控再取 top-k 类。
- 方案 B：把负采样比例从 3.0 提到 20–50，并按难度（同句错误 pair、共享 opinion、近邻边界）分层采样，而不是全局 topk。
- 方案 C：对短句直接全网格 BCE（L≤64 时 cell 数可控），不采样。
- 建议先做 A+B 组合，因为它同时给出可解释的 pair 置信度。

### P1 —— 解码侧收紧

- 类别改为**按类别阈值**而不是每 cell 无条件 top-2；top-2 在 V1 中有 34.5% 的 cell 触发，等于凭空翻倍 FP。
- `min_relation_score` 从 0.05 提升到由 fold-safe 校准得到的 pair 门控阈值。
- 保留 NMS，但 NMS 只解决重复，不解决“错误 pair”，因此优先级低于前两项。

### P2 —— 训练信号

- 负例 `pos_weight=3.0` 在负样本覆盖不足时意义有限；改为按采样后的真实正负比例计算。
- boundary_loss 权重 0.25 目前是唯一约束 span 边界的信号；span-aware head 上线后应重新平衡。
- implicit 权重仅允许 `{1.0, 1.5, 2.0}`，仍是三值筛选，不做大搜索。

### 明确不做

- 不继续扩大解码宽度当作解决方案（已证伪：oracle ↑、F1 ↓）。
- 不做全局阈值暴搜（已证伪：最优单阈值低于已校准值）。
- 不引入新 backbone、不做 seed sweep、不重训 RBT3（锁定计划禁止项）。

## 7. 验收门槛（沿用锁定计划）

- V2 相对 V1 在固定 3-fold 上 strict F1 提升 ≥0.015，或 F1 ≥0.005 且 oracle / implicit FN 明显改善。
- pair 级精确率（在分数 ≥0.8）应从当前 0.569 提升到 0.75 以上，作为“no-relation 建模成功”的直接指标。
- 若 V2 未达门槛，只允许一次基于本审计的 V2.1 修补，之后按计划进入 Task 4。

## 8. 当前状态

- 未启动任何训练；本文档仅为分析与方案。
- V1 三模型已归档：RBT3 0.5504、MacBERT 0.6154、WWM 0.6252（均为 screening_v1，不进入 5-fold）。
- 已有可复用产物：`error_audit_v1.json`（六类错误桶）、`oof_wide_wwm.json`（宽解码 OOF）、`oracle_ceiling_sweep.py`（可复算 oracle）。

## 9. 与并行改造的分工（2026-09-28 15:30）

另一个执行体（GPT）正在同步改造 `grid_model.py` / `grid_trainer.py` / `pipeline.py`。本节记录只读核查结果，供后续对比，不构成对其实现的评价，也不在其编辑期间做任何修改。

已落地（只读确认）：

- `CompactPairRelationHead.pair_logits` 已用于训练与解码，span 表示改为真实 pooled span（`_pair_contexts`），span end 已进入关系打分。
- 新增 `pair_validity_logits`（`pair_validity` MLP + 四状态 bias）作为 no-relation 门控。
- 新增 `implicit_aspect_presence` / `implicit_opinion_presence` 两个二分类头，并进入 `constrained_topk_span_decode`。
- 训练已接 `hard_negative_pairs`（`hard_negative_ratio` 默认 3），并把 `exact_pair_target_matrix` 用于按完整 span 精确匹配的类别监督。
- 解码改为 `pair_logits + pair_validity` 组合打分，`relation_top_k` 默认值由 2 降为 1。
- 损失新增 `pair_relation_loss_weight=0.5`、`pair_validity_loss_weight=0.5`、`implicit_presence_loss_weight=0.25`。
- `pipeline.select_threshold` 已扩展为四状态阈值（explicit / implicit-A / implicit-O / dual-implicit）。

尚未收敛（只读观察，进行中）：

- `tests/test_pipeline_crossfit.py::test_select_threshold_learns_separate_implicit_opinion_state` 当前失败（`KeyError: 'state_thresholds'`）；`pipeline.py` 已在返回体里提供该字段，说明实现与测试正处于编辑中间态。
- 旧的 start-token-only `grid_logits` 仍在损失中占主权重（`grid_loss` 系数 1.0，未加权下调），其影响需要实测确认。
- `min_relation_score` 仍为固定 0.05（`grid_model.py:904`），解码门控尚未改为校准阈值。
- `hard_negative_ratio` 仍为 3，与第 3 节“负样本覆盖 0.04%”的结论相比提升有限。

本文档第 6 节的 P0/P1/P2 与收益预测模型（`scripts/pair_factor_projection.py`）仍可作为验收对照：改造后应直接测 pair 精确率是否从 0.5887 上升，并按 `F1 = 2·c·r·G / (k·r·G/p + G)` 预测落点。
