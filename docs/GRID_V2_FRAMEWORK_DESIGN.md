# Grid V2 优化框架设计（可执行规格，未实施）

日期：2026-09-28
前置：`docs/GRID_V2_STRUCTURAL_ANALYSIS.md`
状态：设计规格，尚未修改训练与解码代码，未启动训练

---

## 0. 收益预测模型（精确复现，已验证）

把 strict F1 按四个可测量因子精确分解：

```
F1 = 2 · c · r · G / ( k · r · G / p + G )
  c = pair 正确时类别正确的比例
  r = pair 级召回率
  p = pair 级精确率
  k = 每个发射 pair 的平均类别数
  G = gold 数量
```

代入 V1 实测 `c=0.9704, p=0.5887, r=0.7177, k=1.0073, G=6632` 得到 **0.6252**，与实测校准 F1 **完全一致**（复算脚本 `scripts/pair_factor_projection.py`）。

投影（c 与 k 保持实测值）：

| pair 精确率 | pair 召回 0.80 | 0.85 | 0.90 |
|---:|---:|---:|---:|
| 0.70 | 0.7217 | 0.7420 | 0.7610 |
| **0.75** | 0.7484 | 0.7703 | **0.7908** |
| **0.80** | 0.7735 | 0.7968 | **0.8188** |
| 0.85 | 0.7970 | 0.8218 | 0.8452 |
| **0.90** | 0.8192 | 0.8454 | **0.8702** |
| 0.95 | 0.8400 | 0.8677 | 0.8938 |

结论：**pair 精确率是决定性变量**。

- 最低目标 0.75 需要 pair 精确率约 **0.75** 且召回维持 0.80 以上。
- 主目标 0.80 需要 pair 精确率 **0.78–0.80** 且召回 ≥0.85。
- 冲刺 0.85 需要 pair 精确率 **≥0.85** 且召回 ≥0.85，或精确率 0.90 + 召回 0.85。

特别提醒：召回从 0.72 提到 0.90 的收益远小于精确率从 0.59 提到 0.80。因此所有 P0 工作都应围绕“把 pair 判准”设计。

前一版本本文档曾用 oracle 召回（0.8489/0.8786）做投影并给出 0.886/0.901 的上限，那组数字是**理论上限而非预期**，已修正为上述可复现投影。

## 1. 目标架构

当前（缺陷）：

```
relation_logits[b, i, j, :] = f(hidden[b,i], hidden[b,j])      # 只有起点，且无 no-relation
decode: score = relation_logits[b, a_start, o_start, class]     # 同样只看起点
```

目标：

```
候选对 (a_span, o_span)  ->  pooled_span(a_span)  ->  关系打分  ->  类别分数
                         ->  pooled_span(o_span)
                         ->  pair_gate 是否构成关系（含 no-relation）
```

关键点：`pooled_span` 用**真实 span 边界**（start 与 end 都参与），而不是单个起点 token。

## 2. 三段改造

### 改造 A：span-aware 关系打分（P0）

复用已存在但未接线的实现：`grid_model.py:478-492` 的 cumulative-sum pooling 与 `pair_logits`（第 507 行）。

接口约定（保持现有数据类不变）：

```python
head.pair_logits(
    hidden: Tensor,                       # [B, L, H]
    pair_spans: Tensor,                   # [N, 5] = (batch, a0, a1, o0, o1)，隐式边用 -1
) -> Tensor                               # [N, num_classes] 多标签 logits
```

已确认的性质：

- padding 安全：cumulative-sum 差分后按宽度取平均，`clamp(min=0)` 保护隐式边。
- 隐式边用可学习向量 `implicit_aspect` / `implicit_opinion` 替代 pooled 表示。
- 四种状态（显式/显式、隐式 A、隐式 O、双隐式）有独立 bias，已实现（`implicit_bias[state]`）。
- 显式边仍拼接 CLS：`aspect_context = [pooled, cls]`，保持与现有训练一致的上下文。

训练路径必须从“全网格 dense logits”改为“对候选对打分”，因为 pooled 表示无法预先算成 `[L, L, H]` 张量（end 维度会使显存爆炸）。这天然引向改造 B。

### 改造 B：候选对采样 + no-relation 门控（P0）

每步为每个样本构造一个候选对集合，而不是对 16,384 个 cell 做采样：

```
候选集 = 全部正例
       + hard negatives（已有实现：hard_negative_pairs，grid_model.py:566）
       + 少量随机负例（控制比例）
```

正负比例目标 **1:20 到 1:50**（当前有效比例约 1:3，且负例覆盖仅 0.04%）。

门控设计：在 `pair_logits` 之外加一个二分类头：

```python
gate_logits = head.pair_gate(hidden, pair_spans)      # [N]，二分类：是否构成关系
class_logits = head.pair_logits(hidden, pair_spans)   # [N, 39]，关系成立时才是哪些类
```

- gate 用 BCE，负例来自 hard negatives + 随机负例。
- class loss 只在“正例 + 被判定为难负例”上计算，避免用 40 倍负例稀释类别信号。
- 推理时 `pair_score = sigmoid(gate) `，只有过门控的 pair 才进入类别 top-k。这就是“无关系独立建模”。

为什么优先 gate 而不是直接加 background 类：

- 39 类 sigmoid 的语义是“多个类可同时成立”，塞入第 40 个 background 类会与多标签语义冲突。
- 单独 gate 输出的 pair 置信度可直接复用为解码阈值，替代当前无效的 `min_relation_score=0.05`。
- gate 的概率天然可校准（Platt / isotonic，fold-safe），符合锁定计划对校准的要求。

### 改造 C：解码收紧（P1）

1. **按类别阈值替代无条件 top-2**：当前 34.5% 的 cell 会发射 2 个类，等于凭空翻倍 FP。改为“类概率 ≥ 该类阈值才发射”，阈值由 fold-safe 校准给出。
2. **pair 门控替代 0.05 下限**：只保留 `sigmoid(gate) ≥ pair_threshold` 的 pair。
3. 保留 top-k span beam 与 NMS：它们解决“同一实体重复”，不解决“错误 pair”，优先级低于 1、2。
4. `pair_threshold` 与 per-class 阈值必须 **leave-one-fold-out** 学习，绝不在同一 OOF 上调参自评（锁定计划硬约束）。

## 3. 训练接线点（精确定位）

| 位置 | 现状 | 改造 |
|---|---|---|
| `grid_trainer.py:189` | `"relation": self.relation_head.grid_logits(hidden)` | 改用候选对采样 + `pair_logits` + `pair_gate` |
| `grid_model.py:514` `grid_logits` | 被训练使用 | 降级为推理期可选的全网格扫描，或直接删除 |
| `grid_model.py:507` `pair_logits` | 从未被调用 | 成为训练主路径 |
| `grid_model.py:695` `pair_relation_loss` | 从未被调用 | 用于 class loss |
| `grid_model.py:566` `hard_negative_pairs` | 从未被调用 | 成为负例来源 |
| `grid_trainer.py:345` 解码 | 用 `relation_logits[a_anchor,o_anchor]` | 改用 pooled 打分 + gate |

注意：改造后 `_Feature.relation`（`[L, 39]` 按 token 的 target）不再对应新路径，需要改为按候选对构造 target。`_Feature.pairs` 保留（含官方 offset 派生的 a0/a1/o0/o1），是候选对 target 的直接来源。

## 4. 预计成本

- 显存：候选对方式把每样本的 relation 计算量从 `L×L=16384` cell 降到约 `N≈60–320` 个对，**显著下降**；这是净收益而非代价，允许提高 batch。
- 速度：pooling 用 cumulative-sum 差值，无循环；`pair_logits` 是两次 `Linear` + `einsum`，与现有网格相比更省。
- 风险：候选采样必须保证“正例永不遗漏”，否则会引入新的召回损失。需要单测断言 `候选集 ⊇ 正例集`。

## 5. 验收门槛

主门槛（沿用锁定计划）：

- V2 相对 V1 在固定 3-fold 上 strict F1 提升 **≥0.015**；或 F1 提升 ≥0.005 且 oracle / implicit FN 明显改善。

新增过程指标（用于在训练早期判断方向）：

- **pair 级精确率从 0.5887 提升到 ≥0.75**。这是 no-relation 建模是否成功的直接证据。
- pair 召回不低于 0.80（防止用门控换精确率时把召回压垮；当前 0.7177）。
- 类别准确率 c 不跌破 0.95（当前 0.9704，这是已达标的强项，不应被新头拖累）。
- 预测数量从每 review 中位 11 降到接近 gold 的 2–4 量级。

预测：pair 精确率 0.80 + 召回 0.85 对应 F1 ≈ **0.797**；精确率 0.90 + 召回 0.90 对应 ≈ **0.870**。若实测 F1 低于投影 0.03 以上，说明类别头或召回出现回归，应停下排查而不是继续加结构。

## 6. 实施顺序（供后续执行）

1. 写失败单测：候选集包含全部正例；pooled 打分对 span 边界敏感（改变 end 必须改变分数）；gate 能把已知错误 pair 压到低分。
2. 实现 `pair_gate` 与候选采样，接 `pair_logits` / `pair_relation_loss`。
3. 改解码：gate 阈值 + per-class 阈值替代 top-2 与 0.05。
4. 跑固定 3-fold 的 WWM 对照（仅一次，不用新 backbone）。
5. 计算 pair 精确率 / 召回 / oracle，与收益模型对齐后决定是否进入 Task 4。

按锁定计划，若 V2 未达门槛，只允许一次基于本分析的 V2.1 修补，之后进入 Task 4（强 backbone 固定对比），不再继续结构探索。

## 7. 明确不做

- 不继续扩大解码宽度（已证伪：oracle ↑、F1 ↓）。
- 不做全局阈值暴搜（已证伪：最优单阈值 0.6081 < 校准 0.6252）。
- 不引入新 backbone、不做 seed sweep、不重训 RBT3。
- 不在同一 OOF 上调参并自评（必须 LOFO）。
