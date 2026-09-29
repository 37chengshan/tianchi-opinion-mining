# Compact implicit-aware relation grid

这份设计对应新增的 [grid_model.py](/Users/cc/code/tianchi-opinion-mining/src/opinion_mining/grid_model.py)。本次只新增这个模块和本文件，没有修改主线代码，也没有加载新的预训练模型或运行大规模 MPS 训练。

## 1. 与当前 neural.py 的对接事实

当前 `neural.py` 已经给出了三个需要保持一致的约定：

| 现有接口 | 形状或格式 | 新模块的处理 |
|---|---|---|
| `_Feature.input_ids`, `attention_mask` | `[L]` | 不重复构造，继续由现有 tokenizer/dataset 提供 |
| `_Feature.offsets` | `L` 个 `(char_start, char_end)` | 只把非空且完整包含在官方字符范围内的 token 视为实体 token |
| `_Feature.pairs` | `(a_left, a_right, o_left, o_right, class_id)` | 转成稀疏网格正例；端点为当前的 inclusive token index |
| `PointerQuadrupleModel` 的 `hidden` | `[B,L,H]` | 传给 `CompactPairRelationHead` |
| `PointerQuadrupleModel` 的四个 pointer 输出 | `[B,L,L]` | 可在主线中按 position 取一行，再喂给独立 span decoder；模块本身不改变 pointer 语义 |
| `PointerQuadrupleModel.relation` | `[B,L,39]` | 保留现有头的兼容性；新 grid head 另外输出 `[B,L,L,39]`，避免把 token 级输出误当成 pair grid |
| `Candidate` | `Candidate(quadruple=Quadruple, score, sources)` | `grid_candidates_to_candidates` 直接产生该格式 |

`_Feature` 对隐式项的当前表示是：隐式 A 为 `(-1, -1)`，隐式 O 为 `(0, 0)`。后者是 index zero 的 CLS sentinel，不是一个真实的表面 token。新模块内部统一用 `-1,-1` 标记隐式侧，网格坐标仍使用 index zero，因此四种状态分别是：

1. 显式 A + 显式 O：`(a_start, o_start)`；
2. 隐式 A + 显式 O：`(0, o_start)`；
3. 显式 A + 隐式 O：`(a_start, 0)`；
4. 隐式 A + 隐式 O：`(0, 0)`。

官方 `Train_labels.csv` 的 `A_start/A_end/O_start/O_end` 是字符范围，结尾为 exclusive。新模块的 `official_char_to_token_span` 与现有 `_span_token_range` 采用相同的完整包含规则，并且不会在偏移缺失时用字符串搜索猜测重复词的出现位置。这一点对同一句评论中重复的观点词很重要。

本地官方数据的静态检查结果是 6633 个标签，其中隐式 A 4735 个、显式 A 1898 个；显式 O 6461 个、隐式 O 172 个；当前数据没有双隐式样本。双隐式仍然是模块支持的合法状态，并通过 `(0,0)` 网格单元和独立 implicit bias 保留给后续训练或合成 smoke。

## 2. 稀疏 target

主入口：

```python
from opinion_mining.grid_model import build_sparse_relation_targets

targets = build_sparse_relation_targets(features, max_length=128)
```

其中 `features` 可以是当前 `_ReviewDataset.features`；也可以传 `ReviewExample` 加同批次的 tokenizer `offsets`，或者传带 `class_id` 的 `PairSpan` 列表。结果 `SparseRelationTargets` 包含：

```text
shape   = (B, L, L, 39)
indices = [4, N]   # batch, aspect_anchor, opinion_anchor, class_id
values  = [N]      # 当前为 1
spans   = [N, 6]   # batch, a_start, a_end, o_start, o_end, class_id
```

`N` 是去重后的正例数。网格只监督 span 的 start anchor；完整 span 端点留在 `spans` 中，供 boundary head 与约束解码使用。这样一个共享观点 span 可以对多个 relation class 产生多个正例，不会像单标签 CE 那样覆盖掉其它类别。

`SparseRelationTargets.to_dense()` 和 `as_coo()` 是显式 opt-in 的调试工具。默认的 `sparse_grid_bce_loss` 不会把 target 物化成 `[B,L,L,39]`，而是读取正例索引，再从有效网格中挑选当前 logit 最高的负例。

如果从 `ReviewExample` 构造 target，必须传每条样本对应的 `offsets`：

```python
targets = build_sparse_relation_targets(
    examples,
    offsets=batch_offsets,
    max_length=128,
)
```

缺少显式项官方 offset 时会跳过这条无法可靠定位的 pair，而不是回退到 `text.find`。使用当前 `_Feature` 时，offset 映射已经在 `neural.py` 的 dataset 阶段完成。

## 3. 轻量 pair relation head

主类是 `CompactPairRelationHead(hidden_size, rank=32)`。它有两条调用路径：

```python
head = CompactPairRelationHead(hidden_size=hidden.size(-1), rank=32)

# 稀疏候选 pair：输出 [N, 39]
pair_logits = head.pair_logits(hidden, pair_spans)

# 全部 anchor cell：输出 [B, L, L, 39]
grid_logits = head.grid_logits(hidden)
```

`pair_spans` 使用 `[N,5] = (batch, a_start, a_end, o_start, o_end)`；隐式侧的两个端点都写 `-1`。也接受 `[B,N,4]` 或 `[N,4]` 的 tensor。显式 span 使用 inclusive endpoint，与 `_Feature.pairs` 一致。

head 的参数是：

- aspect/context 和 opinion/context 各一个 `2H -> rank` 投影；
- 每个侧面各一个 `2H -> 39` additive 投影；
- `39 x rank` 的 class factor；
- 显式/隐式四状态的 bias；
- 一个 implicit aspect vector 和一个 implicit opinion vector。

pair logit 是 class-conditioned diagonal low-rank bilinear interaction 加 additive terms，输出使用 sigmoid 解释为 39 类 multi-label。双隐式不会被当作一个空的显式 pair，而是使用两个 implicit vector 和 state=3 的 bias。

对于训练 pair 候选，可先用 `pair_target_matrix(pair_spans, targets)` 得到 `[N,39]` multi-hot target，再调用 `pair_relation_loss(pair_logits, target_matrix)`。对完整 grid 则调用 `sparse_grid_bce_loss(grid_logits, targets)`。

## 4. 解码和 NMS

三个独立 API 分别负责 span、pair 和 NMS：

```python
aspect_spans = constrained_topk_span_decode(
    aspect_start_logits, aspect_end_logits,
    valid_mask=attention_mask.bool(),
    top_k=8,
    max_span_length=10,
)
opinion_spans = constrained_topk_span_decode(
    opinion_start_logits, opinion_end_logits,
    valid_mask=attention_mask.bool(),
    top_k=8,
    max_span_length=10,
)
raw = constrained_topk_pair_decode(
    aspect_spans, opinion_spans, relation_logits,
    relation_top_k=2,
    top_k=32,
)
kept = nms_grid_candidates(raw, iou_threshold=0.8)
```

约束包括：

- surface span 只能使用正 token index；index zero 只表示 implicit；
- end 不得早于 start，且长度不超过 `max_span_length`；
- 显式 A/O 重叠时默认丢弃；
- relation 使用独立 sigmoid 和 top-k，允许同一 pair 同时拥有多个 39 类标签；
- NMS 默认要求 category/polarity class 相同，且 A、O 两侧 token IoU 都达到阈值才抑制；隐式侧只有和另一个隐式侧才视为 IoU=1。

如果需要主线的字符串候选：

```python
candidates = grid_candidates_to_candidates(
    kept,
    text=review.text,
    offsets=feature.offsets,
    source="relation_grid",
)
```

它返回 `list[Candidate]`，并以 `Quadruple(aspect, opinion, category, polarity)` 为 key 合并重复预测。`decode_grid_candidates` 是把上述四步串起来的单样本 convenience API；不传 `text` 时返回 `GridCandidate`，传 `text` 和 `offsets` 时返回项目 `Candidate`。

## 5. 内存估算

以下按 FP32、`L=128`、`C=39` 估算；MiB 使用 `2**20` 字节：

| 对象 | 公式 | B=4 |
|---|---:|---:|
| dense grid target | `B*L*L*39*4` | 10,223,616 bytes，约 9.75 MiB |
| sparse indices | `4*N*8` | 每个正例 32 bytes |
| sparse values | `N*4` | 每个正例 4 bytes |
| sparse span table | `6*N*8` | 每个正例 48 bytes |
| sparse target 合计 | `84*N` | 若每条样本 8 个正例、B=4，仅约 21 KiB |
| pair logits | `N*39*4` | N=256 时约 39 KiB |

head 在 `H=768, rank=32` 时约 221k 个参数，FP32 约 0.84 MiB，不包含 encoder。`grid_logits` 本身仍会产生 dense logits；若显存紧张，应先从 boundary top-k 形成 pair spans，再走 `[N,39]` 的 `pair_logits` 路径。稀疏 target 的价值是避免额外复制一个同大小的 dense label tensor。

## 6. 固定 3-fold 验证设计

本模块没有在本次任务中运行 3-fold，也没有生成或声称任何 F1。后续主线验证固定采用以下协议，便于不同 head 和阈值之间可比：

1. 以 review id 为样本单位，`KFold(n_splits=3, shuffle=True, random_state=42)`；同一 review 的全部 `LabelSpan` 必须留在同一 fold。
2. 每个 fold 重新建立 encoder 的训练状态、`CompactPairRelationHead`、优化器和 target builder；验证 fold 的 label 不能参与字典、阈值或参数拟合。
3. 固定 `max_length=128`、`max_span_length=10`、`span_top_k=8`、`relation_top_k=2`、`pair_top_k=32`、NMS IoU=0.8。若资源需要变化，应在报告中写明并保持所有 fold 相同。
4. 训练只用 train split，先生成 sparse targets，再用 `sparse_grid_bce_loss` 或 pair loss；验证只做 CPU decode 或小批量推理。阈值只能在该 fold 的 validation predictions 上搜索，然后冻结后汇总三折 OOF candidates。
5. 最终评估调用项目的 strict quadruple metric，分别记录显式 A、隐式 A、显式 O、隐式 O，以及 overall precision/recall/F1。未实际执行的指标保留为“未运行”，不能用 smoke loss、候选数量或静态检查替代 F1。

建议固定记录 `seed=42`、Python/PyTorch 版本、实际有效 offset 比例、每类正例数、每折候选数和阈值。这样 relation grid 的改进不会被不同的 split 或 threshold search 混淆。

## 7. 失败信号和排查顺序

出现以下任一信号时，应先停在当前 fold 排查，不把结果当作可比较的模型结果：

- **offset mapping drop**：官方显式 label 有合法字符范围却无法映射到有效 token；先检查 tokenizer、截断和 exclusive end，不能静默改成首次字符串命中。
- **target empty 或 shape mismatch**：某批次 `num_positives=0`、`[B,L,L,39]` 与 logits 不一致，或 class id 不在 0..38；先检查 `_Feature.pairs` 和 `max_length`。
- **non-finite loss/gradient**：`loss`、logit 或梯度出现 NaN/Inf；检查学习率、pos weight、空负例处理和混合精度设置。
- **implicit collapse**：implicit A/O 的 target 能生成，但 decoder 长期只输出显式 pair；单独统计四种 state 的正例、relation probability 和候选保留数。
- **dual implicit regression**：合成 `(0,0)` relation 无法通过 head、loss、decode 和 NMS；必须保留这个 CPU smoke，即使官方当前统计为 0。
- **decode invalid**：出现越界 token、空字符串、A/O 重叠或同一 quadruple 大量重复；检查 span constraint、offsets 和 NMS 顺序。
- **candidate explosion**：top-k 后每条评论候选数异常增长，或 NMS 抑制率接近 100%；降低 pair top-k 前先查看 relation sigmoid 和 implicit bias 是否塌缩。
- **泄漏**：fold 外的 labels、阈值、retrieval dictionary 或 tokenizer-derived label statistics 被复用；每 fold 要有独立 target/训练/threshold 生命周期。

本次 CPU/dummy smoke 只验证模块导入、四种 implicit 状态、稀疏 target、两种 loss 的反向传播、约束 decode、NMS 和 `Candidate` 转换；它不代表训练质量，也不产生 F1。
