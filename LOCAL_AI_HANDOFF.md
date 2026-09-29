# 天池赛道六：评论观点挖掘 — 本地 AI 完整接手说明

> 目标：从当前资料状态继续，最终产出一个**可直接提交天池**的 `Result.csv`，并保留完整可复现代码、OOF 严格 F1、提交校验与迭代记录。

---

## 0. 最重要的原则

这个比赛的评价不是单独的实体抽取、类别分类或情感分类，而是对同一 `id` 下的完整四元组做严格匹配：

`(AspectTerm, OpinionTerm, Category, Polarity)`

只有四个字段全部正确，才算一个正确预测。

因此：

1. 所有模型选择必须最终看 **strict quadruple micro-F1**，不能只看 token F1 / span F1 / category accuracy。
2. AspectTerm / OpinionTerm 非 `_` 时，输出必须是原评论中的**精确原文子串**。
3. 训练集大量 `AspectTerm = _`，必须显式支持隐式属性。
4. 预测过多会直接降低 Precision；预测过少会降低 Recall，所以阈值选择非常重要。
5. 所有 CV / OOF 过程必须严格防止验证折标签泄漏进词典、检索库、类别映射或阈值统计。
6. 当前机器按 **M1 Pro 16GB、竞赛进程最多约 8GB** 设计；完整资源策略、自主夜间搜索和网页看板协议以 `docs/M1_PRO_8GB_OPTIMIZED_PLAN.md` 为最高优先级覆盖规则。
7. 本地 AI 必须先启动并打开 `dashboard/` 训练看板，确认 PRECHECK 页面正常后才开始训练；训练结束后同一看板进入 Final 模式并导出静态最终报告。

---

# 1. 项目位置

当前项目：

`/Users/cc/code/tianchi-opinion-mining`

已有资料：

```text
tianchi-opinion-mining/
├── README.md
├── LOCAL_AI_HANDOFF.md
├── docs/
│   ├── competition-spec.md
│   ├── data-profile.md
│   ├── baseline-notes.md
│   ├── raw/
│   │   ├── train-readme.md
│   │   └── test-readme.md
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-09-27-tianchi-opinion-mining-design.md
│       └── plans/
│           └── 2026-09-27-tianchi-opinion-mining.md
├── src/
│   └── opinion_mining/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       └── submission.py
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_metrics.py
    └── test_submission.py
```

注意：当前 CodexPro 本地环境曾出现 `pytest: command not found`，所以不要假设依赖已装好。应创建独立虚拟环境。

---

# 2. 官方数据

用户提供两份官方压缩包：

- `初赛训练数据 2019-08-01.zip`
- `初赛测试数据 2019-08-15.zip`

训练包结构：

```text
README.md
TRAIN/
├── Train_reviews.csv
└── Train_labels.csv
```

测试包结构：

```text
README.md
TEST/
├── Test_reviews.csv
└── Result(example).csv
```

已整理出的数据规模：

- 训练评论：3229
- 训练四元组：6633
- 测试评论：2237
- 每评论平均约 2.05 个四元组

Category 官方集合：

```text
包装
成分
尺寸
服务
功效
价格
气味
使用体验
物流
新鲜度
真伪
整体
其他
```

Polarity：

```text
正面
中性
负面
```

数据存在明显长尾：`整体`、`使用体验` 样本多；`尺寸`、`新鲜度`、`成分` 很少。情感也极不均衡，以正面为主。

---

# 3. 最终提交文件硬约束

文件名必须是：

`Result.csv`

格式：

```text
id,AspectTerm,OpinionTerm,Category,Polarity
```

但**没有表头**。

约束：

- UTF-8，无 BOM
- id 升序
- 测试集所有 id 必须出现，不能漏，不能多
- 一个 id 可多行
- 无预测的 id 必须写：

```text
id,_,_,_,_
```

- AspectTerm / OpinionTerm 只要不是 `_`，必须严格存在于该评论原文
- Category / Polarity 必须属于官方集合
- 同一个 id 不要重复相同四元组

已有 `src/opinion_mining/submission.py` 的目的就是将这些约束程序化校验；接手后先把它的测试跑通。

---

# 4. 已知官方/历史 baseline

赛事页面给出：

`https://github.com/eguilg/OpinioNet`

这是 2019 年相关赛事 Top3 方案，核心思想：

- one-stage 端到端实体关系抽取
- BERT / RoBERTa-WWM / ERNIE backbone
- 交叉验证
- 阈值优化
- 多模型集成

历史代码环境极老：Python 3.6、PyTorch 1.1、CUDA 9，不建议直接复刻依赖。

应复用的是：

1. 联合建模思想
2. OOF 阈值优化
3. 多折 / 多模型集成
4. 四元组整体优化

不要直接照搬旧训练环境。

---

# 5. 推荐总方案

> **M1 Pro 16GB / 竞赛进程约 8GB 内存预算覆盖规则：** 本文后续所有神经模型建议必须服从 `docs/M1_PRO_8GB_OPTIMIZED_PLAN.md`。不要按独显工作站思路同时加载多个 Transformer，也不要为 Aspect、Opinion、Pair、Category/Polarity 各训练一套独立大模型。主路线改为“单个紧凑共享 Encoder + 多任务 heads + 轻量 CPU 规则/检索/校准器”，所有 fold 串行训练和串行推理。

建议做成 3 层模型，最终以 OOF strict quadruple F1 决定组合方式。

## Layer A：高精度规则/词典模型

用途：

- 快速得到稳定 baseline
- 给神经模型做候选补充
- 对重复出现的高纯度短语非常有效

训练统计：

### 5.1 OpinionTerm → (Category, Polarity)

统计每个 OpinionTerm 映射到 `(Category, Polarity)` 的频次和纯度：

```text
purity = top_pair_count / opinion_total_count
```

若 OpinionTerm 在目标评论中出现，且映射纯度高，则生成候选。

特别要统计：

```text
OpinionTerm -> AspectTerm == _ 的概率
```

高概率时允许输出隐式 Aspect。

### 5.2 AspectTerm + OpinionTerm 联合记忆

统计：

```text
(AspectTerm, OpinionTerm) -> (Category, Polarity)
```

当两者均在目标评论中出现时，通常比单 Opinion 更高置信。

### 5.3 完整四元组重复记忆

若一个完整四元组在多个训练样本中重复出现，只要其显式字符串在目标评论中命中，可以赋高置信度。

### 5.4 字符 n-gram 检索

用 `TfidfVectorizer(analyzer="char", ngram_range=(2,5))` 建训练评论索引。

对每个目标评论找 top-k 近邻。

仅当：

- cosine similarity 达阈值
- 迁移标签中的 Aspect / Opinion 在目标文本中存在

才迁移。

隐式 Aspect `_` 可直接保留，但 Opinion 仍必须命中目标文本。

---

# 6. 主力模型：字符级 span 抽取 + 配对 + 分类

因为官方要求原文精确字符串，所以字符级建模比中文分词更自然。

推荐拆成两个阶段：

## Stage 1：Aspect / Opinion span 抽取

### 标签设计

分别做两套 BIO/BIES 标注：

```text
Aspect: B-A / I-A / O
Opinion: B-O / I-O / O
```

可以：

- 两个独立模型
- 一个共享 encoder + 两个 token classification head

优先第二种。

### Backbone 优先级

若 GPU 可用：

1. `hfl/chinese-roberta-wwm-ext`
2. `hfl/chinese-macbert-base`
3. 可用的 Qwen encoder / ModernBERT 中文模型（仅当 HuggingFace token 对齐稳定）

若只有 CPU / Apple Silicon：

先用：

- char-level CRF / BiLSTM-CRF
- 或字符 n-gram + CRF

目的不是最终最高分，而是快速完成完整闭环。

### 长文本

评论通常不长，但仍建议：

- max_length 256
- 超长评论做 sliding window
- 合并重复 span

### 精确恢复

使用 tokenizer `offset_mapping`，恢复原文字符边界。

禁止输出经过 normalize 后的文本；必须切原始 review 字符串。

---

# 7. Stage 2：Aspect–Opinion 配对

这是整个比赛最关键且最容易丢分的环节。

对于一个评论中抽到：

```text
A = {a1, a2, ...}
O = {o1, o2, ...}
```

需要判断哪些 `(a, o)` 是真实关系。

同时必须允许：

```text
(_, opinion)
```

## 推荐方案 A：pair classifier

为所有候选 pair 构造特征：

- [CLS] review [SEP]
- Aspect span start/end marker
- Opinion span start/end marker
- 字符距离
- 是否在同一分句
- 中间标点数量
- 相对顺序

输出：

```text
is_relation
```

正样本：训练标签真实 pair。

负样本：同评论内错误交叉配对。

## 隐式 Aspect

为每个 Opinion candidate 额外建立：

```text
(_, opinion)
```

作为一个特殊 pair。

让分类器判断是否为隐式 Aspect。

---

# 8. Category + Polarity 分类

对已经判断为真实关系的 pair 再做分类。

推荐合并成一个 joint label：

```text
13 Category × 3 Polarity = 39 类
```

但因为很多组合不存在，实际只保留训练出现的组合。

优点：

- 直接优化完整四元组一致性
- 减少 Category/Polarity 独立错误组合

输入建议：

```text
[CLS] review [SEP]
+ span markers
```

分类头输出 joint `(Category, Polarity)`。

备选：多任务两个 head。

必须用 OOF 比较：

- joint classification
- two-head classification

谁的最终 strict quadruple F1 高就用谁。

---

# 9. 更强方案：统一四元组抽取

如果有足够 GPU 和时间，可以直接做 one-stage / GlobalPointer / GPLinker 风格模型。

推荐结构：

1. Aspect span head
2. Opinion span head
3. Aspect–Opinion relation matrix
4. Category/Polarity relation label

这更接近 OpinioNet 的思想。

但当前只有约 3k 评论，复杂模型未必一定优于“span + pair + joint classifier”。

所以实施顺序应是：

```text
统计 baseline
→ span + pair + classifier
→ OOF 验证
→ 再决定是否上 unified model
```

不要一开始就把时间花在复杂端到端模型。

---

# 10. OOF 设计

必须 5 折，按 review id 切分。

推荐：

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

如果希望类别分布更稳，可自己按每评论包含的主 Category 做近似 stratification，但不要为了 stratify 造成标签泄漏或复杂化。

每一折：

```text
train_fold
  -> 训练 span model
  -> 构建词典/检索库
  -> 训练 pair classifier
  -> 训练 category/polarity classifier

valid_fold
  -> 预测候选 span
  -> pair
  -> category/polarity
  -> 形成完整四元组
```

汇总全部 OOF prediction 后计算：

```text
Precision
Recall
strict quadruple F1
```

模型和参数选择只能依据 OOF。

---

# 11. 阈值优化

每个最终候选四元组计算综合 score：

示例：

```text
score =
  span_aspect_conf *
  span_opinion_conf *
  relation_conf *
  class_conf
```

隐式 Aspect 时可去掉 aspect span conf。

然后在 OOF 上搜索：

```text
0.05 ~ 0.95
```

不要只搜一个全局阈值，可进一步搜索：

- 显式 Aspect threshold
- 隐式 Aspect threshold
- 高频类别 / 长尾类别 threshold

但优先只做：

```text
explicit_threshold
implicit_threshold
```

防止过拟合。

---

# 12. 统计 baseline 的候选融合

最终模型不要简单替代规则模型。

建议候选来源：

```text
neural
exact_pair_dictionary
opinion_dictionary
retrieval
```

对同一个 quadruple 合并 score。

可试：

```text
final_score = max(neural_score, rule_score)
```

或：

```text
neural_score + bonus_if_rule_confirmed
```

规则候选不要无条件加入；也要在 OOF 上调阈值。

---

# 13. 数据增强建议

历史 OpinioNet 方案提到数据增强曾造成下降，所以谨慎。

优先不要做随机同义改写，因为输出要求原文精确 span，很容易破坏边界。

可以安全尝试：

### 13.1 负样本增强

pair classifier 中增加 hard negatives：

- 同一评论中的错误 Aspect–Opinion 组合
- 相邻但错误的 pair

### 13.2 类别长尾重采样

对 `尺寸/新鲜度/成分/服务` 等少数类：

- WeightedRandomSampler
- class-weighted CE
- focal loss

优先 class-weighted CE，简单稳定。

### 13.3 对 Opinion 的边界扰动负样本

构造错误 span 边界作为训练负样本，有助于精确匹配。

---

# 14. 本地 evaluator 必须先写好

已有：

`src/opinion_mining/metrics.py`

要求：

对每个 id：

```python
gold_set = set(gold_quads[id])
pred_set = set(pred_quads[id])
S += len(gold_set & pred_set)
P += len(pred_set)
G += len(gold_set)
```

然后：

```text
Precision = S / P
Recall = S / G
F1 = 2PR/(P+R)
```

注意：

`(_, _, _, _)` 是提交占位，不是一个预测正例，不应进入训练/OOF 的 pred_set。

---

# 15. 必须实现的代码结构

建议最终结构：

```text
src/opinion_mining/
├── __init__.py
├── data.py
├── metrics.py
├── submission.py
├── baseline.py
├── spans.py
├── pairing.py
├── classifier.py
├── pipeline.py
└── utils.py

scripts/
├── prepare_data.py
├── run_baseline.py
├── train_spans.py
├── train_pairs.py
├── train_classifier.py
├── run_oof.py
├── predict_test.py
└── make_submission.py

tests/
├── test_data.py
├── test_metrics.py
├── test_submission.py
├── test_baseline.py
├── test_spans.py
├── test_pairing.py
└── test_pipeline.py

artifacts/
├── oof/
├── models/
├── reports/
└── submissions/
```

---

# 16. 环境建议

创建虚拟环境：

```bash
cd /Users/cc/code/tianchi-opinion-mining
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

第一阶段最小依赖：

```text
pandas
numpy
scikit-learn
pytest
joblib
```

神经模型：

```text
torch
transformers
datasets
accelerate
seqeval
```

可选：

```text
torchcrf
optuna
```

Apple Silicon：优先检测 `torch.backends.mps.is_available()`。

---

# 17. 推荐执行顺序

## Phase 1：把工程跑绿

1. 安装依赖
2. 跑现有 tests
3. 修复所有失败
4. 把官方 zip 解压到：

```text
data/raw/train/
data/raw/test/
```

5. 写 `prepare_data.py`
6. 验证官方数据统计与 docs/data-profile.md 一致

完成条件：

```text
pytest 全绿
train=3229
labels=6633
test=2237
```

---

## Phase 2：可靠统计 baseline

实现：

- opinion purity
- aspect-opinion mapping
- implicit aspect mapping
- TF-IDF retrieval
- candidate confidence

跑 5-fold OOF。

记录：

```text
OOF precision
OOF recall
OOF strict F1
最佳 threshold
```

不要凭训练集成绩选参数。

---

## Phase 3：span 模型

先只训练：

- Aspect span
- Opinion span

分别计算 exact span F1。

此处指标只作为诊断，不作为最终模型选择标准。

确保：

- offset mapping 精确
- `_` 不当成 span
- 多 span 正确恢复

---

## Phase 4：pair + joint class

使用 gold span 先训练 pair/classifier，确认上限。

然后分别评估：

```text
gold spans + predicted relation/class
predicted spans + predicted relation/class
```

这样能明确瓶颈在：

- span
- relation
- category
- polarity

哪个环节。

---

## Phase 5：完整 OOF

每折必须全部从 train fold 训练。

生成：

```text
artifacts/oof/fold_0.csv
...
artifacts/oof/fold_4.csv
artifacts/reports/oof_metrics.json
```

在完整四元组层面搜索阈值。

---

## Phase 6：集成

优先做：

```text
5-fold model ensemble
```

再考虑：

```text
MacBERT + RoBERTa-WWM ensemble
```

不要一开始就 3-5 个 backbone，数据太少且时间浪费大。

规则 baseline 作为 bonus source。

---

## Phase 7：全量训练 + Test

确定最终配置后：

方案 A：直接使用 5 个 fold 模型对测试预测并平均/投票。

推荐 A，因为：

- 不需要重新全量训练
- 多模型天然集成
- 与 OOF 分布一致

然后生成：

```text
artifacts/submissions/Result.csv
```

---

# 18. 最终 Result.csv 校验

必须程序验证：

1. 文件不是 BOM
2. 无表头
3. 每行恰好 5 列
4. id 可转 int
5. id 单调非降
6. id 集合 == Test_reviews.csv id 集合
7. 非 `_` Aspect 在 review 中
8. 非 `_` Opinion 在 review 中
9. Category 合法
10. Polarity 合法
11. 同 id 无重复 quadruple
12. 空预测 id 只有一行 `_,_,_,_`

再额外输出 SHA256：

```bash
shasum -a 256 Result.csv
```

把 hash 写进 `run_report.json`。

---

# 19. run_report.json 建议字段

```json
{
  "train_reviews": 3229,
  "train_quadruples": 6633,
  "test_reviews": 2237,
  "cv_folds": 5,
  "seed": 42,
  "oof": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "thresholds": {
    "explicit": 0.0,
    "implicit": 0.0
  },
  "submission": {
    "rows": 0,
    "predicted_quadruples": 0,
    "empty_ids": 0,
    "sha256": "..."
  }
}
```

---

# 20. 实验记录与打榜协议

**分数目标：** 0.75 是必须认真争取的最低打榜线，0.80+ 是主目标，0.85 是冲刺目标。达到 0.75 之前，不允许因为已经生成合法 CSV 就结束优化；若 12/24 小时预算耗尽仍未达到 0.75，必须输出结构化错误分解和下一轮最高价值实验，而不是把“可提交”当作“完成”。

每次实验记录到：

`artifacts/reports/experiments.csv`

至少：

```text
experiment_id
model
features
folds
threshold
precision
recall
strict_f1
notes
```

不要只看 leaderboard 盲调。

当前平台页面显示剩余提交次数为 **999**。不要把它理解成应该暴力提交 999 个近似结果，而是建立 `artifacts/submissions/manifest.csv`，每一个准备上传的候选都记录：

```text
submission_id
experiment_id
config_hash
local_oof_f1
precision
recall
threshold_explicit
threshold_implicit
blend
csv_path
csv_bytes
csv_sha256
leaderboard_score
submitted_at
notes
```

优先使用高信息量分批提交：anchor → calibration → model challenger → ensemble → error-correction。若平台仍存在单日或频率限制，以平台实时规则为准，不使用旧页面文字推断当前限制。

最终上传物必须是 CSV，且 **小于 100,000,000 bytes**；任何候选在进入上传队列前都必须通过完整 submission validator、文件大小检查和 SHA256 记录。

---

# 21. 需要重新验证的内容

聊天过程中曾出现过一些“字符 span 模型 OOF 很高”的临时描述，但这些数字在当前工作区里**没有形成可审计的测试输出或报告文件**。

因此本地 AI 必须：

- 不把任何聊天中的临时数值当事实
- 所有指标重新跑
- 把输出写到 `artifacts/reports/`
- 只有真实日志 / 文件可作为后续决策依据

这是非常重要的。

---

# 22. 最低可交付版本

如果 GPU / 模型下载遇到问题，仍然必须先完成一个可提交版本：

```text
统计词典 + TF-IDF retrieval + OOF threshold
```

这个版本必须：

- 代码完整
- OOF 指标真实
- `Result.csv` 格式通过 validator

然后再继续神经模型。

不要因为想追高分而长期没有可提交文件。

---

# 23. 推荐的最高优先级任务

接手后依次执行：

1. 建 venv + 安装 pytest/pandas/numpy/sklearn
2. 跑现有 tests
3. 修正 `data.py / metrics.py / submission.py` 直到全绿
4. 放入官方 zip 并解压
5. 写真实数据 smoke test
6. 实现 baseline.py
7. 5-fold OOF + threshold
8. 先产出第一个合法 `Result.csv`
9. 再实现 span model
10. relation classifier
11. joint Category/Polarity classifier
12. 完整 OOF
13. ensemble
14. 最终 `Result.csv`
15. validator + SHA256 + run_report

---

# 24. 完成定义（Definition of Done）

只有满足以下全部条件才算“完成比赛提交工程”：

- `pytest` 全部通过
- 数据统计与官方数据一致
- 存在真实 5-fold OOF strict quadruple F1
- 所有模型/阈值无验证集泄漏
- 最终 `Result.csv` 已生成
- Result.csv 通过程序 validator
- 2237 个 test id 全覆盖
- UTF-8 无 BOM，无表头
- 保存 `run_report.json`
- 保存最终配置/随机种子
- README 中有一条命令可以复现最终预测

隐藏 leaderboard 分数只有实际提交天池后才能知道，不能本地宣称“完美成绩”。

---

# 25. 给本地 AI 的直接任务指令

你现在是这个竞赛工程的实现者。请不要停留在建议或伪代码层面，从当前 `/Users/cc/code/tianchi-opinion-mining` 继续实际实现。严格遵守 TDD 和 OOF 防泄漏原则。先保证最低可提交版本，再逐步提升模型。每完成一个阶段都运行真实测试和真实 OOF，并把结果写入 `artifacts/reports/`。最终目标是生成并验证 `artifacts/submissions/Result.csv`，同时给出本地 OOF strict quadruple F1、阈值、提交统计和 SHA256。若神经模型训练环境不可用，不得卡住整个项目，应先交付统计/检索 baseline 的合法提交文件，再继续提升。
