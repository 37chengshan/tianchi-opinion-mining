# 天池评论观点挖掘赛可提交方案设计

## 目标

构建一个可复现、可本地验证、能生成官方合法 `Result.csv` 的竞赛工程，并在有限数据（3229 条训练评论）下优先争取严格四元组 F1。

## 输入与输出

输入：官方训练包和测试包，核心文件分别为 `Train_reviews.csv`、`Train_labels.csv`、`Test_reviews.csv`。

输出：无表头、UTF-8 无 BOM、覆盖全部测试 id、id 升序的 `Result.csv`，字段固定为：`id,AspectTerm,OpinionTerm,Category,Polarity`。

## 评价口径

同一 id 内以四元组 `(AspectTerm, OpinionTerm, Category, Polarity)` 做集合级严格匹配。只有四字段完全一致才计为正确。工程内 evaluator 必须与此规则一致，并对重复预测去重。

## 总体架构

采用“可靠基线 + 统计检索增强 + 可升级神经模型接口”的分层方案：

1. 数据层负责读取官方 CSV、标准化 `_`、恢复评论文本与标签集合。
2. 评测层实现严格 micro Precision / Recall / F1。
3. 第一版可提交模型使用训练标签构建高精度词典与短语规则，并用评论相似检索补充四元组候选；全部候选必须经过原文字符串约束。
4. 在训练集上做 K 折 out-of-fold 阈值搜索，选择能最大化严格 F1 的候选置信度阈值。
5. 训练完后在全量训练集拟合规则/检索统计，对测试集生成候选并过滤。
6. 提交层保证全 id 覆盖、字段合法、原文一致性、无 BOM、无表头。
7. 神经模型作为后续升级点：可以在不改变 evaluator / submission API 的前提下接入现代中文 encoder 或指针网络，并与规则候选做并集/投票。

## 第一阶段模型设计

### 精确短语记忆

从训练集中统计 `(AspectTerm, OpinionTerm, Category, Polarity)` 的出现频率以及其在评论中的上下文。若 AspectTerm / OpinionTerm 非 `_`，预测时必须确认对应字符串出现在测试评论原文中。

### Opinion 驱动候选

OpinionTerm 往往比 AspectTerm 更直接表达观点。对在训练中高纯度映射到固定 Category/Polarity 的 OpinionTerm，可在测试评论中命中后产生候选；若对应 AspectTerm 在训练中经常为 `_`，允许生成隐式属性四元组。

### Aspect-Opinion 共现

当训练中某 AspectTerm 与 OpinionTerm 配对稳定，且两者都出现在测试评论时生成高置信候选。对同一短语存在多类别/极性时按训练条件概率打分，不做无依据硬猜。

### 检索迁移

用字符 n-gram TF-IDF 计算测试评论与训练评论相似度。只有高相似度邻居才迁移标签；迁移标签中的显式 Aspect/Opinion 必须在目标评论中原样出现，否则丢弃。隐式 AspectTerm `_` 可保留，但 OpinionTerm 仍需在目标文本中出现。

### 候选融合

每个候选四元组附带来源和置信度，来源包括 exact-pair、opinion-map、aspect-opinion、retrieval。相同四元组合并并提升置信度。最终阈值由 OOF 搜索确定，而不是手工固定。

## 数据切分

使用按评论 id 的 KFold（默认 5 折，固定随机种子）。每折只允许用训练折构建词典/检索库，在验证折上生成预测，汇总所有 OOF 结果后搜索全局阈值。禁止让验证折标签进入候选统计。

## 边界与约束

- AspectTerm / OpinionTerm 只要不是 `_`，必须是当前评论的原文子串。
- Category 必须属于官方 13 类集合。
- Polarity 必须属于 `正面/中性/负面`。
- 一个 id 没有候选时，提交一行 `id,_,_,_,_`；这行仅用于提交格式，不参与训练 evaluator 的正例集合。
- 重复四元组只保留一条。
- 不对原文做会改变字符串的规范化后再输出。

## 工程结构

- `src/opinion_mining/data.py`：数据读取、标签结构。
- `src/opinion_mining/metrics.py`：严格四元组 F1。
- `src/opinion_mining/baseline.py`：词典 + 检索候选模型。
- `src/opinion_mining/pipeline.py`：OOF、阈值搜索、全量拟合、推理。
- `src/opinion_mining/submission.py`：Result.csv 生成和校验。
- `scripts/run_baseline.py`：一键训练、验证、预测、生成提交。
- `tests/`：核心单元测试。

## 完成标准

1. 单元测试覆盖 strict F1、重复预测、隐式属性、原文约束、缺失 id、BOM/表头。
2. OOF 流程可运行并输出阈值和本地 F1。
3. 对官方测试集生成 `Result.csv`。
4. 校验器确认 2237 个测试 id 全覆盖、升序、合法字段、无 BOM、无表头。
5. 保存运行报告，记录模型配置、OOF 指标和提交统计。
