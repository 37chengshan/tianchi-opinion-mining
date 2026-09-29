# 官方 Baseline / OpinioNet 资料笔记

赛事页面给出的 baseline 链接：`https://github.com/eguilg/OpinioNet`。

该仓库说明自身为 2019 之江杯人工智能大赛电商评论观点挖掘赛道 Top3 方案。

## 1. 原始运行环境

仓库 README 记录的历史环境：

- Ubuntu 18.04
- Python 3.6.5
- PyTorch 1.1.0
- CUDA 9.0
- cuDNN 7.1.3
- GTX1080 8GB 单卡
- 16GB 内存

这套环境已经非常陈旧，不能直接作为 2026 年本项目的依赖基线；后续应优先复用算法思想，而不是照搬依赖版本。

## 2. 原方案核心思想

仓库将模型称为 `OpinoNet Only Look Once`，核心是：

- One-stage 端到端建模；
- 不先独立做实体抽取，再单独做关系分类；
- 以 BERT 系预训练模型作为骨架；
- 使用过 RoBERTa-WWM、BERT-WWM、ERNIE 等不同初始化；
- 通过交叉验证与多模型集成提升成绩；
- 使用阈值筛选控制最终预测数量。

从官方 README 可见，历史初赛单模型 CV 成绩约为 `0.7868`，复赛集成 CV 最高记录约为 `0.8224`。这些数字来自原赛事环境，只能作为方法有效性的历史参考，不应直接视为当前天池赛季可复现成绩。

## 3. 仓库主要文件

根目录包括：

```text
README.txt
eval_script.sh
label_corpus.sh
models/
requirements.txt
src/
train_script.sh
```

`src/` 中可见的关键代码包括：

- `config.py`
- `dataset.py`
- `model.py`
- `finetune_cv.py`
- `pretrain.py`
- `eval.py`
- `eval_ensemble.py`
- `eval_ensemble_final.py`
- `eval_ensemble_round2.py`
- `data_aug.py`
- `data_augmentation.py`
- `lr_scheduler.py`

## 4. 原方案训练流程

README 中描述的流程大致为：

1. 用无监督领域语料与有标注跨域数据做预训练 / 多任务训练；
2. 在目标领域有标注数据上做 5 折交叉验证微调；
3. 保存每折最佳筛选阈值；
4. 对 RoBERTa-WWM、BERT-WWM、ERNIE 等模型做集成；
5. 生成最终 `Result.csv`。

原仓库还特别提到：数据增强尝试曾使其复赛成绩下降，因此“简单增加增强数据”不是自动有效的策略。

## 5. 对当前初赛数据最值得复用的部分

优先复用：

1. **端到端四元组思路**：评分是严格四元组匹配，联合建模天然比完全割裂的流水线更契合指标。
2. **交叉验证**：训练样本只有 3229 条评论，5 折 OOF 能更可靠地调阈值和评估泛化。
3. **阈值优化**：当前评分同时惩罚过生成与漏召回，阈值是关键超参数。
4. **模型集成**：小数据下不同随机种子 / 不同 backbone 的集成通常有价值。
5. **显式处理隐式属性**：大量 AspectTerm 为 `_`，必须把“无显式属性”的情况作为模型能力的一部分。

## 6. 不建议直接照搬的部分

- Python 3.6 / PyTorch 1.1 / CUDA 9 的历史环境；
- 依赖旧版预训练模型代码的硬编码；
- 复赛 laptop / makeup 领域的额外语料流程，因为当前上传数据是初赛化妆品评论，数据条件并不相同；
- 未经当前训练集 OOF 验证的历史阈值和历史模型权重。

## 7. 当前实现方向

后续方案应以“现代中文预训练模型 + 四元组结构化抽取 + OOF 严格 F1 验证”为主线，同时保留一个轻量规则/词典基线用于 sanity check。

在正式实现前，应先定义：

- 数据切分方法；
- 四元组标签编码方式；
- 隐式 AspectTerm 的建模方式；
- 本地严格 F1 evaluator；
- OOF 阈值搜索策略；
- 最终提交文件校验器。
