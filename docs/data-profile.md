# 数据概况

本文件基于用户提供的两份官方压缩包做只读统计。

## 1. 原始压缩包结构

### 训练包：初赛训练数据 2019-08-01.zip

```text
README.md
TRAIN/
├── Train_labels.csv
└── Train_reviews.csv
```

文件大小：

- `README.md`：1098 bytes
- `TRAIN/Train_labels.csv`：278079 bytes
- `TRAIN/Train_reviews.csv`：223607 bytes

### 测试包：初赛测试数据 2019-08-15.zip

```text
README.md
TEST/
├── Result(example).csv
└── Test_reviews.csv
```

文件大小：

- `README.md`：864 bytes
- `TEST/Result(example).csv`：97 bytes
- `TEST/Test_reviews.csv`：153835 bytes

## 2. 训练集规模

`Train_reviews.csv`：

- 3229 条评论
- 2 列：`id`, `Reviews`
- id 唯一数：3229

`Train_labels.csv`：

- 6633 条四元组标签
- 9 列：`id`, `AspectTerms`, `A_start`, `A_end`, `OpinionTerms`, `O_start`, `O_end`, `Categories`, `Polarities`
- 覆盖 id 数：3229

每条评论的标签数统计：

- 平均：2.054
- 中位数：2
- 25% 分位：1
- 75% 分位：3
- 最少：1
- 最多：7

这说明绝大多数评论不是单标签任务，而是需要进行多观点抽取。

## 3. 测试集规模

`Test_reviews.csv`：

- 2237 条评论
- 2 列：`id`, `Reviews`

提交时必须覆盖全部 2237 个测试 id。

## 4. Category 分布

训练集四元组的类别计数：

| Category | 数量 |
|---|---:|
| 整体 | 2822 |
| 使用体验 | 1042 |
| 功效 | 726 |
| 价格 | 696 |
| 物流 | 517 |
| 气味 | 225 |
| 包装 | 195 |
| 真伪 | 161 |
| 服务 | 86 |
| 其他 | 65 |
| 成分 | 61 |
| 尺寸 | 24 |
| 新鲜度 | 13 |

明显存在长尾类别问题。`整体` 占比很高，而 `新鲜度`、`尺寸`、`成分` 等类别样本极少。

## 5. Polarity 分布

| Polarity | 数量 |
|---|---:|
| 正面 | 5925 |
| 负面 | 556 |
| 中性 | 152 |

情感极性高度不均衡，正面样本占绝大多数。

## 6. 数据样例

### Train_reviews.csv

```text
1,很好，超值，很好用
2,很好，遮暇功能差一些，总体还不错
3,包装太随便了，连个包装盒都没有，第一感觉很不好
4,宝贝收到了，产品非常的不好，简直就是个垃圾，我都扔了。
5,活动价很是划算，买一送一共60片才花八十五块，天天用都不心疼啊
```

### Train_labels.csv 典型四元组

```text
id=1, AspectTerm=_, OpinionTerm=很好,   Category=整体, Polarity=正面
id=1, AspectTerm=_, OpinionTerm=超值,   Category=价格, Polarity=正面
id=1, AspectTerm=_, OpinionTerm=很好用, Category=整体, Polarity=正面
id=2, AspectTerm=遮暇功能, OpinionTerm=差一些, Category=功效, Polarity=负面
id=3, AspectTerm=包装, OpinionTerm=太随便了, Category=包装, Polarity=负面
id=3, AspectTerm=包装盒, OpinionTerm=没有, Category=包装, Polarity=负面
```

关键观察：

- 大量标签的 `AspectTerm` 为 `_`，说明隐式属性非常常见。
- 一个评论可能同时包含多个 Category 与多个情感极性。
- 属性词、观点词要求逐字对应原评论，边界判断会直接影响最终 F1。

## 7. 测试集样例

```text
1,最近太忙一直没有空来评价，东西已试过是正品，擦在脸上勾称白嫩，是个不错的商品
2,这款气垫，不是我喜欢的，因为颜色明显，还有让人家一眼就看出
3,今天才用上，感觉还是挺细腻的
4,还行吧，拿到就开始使用了
5,十年前用过这个品牌的隔离霜，没想到现在又用上了，效果不错
```

## 8. 建模风险点

1. **隐式 AspectTerm**：不能只做 NER，否则会漏掉大量 `_ + OpinionTerm` 标签。
2. **多观点关系配对**：同一句内多个 Aspect 与 Opinion 的正确对应关系决定四元组是否完全命中。
3. **类别长尾**：简单分类器容易偏向 `整体`、`使用体验` 等头部类别。
4. **极性长尾**：中性、负面样本少，需要避免全部偏向正面。
5. **严格字符串匹配**：抽取边界必须与原文完全一致。
6. **预测数量控制**：四元组过生成会直接降低 Precision。
