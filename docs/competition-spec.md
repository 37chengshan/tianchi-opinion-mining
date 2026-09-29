# 比赛规则与任务定义

## 1. 比赛名称

【天池经典打榜赛】赛道六-评论观点挖掘赛。

赛题来自电商评论观点挖掘场景，当前赛季仅开放初赛测评。

## 2. 任务定义

给定一条商品评论，需要抽取一个或多个独立观点四元组：

`(AspectTerm, OpinionTerm, Category, Polarity)`

字段含义：

- `id`：评论唯一标识。
- `AspectTerm`：商品属性特征词，必须与评论原文中的表述一致；为空时使用 `_`。
- `OpinionTerm`：消费者观点词，必须与评论原文中的表述一致；为空时使用 `_`。
- `Category`：属性种类，必须属于官方限定集合。
- `Polarity`：观点极性，取正面、中性、负面。

同一条评论可以包含多个四元组，四元组之间独立计分。

## 3. 官方类别集合

训练数据 README 给出的 Category 集合：

- 包装
- 成分
- 尺寸
- 服务
- 功效
- 价格
- 气味
- 使用体验
- 物流
- 新鲜度
- 真伪
- 整体
- 其他

Polarity 集合：

- 正面
- 中性
- 负面

## 4. 训练数据

训练包结构：

```text
README.md
TRAIN/
├── Train_reviews.csv
└── Train_labels.csv
```

`Train_reviews.csv` 包含：

- `id`
- `Reviews`

`Train_labels.csv` 包含：

- `id`
- `AspectTerms`
- `A_start`
- `A_end`
- `OpinionTerms`
- `O_start`
- `O_end`
- `Categories`
- `Polarities`

位置字段只用于训练数据中的标注定位；官方明确说明预测结果不需要位置字段。

## 5. 测试与提交格式

测试包结构：

```text
README.md
TEST/
├── Test_reviews.csv
└── Result(example).csv
```

最终提交文件必须命名为：

`Result.csv`

每行 5 个字段，顺序为：

```text
id,AspectTerm,OpinionTerm,Category,Polarity
```

关键约束：

1. 不要表头。
2. 必须是 UTF-8 且无 BOM。
3. 测试集中出现的每一个 id 都必须在提交文件中出现，不能遗漏，也不能多出 id。
4. id 必须升序排列。
5. 一个 id 可对应多行四元组。
6. 若某个 id 完全没有预测结果，也必须保留一行，并将其余字段写成 `_`。
7. AspectTerm / OpinionTerm 若预测为空，写 `_`。

官方样例：

```csv
1,味道,香香的,气味,正面
1,物流,神速,物流,正面
2,_,好贵,价格,负面
3,_,_,_,_
```

## 6. 评分规则

在相同 id 内逐一匹配预测四元组和真实四元组。

只有以下四个字段全部一致时，该四元组才算预测正确：

- AspectTerm
- OpinionTerm
- Category
- Polarity

定义：

- `P`：预测四元组总数
- `G`：真实四元组总数
- `S`：完全匹配的正确四元组数

则：

```text
Precision = S / P
Recall    = S / G
F1        = 2 * Precision * Recall / (Precision + Recall)
```

最终排名指标为 F1-score。

## 7. 赛制与提交限制

根据赛事页面信息：

- 比赛不允许组队，以个人形式参赛。
- 赛程期间实时评测，整点刷新排名。
- 每天最多提交 5 次。
- 获奖后需要在规定时间内公开方案并通过主办方验证。
- 主办方有权要求提交源代码进行审查。

## 8. 对解题方案的直接影响

由于采用“整四元组严格匹配”计分：

- 只抽对实体但类别或极性错，仍然计为整条错误。
- AspectTerm 和 OpinionTerm 必须保持原文字符串，不能随意归一化或改写。
- 预测过多会降低 Precision，预测过少会降低 Recall，因此阈值调优非常重要。
- `_` 形式的隐式属性是任务的重要组成部分，不能只训练显式 AspectTerm 抽取。
