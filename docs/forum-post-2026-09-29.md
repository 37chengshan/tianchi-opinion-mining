# 从 0.7162 到 0.7734：评论观点挖掘 ACOS 四元组方案（含失败清单与复现入口）

> **代码与复现入口**：https://github.com/37chengshan/tianchi-opinion-mining
> **线上成绩**：strict 四元组 F1 **0.7734**（天池评测系统 8 次真实提交，最高一次见下表）
> **训练环境**：Apple M1 Pro / 16GB 统一内存（本地训练，无云端 GPU）
> **可验证性**：`python3 -m pytest -q` → `123 passed, 3 skipped`；全部线上提交文件与 SHA256 已归档

---

## 一、结果先行

| # | 提交内容 | 本地（无泄漏交叉拟合） | 线上 LB |
| --- | --- | ---: | ---: |
| 0 | 历史 Anchor：`hfl/rbt3` one-stage span/relation，5 折 | 0.7209 | 0.7162 |
| 1 | 换 backbone：`hfl/chinese-roberta-wwm-ext` 5 折 × 4 轮 | 0.7543 | 0.7583 |
| 2 | + 长训练成员：WWM 5 折 + WWM 3 折 × 8 轮（1 : 0.7） | 0.7693 | 0.7645 |
| 3 | 单模型试投：WWM 5 折 × 10 轮 / WWM 3 折 × 8 轮 | 0.7731 / 0.7637 | 0.7633 / 0.7559 |
| 4 | + 多样性成员：三源（1 : 0.7 : 0.3） | 0.7682 | 0.7668 |
| 5 | 四源备选权重（1 : 1 : 0.5 : 0.4） | 0.7753 | 0.7702 |
| 6 | **四源融合（最终）**：WWM 5×10 + WWM 3×8 + WWM 5×4 + MacBERT 5（1 : 0.7 : 0.4 : 0.3） | **0.7768** | **0.7734** |

两个必须先说清楚的观察，它们决定了后面所有决策：

1. **线上分系统性地低于本地无泄漏分**（−0.0014 ~ −0.0048）。所以本地分只能当筛选门禁，**任何替换都以线上实测为准**，本地高 0.001 不算高。
2. **上限不在模型，在决策层**。WWM 候选集合的**金标覆盖已达 0.9772**，也就是说金标里有 97.7% 的答案模型其实生成出来了；完美排序的理论上限是 F1 0.9885。而阈值决策实际只做到 0.754。把力气花在"更好的重排序"上，不如花在"更多样、更强的成员 + 折安全的融合"上。

---

## 二、赛题与评测口径

从中文电商评论里同时抽出 **(Aspect, Opinion, Category, Polarity)** 四元组：方面词、观点词、类别、极性。评分是**严格四元组 micro-F1** —— 四个字段全部命中才算一条正确，错一个字段等于零分。这带来两个直接后果：

- **不能靠堆召回拿分**：多输出 1 条错的就是 FP，P 与 R 必须同时成立；
- **解码约束比模型容量更早成为瓶颈**：start/end 位置、`_`（隐式观点）哨兵、类别的 39 类联合空间，任何一处不一致都会整条作废。

数据规模（`artifacts/data/`）：训练 **3229 条评论 / 6632 条金标四元组**，测试 **2237 条评论**；单条评论最长只有 **72 token** —— 这条事实后来直接否掉了一个实验方向（见第五节）。

---

## 三、方案：one-stage span/relation + 折安全决策层

![训练与推理流水线](https://raw.githubusercontent.com/37chengshan/tianchi-opinion-mining/main/assets/readme/workflow.svg)

**模型侧（单阶段，不用 dense 网格）**

| 项 | 取值 |
| --- | --- |
| 主干 | `hfl/chinese-roberta-wwm-ext`（对比 `hfl/rbt3`、`hfl/chinese-macbert-base`） |
| 序列 | `max_length=96`，`batch=4`，梯度累积 2 |
| 优化 | lr `2e-5`（长训练档 `1.2e-5`），head lr `8e-4`，weight decay `0.01` |
| 可训练层 | 后 2 层 + head |
| 轮数 | 4 / 8 / 10（长训练是后来发现的增益来源） |
| Head | objectiveness 打分 + 四个指针（aspect/opinion × start/end）+ `Category × Polarity` 联合 39 类 |
| 解码 | `start ≤ end`、`span ≤ max_span_len(10)`、`_` 哨兵表示隐式观点、NMS、`relation_top_k=2` |

**决策侧（本方案最关键的部分）**

- **折安全（fold-safe）**：阈值与融合权重只在**其他折**上拟合，被评分的那一折只做应用。任何"在当前折上调阈值"得到的分数，在本仓库一律标记为 `in_sample_reference_not_strict`，**禁止用于晋级**。这是"我的本地 0.78 一上线就掉"最常见的原因。
- **单变量门禁**：每次实验只改一个配置项，保留条件是 ΔF1 ≥ 0.004；不满足就回滚，不叠加未验证技巧。
- **成员按异构选，不按最强选**：四源里包含 8 轮/10 轮两个训练长度、3 折/5 折两种划分、WWM/MacBERT 两个主干。第 4 次提交（三源，本地 0.7682）本地比二源低 0.0011，线上却高 0.0023 —— 这条证据直接改变了后面的选人标准。
- **概率级融合**：候选先归一化再按源加权（最终 1 : 0.7 : 0.4 : 0.3），而不是在答案集合层面做规则合并。

---

## 四、单项消融（3 折、固定划分、同参、单变量）

历史 RBT3 基线（同折同参）：**0.6874**

| 实验 | 严格 F1 | Δ | 门禁（≥0.004） |
| --- | ---: | ---: | --- |
| B1 官方 offsets | **0.7140** | +0.0266 | 通过 |
| B2 implicit-O 哨兵 | 0.7116 | +0.0241 | 通过（implicit-O 召回 **0.0000 → 0.3547**） |
| B3 多关系保留 | 0.7064 | +0.0189 | 通过 |
| v1 = B1+B2+B3 | 0.7140 | +0.0266 | 组合**严格等于 B1 单独**，无叠加增益 |

结论很反直觉但很重要：三项修复单独都过门禁，**合在一起却等于 B1 单独**。增益集中在"offsets 与解码一致性"上，B2/B3 的价值被 B1 覆盖（B2 的意义反而是把 implicit-O 的召回从 0 拉起来，避免这一类样本被整体丢弃）。

---

## 五、失败清单（同样重要，避免后来者重复踩）

| 路线 | 实测 | 处理 |
| --- | --- | --- |
| dense Grid V2 + pair filter 融合 | 无泄漏交叉拟合 **0.6337** | 整条路线淘汰，one-stage 才是主线 |
| OpinioNet-style 结构增强（RBT3 控制组） | 0.7088 vs 0.7140 | 未过 −0.002 门禁 |
| 本地 QLoRA（通义千问 Qwen3-4B，MLX 4-bit） | fold-1 **0.6417**（P 0.6688 / R 0.6167） | 量化 + 短训不足，降级为 verifier/challenger |
| reranker 路线（BM25 / prior 重排） | 0.7180 ~ 0.7527 | 未超过同参数 one-stage |
| 加大 `max_length` | 最长评论 72 token，96 已无截断损失 | 不是瓶颈，放弃 |

**关于 LLM 直抽这条路**：我们用通义千问 Qwen3-4B 在本赛题数据上做了真实的 QLoRA 微调实验（MLX 4-bit，batch 1 + 梯度累积、cutoff 512、2 epochs / 2152 iters；输出严格 JSON、含 implicit-O 哨兵，**解析失败 0 条**），fold-1 严格 F1 = **0.6417**（1077 条验证，2023 条预测，2194 条金标），对照同折 BERT 是 0.7442。差距不在"LLM 不行"，而在**本地的量化 + 短训条件**：4-bit 权重 + 单卡统一内存，训练轮数与序列预算都被压住。同赛题的公开方案用云端 bf16 LoRA，同一份数据可以到 Qwen3-4B 0.75 / Qwen2.5-7B 0.78 / 32B 0.81 档 —— 所以这条路线被保留为"云端再战"，而不是在本地硬堆。

---

## 六、阿里云产品使用说明

本项目从平台、模型获取到模型微调实验都在阿里云生态内完成：

**1）天池平台（评测与提交）**
全部线上分数来自天池评测系统的真实提交：共 **8 次评测**（历史 Anchor 0.7162，本轮 7 次：0.7559 / 0.7583 / 0.7633 / 0.7645 / 0.7668 / 0.7702 / **0.7734**）。每次提交的候选文件、行数、预测四元组数、SHA256 都归档在仓库 `artifacts/submissions/`，可逐条复核。

**2）魔搭 ModelScope（模型获取，阿里云）**
仓库提供一条命令取主干权重，内置 Hugging Face → 魔搭的镜像映射（`hfl/rbt3` → `dienstag/rbt3`，`hfl/chinese-roberta-wwm-ext` → `dienstag/chinese-roberta-wwm-ext`），下载后自动校验 `config.json` / tokenizer / 权重文件并输出 JSON 摘要：

```bash
python3 scripts/fetch_model.py --model hfl/chinese-roberta-wwm-ext --out models/wwm
# 默认 --source modelscope（魔搭，国内直连）；需要时可用 --source hf 切回 Hugging Face
```

**3）通义千问 Qwen3-4B（阿里云开源模型）微调实验**
如第五节所述，我们在本赛题数据上真实跑过 **Qwen3-4B 的 QLoRA 微调**（MLX 4-bit，LoRA rank 8、cutoff 512、batch 1 + 梯度累积），并用与 BERT 主线**完全相同的严格评分器**打了 fold-1：**0.6417**，且 0 条解析失败（说明 prompt/JSON schema/子串校验这条工程链路是通的）。这个负数结论直接决定了 Qwen 在本方案中的角色：**只做长尾/隐式样本的 verifier 与 challenger，不做主模型，也不允许直接覆盖高置信的 BERT 集成输出**。

**4）PAI-DSW 云端复现路径（云端路线规划，非本机实跑）**
本仓库训练脚本为纯 PyTorch，可直接在 PAI-DSW Notebook 中以相同命令、相同 fold 划分运行；同时仓库已提供云端配方与打包脚本：`run_cloud_lora.yaml`（bf16 LoRA，lr 1e-4 + cosine/warmup、rank 16/alpha 32、cutoff 1024、batch 4 × accum 8、3 epochs）与 `scripts/make_cloud_bundle.sh`（把数据、prompt 契约、评分脚本、训练配置打成一个不含量密钥的 tar 包）。评测口径与本地一致，回传 `predictions.jsonl` 即可用同一评分器对齐比较。

**参考的真实实践案例**（同类云上路径，供核对）：
- 阿里云官方教程《在 PAI 上使用 Python SDK 部署与微调 ModelScope 模型》（Qwen1.5-7B-Chat 完整示例）：https://developer.aliyun.com/article/1520987
- PAI-ArtLab × 魔搭训练工具（基于魔搭模型做大模型 LoRA 微调）：https://help.aliyun.com/zh/pai/pai-artlab-modelscope-model-training
- 天池《通义千问 AI 挑战赛》Code Qwen 赛道获奖方案（Qwen-72B LoRA，1 万条样本约 4.5 小时）：https://tianchi.aliyun.com/forum/post/659750
- 阿里云 PAI-DSW 产品页（Notebook 云端深度学习开发环境）：https://www.aliyun.com/activity/bigdata/pai-dsw

---

## 七、复现步骤

```bash
git clone https://github.com/37chengshan/tianchi-opinion-mining.git
cd tianchi-opinion-mining
export PYTHONPATH=src

# 1) 环境自检（不需要数据、不需要 GPU）
python3 -m pytest -q                 # 123 passed, 3 skipped

# 2) 取主干权重（默认走魔搭 ModelScope）
python3 scripts/fetch_model.py --model hfl/chinese-roberta-wwm-ext --out models/wwm

# 3) 放入赛题数据（Train_reviews.csv / Train_labels.csv / Test_reviews.csv）
#    artifacts/data/train/TRAIN/ 与 artifacts/data/test/TEST/

# 4) 训练 + 严格评估 + 生成提交
python3 scripts/run_anchor.py --config configs/anchor_wwm_5fold.json
python3 scripts/build_submission.py --experiment anchor_wwm_5fold

# 5) 概率级融合（多源）
python3 scripts/ensemble_submit.py --source wwm5=... --source wwm3long=... --weights 1:0.7
```

**硬件与时间参考**：M1 Pro 16GB，全部本地训练。RBT3 3 折筛选约 14 分钟（每折 290 秒）；WWM 单折约 29 分钟（5 折 × 4 轮）；MacBERT 单折约 14 分钟。所有实验都是本地单机可复现规模。

---

## 八、领奖与验证信息（按赛制要求）

- **方案发布**：本文发布于天池论坛（**发布后请在此处填入本贴链接**）；
- **代码/复现地址（可填领奖表单的 notebook 或文章地址）**：https://github.com/37chengshan/tianchi-opinion-mining
- **数据**：未随仓库分发，请从比赛页面获取后放入 `artifacts/data/`（仓库 `.gitignore` 已排除）；
- **结果可核对**：线上 8 次评测的候选文件与 SHA256、每折严格 F1、阈值与融合权重均归档在仓库内；
- **云产品使用**：见第六节（天池平台 + 魔搭 ModelScope 模型获取 + 通义千问 Qwen3-4B QLoRA 微调实验 + PAI-DSW 云端复现路径）。若申请云产品相关奖励，表单可备注"**盲盒**"。

---

## 九、下一步（冲 0.78 → 0.80）

按"预期收益 / 小时"排序，全部建立在上面已经证明的三条结论上：

1. **更大的中文 encoder**：`hfl/chinese-roberta-wwm-ext-large`（24 层）走 3 折筛选 → 5 折 → 与现有四源融合。历史上换更强的 WWM 一次就带来 +0.0334 本地分，规模差异通常还能再给 +0.01~0.02。
2. **云端 bf16 LoRA 的 Qwen2.5-7B**：用同一份 fold-1 协议先验证（fold-1 ≥ 0.72 才值得全量重训），只做**加法补召**，绝不允许它删除 BERT 集成的高置信输出。
3. **TAPT / DAPT**：在赛题语料上做领域自适应预训练（是否可使用测试集文本会先核对赛规），用下游 3 折决定是否保留。
4. **长尾定向**：历史 Anchor 的错误画像显示 FN 主要落在 implicit-A（1218 条）、FP 主要在 span / 未见观点（1136 条）；新版集成尚未做同口径拆解，但这两类样本的定向构造可能比"再加大一个模型"更便宜。

---

## 附录：关键数字与文件

| 名称 | 值 |
| --- | --- |
| 冠军提交 SHA256 | `5ad6c1335879a33c5232ee6f8f0f6ea823524c331f82f6cba1bddfbf5ac924e7` |
| 历史 Anchor SHA256 | `f9eac4b6d4ea769e13ea62cac9d3fa9364a70ddf9a0297c8332cc7a80d40d43f` |
| 历史 Anchor 误差画像 | TP 4626 / FP 1576 / FN 2006（FN implicit-A 1218，FP span/unseen-opinion 1136） |
| 候选金标覆盖（WWM 3 折） | 0.9772（完美排序上限 F1 0.9885） |
| Qwen3-4B QLoRA（fold-1） | P 0.6688 / R 0.6167 / F1 0.6417，解析失败 0 |
| 测试规模 | 2237 条评论 / 融合候选提交约 4000+ 条预测四元组 |

*文中所有分数均来自仓库内归档的实验报告与提交记录，欢迎逐项复核；负结果同样欢迎质疑与复现。*
