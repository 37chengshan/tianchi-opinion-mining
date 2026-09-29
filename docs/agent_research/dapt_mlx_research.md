# DAPT/TAPT 与 MLX QLoRA 研究前置

更新时间：2026-09-28

本文是小预算研究方案和泄漏边界。当前只完成静态研究与只读预检，没有下载模型、安装依赖、启动 DAPT/TAPT、启动 MLX、生成或训练 Qwen。

## 1. 当前数据与可用语料

本地官方数据的只读统计：

| 数据 | 数量 | 用途 |
|---|---:|---|
| `Train_reviews.csv` | 3,229 条 review、3,229 个 id | 有标签任务的原文；可作为 fold 内 TAPT 原文 |
| `Train_labels.csv` | 6,633 行、覆盖 3,229 个 id | 只允许在训练 fold 做监督训练和统计 |
| `Test_reviews.csv` | 2,237 个 id | 最终预测输入；默认不进入 DAPT/TAPT |
| 训练集规范化重复文本组 | 17 组、47 行、最大重复 8 次 | 切分时必须按文本组一起放入同一 fold |

仓库当前没有单独的无标签外部商品评论语料。因而：

- **DAPT** 只有在另有合规、可追溯的外部商品评论语料时才有意义；它应模拟产品评论领域，而不是偷偷使用验证或测试文本。
- **TAPT** 可以使用当前任务的 review 原文，但每个 fold 只能使用该 fold 的 train review。没有外部语料时，优先做 fold-local TAPT，不把全部 3,229 条文本预先混合后再切 fold。

## 2. 严格防泄漏协议

切分单位不是 label 行，而是 review group。先以 `id` 聚合所有 label，再按 `NFKC(text).strip()` 的规范化文本把重复 review 合并为同组，最后固定 `KFold(n_splits=3 或 5, shuffle=True, random_state=42)`。这样同一评论的多个四元组和重复文本不会跨 train/valid。

每一个 fold 必须单独创建：

1. 原始 pretrained encoder 的新实例。
2. tokenizer 实例和 TAPT/DAPT 数据集。
3. MLM checkpoint、任务 head、optimizer、scheduler 和阈值搜索输入。
4. 只由训练 fold 生成的文本统计、类别先验、候选字典或 retrieval 资源。

验证 fold 的原文不能进入该 fold 的 DAPT/TAPT；测试原文默认也不进入任何 unsupervised adaptation。tokenizer 不重新训练词表，不用全数据拟合归一化、阈值或标签映射。每个验证样本的输出只能在预测完成后参与该折阈值搜索，不能反向改 MLM 或任务参数。

若未来要研究 transductive adaptation，必须另建实验名并明确记录“使用了测试原文”，不能把它和严格无泄漏结果放在同一个晋级表中。当前任务不采用该变体。

## 3. 小预算 DAPT/TAPT 方案

目标是用很小的 MLM 预算观察领域适配是否值得进入任务微调，不把无标签 loss 当作比赛分数。

| 阶段 | 语料 | max length | micro batch | 累积 | 学习率 | 上限 | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| 3-fold screen：TAPT | 当前 fold 的 train review 原文 | 128 | 2 | 4 | `1e-5` | 200 steps 或 1 pass，先到者为准 | 动态 MLM mask，建议 15%；每 fold 从原始强 encoder 开始 |
| 3-fold screen：外部 DAPT（可选） | 独立外部商品评论，去重后只读 | 128 | 2 | 4 | `1e-5` | 200 steps 或 1 pass，先到者为准 | 没有外部语料时不运行、不伪造结果 |
| 任务微调 | 同一 fold 的监督训练集 | 96/128 | 2 | 4 | encoder `2e-5`，head `8e-4` | 沿用现有小预算 early stop | 只在 3-fold screen 通过后比较 5-fold |

建议先做三条互斥候选：`MacBERT-base`、`RoBERTa-wwm-ext-base`、以及其中 OOF 较好的一个 TAPT 版本。每条候选都必须经过同一 strict decoder、同一 threshold search 和同一 3-fold split。若 3-fold 没有超过 rbt3 参考，不进入更贵的 5-fold。

MLM 适配的成功标准是“下游 strict F1 的 OOF 改善”，不是 MLM loss 下降。TAPT checkpoint 只服务于该 fold 的监督任务；不能把一个 fold 的适配权重复制给另一个 fold。

## 4. 晋级门槛

仓库已有 rbt3 参考：`neural_3fold_rbt3` strict F1=`0.6874480465502909`，`neural_5fold_confirm` strict F1=`0.7208976157082748`。这两个数来自已有 artifact，仅作为比较基线，未由本次任务重新计算。

固定流程：

1. 3-fold screen 只比较同一 split、同一 `max_length`、同一解码和阈值协议。候选均值必须超过 rbt3 的 3-fold 参考，且至少 2 折不低于对应参考折。
2. 通过后进入 5-fold confirm；每折从原始模型或该折自己的 DAPT/TAPT checkpoint 开始，不能 warm-start 另一个 fold。
3. 5-fold OOF 阈值只用 OOF 预测搜索。最终比较必须相对 rbt3 的 5-fold 参考，检查均值、每折分数、显式/隐式 A/O 分组和候选数量。
4. 不能因为某一折或某个阈值偶然上升就晋级；没有稳定 OOF 改善时保持当前冠军。

## 5. Qwen3-4B 4bit MLX QLoRA 前置条件

### 模型与官方路径

Qwen 官方模型 ID 是 `Qwen/Qwen3-4B`：4.0B 参数、36 layers，原生 context length 32,768。[Qwen3-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)

Apple MLX challenger 使用 `mlx-community/Qwen3-4B-4bit`。该模型卡标记为 MLX、4-bit、约 2.26 GB，并说明它由 `Qwen/Qwen3-4B` 转换而来；它不是当前本地已缓存的模型。[MLX Qwen3-4B-4bit 模型卡](https://huggingface.co/mlx-community/Qwen3-4B-4bit)

MLX-LM 官方 LoRA 文档说明：`mlx_lm.lora` 接受本地转换目录或 Hugging Face repo；量化模型会走 QLoRA；训练数据目录需要 `train.jsonl`，可以有 `valid.jsonl`；内存紧张时优先 batch size 1、gradient accumulation、少量 layers 和 gradient checkpointing。[MLX-LM 官方 LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

### 必须全部满足的 gate

1. Apple Silicon macOS，总统一内存至少 16 GiB；可用内存低于 3 GiB 时禁止启动，建议至少 5 GiB 再进入 tiny smoke。16 GiB 不是训练保证，2.26 GB 权重之外还需要 KV/activation、adapter、Python 和系统空间。
2. `mlx`、`mlx-lm` 和 `mlx_lm.lora` 可导入，版本和 parser 选项可记录。当前预检结果是三者均未安装，不能运行。
3. `mlx-community/Qwen3-4B-4bit` 已完整存在本地，至少有 `config.json`、tokenizer 和 safetensors 权重；预检不得访问 Hub。当前 `/Users/cc/.cache/huggingface` 只发现 `hfl/rbt3`，Qwen 4bit 未缓存。
4. 训练数据由当前 fold 的 train review 生成 `train.jsonl`，验证 review 生成 `valid.jsonl`；不要把 test review 写入训练或验证文件。任务格式必须固定，输出能解析为项目四元组并用同一 strict evaluator 评分。
5. 第一轮只允许极小配置：batch 1、gradient accumulation 8、max sequence 512、`num_layers=4`、gradient checkpointing；先做 parser/config smoke，再由用户另行批准真实训练。
6. adapter、config、训练日志和解析后的候选要按候选 id 保存；不 fuse 回原模型，不用生成结果替代 strict F1，不把 Qwen challenger 结果直接混入 encoder champion。

MLX QLoRA 路径的量化依赖是 MLX/MLX-LM；它不因为 `bitsandbytes` 缺失就自动可用，也不需要把 Transformers 的 PEFT 状态误报为已准备好。当前 `mlx`、`mlx-lm`、`peft`、`bitsandbytes` 都未安装，所以 Qwen 只停留在前置条件设计。

## 6. 当前执行边界

- 本次只运行了 `python3 scripts/encoder_preflight.py` 默认预检和 `py_compile` 语法检查。
- 默认预检输出 `status=queue`、`reason=strong_encoder_models_not_cached`。
- 没有执行 `--probe-load`；因此没有声称任何 MacBERT、RoBERTa 或 Qwen 权重被加载。
- 没有下载、安装、训练、生成、提交或上传。
