# 强编码器研究与本地前置检查

更新时间：2026-09-28

本文只记录模型标识、当前本地状态、资源边界和后续验证门槛。本次没有下载模型、安装依赖、启动训练或修改训练核心。

## 1. 真实模型标识

项目现有 [grid_trainer.py](/Users/cc/code/tianchi-opinion-mining/src/opinion_mining/grid_trainer.py) 已把 `macbert` 和 `wwm` 别名解析到下面两个 ID。模型卡的真实 ID 不是项目别名：

| 研究候选 | 真实 Hugging Face ID | 官方模型卡 | 本地状态（2026-09-28） |
|---|---|---|---|
| MacBERT-base | `hfl/chinese-macbert-base` | [HFL MacBERT-base](https://huggingface.co/hfl/chinese-macbert-base) | **未缓存**；没有发现完整的 `config.json`、tokenizer 和权重目录 |
| RoBERTa-wwm-ext-base | `hfl/chinese-roberta-wwm-ext` | [HFL RoBERTa-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | **未缓存**；没有发现完整的 `config.json`、tokenizer 和权重目录 |
| 当前项目基线 | `hfl/rbt3` | 项目本地引用 | **完整缓存**：`/Users/cc/.cache/huggingface/hub/models--hfl--rbt3/snapshots/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c` |

HFL 模型卡要求使用 BERT 相关加载函数。MacBERT 卡说明其预训练改进包含 MLM-as-correction、WWM、N-gram masking 和 SOP；RoBERTa 卡说明它是中文 Whole Word Masking BERT。两者都可以作为当前 `AutoModel` encoder 适配器的候选，实际训练前仍必须先拿到本地完整快照。[MacBERT 模型卡](https://huggingface.co/hfl/chinese-macbert-base)、[RoBERTa 模型卡](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

本地唯一完整快照的 `config.json` 是 `hfl/rbt3`：`model_type=bert`、`hidden_size=768`、`num_hidden_layers=3`、`num_attention_heads=12`、`vocab_size=21128`，权重文件为 `pytorch_model.bin`，目录大小约 150 MiB。这个事实只证明 rbt3 缓存完整，不代表两个强编码器已经可用。

## 2. 依赖、MPS 和 16 GB 统一内存

`python3 scripts/encoder_preflight.py` 的真实默认输出摘要如下；它只做依赖元数据、`find_spec`、本地文件系统和 MPS 能力检查：

| 检查项 | 实际结果 |
|---|---|
| 主机 | macOS 27.0，Apple arm64，Apple M1 Pro，16.0 GiB |
| Python | 3.9.6 |
| PyTorch | 2.8.0；MPS built=`true`，available=`true` |
| Transformers | 4.57.6 |
| Accelerate | 1.10.1 |
| safetensors / tokenizers | 0.7.0 / 0.22.2 |
| sentencepiece | 未安装；当前缓存的 rbt3 有 `tokenizer.json` 和 `vocab.txt` |
| MLX / MLX-LM | 均未安装 |
| PEFT / bitsandbytes | 均未安装 |
| 近似可用统一内存 | 4.51 GiB，黄色区间；内存压力 free percentage=57%；swap 已用约 6.79/8.00 GiB |
| 预检状态 | `queue`，原因 `strong_encoder_models_not_cached` |

PyTorch 官方 MPS 文档把 `torch.backends.mps.is_built()` 和 `torch.backends.mps.is_available()` 作为能力检查入口；本机两项都通过。[PyTorch MPS 官方文档](https://docs.pytorch.org/docs/main/notes/mps.html)

16 GiB 是统一内存总量，不是可供模型独占的显存。模型权重、激活、优化器状态、Python 进程、文件缓存、压缩内存和 swap 共用这一空间。当前近似可用内存只有 3.66 GiB，因此处于预检的黄色区间；脚本把低于 3 GiB 设为禁止探测加载，5 GiB 以上才记为绿色。后续运行必须逐折串行、单 encoder、短序列和小 batch，并在每折结束后释放模型、优化器和 MPS 缓存。

Transformers 的 `from_pretrained` 支持本地目录和 `local_files_only`；本项目的 [CompactGridEncoderAdapter.from_pretrained](/Users/cc/code/tianchi-opinion-mining/src/opinion_mining/grid_trainer.py:155) 已采用 `local_files_only=True`。本次的 `encoder_preflight.py` 在 `--probe-load` 中进一步只传入已经检查为完整的本地快照，禁止把模型 ID 交给 Hub 解析。[Transformers 模型加载文档](https://huggingface.co/docs/transformers/main_classes/model)

### 依赖判断

- **普通强 encoder 微调**：当前 `torch`、`transformers`、`accelerate`、`safetensors` 和 `tokenizers` 已满足基础软件入口；模型文件仍缺失，所以不能进入加载或训练。
- **MPS**：当前可用，但 MPS 使用统一内存。`mps_available=true` 只说明后端可用，不说明 16 GiB 主机适合任意 batch、序列长度或折数。
- **Transformers 量化/QLoRA**：当前没有 `peft` 或 `bitsandbytes`，不能把该路径当作已准备好。Hugging Face 的 bitsandbytes 文档现在列出 Apple Silicon 支持，但本机没有安装，且本项目没有执行安装；因此暂不把它作为 MPS 强 encoder 路径的前置承诺。[bitsandbytes 官方安装文档](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- **MLX QLoRA**：走独立的 `mlx`/`mlx-lm` 路线，不等同于 Transformers+bitsandbytes；当前二者均未安装，不能运行 Qwen challenger。

## 3. 只读 preflight 行为

入口：[encoder_preflight.py](/Users/cc/code/tianchi-opinion-mining/scripts/encoder_preflight.py)

默认命令：

```bash
cd /Users/cc/code/tianchi-opinion-mining
python3 scripts/encoder_preflight.py
```

默认行为：

1. 检查 Python 包是否可发现并读取已安装版本，不安装任何东西。
2. 检查 macOS/arm64、总内存、近似可用内存、内存压力和 swap；不创建 tensor。
3. 只在已知 Hugging Face cache 根目录中检查两个强 encoder 和当前 rbt3；要求 `config.json`、tokenizer 文件和权重文件同时存在才算完整缓存。
4. 输出单行 JSON，包含 `status`、`reason`、模型状态、依赖状态、MPS 状态和安全边界。

显式 `--probe-load` 时，脚本只尝试已经判定为完整的本地模型，使用 `local_files_only=True`，默认在 CPU 上加载；`finally` 中删除 tokenizer/model 引用、运行 `gc.collect()`，若选择 MPS 则调用 `torch.mps.empty_cache()`。未缓存模型不会被尝试，Hub 不会被访问。当前默认预检没有执行 `--probe-load`，所以没有声称任何目标模型加载成功。

## 4. 强 encoder 实验设计

保持现有 `CompactGridEncoderAdapter`、strict quadruple evaluator 和候选格式不变，只替换 `model_name` 并为每折重新创建 encoder、head、optimizer 和 tokenizer。推荐顺序：

1. `hfl/chinese-macbert-base`，作为 correction-style 中文预训练候选。
2. `hfl/chinese-roberta-wwm-ext`，作为 WWM 中文候选。
3. 只有在有明确 OOF 证据时，才把 DAPT/TAPT 版本送入同一晋级流程。

每折的资源上限先固定为 `max_length=96` 或 `128`、batch size 2、gradient accumulation 4、只训练最后 1–2 个 encoder layer 加任务 head；遇到黄色内存状态时先暂停，不能依靠 swap 继续扩大预算。所有折串行执行，禁止同时常驻两个 base encoder。

## 5. 3-fold 晋级 5-fold

当前已有本地参考结果：`neural_3fold_rbt3` 的 OOF strict F1 为 `0.6874480465502909`，`neural_5fold_confirm` 的 OOF strict F1 为 `0.7208976157082748`。这些是历史产物中的 rbt3 结果，不是本次新运行结果。

推荐固定晋级规则：

- **3-fold screen**：按 review group 固定 `KFold(n_splits=3, shuffle=True, random_state=42)`；阈值只由该次 OOF 预测搜索。候选必须高于当前 3-fold 参考，且至少 2/3 折不低于对应参考折，才进入 5-fold。
- **5-fold confirm**：从原始预训练快照重新初始化 5 个 fold，不能复用 3-fold checkpoint 或阈值。用 5-fold OOF 统一搜索阈值后，与 `0.7208976157082748` 的 rbt3 参考比较；若均值没有稳定提升或最差折明显崩溃，保留 rbt3。
- **严格指标**：只看四元组全部字段匹配的 precision/recall/F1；loss、候选数、单折 early-stop 分数不能代替晋级证据。
- **提交边界**：本研究任务不生成新提交、不上传线上评测；没有真实线上返回值时不得填写 leaderboard 分数。
