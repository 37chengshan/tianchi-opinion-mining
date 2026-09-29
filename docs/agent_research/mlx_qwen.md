# Qwen3-4B MLX challenger 安全入口

更新时间：2026-09-28

这份记录只准备 challenger 的可复现入口和后续接入边界。当前任务没有下载模型、安装依赖、启动生成或训练，也没有创建 MLX/MPS 张量。

## 当前状态

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 安全 smoke 入口 | 已完成 | [`scripts/mlx_qwen_smoke.py`](../../scripts/mlx_qwen_smoke.py) |
| 本机内存门槛探测 | 已完成 | 入口读取 macOS `hw.memsize` 与 `vm_stat`；验证快照在 2.70–3.02 GiB 间波动 |
| `mlx` / `mlx_lm` import | 未执行 | 本次环境预检发现两者均未提供 |
| Qwen3-4B 4bit 本地缓存 | 未发现 | 入口只检查本地文件，不访问 Hub |
| 生成、LoRA、QLoRA、RAG 端到端 | 未验证 | 本 bounded 任务明确禁止执行 |

本机预检的硬件总内存为 16 GiB，验证期间可用内存近似值在 2.70–3.02 GiB 间波动。16 GiB 是运行门槛，不代表 Qwen 训练已经可运行；统一内存还要留给系统、数据处理和 MLX 工作区。

## 安全入口

从仓库根目录执行：

```bash
cd /Users/cc/code/tianchi-opinion-mining
python3 scripts/mlx_qwen_smoke.py
```

也可以把已经存在的本地模型目录明确传入：

```bash
python3 scripts/mlx_qwen_smoke.py \
  --model mlx-community/Qwen3-4B-4bit \
  --model-cache /absolute/path/to/already-cached/Qwen3-4B-4bit
```

入口的检查顺序固定为：

1. 用 `importlib.util.find_spec` 检查 `mlx` 与 `mlx_lm`，不导入模块。
2. 检查显式模型目录、项目目录和现有 Hugging Face cache 中的 `config.json`、tokenizer 文件与 safetensors 文件，不读取权重内容，不发网络请求。
3. 读取 macOS 总内存和可用内存近似值。
4. 只有前面全部通过时，才 import `mlx`、`mlx_lm`、`mlx_lm.lora`，构造 LoRA parser，并解析未来配置：batch 1、gradient accumulation 8、max sequence 512、4 层与 gradient checkpointing。

它不会调用 `mlx_lm.load`、`generate`、`lora --train`、`convert`、`fuse`，不会创建 MLX tensor，也不会占用 MPS。输出是一行 JSON，便于调度器直接消费：

- `status=queue`：缺依赖、未缓存或无法测量内存；等待外部准备后重跑。
- `status=skip`：不是 Apple Silicon、总内存低于 16 GiB、可用内存低于 3 GiB，或 import/config smoke 失败。
- `status=ready`：只表示 import/config smoke 通过，不表示权重加载、生成或训练通过。

## 官方接口与已验证边界

官方 MLX-LM LoRA 文档规定：主入口是 `mlx_lm.lora`；`--model` 可以是 Hugging Face repo 或本地转换目录；量化模型走 QLoRA；本地训练数据目录需要 `train.jsonl`，可选 `valid.jsonl`；支持 `chat`、`completions` 与 `text` JSONL 格式。[官方 LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

当前 `main` 分支的 LoRA parser 还提供 `--batch-size`、`--grad-accumulation-steps`、`--max-seq-length`、`--num-layers` 与 `--grad-checkpoint`；入口只解析这些选项，不执行它们。[官方 `lora.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py)

本记录使用 `mlx-community/Qwen3-4B-4bit` 作为默认模型名。模型卡将它标为 Qwen3-4B 的 4-bit MLX safetensors 模型，页面显示权重约 2.26 GB；磁盘大小不能直接当成训练峰值内存。[模型卡](https://huggingface.co/mlx-community/Qwen3-4B-4bit)

以下内容属于**建议配置**，不是本机实测结果：

- 4-bit QLoRA，`--batch-size 1`。
- `--grad-accumulation-steps 8` 起步；必要时只在同一配置上升到 16。有效 batch 分别为 8 与 16，但 accumulation 不会降低单个 micro batch 的激活峰值。
- `--max-seq-length 512`，评论任务不使用 32K 上下文。
- `--num-layers 4`、`--grad-checkpoint` 作为 16 GiB 机器的保守起点。
- LoRA rank 8 作为起点；如果用 YAML 配置，保持 `lora_parameters.rank: 8`，不要在首轮扩大网格。

## 数据接入约定

仓库当前原始数据是 CSV：

```text
artifacts/data/train/TRAIN/Train_reviews.csv
artifacts/data/train/TRAIN/Train_labels.csv
artifacts/data/test/TEST/Test_reviews.csv
```

训练集有 3229 条评论，测试集有 2237 条评论，任务要求输出完整四元组。MLX-LM 不会理解本仓库的 CSV 标签，因此需要先按固定 fold 转成 `chat` JSONL。建议 assistant 只输出 JSON 数组，每个对象固定为：

```json
[{"AspectTerm":"包装","OpinionTerm":"太随便了","Category":"包装","Polarity":"负面"}]
```

训练 prompt 必须要求：四个字段逐字来自评论或使用 `_`；只输出合法 JSON；不得输出解释性文字。推理结果仍要回到现有严格四元组校验、原文子串校验和提交格式校验，不能把生成文本直接写进 `Result.csv`。

正式 OOF 时，每个 fold 的 `train.jsonl` 只能由 train split 生成，`valid.jsonl` 只能由 valid split 生成。词典、检索库、RAG 上下文和阈值都必须在 fold 内隔离。

## RAG 串行预计算

Qwen 与检索器不要同时常驻。先用 CPU 对每个 fold 预计算候选上下文，落盘后退出检索进程，再单独启动 MLX QLoRA。当前仓库已有字符 TF-IDF 检索基线，所以不需要新增依赖即可产生确定性的 `top20 -> top5` 文件；这不是已经验证的语义 reranker。

下面命令是后续获准运行时的可复现预计算命令。它只读已有 CSV 和 Python 包，按 fold 排除自身样本，valid 只从 train split 检索；本任务没有执行它：

```bash
cd /Users/cc/code/tianchi-opinion-mining
export PYTHONPATH="$PWD/src"
mkdir -p artifacts/mlx_qwen_data/fold_1

python3 - artifacts/mlx_qwen_data/fold_1 <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from opinion_mining.data import load_train_data
from opinion_mining.folds import fixed_splits

out = Path(sys.argv[1])
rows = load_train_data(
    "artifacts/data/train/TRAIN/Train_reviews.csv",
    "artifacts/data/train/TRAIN/Train_labels.csv",
)
train_indices, valid_indices = fixed_splits(rows, n_splits=3, seed=42)[0]
train_rows = [rows[i] for i in train_indices]
queries = {
    "train": [rows[i] for i in train_indices],
    "valid": [rows[i] for i in valid_indices],
}

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=1,
    max_features=30000,
    sublinear_tf=True,
    norm="l2",
)
matrix = vectorizer.fit_transform([row.text for row in train_rows])

def quad_dict(quad):
    return {
        "AspectTerm": quad.aspect,
        "OpinionTerm": quad.opinion,
        "Category": quad.category,
        "Polarity": quad.polarity,
    }

for split, split_rows in queries.items():
    top20_path = out / ("rag_top20_" + split + ".jsonl")
    top5_path = out / ("rag_top5_" + split + ".jsonl")
    with top20_path.open("w", encoding="utf-8") as top20, top5_path.open("w", encoding="utf-8") as top5:
        for query in split_rows:
            scores = cosine_similarity(
                vectorizer.transform([query.text]), matrix
            ).ravel()
            hits = []
            for index in np.argsort(-scores):
                source = train_rows[int(index)]
                if source.id == query.id or float(scores[int(index)]) <= 0.10:
                    continue
                hits.append({
                    "id": source.id,
                    "score": round(float(scores[int(index)]), 6),
                    "review": source.text,
                    "quadruples": [
                        quad_dict(q)
                        for q in sorted(source.labels)
                    ],
                })
                if len(hits) == 20:
                    break
            record = {"id": query.id, "hits": hits}
            line = json.dumps(record, ensure_ascii=False)
            top20.write(line + "\n")
            top5.write(json.dumps({"id": query.id, "hits": hits[:5]}, ensure_ascii=False) + "\n")
PY
```

把 `rag_top5_*.jsonl` 的 `review` 与 `quadruples` 序列化到 user prompt 后再写 `chat` JSONL。不要把 valid 标签放进 prompt；RAG 只提供 train split 的示例。BM25 或独立 reranker 属于下一步建议，若启用也必须先离线完成并退出，再运行 Qwen。

## 后续获准训练命令

只有以下条件全部满足后才允许由操作者单独启动；本任务不会执行：

1. `python3 scripts/mlx_qwen_smoke.py` 返回 `status=ready`。
2. 模型目录已经存在且是目标 4-bit MLX 模型；禁止让 `mlx_lm` 通过 repo 名称触发隐式下载。
3. `artifacts/mlx_qwen_data/fold_1/train.jsonl` 与可选的 `valid.jsonl` 已由 fold-safe 流程生成。
4. 可用统一内存至少 5 GiB，`memory_pressure` 为 normal，且没有快速增长的 swap；运行期间只保留一个 MLX/PyTorch 模型进程。

受控的第一轮配置如下，`--iters 120` 是有界 pilot，不是长训；只有 OOF 有证据时才考虑增加迭代数：

```bash
cd /Users/cc/code/tianchi-opinion-mining
MODEL_DIR="/absolute/path/to/already-cached/Qwen3-4B-4bit"
DATA_DIR="$PWD/artifacts/mlx_qwen_data/fold_1"
ADAPTER_DIR="$PWD/artifacts/experiments/qwen3_mlx/fold_1/adapter"

python3 scripts/mlx_qwen_smoke.py \
  --model mlx-community/Qwen3-4B-4bit \
  --model-cache "$MODEL_DIR"

mlx_lm.lora \
  --model "$MODEL_DIR" \
  --train \
  --fine-tune-type lora \
  --data "$DATA_DIR" \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --max-seq-length 512 \
  --num-layers 4 \
  --grad-checkpoint \
  --mask-prompt \
  --iters 120 \
  --val-batches 10 \
  --steps-per-eval 60 \
  --adapter-path "$ADAPTER_DIR" \
  --save-every 60 \
  --seed 42
```

`--mask-prompt` 只在 `chat` / `completion` 数据格式下使用，并让 loss 聚焦 assistant completion；如果实际数据 builder 不是这两种格式，应先删掉该 flag 并重新做短 pilot。训练命令中的 `--train`、`--model` 加载和所有 MPS 使用都属于未来显式批准后的动作。

## 资源门槛与停止线

| 状态 | 可用统一内存 | 行为 |
| --- | ---: | --- |
| GREEN | `>= 5 GiB` | 才可考虑 batch 1 的有界 pilot；Qwen 单独运行 |
| YELLOW | `3–5 GiB` | 只做 import/config smoke；若要 pilot，先释放其他模型并重新预检 |
| RED | `< 3 GiB` | `skip`，不启动 MLX；swap 快速增长或 pressure critical 也立即停止 |

这些数值是针对当前 16 GiB M1 Pro 的运行建议，不是 MLX-LM 官方保证，也没有在本机完成训练验证。官方文档同样建议用量化、较小 batch、gradient accumulation、减少可训练层数和 gradient checkpointing 来降低 LoRA 内存；其示例是在 32 GB M1 Max 上运行，不能直接外推到本机。[官方 Memory Issues 说明](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#memory-issues)

## 验证记录

本 bounded 任务中允许且已使用的验证命令：

```bash
cd /Users/cc/code/tianchi-opinion-mining
python3 - <<'PY'
import ast
from pathlib import Path
ast.parse(Path("scripts/mlx_qwen_smoke.py").read_text(encoding="utf-8"))
print("AST OK")
PY
python3 scripts/mlx_qwen_smoke.py
```

两次实际运行都保持了安全边界：可用内存约 2.70 GiB 时输出 `status=skip`、`reason=available_memory_below_red_gate`；最后一次约 3.02 GiB 时输出 `status=queue`、`reason=missing_dependency`，并在检查字段中记录 `mlx` / `mlx_lm` 缺失与目标模型未缓存。两次都没有进行 import/config smoke，更不代表 challenger 已可训练。
