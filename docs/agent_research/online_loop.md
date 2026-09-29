# 线上候选 manifest / queue 基础

`scripts/submission_manifest.py` 只负责本地候选物料登记。它接收一个已经通过项目 validator 的 CSV、一个本地 OOF JSON 和一个本地 config JSON，写出 `manifest.csv` 与 `queue.json`。脚本不打开浏览器、不上传天池、不发送网络请求，也不读取凭证。

## 创建候选

建议显式指定输出目录和候选目录：

```bash
python3 scripts/submission_manifest.py create \
  --csv artifacts/submissions/Result_group_calibrated.csv \
  --oof-json artifacts/experiments/neural_5fold_confirm/oof_candidates.json \
  --config-json artifacts/experiments/neural_5fold_confirm/threshold.json \
  --output-dir artifacts/online_loop \
  --candidate-dir artifacts/online_loop/candidates
```

也支持不写 `create`、使用前三个位置参数的形式。`--local-score 0.72` 只用于 OOF/config JSON 没有可识别本地分数时的显式补充。脚本优先从 OOF JSON 读取分数，之后从 config JSON 中的 `local_score`、`f1`、`score.f1`、`metrics.f1` 等已知位置读取；两者都没有时会失败，避免生成缺少本地依据的候选。

`--candidate-dir` 传入后，CSV 会被复制到：

```text
<candidate-dir>/<candidate-id>/<原文件名>
```

复制按原始字节进行，并再次核对 SHA256。若省略 `--candidate-dir`，只有在 `<output-dir>/candidates` 已经存在时才会使用它；否则队列直接引用源 CSV。源 CSV 不会被改写。

默认 candidate id 是 CSV SHA256 前 12 位和 config hash 前 12 位组成的稳定名称：

```text
candidate-<csv_sha256[:12]>-<config_hash[:12]>
```

相同 CSV/config 重复执行会更新同一条本地记录。已经被本地回填为 `scored` 的记录会保留其状态和 leaderboard 分数。

## 记录内容

`manifest.csv` 每行对应一个候选，关键列如下：

| 列 | 含义 |
| --- | --- |
| `candidate_id` | 稳定的候选标识 |
| `source_csv` | 原始 CSV 的绝对路径 |
| `candidate_csv` | 主进程应上传并核对的 CSV 路径；有副本时指向副本 |
| `encoding` / `bom` / `utf8_no_bom` | 固定记录为 `UTF-8`、`false`、`true` |
| `rows` | CSV logical records 行数，包含每一行提交记录 |
| `sha256` | `candidate_csv` 应具有的文件 SHA256 |
| `config_hash` | config JSON 解析后按 UTF-8、排序键、紧凑分隔符规范化，再计算的 SHA256 |
| `local_score` | OOF/config 中得到的本地 F1 依据；不是 leaderboard 分数 |
| `leaderboard_score` | 初始为空，只有显式本地回填后才有值 |
| `status` | 创建时固定为 `pending`；显式回填后变为 `scored` |

`queue.json` 的结构是：

```json
{
  "schema_version": 1,
  "queue": [
    {
      "candidate_id": "candidate-...",
      "upload_path": "/absolute/path/to/Result.csv",
      "sha256": "...",
      "config_hash": "...",
      "local_score": 0.72,
      "leaderboard_score": null,
      "status": "pending"
    }
  ]
}
```

队列会保留已完成的条目，`status` 是串行主进程的状态边界。这样可以回看每个候选曾经使用的文件、哈希和本地依据。

## 有浏览器能力的主进程如何串行上传

脚本不会执行以下流程；这些步骤由有浏览器能力并且得到相应授权的主进程逐项完成：

1. 读取 `queue.json`，按文件顺序选取一条 `status=pending` 记录。一次只处理一个候选，不并发上传。
2. 在上传前读取 `upload_path` 的本地字节，确认文件仍为 UTF-8 无 BOM，重新计算 SHA256，并与该条目的 `sha256` 和 `manifest.csv` 对照。任何不一致都停止该条目，不上传。
3. 通过浏览器人工选择这个已经核对过的 `upload_path`，完成一次提交。上传动作不由本脚本触发。
4. 等待页面返回真实结果后，在浏览器页面上核对候选标识、提交时间或页面显示的文件信息。页面没有可核对的文件信息时，以提交前的本地 SHA 记录作为审计依据，不把页面上其他候选的分数归给当前候选。
5. 只有在 leaderboard 分数真实显示并已人工核对后，才使用下面的本地回填命令。没有返回分数时保持 `pending`，不要填写估计值。

回填仍然只改本地两个记录文件：

```bash
python3 scripts/submission_manifest.py backfill \
  --manifest artifacts/online_loop/manifest.csv \
  --queue artifacts/online_loop/queue.json \
  --candidate-id candidate-... \
  --leaderboard-score '<从已核对页面抄录的真实分数>'
```

对应的 Python 本地函数是：

```python
from scripts.submission_manifest import backfill_leaderboard

backfill_leaderboard(
    "artifacts/online_loop/manifest.csv",
    "artifacts/online_loop/queue.json",
    "candidate-...",
    verified_score,
)
```

`backfill_leaderboard` 不会联网，也不会提交；它只在调用者明确提供有限数值后，把分数写入 manifest/queue，并将状态改为 `scored`。它还会先核对 manifest 和 queue 中的 SHA256 是否一致。

## 实验边界

- `local_score` 是本地 OOF 证据，不能当作线上分数，也不能替代线上核对。
- 每次线上候选应有独立的 CSV SHA、config hash 和候选 id，上传前后都核对同一条记录。
- 线上尝试按预先批准的候选顺序串行执行；禁止围绕相邻阈值进行暴力枚举或用线上分数反向搜阈值。
- 线上分数未真实返回前，`leaderboard_score` 必须保持为空；smoke 或离线实验不得伪造 LB 分数。
