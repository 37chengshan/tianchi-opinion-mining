# M1 Pro 16GB / 竞赛进程约 8GB：夜间自主进化方案

## 1. 目标与边界

硬件是 M1 Pro + 16GB unified memory，但竞赛进程按 **8GB 总预算**设计，给 macOS、IDE、浏览器和本地 AI 留出空间。

工程目标是在 12–24 小时墙钟预算内持续寻找更高的 **无泄漏 strict quadruple OOF F1**，并以真实天池 leaderboard 反馈做第二信号。分数目标分三级：**0.75 是必须认真争取的最低打榜目标，0.80+ 是主目标，0.85 是冲刺目标**。在达到 0.75 之前，Autopilot 不得因为“已有合法 CSV”就停止优化；达到 0.75 后继续切换到更强的关系建模、融合和 ensemble 搜索，直至时间预算/收敛条件触发。最终自动选出验证最可靠且 leaderboard 表现最好的候选并生成合法 `Result.csv`。

硬约束：

- 训练进程目标 RSS ≤ 6.5GB；7GB 进入降级；7.5GB 立即终止当前 trial 并回退。
- MPS 上同一时刻只允许一个 Transformer 常驻。
- 所有 fold 串行训练；一折结束立即保存需要的权重/预测并释放模型、optimizer、dataset cache。
- 不并行跑神经训练 trial；CPU 规则/统计任务也只允许低并发。
- 所有最终比较以 strict quadruple OOF F1 为准。

## 2. MPS 训练策略

MPS 使用 CPU/GPU 共享统一内存，因此不要照搬 CUDA 多模型并行方案。

默认：

- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- PRECHECK 检测 macOS / PyTorch / Transformers 版本；macOS 14+ 且实际 smoke test 稳定时优先 `bf16`，否则比较 `fp16` 与 `fp32` 的稳定性/速度后选择
- 固定/少量 sequence buckets，避免大量动态 shape；这是夜间防止 MPS graph cache 持续增长的硬要求
- `max_length=128` 起步；只有截断率明显影响结果才提升到 160/192
- micro batch 从 2 起；内存压力则降到 1
- gradient accumulation 组成 effective batch 8–16
- gradient checkpointing 只在 base encoder 超预算时开启，因为它省内存但耗时
- 周期性 `torch.mps.empty_cache()`；若 PyTorch >=2.13 且当前 Transformers 支持，则启用 `torch_empty_cache_steps`，先以 20–50 step 周期 smoke benchmark，在内存稳定与吞吐之间选值，不默认每 step 清理
- 若 PyTorch >=2.9，优先使用 safetensors 权重路径，减少 MPS 模型加载时的中间内存峰值

禁止：

- 5 个 fold 模型同时加载做 ensemble
- 同时加载两个 BERT-base backbone
- `device_map=auto` 幻想把 MPS 模型自动分层 offload
- 大量动态 padding shape 长时间训练

## 3. 模型架构：一个共享 Encoder，多任务 heads

主模型只保留一个中文 encoder，挂多个轻量 head：

1. Aspect span head
2. Opinion span head
3. Pair relation head
4. Joint `(Category, Polarity)` head
5. Implicit-aspect head：判断 `(_, opinion)`

不要为 span、pair、category、polarity 各训练一套独立 Transformer。Pair/classification head 直接消费共享 encoder 的 span pooling 表示。

### Backbone 梯度

**Tier S（搜索默认）**：小型中文 encoder（3–6 层、hidden 384–512 级别优先）。

**Tier B（最终候选）**：MacBERT/RoBERTa-WWM base 级，只在 Tier S 证明神经路线确实带来明显增益后才进入。

如果 base 模型进入最终候选，优先冻结底部若干层或使用 LoRA/adapter，只训练上层 + heads，以降低 optimizer state 和反向图内存。

## 4. 规则/检索层：CPU 高性价比

保留并强化：

- OpinionTerm purity
- `(AspectTerm, OpinionTerm) -> (Category, Polarity)`
- 隐式 Aspect 映射
- 完整四元组重复记忆
- char TF-IDF 检索
- 同分句、字符距离、标点距离特征

这些任务放 CPU，内存占用低，并给神经候选加 confirmation bonus。

## 5. 验证设计优化

### 5.1 搜索阶段：3-fold

绝大多数实验只跑固定 3-fold screening CV，用来筛架构、损失、融合和阈值策略。

只有相对当前 champion 达到晋级门槛的配置进入 5-fold confirmation。建议晋级条件：

- 3-fold strict F1 比 champion screening F1 提升 ≥ 0.004；或
- F1 相近但 Precision/Recall 明显更适合融合；或
- 属于结构显著不同、有 ensemble 多样性的候选。

### 5.2 最终阶段：5-fold

对少量 champion/challenger 跑完整 5-fold OOF，fold assignment 固定保存。

### 5.3 阈值校准不得自我评分

使用 leave-one-fold-out calibration：对第 k 个 fold 的阈值，只用其余 folds 的 OOF 候选搜索，再将阈值应用到第 k fold。最终报告拼接后的 calibrated OOF strict F1。

## 6. 自主实验 Autopilot

新增 `scripts/autopilot.py`，以 wall-clock budget 运转。

输入：

- `--hours 12` 或 `--hours 24`
- `--memory-budget-gb 8`
- `--seed 42`
- `--mode mps`

状态机：

1. PRECHECK：环境、数据、测试、MPS、磁盘、内存检查
2. BASELINE：规则/检索 baseline + 3fold
3. SEARCH：低成本神经/融合 trial
4. PROMOTE：优胜候选进入完整 5fold
5. ENSEMBLE：仅对互补 champion 尝试融合
6. FINALIZE：选定 final config，测试集推理、validator、hash
7. REPORT：生成最终网页报告

每个 trial 必须记录：

- config hash
- start/end time
- peak RSS / MPS memory（能采到则记录）
- fold metrics
- strict F1 / Precision / Recall
- wall-clock
- checkpoint path
- status：completed / pruned / oom / failed

## 7. 自动淘汰与早停

- 第 1 fold 明显低于 champion 下界时可 prune
- span exact F1 大幅低于已知候选时无需继续完整 pair/classification
- 连续若干 trial 无提升，切换搜索维度，不重复微小参数
- OOM：batch 2→1；仍超预算则 max_length 128→96；仍失败则降级 backbone
- MPS unsupported op：允许 CPU fallback；若速度严重恶化，换实现而不是无限等待
- trial 崩溃最多重试 1 次，之后标记失败继续队列

## 8. 夜间搜索空间

不做大规模 Optuna 笛卡尔积暴搜，采用 successive-halving：

- encoder tier：small-3L / small-6L / base-finalist
- max_length：96 / 128 / 160
- trainable encoder layers：top2 / top4 / all-small
- LR：1e-5 / 2e-5 / 3e-5，heads 可单独更高
- loss weighting：span-heavy / balanced / relation-heavy
- implicit threshold：3 个候选
- explicit threshold：3 个候选
- rule confirmation bonus：0 / small / medium
- retrieval cutoff：2–3 个候选

## 9. 进度网页：训练前必须先启动

目录：`dashboard/`。

训练前执行顺序：

1. 生成 `dashboard/state.json`
2. 启动本地 HTTP server
3. 自动打开 `http://127.0.0.1:8765/`
4. 页面显示 PRECHECK 且可刷新后，才允许启动 Autopilot

训练中页面每 2–5 秒轮询：

- 当前阶段 / trial / fold / epoch
- 已运行时间 / 剩余预算
- 当前 strict F1、P、R
- champion F1
- 实验排行榜
- fold 曲线
- loss 曲线
- RSS/MPS 内存
- trial 成功/剪枝/OOM/失败计数
- 当前候选配置
- 最近事件日志

训练程序原子写 `dashboard/state.json` 和 `dashboard/history.json`，网页不直接读取半写入 JSONL。

## 10. 最终网页

同一个 dashboard 在 `status=completed` 后切换 Final 模式，展示：

- champion 模型结构和 config hash
- 3-fold screening 与 5-fold calibrated OOF 对比
- P/R/F1
- 每 fold 分数
- Category / Polarity / implicit vs explicit 诊断
- 最佳与次佳实验对比
- 整夜/全天实验时间线
- 资源占用曲线
- 最终提交行数、空 id 数、预测四元组数
- Result.csv validator 检查项
- SHA256
- reproduction command

同时导出静态快照 `artifacts/reports/final_dashboard.html`。

## 11. 文件设计

```text
dashboard/
  index.html
  state.json
  history.json
scripts/
  serve_dashboard.py
  autopilot.py
  resource_guard.py
  experiment_runner.py
  export_final_dashboard.py
artifacts/
  experiments/
  models/
  oof/
  reports/
  submissions/
```

每个 trial 写自己的目录，禁止互相覆盖。

## 12. Autopilot 停止条件

任一条件触发即可 FINALIZE：

- 到达 wall-clock budget（默认 12h，可设 24h）
- 剩余时间不足以完成任何有意义的已排队 trial
- 连续 8 个有效 trial 没有 ≥0.001 strict F1 改善，且没有未试的结构级候选
- 用户手动创建 `artifacts/STOP`
- 系统持续内存压力或磁盘不足

这里的“完美”定义为：**在给定硬件、时间、内存和无泄漏验证约束下，Autopilot 已穷尽计划中的高价值候选并选出当前最可靠 champion**；不宣称未知隐藏测试集的理论最优。

## 13. 推荐执行优先级

1. Dashboard + resource guard + state protocol
2. evaluator/submission tests 全绿
3. 数据完整性 smoke test
4. 规则/检索 3-fold baseline
5. 共享 small encoder 多任务模型 3-fold
6. 神经 + 规则融合
7. champion 5-fold confirmation + calibrated threshold
8. 时间充足再让 base encoder challenger 晋级
9. 少量 ensemble
10. Finalize Result.csv + final dashboard

## 14. 资源默认参数

```text
memory_budget_gb = 8.0
soft_rss_limit_gb = 6.5
hard_rss_limit_gb = 7.5
max_concurrent_neural_trials = 1
max_length = 128
micro_batch = 2
gradient_accumulation = 4 or 8
num_workers = 0 or 1
search_cv_folds = 3
final_cv_folds = 5
dashboard_port = 8765
dashboard_refresh_seconds = 3
```

PRECHECK 可以根据 2–3 分钟 smoke benchmark 自动降级参数，但不得自动扩大到突破 8GB 预算。

## 15. 分数阶梯与错误驱动进化

Autopilot 不按“多跑几个 epoch”定义进化，而按当前 strict F1 所处区间切换优化重点：

- **< 0.70**：优先修正数据/边界/配对/隐式属性等结构性问题，禁止沉迷微调阈值。
- **0.70–0.75**：重点做 relation pairing、隐式 Aspect、Category/Polarity 联合分类和高纯度规则补召回，目标尽快跨过 0.75。
- **0.75–0.80**：进入强模型阶段；优化共享 encoder、多任务 loss、hard negative、显式/隐式双阈值、规则确认 bonus。
- **0.80–0.85**：只保留高价值 challenger；允许 base encoder finalist、不同 seed/结构的少量 ensemble、类别分层阈值、错误簇定向修正。
- **>= 0.85**：主要做稳健性确认、Precision 防守、cross-fold/cross-seed 验证和最终提交选择，避免为了极小公开榜提升破坏泛化。

每个阶段都要输出错误分解：span miss、span boundary、wrong pair、wrong category、wrong polarity、implicit miss、over-generation。下一轮实验优先攻击占比最高的错误类型。

## 16. 999 次提交的 leaderboard 反馈协议

当前页面显示剩余提交次数为 **999**。这意味着可以把 leaderboard 作为有价值的第二验证信号，但绝不能无脑穷举 999 个近似 CSV。

所有准备上传的候选必须先写入 `artifacts/submissions/manifest.csv`，字段至少包括：

`submission_id, experiment_id, config_hash, local_oof_f1, precision, recall, threshold_explicit, threshold_implicit, blend, csv_path, csv_sha256, csv_bytes, leaderboard_score, submitted_at, notes`

建议分批：

1. **Anchor batch**：3–5 个结构明显不同的 baseline / neural / fusion 候选，确认本地 OOF 与线上排序相关性。
2. **Calibration batch**：围绕最佳候选做 10–20 个高信息量显式/隐式阈值与候选数量调整，不做极细粒度暴搜。
3. **Model batch**：只提交 OOF 有合理依据的 pair/joint-class/encoder challenger。
4. **Ensemble batch**：提交少量真正互补的模型组合，而不是相同模型近似权重。
5. **Error-correction batch**：根据线上与 OOF 差异，针对 implicit、长尾 Category、Precision/Recall 失衡做定向修正。

即使有 999 次，也建议优先使用前 50–150 次高信息量提交，并保留大量余量给最后阶段。若平台还有单日/频率限制，以平台实时规则为准。

### Leaderboard 防过拟合

候选晋级 final 至少满足以下之一：

- calibrated OOF 同步提升；
- OOF 基本持平但模型错误分布明显互补，ensemble 后 OOF 提升；
- leaderboard 提升在至少两个结构不同的相邻候选中重复出现，并且没有明显破坏 OOF。

如果某配置线上暴涨但 OOF 明显下降，将其标记为 `leaderboard_only`，不能直接成为 final champion；必须再做独立 fold/seed 验证。

## 17. 最终 CSV 硬校验

最终上传物只允许是 `Result.csv`。在 FINALIZE 阶段新增硬门：

- 文件扩展名 `.csv`
- UTF-8 无 BOM
- 无表头
- 每行 5 列
- 全部测试 id 覆盖且升序
- 非 `_` Aspect/Opinion 为原文精确子串
- Category / Polarity 合法
- 同 id 无重复四元组
- 空预测 id 仅一行 `_,_,_,_`
- **文件大小必须 < 100,000,000 bytes**（并同时记录 MiB）
- 计算 SHA256

若任何一项失败，不允许进入上传队列。最终 Dashboard 必须显示 CSV 字节数、MiB、validator 结果和 SHA256。
