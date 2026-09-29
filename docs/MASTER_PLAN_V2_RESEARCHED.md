# 天池评论观点挖掘 — Master Plan V2（深度调研 + 多维审查版）

更新时间：2026-09-28
硬件：Apple M1 Pro / 16GB unified memory；允许竞赛进程动态使用约 8–12GB，但必须以系统 memory pressure 和 available memory 为最终约束。
目标：strict quadruple F1 0.75 为必须认真争取的门槛，0.80+ 为主目标，0.85 为冲刺目标。

## 1. 最终技术决策

不再把 RBT3 当主模型。采用“两条强主线 + 一个轻量守门模型”：

A. 主线 A：**MacBERT-base / Chinese RoBERTa-WWM-ext + implicit-aware relation grid**
- 精确抽取 Aspect/Opinion 原文 span。
- 使用官方 A_start/A_end/O_start/O_end 监督，禁止 `.find()` 代替 offset。
- 加 `[IMPLICIT_A]`、`[IMPLICIT_O]` 两个 pseudo-token，原生支持 Aspect=`_`、Opinion=`_`。
- 使用低秩 biaffine/grid relation head，直接表达 Aspect–Opinion–Category–Polarity 关系，避免单 token 多关系覆盖。
- span boundary + objectiveness + relation/grid + category/polarity 多任务训练。
- constrained top-k beam + NMS 解码。

B. 主线 B：**Qwen3-4B 4-bit MLX QLoRA + RAG**
- Apple Silicon 原生 MLX-LM。
- 只训练 LoRA/QLoRA，不在 16GB M1 Pro 上做 4B 全参微调。
- BM25 Top-20 → reranker Top-5 → few-shot。
- 输出经过严格 substring / enum / dedupe validator。
- 作为长尾、隐式、复杂语义 challenger，与主线 A 做真正异构 ensemble。
- 7B 只在 4-bit smoke benchmark 证明内存稳定后启用；默认不作为第一夜主力。

C. RBT3：只保留为 **5–10 分钟 smoke / regression baseline**，用于验证数据、metric、grid/decoder 改动是否正确，不再继续 seed sweep。

## 2. 立即修复的 P0 问题

1. **Autopilot champion promotion**：challenger 超过 champion 后必须真正更新 champion、test prediction、Result.csv、dashboard、manifest，不能只 record trial。
2. **官方 offset**：数据层保存 A_start/A_end/O_start/O_end；重复词必须使用官方位置。
3. **Opinion=`_`**：当前实现会跳过缺失 Opinion；改为独立 implicit opinion token/pointer。
4. **多关系覆盖**：同一 span/token 可对应多个四元组，不能用单 category/polarity 覆盖。改成 pair/grid 多标签关系表达。
5. **OOF 校准泄漏**：group threshold 必须 leave-one-fold-out；小样本 group shrink 到 global/type threshold。
6. **核心测试缺失**：新增 neural/grid/offset/implicit/multi-relation/promotion/LOFO calibration 单测。
7. **Dashboard**：显示 real champion、calibrated OOF、leaderboard score、current candidate、MPS/系统内存。

## 3. 为什么选择 implicit-aware grid

当前错误分析显示 implicit Aspect 是最大 FN 来源；当前 pointer/token 标签又存在 multi-relation overwrite。现代 ASQP 工作明确把“多个隐式表达”和“元素间关系”作为核心难点。

实现选择：不照搬完整重型论文网络，而做 M1 友好的 compact grid：
- Encoder hidden 768。
- pair projection 128–192。
- `L x L` grid，L 默认 96/128；评论很短，因此开销可控。
- biaffine / conditional layer norm 二选一先做简单版；只有 OOF 证明收益才加 triaffine。
- 对 explicit pair、implicit-A、implicit-O、dual-implicit 统一编码。
- hard negative sampling：同句错误 pairing、近邻 span、相同 opinion 不同 aspect 优先。

## 4. Backbone 顺序

固定同一 folds、同一 decoder、同一 loss，先比较 backbone，不混合其他变化。

1. `hfl/chinese-macbert-base`
2. `hfl/chinese-roberta-wwm-ext`
3. 若本地已有缓存且资源允许，再测 NEZHA-base/large 或 MacBERT-large；large 不进入第一批。

晋级规则：
- 3-fold calibrated strict F1 比当前强基线提升 >= 0.005；或
- F1 基本持平但 error overlap 显著下降、ensemble oracle 明显提升。

只有晋级候选跑 5-fold。

## 5. DAPT / TAPT

强 encoder 结构跑通后做一次高价值自适应预训练，而不是一开始烧时间。

优先级：
1. TAPT：本赛题评论文本继续 MLM。
2. DAPT：若规则允许外部公开电商/化妆品评论，再加入同域无标注评论。
3. 是否使用 test 文本做无监督 TAPT 先过比赛规则 gate；未明确允许时默认只用 train/公开无标注语料。

策略：
- MLM continuation，短序列 96/128。
- 对 MacBERT 和 WWM winner 各只做一次短程适配。
- 用固定 3-fold downstream OOF 判断 DAPT/TAPT 是否真正有效。
- 无提升立即淘汰，不因 pretrain loss 下降而保留。

## 6. Qwen/MLX challenger

来源证据：同赛题 2026 公开方案报告 Qwen3-4B / Qwen2.5-7B / 32B 随规模提升，4B 已可做有效 challenger；RAG 用于长尾和隐式属性。

M1 Pro 16GB 配置：
- MLX-LM 4-bit QLoRA。
- Qwen3-4B-Instruct 优先。
- batch 1 起，gradient accumulation 8–16。
- max sequence 512 足够，本赛题评论短，禁止 32K 上下文浪费内存。
- LoRA rank 8/16；优先 8 或少量层 adapter。
- 只跑 1–2 个高价值配置，不做大网格。

RAG：
- BM25 Top-20。
- reranker Top-5；若 reranker 与 Qwen 同时常驻造成 memory pressure，则串行离线预计算 Top-5。
- prompt 中包含相似训练评论 + 标准四元组。
- 强制 JSON schema；随后再做 exact-substring validator。

Qwen 只在真实 OOF / leaderboard 证明价值时进入 ensemble。

## 7. 数据与 loss

数据层：
- 保留原始 review、四元组、四个官方 offset。
- Aspect/Opinion 缺失统一转 pseudo-token，而不是丢样本。
- 对重复四元组同时保留 raw row count 和 unique metric count，报告两者。
- 固定 folds 文件化，所有模型共用。

Loss：
- span/grid 使用 class-balanced CE 或 focal loss。
- Category+Polarity 可联合成 39 类 relation label，同时保留分解 head 作为辅助 loss。
- implicit loss 单独加权，权重通过 3-fold 小范围搜索。
- hard negative ratio 2–4 倍 positive 起步。

## 8. 解码

禁止只用 argmax endpoint。

使用：
- endpoint top-2 / top-3 constrained beam；
- max span length；
- Aspect 与 Opinion overlap 禁止；
- pair/grid score；
- NMS / duplicate suppression；
- explicit / implicit 独立 threshold；
- category/polarity 只接受合法集合；
- 最终 term 必须是原文 exact substring 或 `_`。

先测候选 oracle recall。若 oracle recall 不足，则优先改 span/grid；若 oracle recall 高而实际 F1 低，则优先改 calibration/ranking。

## 9. 验证体系

三层指标：
1. **3-fold screening**：快速架构筛选。
2. **5-fold confirmation**：只跑 finalists。
3. **Leaderboard**：第二验证信号，不替代本地 OOF。

强制：
- fixed folds。
- fold 内所有词典、RAG index、阈值、统计都只用 train fold。
- group threshold = leave-one-fold-out calibration。
- 小 group threshold 做 shrinkage；禁止几十个自由 threshold 在同一 OOF 自我评分。
- 保存每个模型完整 OOF candidate map，支持离线 ensemble，不重复训练。

## 10. Ensemble

只集成“错误结构不同”的模型：

候选池：
- MacBERT-grid
- RoBERTa-WWM-grid
- Qwen3-4B QLoRA+RAG
- 少量规则/高纯度 memory 仅做 confirmation bonus

先算：
- pairwise prediction overlap
- pairwise TP/FP overlap
- ensemble oracle recall

只有互补性明显才进入 ensemble。

优先：
- rank/score calibration 后 weighted voting；
- 至少两个模型一致的 quadruple 提升置信度；
- Qwen 独有候选必须经过 substring + confidence + RAG consistency gate。

不要把当前低 precision statistical baseline 直接大权重平均进神经模型。

## 11. M1 Pro 动态内存策略

不再固定“只能 8GB”。按实时状态动态利用 8–12GB unified memory。

监控：
- `psutil.virtual_memory().available`
- macOS `memory_pressure`
- `torch.mps.current_allocated_memory()` / `driver_allocated_memory()`
- swap 增长

状态：
- GREEN：pressure normal 且 available >= 5GB → 正常配置，可逐步加 batch/trainable layers。
- YELLOW：available 3–5GB 或 pressure warn → batch 减半、启用 grad accumulation/gradient checkpointing、清 cache。
- RED：available < 3GB、pressure critical 或 swap 快速增长 → 停止当前 trial，落盘 checkpoint，降 batch/max_length/model tier 后重试。

PyTorch/MPS：
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- macOS/PyTorch 支持时优先 bf16；否则 smoke 比较 fp16/fp32。
- 使用 safetensors。
- fold/trial 串行；模型逐个加载、预测落盘、卸载、清 cache。

MLX：Qwen challenger 单独运行，禁止与 PyTorch base 模型同时常驻。

## 12. Autopilot 新状态机

`PRECHECK -> P0_FIX_VERIFY -> RBT3_REGRESSION -> STRONG_ENCODER_SCREEN -> GRID_ABLATION -> DAPT_TAPT -> QWEN_CHALLENGER -> PROMOTE_5FOLD -> ENSEMBLE -> ONLINE_VALIDATE -> FINALIZE`

停止 seed-only evolution。

每个 trial 必须输出：
- config hash
- fold P/R/F1
- calibrated F1
- oracle recall
- implicit/explicit F1
- category F1
- error buckets
- peak RSS/MPS/available memory
- checkpoint / OOF predictions

Promotion 必须影响真实 champion，不只是 Dashboard 排名。

## 13. 线上验证闭环

用户已说明本地 Codex 浏览器会话已登录天池，可进行上传与查看成绩。

实施规则：
1. 本地 executor 启动时先检测是否存在 browser automation 能力和已登录状态。
2. 只允许上传已通过 validator 的 `artifacts/submissions/candidates/<id>/Result.csv`。
3. 上传前记录 SHA256、local OOF、config hash、模型组合。
4. 上传后读取真实 leaderboard 分数并回填 `artifacts/submissions/manifest.csv`。
5. Dashboard 同时显示 local calibrated F1 与 LB F1。
6. 单次只提交一个候选；确认该次成绩对应正确 SHA/config 后才允许下一次。
7. 若浏览器能力暂时不可用，只生成 submission queue，不阻塞训练。

提交选择分层：
- Anchor：MacBERT / WWM / Qwen / ensemble 各 1–2 个结构明显不同候选。
- Calibration：围绕最好候选做少量 high-information threshold/decoder 调整。
- Challenger：只提交 OOF 或互补性有证据的结构变化。
- Ensemble：只提交 OOF ensemble 有增益的组合。

不要用 999 次做近似阈值暴力搜索。Leaderboard 是第二目标：线上提高但 OOF 明显下降的模型标记 `leaderboard_only`，必须额外 fold/seed 验证。

## 14. 线上反馈决策规则

维护指标：`local_f1, lb_f1, precision, recall, model_family, decoder, implicit_strategy, hash`。

优先保留：
- local 与 LB 同升；
- local 持平、LB 稳定提升且模型结构不同；
- ensemble local 提升且 LB 验证。

警报：
- 连续多个非常相似 config 只有 LB 波动，没有 OOF 解释 → 停止该方向。
- 某 threshold 仅在线上单点提升 → 不立即推广。
- 线上/线下排序长期反向 → 检查 CV 分布、重复评论、train/test shift，再重新设计 split。

## 15. 实验顺序（最佳性价比）

### Phase 0 — 立即
- 等当前 fold 安全落盘后停止 seed-only queue。
- 快照现有 0.7209 champion 和所有产物。

### Phase 1 — P0 修复
- offset / Opinion `_` / multi-relation / promotion / LOFO / tests。
- RBT3 只跑一次 3-fold regression，确认修复方向。

### Phase 2 — 强 encoder
- MacBERT-base grid 3-fold。
- RoBERTa-WWM-ext grid 3-fold。
- 对 winner 做 grid/beam/implicit loss 小范围 ablation。

### Phase 3 — adaptive pretraining
- 对 strong winner 做一次 TAPT/DAPT。
- 3-fold 验证，无提升即丢弃。

### Phase 4 — Qwen challenger
- Qwen3-4B MLX 4-bit QLoRA。
- 无 RAG / 有 RAG 两个版本最多各 1 个主配置。

### Phase 5 — 5-fold
- 最强 1–2 个 encoder + 最有价值 Qwen（若 OOF 可公平验证）进入确认。

### Phase 6 — ensemble
- 基于 OOF complementarity 搜少量权重/一致性规则。

### Phase 7 — online loop
- 自动上传高信息量候选，回填 LB，决定下一 challenger。

### Phase 8 — final
- 选 local/LB 都稳健的 champion。
- 生成最终 `Result.csv`、SHA256、manifest、final dashboard、reproduce command。

## 16. 分数预期与决策线

不要保证隐藏榜成绩，但设明确决策：
- <0.73：仍有结构/数据实现问题，禁止扩大模型规模。
- 0.73–0.75：重点 implicit/grid/beam/calibration。
- >=0.75：强 encoder 方向成立，进入 DAPT/异构 ensemble。
- >=0.78：重点 5-fold、MacBERT/WWM ensemble、Qwen challenger。
- >=0.80：进入 LB 精细验证和稳健性阶段。
- 0.85：冲刺，不承诺；需要强架构 + 数据适配 + 真正异构 ensemble + 良好 LB 泛化共同成立。

参考不是目标保证：历史 OpinioNet README 报告早期单模型 CV 0.7868、CV/ensemble 0.8109/0.8224；2026 同赛题公开仓库报告 Qwen 路线最高约 0.81。两者说明 0.80 级别值得认真争取，但数据版本/赛季/实现可能不同，不能直接当当前隐藏榜预期。

## 17. 交付物

- `artifacts/submissions/Result.csv`
- `artifacts/submissions/manifest.csv`
- `artifacts/submissions/candidates/<submission_id>/Result.csv`
- `artifacts/reports/experiments.csv`
- `artifacts/reports/oof_metrics.json`
- `artifacts/reports/error_analysis_v2.json`
- `artifacts/reports/oracle_ceiling.json`
- `artifacts/reports/final_report.json`
- `artifacts/reports/final_dashboard.html`
- 固定 folds、配置、模型 checkpoint、OOF predictions、online score history

最终 CSV：UTF-8 无 BOM、无表头、每行 5 列、全 test id、字段合法、exact substring、无重复、<100MB、SHA256 记录。

## 18. 参考资料

- OpinioNet Top3: https://github.com/eguilg/OpinioNet
- 同赛题 2026 Qwen/RAG 方案: https://github.com/9yly/TianChiTrack6_ABSA
- 天池 2025 BERT/LLM 方案: https://tianchi.aliyun.com/forum/post/940230
- 天池 2026 Qwen+RAG: https://tianchi.aliyun.com/forum/post/949436
- MacBERT: https://aclanthology.org/2020.findings-emnlp.58/
- DAPT/TAPT: https://aclanthology.org/2020.acl-main.740/
- UGTS implicit grid: https://aclanthology.org/2025.coling-main.269/
- CACA implicit cross-attention: https://aclanthology.org/2025.coling-main.635/
- Span + Table Filling: https://aclanthology.org/2024.lrec-main.742/
- Implicit ASQP contrastive learning: https://aclanthology.org/2024.findings-emnlp.453/
- ASQP pseudo-label self-training: https://aclanthology.org/2024.acl-long.640/
- MLX-LM QLoRA: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- PyTorch MPS: https://docs.pytorch.org/docs/main/notes/mps.html
- Transformers Apple Silicon: https://huggingface.co/docs/transformers/perf_train_special

## 19. 执行原则

- 先修 P0，再扩模型。
- 所有新模型先 3-fold，再 5-fold。
- 所有线上提交都有本地证据和 SHA256。
- 不重复 seed sweep。
- 不以单个 leaderboard spike 作为最终结论。
- 不因“已有合法 CSV”停止；达到时间预算时必须保留当前最优可提交物。
- Browser automation 若可用，则把线上提交纳入 Autopilot；若不可用，则自动排队并继续训练，不阻塞实验。
