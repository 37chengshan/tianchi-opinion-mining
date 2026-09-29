# 天池评论观点挖掘 Locked Final Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 冻结一条可持续执行到最终提交的主线，不再因为单次实验波动切换方向；目标为严格四元组 F1 0.80+，0.85 作为冲刺。

**Architecture:** 主模型固定为 Chinese RoBERTa-WWM-ext / MacBERT-base + implicit-aware relation grid V2；Qwen3-4B 仅作为异构 verifier/teacher/challenger；DAPT/TAPT 和高置信伪标签属于固定增强阶段；最终只对通过统一 3-fold gate 的模型做 5-fold、ensemble 和线上验证。RBT3 只保留 Legacy baseline，不再重新训练。

**Tech Stack:** PyTorch MPS, Hugging Face Transformers, hfl/chinese-roberta-wwm-ext, hfl/chinese-macbert-base, MLX-LM(Qwen3-4B), local dashboard, strict OOF evaluator.

**Spec:** `docs/MASTER_PLAN_V2_RESEARCHED.md`

## Global Constraints

- Apple M1 Pro / 16GB unified memory；竞赛进程动态使用 8–12GB，以 memory pressure / available memory 为最终约束。
- 所有模型共用固定 folds、官方 offsets、同一 strict quadruple evaluator。
- 禁止 `.find()` 替代官方 offset；Aspect/Opinion `_` 必须原生建模。
- 同一 span 允许多个四元组关系；不得覆盖。
- 3-fold 只用于 screening；5-fold 只用于 finalists。
- 5-fold 分数必须 cross-fit/LOFO 校准，不能同 OOF 调阈值再自评。
- 每个实验必须保存 OOF candidate map、config hash、资源快照、错误桶和 Result 候选。
- RBT3 仅为 Legacy baseline，禁止再进入正式训练队列。
- leaderboard 只做第二验证信号，禁止阈值暴搜。

## Review Focus

1. implicit Aspect / implicit Opinion / dual-implicit 是否被真实预测，而不是训练时有标签、解码时丢掉。
2. 同一个 opinion 对多个 aspect/category 的多关系是否完整保留。
3. exact substring + official offset 是否在重复词情况下仍正确。
4. OOF threshold / pseudo-label scorer / RAG index 是否严格 fold-safe。
5. RED memory pressure 时是否安全保存并重启 fold，而不是在 optimizer 中途改变 trainable layers。

---

## 冻结后的总路线

`WWM 当前 3-fold完成 → Error Audit → Grid V2 → WWM/MacBERT V2 3-fold → TAPT/DAPT → Pseudo-label → Qwen verifier/teacher → Finalists 5-fold LOFO → Heterogeneous Ensemble → Online Validate → Finalize`

任何未满足下文“改道条件”的实验结果，都不得改变这条顺序。

### 2026-09-28 implementation checkpoint

- Task 1/2 已完成：WWM V1 3-fold strict OOF = 0.625203；raw candidate oracle = 0.878770，pair oracle = 0.916315，implicit-pair oracle = 0.919299。
- Grid V2 代码已实现并通过全量测试（80 passed）：full-span pair refinement、pair-validity/no-relation、balanced structural hard negatives、pair ranking、contextual implicit A/O、objectiveness-aware span decode、四状态全局 F1 calibration、wide span candidate budget (16/12) + relation top-1、训练 target 精确去重。
- V1 最大 FP 为 implicit-A；V2 的 state-confusion negatives 与 pair-validity 专门针对这一根因。四状态 calibration 单独作用于 V1 candidates 仅把 WWM 0.625203 提到约 0.626020，因此后续收益必须由 V2 重训验证，禁止继续阈值微调。
- 正式 fold 固定训练到预设 epoch；fold-local calibrated F1 仅用于 Dashboard display，不再用于 checkpoint/epoch selection。
- 下一步固定为 Task 4 的第一步：WWM-V2 3-fold screening。未得到该 OOF 前，不再增加新 loss/head/backbone。

### Task 1: 完成当前 WWM V1 作为一次性基准

**Files:**
- Read/produce: `artifacts/experiments/grid_wwm_*`
- Modify: `dashboard/archive.json`
- Test: existing grid/runtime tests

**Interfaces:**
- Consumes: 当前正在运行的 WWM 3-fold job。
- Produces: `wwm_v1_oof.json`, OOF candidates, error buckets, runtime report。

- [ ] 让当前 WWM 3-fold 自然完成，不中断、不重启。
- [ ] 只记录其 calibrated 3-fold OOF，不据此直接跑 5-fold。
- [ ] 写入 archive，标签为 `screening_v1`。
- [ ] 把 MacBERT V1 / WWM V1 放到同一 error-audit 脚本比较。

**Gate:** 本任务完成后，V1 backbone screening 永久结束；禁止继续 seed sweep、禁止 RBT3 重跑。

### Task 2: 建立最终错误审计与候选上限诊断

**Files:**
- Create/modify: `scripts/error_analysis_grid_v2.py`
- Create/modify: `scripts/diagnose_candidate_ceiling.py`
- Test: `tests/test_error_analysis_grid_v2.py`

**Interfaces:**
- Consumes: MacBERT V1 / WWM V1 OOF candidates + gold labels。
- Produces: 六类错误桶、oracle recall、pair oracle、implicit oracle、category/polarity confusion。

- [ ] 写 failing tests：explicit span / implicit-A / implicit-O / pair / category / polarity 六类可区分。
- [ ] 计算 candidate oracle recall；若 oracle recall < 0.90，优先修 candidate generation；>=0.90 才优先 ranking/calibration。
- [ ] 输出每类 FN/FP 数量和理论最大可回收 TP。
- [ ] 固定后续 V2 只针对前三大错误桶，不增加与错误无关的复杂度。

### Task 3: Grid V2 — 最终主结构

**Files:**
- Modify: `src/opinion_mining/grid_model.py`
- Modify: `src/opinion_mining/grid_trainer.py`
- Modify: `src/opinion_mining/data.py`
- Test: `tests/test_grid_model.py`, `tests/test_grid_trainer.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: official offsets + fixed folds。
- Produces: unified explicit/implicit multi-relation candidate scores。

- [ ] 加 `[IMPLICIT_A]`、`[IMPLICIT_O]` 两个 pseudo-token，并写解码回 `_` 的 round-trip test。
- [ ] relation head 固定为 biaffine/CLN compact grid；Category×Polarity 联合 relation label，同时保留辅助 category/polarity loss。
- [ ] 同 cell 多 relation 使用 multi-label target；写多四元组不覆盖测试。
- [ ] hard negatives 固定包含：同句错误 pair、共享 opinion、近邻 boundary、隐式/显式混淆。
- [ ] endpoint top-k beam + max-span + NMS；禁止单 argmax span。
- [ ] implicit loss 单独加权，但只允许 `{1.0,1.5,2.0}` 三个 screening 值，不做大搜索。

**Gate:** 只有 V2 相对 V1 在 3-fold strict F1 提升 >=0.015，或 F1 提升 >=0.005 且 oracle/implicit FN 明显改善，才进入 Task 4；否则只允许一次基于错误审计的 V2.1 修补，之后无论如何进入 Task 4，不再继续结构探索。

### Task 4: 强 backbone 固定对比

**Files:**
- Modify: `scripts/run_grid_screen.py`
- Test: `tests/test_grid_screen.py`

**Interfaces:**
- Consumes: Grid V2。
- Produces: WWM-V2 / MacBERT-V2 同 folds 3-fold OOF。

- [ ] 用完全相同 folds/config 训练 WWM-V2。
- [ ] 用完全相同 folds/config 训练 MacBERT-V2。
- [ ] 只允许 backbone LR 在 `{1e-5,2e-5}`，head LR 固定一个值。
- [ ] 记录 pairwise TP/FP overlap。
- [ ] 保留 top-2：按 strict F1 + error complementarity 决定，不允许再新增第三个 BERT 类 backbone。

### Task 5: TAPT/DAPT — 固定一次增强

**Files:**
- Create: `scripts/run_mlm_adaptation.py`
- Create: `scripts/build_domain_corpus.py`
- Test: `tests/test_domain_corpus.py`, `tests/test_mlm_adaptation.py`

**Interfaces:**
- Consumes: top-2 backbone, train reviews, 经规则 gate 的无标注电商评论。
- Produces: adapted backbone checkpoints。

- [ ] 先做 TAPT：仅训练集评论 MLM continuation。
- [ ] 若比赛规则允许外部公开无标注语料，再加入 5万–20万条中文电商评论做 DAPT。
- [ ] 不直接引入外部人工四元组标注，除非规则明确允许。
- [ ] 每个 winner 只跑一个短程 adaptation 配置；不做 epoch sweep。
- [ ] downstream 只用固定 3-fold 判断；提升 <0.005 则淘汰 adapted 版本。

### Task 6: 高置信 pseudo-label 固定增强

**Files:**
- Create: `src/opinion_mining/pseudo_label.py`
- Create: `scripts/build_pseudo_labels.py`
- Test: `tests/test_pseudo_label.py`

**Interfaces:**
- Consumes: top-2 V2 模型 + 可用无标注评论。
- Produces: scored pseudo-label dataset。

- [ ] 只有两个结构不同模型一致，或单模型高置信 + verifier 通过，才接受伪标签。
- [ ] pseudo-label scorer 对 exact-substring、关系一致性、implicit 类型、类别/极性一致性打分。
- [ ] 每类设置最小/最大采样，防止“整体/正面”淹没长尾。
- [ ] 先加入 3k–10k 高置信样本，不一次灌满全部无标注数据。
- [ ] 若 3-fold 提升 <0.005，停止 self-training，不做第二轮伪标签循环。

### Task 7: Qwen3-4B 固定角色

**Files:**
- Create/modify: `scripts/qwen_verifier.py`
- Create/modify: `scripts/qwen_qlora.py`
- Test: `tests/test_qwen_schema.py`

**Interfaces:**
- Consumes: review + encoder candidates + retrieved train exemplars。
- Produces: verification score / optional challenger quadruples。

- [ ] Qwen3-4B 4-bit MLX QLoRA，只跑一个 rank=8 主配置。
- [ ] 第一角色：implicit/pair verifier；第二角色：pseudo-label teacher；第三角色才是独立 challenger。
- [ ] RAG index 在 fold 内只用 train fold。
- [ ] 强制 JSON schema + exact substring validator。
- [ ] Qwen 独立 challenger 只有在 3-fold-like held-out 验证证明互补时才进入 ensemble；否则只保留 verifier/teacher 角色。

### Task 8: Finalists 5-fold confirmation

**Files:**
- Modify: `scripts/run_grid_screen.py`
- Modify: `scripts/calibrate_groups.py`
- Test: LOFO / promotion tests

**Interfaces:**
- Consumes: 最多 2 个 encoder finalists + 可选 Qwen challenger。
- Produces: trusted 5-fold cross-fit metrics + test candidate maps。

- [ ] finalists 固定 epoch count，不在 held-out fold 上 early-stop 选 epoch。
- [ ] 每个 fold threshold 只能由其他 folds 学习。
- [ ] 报告 cross-fit F1、full-OOF-fit test threshold，两者明确分开。
- [ ] 保存 implicit/explicit/category/error bucket 分数。
- [ ] 只允许这一步产生 `confirmed` 标签。

### Task 9: Heterogeneous Ensemble

**Files:**
- Create/modify: `src/opinion_mining/ensemble.py`
- Test: `tests/test_ensemble.py`

**Interfaces:**
- Consumes: confirmed OOF candidate maps。
- Produces: calibrated ensemble candidates。

- [ ] 先算 pairwise TP/FP overlap 和 oracle recall。
- [ ] 只有 error complementarity 明显的模型才融合。
- [ ] encoder+encoder 用 calibrated weighted voting；Qwen 独有项必须通过 verifier gate。
- [ ] ensemble 权重只允许少量离散组合，不做连续暴搜。
- [ ] ensemble OOF 至少比最佳 confirmed 单模高 0.004 才生成正式线上候选。

### Task 10: Online validation 与最终提交

**Files:**
- Modify: `artifacts/submissions/manifest.csv`
- Modify: dashboard online view
- Test: submission validator tests

**Interfaces:**
- Consumes: confirmed single models + proven ensemble。
- Produces: leaderboard evidence + final Result.csv。

- [ ] 上传顺序固定：best single → second structurally different single → best ensemble → 至多两个高信息 calibration challenger。
- [ ] 每次上传前记录 SHA256/config/local OOF；上传后回填 LB。
- [ ] 若 local 与 LB 排序长期反向，停止上传并审查 split shift；不得继续阈值扫榜。
- [ ] 最终 Result.csv 通过无 BOM/无 header/全 id/合法枚举/substrings/dedupe/<100MB validator。

## 改道条件（只有这些情况允许改变主路线）

1. Grid V2 candidate oracle recall <0.80：说明结构根本无法覆盖 gold，可替换 relation head/decoder；否则禁止换架构。
2. WWM-V2 与 MacBERT-V2 都比各自 V1 下降 >0.02：说明 V2 实现存在回归，回滚检查，不引入新模型。
3. TAPT/DAPT、pseudo-label、Qwen 任一增强连续两次不提升：永久关闭该增强，不再搜索。
4. 5-fold 与 3-fold 差异 >0.03：优先检查 split/calibration/data leakage，不换 backbone。
5. 系统 memory pressure RED：只降低资源配置，不改变算法路线。

## 明确禁止的事情

- 禁止重新训练 RBT3。
- 禁止 seed sweep。
- 禁止继续试一堆 BERT/NEZHA/ELECTRA backbone。
- 禁止几十个 group threshold 自由优化。
- 禁止为了 leaderboard 单点增益反复上传近似候选。
- 禁止在没有错误分析证据时再加新 head / 新 loss / 新模型。

## 最终完成定义

项目只有同时满足以下条件才算完成：

- 至少一个 `confirmed` 5-fold 模型；
- LOFO/cross-fit strict quadruple 评估完整；
- 至少完成一次 DAPT/TAPT gate、一次 pseudo-label gate、一次 Qwen verifier gate（可以失败关闭，但必须有结果）；
- ensemble 有 OOF 证据或明确证明无收益；
- 至少完成一轮线上 anchor 验证；
- Dashboard 可按模型查看训练、数据分页、local/LB 分数并存；
- 最终 `Result.csv` + manifest + config hash + SHA256 + final report 完整。
