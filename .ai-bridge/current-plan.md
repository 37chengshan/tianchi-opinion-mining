# 天池赛道六 Master Plan V2 执行

Updated: 2026-09-28T02:15:23.476Z
Workspace: /Users/cc/code/tianchi-opinion-mining
Target agent: Codex (codex)

## Plan

接手当前工程并以 `docs/MASTER_PLAN_V2_RESEARCHED.md` 为最高优先级执行规范。停止继续 seed-only evolution；等当前 fold 安全落盘后切换路线。

必须先修 P0：1) champion promotion 必须真正更新 test prediction/Result.csv/dashboard/manifest；2) 数据层保留并使用官方 A_start/A_end/O_start/O_end，禁止 `.find()` 代替 offset；3) 原生支持 Aspect=`_` 和 Opinion=`_`；4) 修复同一 span/token 多四元组覆盖，改成 pair/grid 多关系监督；5) group threshold 改成 leave-one-fold-out + small-group shrinkage；6) 为 neural/grid/implicit/multi-relation/promotion/calibration 增加测试。

主模型改为：MacBERT-base 与 Chinese RoBERTa-WWM-ext + compact implicit-aware relation grid（pseudo-token `[IMPLICIT_A]`/`[IMPLICIT_O]`，低秩 biaffine/grid，hard negatives，constrained top-k beam + NMS）。RBT3 仅作为 regression/smoke baseline，不再做 seed sweep。所有新结构先固定 3-fold screening，只有 strict calibrated F1 提升 >=0.005 或 ensemble 互补性明确才晋级 5-fold。

强 encoder 跑通后，对 winner 做一次短程 TAPT/DAPT（是否使用 test 文本先核对比赛规则）；用 downstream 3-fold 决定保留与否。随后增加异构 challenger：Qwen3-4B-Instruct 使用 MLX-LM 4-bit QLoRA，batch 1 起、grad accumulation 8–16、短上下文 <=512；RAG 为 BM25 Top20 -> reranker Top5，reranker 可离线预计算以避免与 Qwen 同时占内存。7B 只在 4-bit smoke benchmark 证明内存稳定后尝试。

M1 Pro 16GB 使用动态 8–12GB 策略，不写死 8GB：监控 available memory、memory_pressure、MPS allocated/driver memory、swap。GREEN available>=5GB 正常；YELLOW 3–5GB 降 batch/启 checkpoint/清 cache；RED <3GB 或 critical/swap 快增则 checkpoint 当前 trial 并降级。PyTorch 模型与 MLX Qwen 不同时常驻。

验证：fixed folds；所有 fold 内统计/RAG/阈值只用 train fold；保存完整 OOF candidate maps；先计算 candidate oracle recall，再决定优化 span/grid 还是 calibration；5-fold finalists 做 LOFO calibration。Ensemble 只合并错误互补的 MacBERT-grid / WWM-grid / Qwen+RAG，高纯度规则仅作 confirmation bonus，不再把低 precision statistical baseline 大权重平均进去。

线上闭环：用户已说明本地 Codex 浏览器会话已登录天池。启动时检测 browser automation 与登录状态；若可用，仅上传通过 validator 的 `artifacts/submissions/candidates/<id>/Result.csv`，上传前写 config hash/local calibrated F1/SHA256，上传后读取真实 leaderboard score 并回填 `artifacts/submissions/manifest.csv` 与 Dashboard。单次只提交一个，确认 score 与 SHA/config 对应后再继续。若浏览器工具不可用，只排队 candidate，不阻塞训练。不要用剩余 999 次暴力扫相邻阈值，按 anchor -> calibration -> challenger -> ensemble 分批提交；LB 是第二信号，OOF 明显下降但 LB 单点提高的模型标记 leaderboard_only 并额外验证。

目标：0.75 必须认真争取，0.80+ 主目标，0.85 冲刺。最终生成严格校验的 `Result.csv`（UTF-8 无 BOM、无表头、全 test id、exact substring、<100MB）、manifest、OOF/error/oracle 报告、final dashboard、完整复现命令。不要在中途询问；除非遇到账户安全/平台规则/需要人工验证码，否则自主执行到预算或收敛。

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
