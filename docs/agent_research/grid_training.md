# Relation grid training chain audit

更新时间：2026-09-28

## 已完成的 bounded 范围

`grid_trainer.py` 现在保留本地缓存模型路径，并增加了 tokenizer/encoder 注入点供 smoke 和测试使用。`train_grid_fold()` 使用配置副本，资源降档不会原地污染调用者的后续 fold 配置。

`run_grid_screen.py --dummy-train` 使用合成数据、`DummyTokenizer`、`DummyEncoder` 和 1 个 CPU epoch，覆盖 optimizer、验证解码和 checkpoint 写出；不会下载模型，也不产生官方 F1。

## 真实阻塞与边界

- 原 CLI 只有单 batch smoke 或 3-fold plan，已经用 `--dummy-train` 补上最小端到端可运行路径。
- 官方训练仍缺少外层 OOF 编排、候选汇总和阈值校准；本次没有扩大到这些工作。
- 真实 backbone/tokenizer 只用 `local_files_only=True`。本机缓存缺失时会在加载阶段失败；本次没有下载或启动真实模型训练。

## 复核命令

```bash
PYTHONPATH=src python -m py_compile src/opinion_mining/grid_trainer.py scripts/run_grid_screen.py
PYTHONPATH=src python scripts/run_grid_screen.py --dummy --output-dir /tmp/tianchi-grid-audit-dummy
PYTHONPATH=src python scripts/run_grid_screen.py --dummy-train --output-dir /tmp/tianchi-grid-audit-train
PYTHONPATH=src python -m pytest tests/test_grid_trainer.py tests/test_grid_model.py -q
```

这些命令只覆盖 CPU smoke、静态编译和已有 grid 合约测试；不能替代官方 3-fold F1。
