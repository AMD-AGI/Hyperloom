# Phase B · 步骤 02 — 拆瘦 cli.py(6512 行)

## 现状

god-module / 组合根:参数解析(数百 flag)、executor 注册表、backend 构造、KB bootstrap、`_run_optimize` 主流程、多节点 provision —— 全在一个文件。

## 策略

### A. 先删(部分 Phase A 已做)

- 退役 flag(`_RetiredFlag` roofline 拼写)、KB flusher shim(`_maybe_spawn_kb_flusher`/`_stop_kb_flusher`)、marathon 路径(~5051)、`last_trace_analyze_baseline` 退役管线(~4442)、`_REAL_EXECUTORS_KERNEL_ONLY` 空表。

### B. 按关注点抽出(单向依赖 cli → 子模块)

| 抽出内容 | 目标模块(建议) |
|---|---|
| `_build_parser` + flag 定义(~5048+) | `cli_parser.py` |
| executor 注册表(`_REAL_EXECUTORS_FULL`、`_register_executors`、各 `_build_*_executor`,~1028–1757) | `cli_executors.py` |
| backend 构造(`_build_backends`,709–805) | `cli_backends.py` |
| RecipeKB / KnowledgePlane bootstrap(`_bootstrap_cortex_kb`、`_build_recipe_kb_dispatcher`,3266–3378) | `cli_kb.py`(命名 Phase A 已收口为 recipe) |
| 多节点 provision(`_provision_multi_node_rayjob_stack`,3739+) | `cli_multinode.py` 或并入 `multi_node/` |

`cli.py` 主体只留 `main()` + `_run_optimize` 编排骨架。

### C. 依赖检查

```bash
python -m inference_optimizer.cli --help   # 必须仍可运行
rg -n 'from .* import .*cli\b|import .*\.cli\b' inference_optimizer/cli_*.py  # 子模块不得反向 import cli
```

## 目标

- `cli.py` 从 ~6.5k 降到 < 2k(主流程编排)。
- 子模块各自聚焦、可独立阅读。

## 验收

- [ ] `--help` 金标准 diff:仅退役 flag 消失,其余 flag 不变(§1 契约)。
- [ ] mock 全流程仍跑通(金标准形状不变)。
- [ ] 无循环引用;总 LOC 不增。
- [ ] commit:`Extract cli parser/executors/backends/kb into modules`(可拆多个)。

## ⚠️ 注意

- **CLI flag 面是 §1 对外契约**:抽出 parser 时严禁改 flag 名/默认值/退出码(退役 flag 除外,已在 Phase A 登记删除)。
- executor 注册是动态字典:vulture 会误报"未使用",别据此删;以 `kind` 字符串 call-site 为准。
