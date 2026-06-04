# Phase D — 文件结构 + 引用结构整理

## 目的

降低整体引用复杂度:消除职责重叠的目录、合并碎文件、理顺 import 层次(单向、无循环、浅依赖)。

> 放在注释精简之后:此时代码与注释都已稳定,移动文件造成的 diff 噪音最小。

## 步骤文件

- [`01_import_graph.md`](01_import_graph.md) — 测量并降低 import 复杂度,消除循环/跨层引用。
- [`02_protocol_intent_consolidation.md`](02_protocol_intent_consolidation.md) — 厘清 `protocol/` 与 `orchestrator/intent_parser.py` 职责重叠。
- [`03_file_consolidation.md`](03_file_consolidation.md) — 合并碎文件、整理目录。
- [`conduct.md`](conduct.md) — 行为准则。

## 入口标准

- Phase C 出口达标。

## 出口标准

- [ ] 无循环 import(检测脚本通过)。
- [ ] `protocol/` 与 intent schema 职责清晰(二选一:合并或明确文档化分工)。
- [ ] 碎文件(极小、单一用途、只被一处引用)已合并到合理归属。
- [ ] 护栏全绿;`import inference_optimizer.cli` 等关键 import 正常。
- [ ] 文件数净增 ≤ 个位数(Phase B 拆分 vs 本相位合并相抵)。

## 进度记录

独立复核(本轮):结构性工作已达标,逐项实测如下。

步骤 01 — import 复杂度 / 循环引用:
- AST 全包扫描 `inference_optimizer`(343 模块 / 684 边)→ **模块级循环 = 0**。
- 抽出的 god-module 拆分件严格单向:`cli_backends/cli_kb/cli_executors` 均**不** import `cli`;`coordinator_helpers` 文件头声明且实测**不** import `coordinator`。
- 21 处函数内 internal import 复核后保留:全部是**真·破环**(`shared_state`↔`phase_state`、executors 包内 `profile/baseline`↔`_grid_runner`/`_multi_node_*` 等)或 `cli` 的**启动延迟**优化。此时循环=0 正是靠这些局部 import 维持,上提会**重新制造循环**,违背 conduct(禁止假性破环 / 禁止上提造环)→ 不动。

步骤 02 — protocol/intent 归并(方案 A,已落地):
- `orchestrator/intent_parser.py` 已删;schema 迁入 `protocol/intent.py`,与 `protocol/action_surfaces.py` 同属协议层。
- `protocol/` 纯净:`intent.py`/`action_surfaces.py` 仅 import stdlib(dataclasses/enum/typing),**不**反向依赖上层 → 无环。
- envelope 平价/契约护栏绿(protocol_layer + critic intent_envelope + robustness role_envelope:82 passed)。

步骤 03 — 碎文件 / 目录整理:复核后**无安全合并项**。
- "小且单引用"候选实测全部**活跃**:`multi_node/_internal/log.py`(被 cli/safe_client/ray_dashboard 3 处用)、`tracelens_md.py`(被 shared_state 用)。
- `breakdown/reporters/_renderers/*`(19 个 ~100 行的 section renderer,compose.py 侧效应注册)是**单一职责的好结构**;按主纲领"单文件复杂度 > 文件数量",合并成 2000 行大文件反而增复杂度 → 保留。
- 其余单引用模块(`cursor_store`/`specialist_mcp_config`/`dynamic_action`/`name_mapping`,均 ~80–118 行)的唯一消费方都是大模块;下沉会**膨胀 god-module**,与 Phase B 瘦身目标相悖 → 保留。
- `compat/payload_aliases.py` 仍是 §1 alias 契约(有护栏),非空 → 保留;`benches/` 是 Phase A 已决定保留的 disabled scaffolding(`BENCH_TOOL_ENABLED_V1=False`)→ 不在本相位重判。

出口校验:
- [x] 无循环 import(检测=0)。
- [x] protocol/intent schema 与 action_surfaces 同属 `protocol/`,职责清晰。
- [x] 碎文件复核:无满足"极小+单引用且合并能降复杂度"的项;空目录无。
- [x] 护栏绿;`import inference_optimizer.cli` 等关键 import 正常(全部 OK)。
- [x] 文件数未净增(本相位零代码改动);io 非测试 .py = 169。

结论:Phase D 结构性目标(零循环 / 协议层内聚 / 严格单向 / 无有害碎片)均已满足且本轮独立复核通过;按 conduct「四项判据 + 禁止churn」无进一步安全收益,故本相位仅记录复核结果,不做无谓改动。
