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

(完成后填:文件数前后 / 循环引用数 / commit)
