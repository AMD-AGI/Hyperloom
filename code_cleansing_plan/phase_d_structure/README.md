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

### Step 01 — import 复杂度 / 循环(已核验,无需改动)

- 用 AST 全量构图(含**函数内 import**)跑 Tarjan SCC:**循环引用数 = 0**。
- Phase B 抽出的 `coordinator_helpers` / `cli_executors` / `cli_kb` / `cli_backends` 均严格单向(只 import orchestrator,
  不回 import `coordinator`/`cli`),已复核。依赖方向符合"底层→顶层"理想层次。
- 结论:无环可破、无跨层引用需下沉,Step 01 验收达标,无代码改动。

### Step 02 — intent schema 归入 protocol 层(方案 A,`48bd57a9`)

- 将 `orchestrator/intent_parser.py` 的全部 schema+校验(`IntentType` / `Intent` / `INTENT_ENVELOPE_SCHEMA` /
  `EMIT_INTENT_TOOL_SCHEMA` / `validate_envelope` / 异常 / `_PAYLOAD_REQUIRED`)整体下沉到**最底层** `protocol/intent.py`,
  与 `action_surfaces.py` 同属协议层 → `protocol/` 名实相符、内聚。
- `protocol/intent.py` 纯净(只 import stdlib,**无任何包内 import**),不制造环(复核后 cycles 仍 = 0)。
- 全仓 62 个 call-site 直接改指向 `protocol.intent`(3 种 import 形态:绝对路径 / orchestrator 顶层 `.` / backends `..`),
  **不留 re-export shim**;`intent_parser.py` 删除。robustness-agent 平价测试 import 同步更新;镜像 `envelope.py` 仍保留(独立包)。
- 守 §1 契约:envelope 字段/形状未变;envelope 校验 + robustness 平价护栏全绿。

### Step 03 — 碎文件合并(已核查,无安全合并项)

逐一核查 plan 候选,均**不宜动**(强行合并会违背"不造大杂烩 / 不喂大 god-module"):

| 候选 | 核查结论 |
|---|---|
| `protocol/__init__.py` | 包标记(承载 intent.py / action_surfaces.py),保留 |
| `compat/` | **非空**——`payload_aliases.py` 仍是活的 back-compat shim,保留 |
| `action_executors/dynamic_action.py` | **在用**(`_register_executors` 的 stub fallback),非死 |
| `breakdown/reporters/_renderers/*` | 插件集:`compose.py` side-effect import 注册 + `test_reporters_smoke` 锁定稳定顺序,勿动 |
| `benches/` | `BENCH_REGISTRY` 被 `test_dynamic_action_invariants` 导入遍历,保留 |
| `multi_node/_internal/*` | 多引用工具(`log.py` 7 refs 等),非单引用碎文件 |
| `tracelens_md.py` | 独立单测 + 刻意分离(并入会撑大 `shared_state` god-module),保留 |

> Phase A 已清掉计划预期的死 stub,残留小文件都是合法的包标记/入口/注册插件/非空 compat。无安全合并项,按"宁可漏删"留置。

### 文件数 / 复杂度账(B+D 合计)

- Phase B 拆出 4 个(`coordinator_helpers`, `cli_executors`, `cli_kb`, `cli_backends`);Phase D 新增 1 个(`protocol/intent`)、删 1 个(`intent_parser`)。
- **净增文件 = +4(个位数,达标)**;最大单文件行数较 Phase 0 基线大幅下降(coordinator 12623→12119、cli 6368→5436)。
- 循环引用 = 0;依赖严格单向。四项判据同时改善。

### 出口校验

- [x] 无循环 import(AST+Tarjan,含函数内 import)。
- [x] `protocol/` 与 intent schema 职责清晰(方案 A 合并)。
- [x] 碎文件已核查(无安全合并项,记录理由)。
- [x] 护栏全绿;`import inference_optimizer.cli` 正常;envelope/平价契约不变。
- [x] 文件数净增 ≤ 个位数。
