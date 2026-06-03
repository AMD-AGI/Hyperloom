# Step 05: Clean SharedState And Compatibility

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- `state.json` 是端到端 artifact, 但不是所有历史字段都必须永久保留。
- 新 session 的写路径必须简洁。
- 旧字段兼容只允许集中在 migration 或 reader 层。
- 不再服务端到端 resume/report 的历史字段应删除。

## 目标

`SharedState` 是系统所有 agent 和 executor 的事实平面。它承载了优化进度、phase、baseline、current best、kernel attempts、explore ledger、specialist rounds、gaps、warm start、close sequence 等信息。随着功能叠加, `SharedState` 也积累了大量历史字段、旧 migration、兼容 alias 和 prompt rendering 逻辑。

本阶段目标:

- 明确 `state.json` 当前 schema。
- 删除新写路径中的旧字段。
- 集中兼容迁移。
- 降低 prompt summary 和 state mutation 的复杂度。
- 让 `SharedState` 只保存端到端需要的事实。

## State 字段分类

每个字段应归入以下类别:

### Core runtime field

运行当前端到端流程必须保留。

例如:

- session identity。
- workload config。
- baseline / current best。
- cumulative gain。
- phase and phase history。
- optimization stack。
- current action。
- crash count。

### Action ledger field

用于记录 live action 的尝试和结果。

例如:

- explore search。
- sweep results。
- kernel attempts。
- specialist rounds。
- framework PR progress。
- dynamic action summary。

这类字段可以保留, 但应避免为每个历史实现保留一套 ledger。

### Contract export field

不是 runtime 决策必需, 但 downstream reporting 需要。

例如:

- breakdown collector 需要的 source path。
- report summary 需要的 final stack detail。
- close sequence evidence。

这类字段应尽可能写在 action result 或 reports 中, 不应无限膨胀 `state.json`。

### Compat field

只为旧 session 读取或旧字段迁移服务。

要求:

- 不允许新写。
- 集中在 load/migration 层。
- 有删除条件。
- 有最小测试覆盖。

### Dead field

当前不读、不写、不导出、不影响端到端。

处理方式:

- 删除字段。
- 删除 writer。
- 删除 prompt rendering。
- 删除测试。

## 兼容策略

### 新写路径只写 canonical field

不应再由新 session 写 legacy field。所有 legacy name 应该只在 read/migration 层出现。

如果历史字段需要被接受:

- load 时转换成 canonical field。
- save 时只输出 canonical field。
- prompt 中只显示 canonical field。

### Migration 集中

不要让每个 executor、handler、collector 都自己读 legacy 字段。兼容逻辑应该集中在:

- payload alias helper。
- SharedState load migration。
- breakdown archived-session reader。

其他 live code 应只看到 canonical shape。

### 删除旧 session 兼容的条件

如果决定不再支持旧 session:

- 删除 SharedState old schema migration。
- 删除 breakdown old reader。
- 删除 legacy tests。
- 删除 docs 中的旧 layout 说明。

如果暂时支持:

- 保留 migration。
- 保留最小 test。
- 不再在主流程中讨论旧字段。

## Prompt Surface 清理

`SharedState` 通常会被渲染进 agent prompt。Prompt surface 越大, LLM 越容易被旧状态误导。

需要清理:

- retired action 的 ledger。
- old aliases。
- 过程性字段。
- 内部调试字段。
- 不再影响决策的历史 evidence。

Prompt 应只包含:

- 当前目标。
- 当前 phase。
- 必须执行的下一步。
- 最近重要 action 结果。
- 当前 best 和 remaining gap。
- 有效的 warnings / blockers。

## Mutation Surface 清理

Core state 应保持 Coordinator-only writer。Agent 通过 intent 提交变更时, PolicyGate 应继续拦截核心字段。

清理方向:

- 减少可被 agent update 的字段数量。
- 把 derived field 从 state 中移除, 改为 runtime 计算。
- 把只用于 report 的 detail 移到 reports artifact。
- 把只用于 debug 的 detail 移到 event log 或 task result。

## 测试调整

保留:

- 新 session 写出的 state shape。
- old session migration, 如果兼容策略保留。
- core state field 被 PolicyGate 保护。
- phase history 和 optimization stack 的端到端更新。

删除或合并:

- 对旧字段逐项迁移的重复测试。
- 对 prompt string 中历史字段文案的测试。
- 对不再写入字段的正向测试。
- 对 private state helper 的实现细节测试。

## 风险

主要风险:

- 删除字段后 breakdown 缺数据。
- 删除 migration 后 resume 旧 session 失败。
- 删除 prompt 字段后 Orchestration 缺少必要事实。
- 删除 derived field 后某个 gate 仍读取它。

控制方式:

- 建立字段读写表。
- 区分 runtime read 和 reporting read。
- 先停止新写, 再删除 old read。
- 每次删除字段后检查 prompt, gate, collector, report。

## 输出物

本阶段应产出:

- `SharedState` 字段分类表。
- canonical state schema。
- compat migration 保留/删除清单。
- prompt surface 简化方案。
- state tests 重组方案。

本阶段第一批的字段分类表见
`code-cleansing-plan/05-shared-state/01-field-classification.MD`,
执行记录见
`code-cleansing-plan/05-shared-state/02-execution-status.MD`。

## 完成标准

完成后应达到:

- 新 session 的 `state.json` 明显更干净。
- legacy 字段不再从 live writer 出现。
- 兼容逻辑集中且有删除条件。
- Prompt 只展示当前决策需要的信息。
- `SharedState` 不再承担过多历史叙事。
