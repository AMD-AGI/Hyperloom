# Step 03: Converge Action Runtime Surfaces

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 以端到端 action 行为为准, 不以历史 action 名称为准。
- live action 必须少而清晰。
- action metadata、phase allowlist、executor wiring、path whitelist 不应长期分裂。
- 已合并进新 action 的旧动作应从主流程、prompt、测试中删除。

## 目标

本阶段目标是把 action 体系整理成单一、清晰、可维护的 runtime surface。当前系统中 action 信息散落在多个位置: metadata, CLI executor 注册, phase allowlist, PolicyGate, session path whitelist, prompt builder, tests。长期维护时, 任何一个位置漂移都会造成 LLM 提出不可执行 action、executor 找不到 workspace、phase gate 误拒绝等问题。

本阶段要把这些来源收敛为一个清晰模型:

- 哪些 action 是 live action。
- 哪些 action 可由 LLM 提出。
- 哪些 action 只能由 Coordinator 内部 enqueue。
- 哪些 action 是 Kernel-owned request。
- 哪些 action 只是 compat reader, 不再是 runtime action。

## 核心分类

### LLM-proposable action

这类 action 是 Orchestration 可以主动提出或 delegate 的优化动作。它们应该少而明确。

要求:

- 有 action metadata。
- 有 phase allowlist。
- 有 executor 或 request path。
- 有 prompt 说明。
- 有端到端行为测试。

### Coordinator-internal action

这类 action 不应该由 LLM 主动提出, 只能由 Coordinator 根据阶段和状态自动 enqueue。

典型类别:

- baseline 后自动分析。
- framework PR phase pump。
- sweep 后 concurrency comparison。
- close phase report 和 session breakdown。
- warm replay。

要求:

- PolicyGate 明确拒绝 LLM 直接提出。
- Prompt 中不要鼓励 LLM 提出。
- Executor wiring 清楚。
- 测试关注 Coordinator 自动触发行为, 不关注 LLM 正向提出。

### Kernel-owned action

这类 action 不应该走普通 delegate, 而是通过 `request{target_agent='kernel'}` 进入 kernel request handler。

要求:

- action 名称可以保留在 catalogue 中, 但 runtime 入口必须清楚。
- executor 表中的 no-op 注册必须有明确目的: 避免 stale task 或兼容队列失败。
- Prompt 必须强调 request path, 不让 LLM delegate。
- PolicyGate 必须拒绝错误 transport。

### Retired action

这类 action 不再是 runtime 行为。它们只允许出现在:

- migration reader。
- old session breakdown reader。
- 明确的 rejection test。
- 一处简短升级说明。

不允许出现在:

- enabled action set。
- prompt action catalogue。
- executor registry。
- new state writer。
- new breakdown writer。

## 收敛策略

### 统一 live action source

长期目标是让 live action 集合只有一个权威来源。其他位置应该派生或校验, 而不是手写复制。

需要收敛的来源:

- action metadata。
- executor registry。
- phase allowlist。
- path whitelist。
- prompt enabled action set。
- tests expected action set。

建议方向:

- action metadata 表达 action 的静态属性。
- phase allowlist 表达状态机的 runtime 权限。
- executor registry 表达具体执行能力。
- tests 只校验这些来源不漂移。

不要让 tests 成为新的隐形 source of truth。测试应该验证 source of truth, 不应该替代它。

### 删除旧 action 叙事

如果旧 action 已经合并进新 action, 只在迁移文档中提一次。不要在主 prompt 和核心 docstring 中持续解释旧动作。

例如:

- `backends` 和 `params` 的历史可以在升级说明里出现。
- 运行时 prompt 应只说 `explore`。
- breakdown 如果暂时保留 alias, 应明确这是 archived session reader。

### 精简 action family

当前 action family 如果只是历史设计分类, 但 runtime 决策不再依赖, 应考虑降级为 display-only 或删除。不要保留多个分类维度同时指导调度, 这样会增加心智负担。

应保留的维度:

- phase。
- executor/request ownership。
- resource lane。
- side effect level。

可弱化的维度:

- 历史 family。
- 旧 scoring 分类。
- PR 过程中的 milestone 分类。

## 测试调整

保留:

- live action catalogue 不漂移。
- LLM-proposable 和 Coordinator-internal action 的边界。
- Kernel-owned action transport guard。
- path whitelist 与 executor registry 对齐。

删除或合并:

- retired action 的正向执行测试。
- 对旧 family 排序的测试。
- 对 prompt 中旧 action 文案的测试。
- 只验证 metadata 字段默认值的小测试。

## 风险

主要风险:

- 删除 action metadata 后 path helper 无法创建 workspace。
- phase allowlist 未同步导致 action 被拒。
- prompt 仍提旧 action 导致 LLM 卡住。
- executor registry 缺失导致 queued task 失败。
- old session 中旧 action 无法被 breakdown 读取。

控制方式:

- 先整理 live action 表。
- 每次删除一个 action family 的残留。
- 同步更新 prompt 和 tests。
- 对 old session 兼容单独做保留/删除决策。

## 输出物

本阶段应产出:

- live action 分类表。
- LLM-proposable action 表。
- Coordinator-internal action 表。
- Kernel-owned action 表。
- retired action 删除清单。
- action source drift guard 的简化测试。

本阶段第一批 action surface 收敛的执行记录见
`code-cleansing-plan/03-action-surfaces/01-execution-status.MD`。

当前 action 分类表见
`code-cleansing-plan/03-action-surfaces/02-action-surface-table.MD`。

## 完成标准

完成后应达到:

- 新维护者能一眼看出系统有哪些 action。
- 增加或删除 action 不再需要改多个互相漂移的列表。
- Prompt 不再鼓励 retired action。
- Tests 只保护端到端 action contract, 不锁死旧内部实现。
