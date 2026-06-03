# Step 01: Establish Baseline And Live Contracts

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 以端到端优化流程为最终判断标准, 不以私有 helper 或单元测试细节为判断标准。
- 优先保护用户、平台和外部系统可观察的行为。
- 对没有端到端价值的内部兼容、历史路径、过度测试和过程性注释保持删除倾向。
- 所有保留的复杂度都必须能说明它保护了哪个端到端能力或外部契约。

## 目标

本阶段不做大规模删除, 只建立清理边界。没有边界就直接删代码, 很容易误删通过字符串、JSON、SQLite、subprocess、artifact path 连接的运行路径。

本阶段要回答三个问题:

- 当前系统真正需要保证的端到端能力是什么?
- 哪些文件格式、协议和路径属于外部契约?
- 哪些内部单元行为可以在后续清理中被改写或删除?

## 端到端能力边界

必须保留的端到端能力:

- 能通过 CLI 启动一个优化 session。
- 能建立 baseline 和 target analysis。
- 能完成 profile 或 roofline 分析。
- 能进入 framework PR phase, explore phase, kernel phase, sweep phase 和 close/report phase。
- 能调度 Orchestration, Kernel, Critic, Robustness 四类角色或其降级实现。
- 能执行 benchmark, profile, kernel optimization, framework patch integrate, sweep 和 report。
- 能写出最终可消费的 session artifact。
- 能在必要场景下 resume 或至少明确拒绝不可支持的历史 session。

这些能力是后续清理的最低保留面。只要这些能力保持, 内部函数、类、helper、测试布局都可以改变。

## 外部契约边界

需要先列为保护对象:

- CLI 输入输出: 命令、主要参数、关键 env、启动失败时的错误语义。
- Session artifact: `manifest.json`, `state.json`, `storage/coordinator.db`, `runs/`, `reports/`, `session_breakdown.json`。
- Intent envelope: agent 输出的 intent vocabulary 和 payload shape。
- Policy contract: 哪些角色可以发哪些 intent, 哪些 action 可在什么 phase 执行。
- Kernel bridge: kernel request 到 kernel-agent 工具的 JSON 和 artifact 约定。
- Critic bridge: prepare-review / commit-review 的 request 和 emit envelope。
- Robustness bridge: robustness tick 的 request 和 emit envelope。
- Framework bridge: framework-agent discovery 的 request 和候选 PR 结果。
- Dashboard contract: `session_breakdown.json` 的字段语义。

不属于外部契约的内容:

- 私有 helper 的函数名和参数。
- 单元测试中临时构造的内部 dict 形状。
- 旧 PR 过程中的中间字段。
- 为了迁移期间方便调试而存在的过程性日志或注释。
- 未被 live runtime 使用的 demo 和 backup 文件。

## 兼容决策清单

本阶段需要形成明确判断:

- 是否继续支持旧 session resume?
- 是否继续支持旧 session breakdown 的读取?
- 是否继续支持 legacy field alias?
- 是否继续支持 retired action 的兼容识别?
- 是否继续支持旧目录布局?
- 是否继续保留历史 demo、legacy mode 和未发布 feature scaffold?

如果决定保留某项兼容, 后续文档和代码里必须写清楚:

- 它保护哪个端到端场景。
- 它预计什么时候可以删除。
- 它是否只读兼容, 还是仍允许新写入。

如果决定不保留, 后续清理应直接删除代码和测试, 不再加一层 warning 或 fallback。

## 基线建立方式

建议建立三层基线:

### 行为基线

记录一次当前系统的关键端到端行为, 例如:

- no-kernel smoke 的期望流程。
- kernel-enabled smoke 的期望流程。
- resume smoke 的期望流程。
- session breakdown dump 的期望结构。

这里不需要追求所有测试全绿, 只需要知道清理前后哪些行为变化是可接受的。

### 契约基线

列出当前 live contract:

- live action 集合。
- phase 顺序和 phase allowlist。
- kernel request kind。
- agent intent vocabulary。
- session artifact 目录结构。
- downstream breakdown 必需字段。

后续每次删除或合并都要回到这份清单判断。

### 测试基线

测试基线不等于必须保留所有测试。它的作用是知道当前测试状态:

- 哪些测试是真正 contract test。
- 哪些测试只是内部实现 test。
- 哪些测试是 legacy migration test。
- 哪些测试是 retired feature test。

后续测试清理时, 只保留能证明端到端行为和外部协议的测试。

## 输出物

本阶段结束时应产出:

- 一份 live contract 列表。
- 一份兼容策略列表。
- 一份端到端 smoke 场景列表。
- 一份测试分层列表。
- 一份 dead code 初筛清单。

本阶段的执行产物已经整理在 `code-cleansing-plan/01-baseline-contracts/`:

- `README.MD`: Step 01 产物索引和使用方式。
- `01-implementation-status.MD`: 已实现/未实现状态对照。
- `02-live-contracts.MD`: 清理期间必须保护的 live contracts。
- `03-compatibility-decisions.MD`: 需要产品或维护者确认的兼容策略。
- `04-e2e-smoke-scenarios.MD`: 判断端到端行为是否保持的 smoke 场景。
- `05-test-baseline-and-tiering.MD`: 清理导向的测试分层。
- `06-dead-code-initial-candidates.MD`: Step 02 的初始 dead-code 候选。

## 完成标准

本阶段完成后, 后续每一处删除都应该能回答:

- 它是否在 live contract 内?
- 它是否只服务内部单元行为?
- 它是否只服务旧 session 或旧测试?
- 删除它是否影响端到端流程?
- 如果影响, 这个影响是否被产品上接受?

如果这些问题无法回答, 不应进入删除阶段, 应先补充理解或向维护者确认。
