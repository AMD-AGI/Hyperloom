# 这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度, 下面的代码清理只需要保证端到端的功能一样即可, 不需要保证单元性的功能一致, 当前的代码库已经非常庞大了, 所以不要吝啬cut off代码.

## 1. 清理目标

本次代码清理不是小规模格式整理, 也不是保持所有内部实现兼容的重构。目标是大幅降低 HyperloomV2, 尤其是 `inference_optimizer`, 的工程复杂度, 让后续维护者能够用更少的文件、更少的分支逻辑、更短的上下文理解完整端到端优化流程。

验收标准以端到端能力为准:

- `inference_optimizer optimize` 能完成现有主流程: baseline, target analysis, framework PR phase, explore, specialist/dynamic action, kernel request, sweep, report, session breakdown.
- 关键外部契约保持可用: CLI 参数、session 目录、`state.json`, `manifest.json`, `coordinator.db`, `session_breakdown.json`, agent intent envelope, kernel/framework/critic/robustness subprocess JSON 协议。
- 已发布或实际依赖的运行路径保持可用; 未发布、未使用、只为历史开发过程服务的路径可以删除。
- 允许删除大量单元级兼容逻辑、历史中间态测试、过程性注释和过细的 helper 抽象, 只要端到端行为不变。

## 2. 核心原则

### 2.1 端到端行为优先

本次清理只要求端到端行为一致, 不要求内部单元行为逐字一致。内部函数签名、helper 分层、测试文件布局、旧单元测试断言都可以改变。

需要保留的是用户和外部系统可观察的行为:

- 优化流程能启动、恢复、停止、产出结果。
- agent 之间的 intent 协议不破。
- benchmark/profile/kernel/framework PR 的实际执行能力不破。
- downstream dashboard 能消费 `session_breakdown.json`。

### 2.2 不为未使用路径保留复杂度

如果一段代码只服务于已经退场的设计, 且没有 live runtime 调用, 应默认删除, 而不是继续包一层兼容。

典型候选:

- retired action: `setup`, `classify`, `backends`, `params`, `validate_stack`, `select_kernels`。
- retired system: parliament/vote/objection, old scoring module, old Cortex v1 graph KB path。
- 开发过程中的临时 PR/GAP/Nxx 叙事注释。
- 只服务旧单元测试、但不影响端到端的 helper。

### 2.3 兼容只保留明确产品价值

历史兼容不应无限期保留。对每一类兼容逻辑做明确决策:

- 是否继续支持旧 session resume?
- 是否继续支持旧 session breakdown 读取?
- 是否继续支持旧 payload 字段 `extra_sglang_args`?
- 是否继续支持 legacy flat session layout?
- 是否继续支持 robustness legacy mode?

如果答案是否, 直接删除兼容路径和对应测试。如果答案是暂时保留, 必须给兼容代码加一句短注释说明删除条件。

### 2.4 注释只解释不变量和意图

清理后注释应该短, 只解释代码本身看不出的原因。

保留:

- 外部协议约束。
- 运行时不变量。
- 不明显的失败模式。
- 兼容逻辑的保留原因和删除条件。

删除或压缩:

- PR 编号、Gap 编号、Nxx 编号的过程叙事。
- 与函数名/docstring 重复的逐行解释。
- 已经过期的实现历史。
- 长篇事故复盘, 除非它直接解释当前 guard。

### 2.5 测试为行为服务

测试也要清理。保留能证明端到端行为和协议不变的测试; 删除只锁死旧内部实现的小测试。

保留重点:

- action catalogue / executor wiring / phase allowlist drift tests。
- intent envelope / PolicyGate contract tests。
- session layout / state persistence / resume 关键路径测试。
- `session_breakdown.json` schema 和 downstream contract 测试。
- kernel request handler 的 path/shape/safety gate 测试。
- explore/integrate/sweep/report 的端到端行为测试。

可删除或合并:

- 只验证 helper 内部细节的测试。
- 与更高层测试重复的小型参数化测试。
- 为某次开发过程临时添加、但不再代表用户行为的测试。
- retired action 的正向测试。

## 3. 清理前置工作

### 3.1 建立 live contract 清单

先列出不可破坏的 live contract, 并用它指导删除。

核心文件:

- `inference_optimizer/cli.py`
- `inference_optimizer/orchestrator/coordinator.py`
- `inference_optimizer/orchestrator/phase_state.py`
- `inference_optimizer/orchestrator/policy.py`
- `inference_optimizer/orchestrator/intent_parser.py`
- `inference_optimizer/orchestrator/shared_state.py`
- `inference_optimizer/orchestrator/action_registry.py`
- `inference_optimizer/session_paths.py`
- `inference_optimizer/orchestrator/kernel_request_handlers.py`
- `inference_optimizer/breakdown/schema.py`
- `inference_optimizer/breakdown/exporter.py`

核心 live action 来源:

- `inference_optimizer/tests/test_action_catalogue.py`
- `inference_optimizer/actions/_meta/*.yaml`
- `cli._REAL_EXECUTORS_FULL`
- `cli._register_executors`
- `phase_state.PHASE_ALLOWED_ACTIONS`
- `policy.KERNEL_OWNED_ACTIONS`
- `session_paths._runs_actions`

### 3.2 建立代码分类

每个模块先标记为以下一类:

- `core`: 当前端到端流程必需。
- `contract`: 外部协议或文件格式必需。
- `compat`: 历史兼容, 等待产品决策。
- `test_only`: 只被测试引用。
- `dead`: 无 live runtime 价值, 可删除。
- `docs_only`: 文档或示例, 可压缩或删除。

清理时优先删除 `dead`, 然后处理 `test_only`, 再合并 `core` 中重复逻辑。

### 3.3 明确兼容策略

需要尽快做决策:

- 是否保留旧 session resume?
- 是否保留旧 `session_breakdown.json` reader?
- 是否保留 `extra_sglang_args` 读别名?
- 是否保留 old Cortex v1 KB 方法?
- 是否保留 multi-node 历史 `.bak` 和旧脚本路径?
- 是否保留 `examples/` 中历史 P0/P1/P2 demo?

建议策略:

- 生产端到端功能相关的兼容保留一轮。
- 只服务旧单元测试或旧文档的兼容直接删除。
- 对 `breakdown/collectors.py` 的历史 session reader 采取分批裁剪, 因为它可能影响 dashboard 和离线分析。

## 4. 分阶段执行计划

## Phase 0: 建立基线

目标: 知道清理前系统状态, 避免删除后无法判断破坏面。

动作:

- 跑最小关键测试集:
  - action catalogue
  - phase state
  - policy / intent parser
  - session paths
  - explore executor
  - kernel request handlers
  - breakdown exporter
  - coordinator runtime smoke
- 记录当前失败测试, 不把已有失败误认为清理引入。
- 生成 live action / executor / phase allowlist 对照表。
- 建立一个 `dead_code_candidates` 临时清单, 每个候选写明证据。

产出:

- 当前测试基线。
- live runtime map。
- 第一批确定可删清单。

## Phase 1: 删除确定无用文件和显式 dead code

目标: 先降低噪音, 不碰复杂 runtime。

候选:

- `.bak` 文件。
- coverage omit 中已经不存在的文件项。
- 明确未使用的旧脚本或旧示例。
- 指向不存在模块的文档引用。
- 已经被 tests 明确标记 retired 的正向路径。

重点检查:

- `multi_node/*.bak*`
- `examples/`
- `pyproject.toml` coverage omit stale entries
- `docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md` stale classify 内容
- `framework-agent` 中未实现 5th-role 的过度承诺文档

验收:

- 关键测试不新增失败。
- 文件树噪音减少。

## Phase 2: 收敛 action / executor / phase 三套来源

目标: 降低 action 体系复杂度, 避免一个 action 要在多处手动同步。

动作:

- 以 `ActionRegistry` 为中心, 收敛 action metadata。
- 检查 `phase_state.PHASE_ALLOWED_ACTIONS` 与 action yaml 是否有重复信息。
- 检查 `cli._REAL_EXECUTORS_FULL` 与 `session_paths._runs_actions()` 是否可以减少手写列表。
- 删除 retired action 的残余提示、注释和测试。
- 把 Coordinator-internal action 和 LLM-proposable action 的区别写成短表或常量, 避免散落在注释中。

重点:

- `target_analysis`
- `replay_warm_recipe`
- `profile`
- `roofline`
- `framework_pr`
- `conc_sweep`
- `specialist`
- `dynamic_action`
- `integrate_patch`
- kernel-owned actions

验收:

- live action 数量清晰。
- 删除 `backends/params/validate_stack/setup/classify/select_kernels` 的非兼容残留。
- `test_action_catalogue.py` 简化但仍能防 drift。

## Phase 3: 拆解和瘦身 Coordinator

目标: `coordinator.py` 是最大复杂度中心, 需要先消重再拆分。

先做低风险消重:

- 合并重复 backend error streak 初始化。
- 抽出 `tick()` 和 `run()` 共享的 resume rebuild / drain / abandon dynamic action 逻辑。
- 合并 FRAMEWORK_PR pump 的 try/except 包装。
- 合并 roofline pending gate 的重复检查。
- 合并 time budget prompt block。
- 合并 `extra_server_args` 兼容读取入口。

再做责任拆分:

- `coordinator_phase_hooks.py`: phase entry hooks, prelude/kernel/sweep/close enqueue。
- `coordinator_framework_pr.py`: framework PR discovery, critic gate, candidate enqueue。
- `coordinator_specialist.py`: specialist warmup, result recording, steward verdict。
- `coordinator_dynamic_action.py`: dynamic action dispatch/result/integrate lifecycle。
- `coordinator_kernel_flow.py`: kernel request response, kernel promotion, integrate keep。
- `coordinator_fact_write.py`: journal and RecipeKB fact writes。

拆分原则:

- 不改变 public runtime 行为。
- 私有 helper 可以移动和改名。
- 每次只迁移一个责任域。
- 每次迁移后跑对应测试子集。

验收:

- `coordinator.py` 行数明显下降。
- phase / proposal / dispatcher 主循环更容易阅读。
- 没有新增跨模块循环 import。

## Phase 4: SharedState 和兼容迁移清理

目标: 减少 `state.json` schema 和历史字段迁移复杂度。

动作:

- 列出 `SharedState` 中所有字段, 标记 live / compat / dead。
- 删除已经不写、不读、不进 breakdown 的字段。
- 对保留的 compat migration 写短注释和删除条件。
- 把 `extra_sglang_args -> extra_server_args` 迁移集中到单一模块。
- 删除旧 scoreboard / old select_kernels / old T2/T3 的无用字段。

谨慎区域:

- `optimization_stack`
- `explore_search`
- `gaps`
- `phase_history`
- `last_trace_analyze`
- `kernel_opt_attempts`
- `warm_start_recipe`
- `close_sequence_done`

验收:

- 新 session 的 `state.json` 只含当前运行需要的字段。
- 旧字段不会在新写路径出现。
- compat reader 如果保留, 只在 load/migration 层出现。

## Phase 5: Knowledge / KB / research priors 清理

目标: 统一当前 RecipeKB 与 KnowledgePlane 逻辑, 删除旧 Cortex v1 残留。

动作:

- 确认 CLI 当前是否永远传入 RecipeKB 而不是 v1 Cortex client。
- 删除或隔离 `knowledge_plane.py` 中标记为 dead/grep-stability 的 cortex 方法。
- 检查 `cortex_kb_constants.py` 是否还需要。
- 收敛 `recipe_snapshot_constants.py`, `recipe_kb/schema.py`, `cortex_t0.py` 的职责。
- 简化 research scout hints 的读取和 prompt 注入路径。

验收:

- KB 路径只剩当前 v2 recipe snapshot 和必要 PR feed。
- prompt 中不再出现旧 KB 协议叙事。

## Phase 6: Kernel / framework / robustness 外围边界清理

目标: 外围系统保留清晰边界, 避免主包里复制过多逻辑。

Kernel:

- 保留 `kernel_request_handlers.py` 作为唯一 Coordinator-side kernel bridge。
- 删除已退场 request kind。
- 保留 shape/path/safety gates。
- 精简 GEAK/OOB/Cursor backend 顺序的长注释。

Framework:

- 明确当前只接入 `fa phase-discover`。
- 删除或修正文档中 `fa agent` / 5th-role 已实现的暗示。
- 保留 `FrameworkPrExecutor` 的 apply/bench/KEEP/REVERT 主流程。
- 去掉未调用的 wrapper 和重复 repo map。

Robustness:

- 保留 envelope mirror 和 contract tests。
- 清理 legacy mode 文档和未使用 provider。
- 缩短 `role/envelope.py` 里和 upstream 同步相关的长注释。

Critic:

- 保留 prepare-review / commit-review subprocess bridge。
- 删除旧 bare Codex path 的未使用提示, 如果生产不再支持。

验收:

- 主包对外围系统的依赖边界清楚。
- 外围系统不再在主包中复制过多策略。

## Phase 7: Breakdown 和对外报告清理

目标: 保留对外 dashboard contract, 删除内部历史噪音。

动作:

- 对齐 `docs/INTEGRATION_SESSION_BREAKDOWN.md` 与 `breakdown/schema.py` 的版本描述。
- 明确 `schema_version` 当前值和 additive fields 策略。
- 清理 `breakdown/collectors.py` 中只服务很老 session 的 reader。
- 将 collector 按 section 拆分, 减少单文件复杂度。
- 删除 `param_search` 中旧 `backends/params/validate_stack` 兼容 alias, 如果 dashboard 不再需要。

谨慎:

- dashboard 和 `claw-stats-service` 是否仍依赖旧字段。
- 旧 session 离线分析是否需要保留。

验收:

- 新 session breakdown 简洁。
- 旧兼容路径有明确保留/删除策略。
- schema 文档和代码一致。

## Phase 8: 注释和文档压缩

目标: 大幅降低阅读成本。

动作:

- 对每个 core 文件做注释 pass。
- 删除 PR/GAP/Nxx 过程叙事。
- 长注释改为一两句功能性说明。
- docstring 只保留 API 意图和重要契约。
- README / SKILL / docs 中删除已退场流程。

优先文件:

- `coordinator.py`
- `shared_state.py`
- `policy.py`
- `phase_state.py`
- `cli.py`
- `kernel_request_handlers.py`
- `breakdown/collectors.py`
- `specialist_runner.py`
- `specialist_prompt_builder.py`
- `robustness-agent/src/robustness_agent/role/envelope.py`

注释压缩示例:

Before:

```text
PR-A8 (Arbor-into-Hyperloom): ... long historical explanation ...
```

After:

```text
Track intervention mix so robustness can detect config-only exploration loops.
```

验收:

- 注释主要解释当前行为, 不再讲开发历史。
- 文件开头不再出现过期架构描述。

## Phase 9: 测试体系重组

目标: 保留行为护栏, 删除实现细节负担。

动作:

- 按功能域重组测试:
  - `test_runtime_contracts.py`
  - `test_phase_and_policy.py`
  - `test_action_catalogue.py`
  - `test_explore_flow.py`
  - `test_kernel_flow.py`
  - `test_framework_pr_flow.py`
  - `test_specialist_dynamic_flow.py`
  - `test_breakdown_contract.py`
  - `test_cli_smoke.py`
- 合并小测试文件。
- 删除只验证私有 helper 的重复测试。
- 删除 retired feature 的正向测试。
- 对复杂 helper 保留少量代表性边界测试。

保留测试原则:

- 保证端到端流程不破。
- 保证协议不破。
- 保证关键安全门禁不破。
- 保证输出 artifact 可消费。

不保留:

- 对旧内部函数名字的断言。
- 对中间私有 dict 形状的过度断言。
- 开发过程中为某个 PR 临时加的重复回归测试。

验收:

- 测试数量下降。
- 测试失败更能指向用户可见行为。
- 清理后的测试不再阻止内部实现简化。

## Phase 10: 文件结构和 import 结构整理

目标: 降低引用复杂度。

建议结构:

```text
inference_optimizer/
  cli.py
  paths.py
  session_paths.py
  runtime/
    coordinator.py
    phase_hooks.py
    dispatcher.py
    proposal_flow.py
    kernel_flow.py
    specialist_flow.py
    framework_pr_flow.py
    fact_write.py
  protocol/
    intent_parser.py
    policy.py
    agent_role.py
  state/
    shared_state.py
    migrations.py
  actions/
    _meta/
    playbooks/
  executors/
    baseline.py
    explore.py
    integrate_patch.py
    framework_pr.py
    roofline.py
    sweep.py
    report.py
  integrations/
    kernel_agent.py
    critic_agent.py
    robustness_agent.py
    framework_agent.py
  breakdown/
  recipe_kb/
```

迁移策略:

- 不一次性大搬家。
- 先创建新模块并移动一个责任域。
- 保持旧 import 兼容一小段时间, 或一次性替换所有 import。
- 每次迁移后跑对应测试。

验收:

- import 方向单向: CLI -> runtime -> executors/integrations/state/protocol。
- `policy` 和 `phase_state` 保持纯逻辑, 不依赖 runtime heavy modules。
- executors 不反向依赖 Coordinator。

## 5. 优先清理清单

### 5.1 第一批可立即处理

- 删除 `.bak` 文件。
- 删除 coverage omit 中不存在的路径。
- 修正 stale 文档和注释。
- 删除 `Coordinator.__init__` 重复初始化。
- 合并 `tick()` / `run()` resume drain 逻辑。
- 清理 `agent_role.py` 中与当前 robustness 状态矛盾的说明。
- 修正 `docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md` 中 stale classify 描述。

### 5.2 第二批需要确认后处理

- 删除 old Cortex v1 KB 路径。
- 删除 `extra_sglang_args` 兼容别名。
- 删除旧 session breakdown reader。
- 删除 legacy flat session layout。
- 删除 robustness legacy mode。
- 删除 examples 历史 demo。

### 5.3 第三批结构性整理

- 拆分 `coordinator.py`。
- 拆分 `breakdown/collectors.py`。
- 收敛 action metadata / executor registry / session path action whitelist。
- 重组 tests。
- 重组 package 目录。

## 6. 风险和控制

### 6.1 主要风险

- 删除了通过字符串或 subprocess 调用的代码, import 图看不出来。
- 删除了 dashboard 仍依赖的 breakdown 字段。
- 删除了 resume 仍需要的 migration。
- 删除了 PolicyGate / intent envelope mirror 的同步逻辑。
- 删除了 kernel-agent/framework-agent 的隐式 artifact path。
- 测试删太多导致协议漂移无人发现。

### 6.2 控制方式

- 每个删除候选必须写明证据: 无 runtime wiring, 无 action metadata, 无 subprocess bridge, 无 artifact contract。
- 端到端 contract 测试优先保留。
- 删除 compat 前先做产品决策。
- 每个阶段单独提交, 不混合大规模移动和行为删除。
- 每轮跑最小关键测试集。

## 7. 建议测试分层

### Tier 0: 快速 contract tests

每次小改都跑:

- action catalogue
- phase state
- policy / intent parser
- session paths
- payload alias, 如果还保留

### Tier 1: runtime flow tests

每个阶段完成后跑:

- coordinator runtime
- explore executor
- kernel request handlers
- framework PR executor
- specialist / dynamic action flow
- breakdown exporter

### Tier 2: full package tests

大阶段完成后跑:

- `inference_optimizer/tests`
- `robustness-agent/tests`
- critic integration tests, 如果环境允许

### Tier 3: real smoke

最终验收:

- 一次 no-kernel optimize smoke。
- 一次 kernel-enabled dry/smoke。
- 一次 session breakdown dump。
- 一次 resume smoke。

## 8. 最终验收标准

清理完成后应达到:

- 代码文件数量下降。
- `coordinator.py`, `shared_state.py`, `breakdown/collectors.py`, `cli.py` 明显瘦身。
- retired action 和 retired system 不再污染主流程。
- 注释长度下降, 文件头描述和当前实现一致。
- 测试数量减少, 但端到端 contract 更清晰。
- 新工程师能从 CLI -> Coordinator -> phase -> executor -> artifact 的路径快速理解系统。
- 对维护者来说, 增加一个新 action 或新 phase 不再需要修改多个互相漂移的手写列表。

## 9. 推荐执行顺序摘要

1. 建立 baseline 和 live contract。
2. 删除明显 dead 文件和 stale 文档。
3. 收敛 action catalogue / executor / phase / path whitelist。
4. 消重 Coordinator。
5. 清理 SharedState migration 和旧字段。
6. 清理 KB / KnowledgePlane 旧路径。
7. 清理外围 agent bridge 的 stale 注释和未实现承诺。
8. 清理 breakdown legacy reader 和 schema 文档漂移。
9. 压缩注释。
10. 重组测试。
11. 最后再做文件结构搬迁。

核心态度: 这次清理应该大胆。只要端到端功能、外部协议和最终产物不变, 内部单元实现、历史兼容、过度测试和过程性注释都可以被合并或删除。
