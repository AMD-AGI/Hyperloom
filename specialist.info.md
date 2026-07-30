<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Hyperloom 与 Arbor Specialist 机制调研

> 本文基于 `/wekafs/zgong/Hyperloom` 与 `/wekafs/zgong/Arbor` 的当前源码，重点分析 specialist 派发、模型间通信、上下文管理、自治边界，以及 Hyperloom 为稳定性加入的兜底和限制。
>
> 调研采用分片源码审查、独立综合、反驳式验证和证据复核。第一次全量 safeguard inventory 子代理因任务过大而停滞；随后将审查拆为 Policy/schema、生命周期、资源隔离、下游准入、软约束五部分重新运行。重跑中的一个 evidence auditor 因网关要求 NTID 返回 HTTP 400，但其任务随后由独立证据审查补齐。最终 47 项控制全部复核，发现并修正两处行号引用，计数结论保留。
>
> **更新（分支 `feat/zgong/explore-opt-11`，commit `d2d82cf5c`）**：第 3 节的 inbox 尾窗与第 4.2 节的 compaction 阈值已按实现改动重写——inbox 尾窗删除（本 tick 全量渲染），hard 水位/anti-thrash/emergency bypass 全部删除。第 8 节的 47 项 safeguard 表不含这两项，计数不变。同时修正了若干因 `orchestrator/{loop,state,bus}/` 重组而失效的路径引用。

## 1. 执行摘要

Arbor 和 Hyperloom 不是两个完全独立的系统：

- Arbor 的 `CLAUDE.md` 将其称为 “Arbor (internal name: Hyperloom)”。
- Hyperloom 的 `CLAUDE.md` 说明其树搜索编排以 Arbor 名义发布。
- Hyperloom 的 specialist 文档明确说这里的 specialist 就是 Arbor Domain Specialist：`src/hyperloom/inference_optimizer/actions/specialist.md:12-13`。

源码不能证明“Arbor 是旧版、Hyperloom 是后来的产品化重写”这一时间顺序。更稳妥的表述是：**两者是同一套系统思想的不同实现表面**。

两者都采用类似的主循环：

```text
profile -> gap -> dispatch specialist -> author proposal/patch
        -> integrate -> benchmark -> KEEP/REVERT
```

核心差别不是“有没有能力”，而是“能力由谁约束”：“

- **Arbor DFS specialist** 的约束主要写在 prompt 和文件协议里，结果合同开放，更多判断留给模型。
- **Hyperloom specialist** 的约束主要写进 PolicyGate、任务状态机、资源租约、worktree、结构化输出、Critic 准入和 benchmark/rollback 逻辑里。

因此：

> Arbor 给 specialist 更大的**语义和治理自由**；Hyperloom 给 specialist 不弱、部分场景甚至更强的**操作能力**，但用确定性 runtime 把派发、资源、输出、集成和失败恢复包起来。

经重新分片审查和逐项证据复核，Hyperloom 的控制分两层：

| 范围 | 默认硬控制 | 条件控制 | 软约束 | 合计 |
|---|---:|---:|---:|---:|
| A. Specialist 生命周期专属 | 26 | 9 | 4 | 39 |
| B. 通用下游优化正确性 | 2 | 5 | 1 | 8 |
| 并集 | 28 | 14 | 5 | 47 |

这里按**独立执行边界/不变量**计数，而不是按 PolicyGate 错误字符串、调用次数或重复检查计数。

---

## 2. Hyperloom 的 specialist 派发

### 2.1 正常链路

```text
Orchestration LLM
    │
    │ DELEGATE(action_name="specialist", params=...)
    ▼
IntentRouter._handle_intent
    ▼
PolicyGate.validate_intent
    │  检查发送角色、dispatch payload、GPU 请求等
    ▼
IntentRouter._handle_delegate
    ▼
TaskRegistry.create_or_return_existing
    │  SQLite 持久化、幂等键、任务状态机
    ▼
Dispatcher / SubAgentRunner / SpecialistRunner
    │
    ├─ 获取 research lane 和 GPU/Ray lease
    ├─ 尝试创建每任务 git worktree
    ├─ 组装 specialist prompt
    └─ 启动 claude --print --output-format stream-json
           │
           ├─ heartbeat / process.log
           ├─ specialist_partial.json
           ├─ specialist_done.json
           └─ patch / artifact
    ▼
Runner._finalize
    │  清洗、重新盖章、patch provenance、失败结果合成
    ▼
proposal_set / patch autosubmit
    ▼
Critic review
    ▼
integrate_patch gate
    ▼
apply -> benchmark/correctness -> KEEP / REVERT
```

入口位置：

- `src/hyperloom/orchestrator/loop/intent_router.py:70`
- `src/hyperloom/orchestrator/loop/intent_router.py:343`
- `src/hyperloom/orchestrator/loop/intent_router.py:458`
- `src/hyperloom/orchestrator/policy/gate.py:1606-1635`

只有 orchestration 可以通过路由入口派发 specialist：

```python
SPECIALIST_DISPATCH_SOURCE_ALLOWLIST = frozenset({"orchestration"})
```

这使角色身份同时成为授权主体，而不只是显示名称。

### 2.2 Specialist 入口不经过 Critic，产物集成才经过 Critic

Specialist 启动前主要由 PolicyGate 检查派发身份、payload、GPU 等，并不先让 Critic 审核研究任务。Critic 的强制门位于 patch 集成阶段：

- `src/hyperloom/orchestrator/policy/gate.py:1530-1604`
- `src/hyperloom/orchestrator/actions/executors/integrate_patch.py:1515-1527`

Specialist patch 必须携带 `specialist_task_id`，并已有持久化的 `approve` 或 `advise` verdict，才能进入副作用阶段。只有 critic 身份可以发 review verdict：

- `src/hyperloom/orchestrator/policy/gate.py:419`
- `src/hyperloom/orchestrator/policy/gate.py:1214`

这形成“研究宽、写入窄”的边界：

```text
允许 specialist 自由研究
          !=
允许 specialist 结果直接进入 serving workspace
```

注意：`enablement_launch_only` 有豁免；`advise` 也属于 permissive verdict；使用 `--critic-mock` 时会机械批准，因此“有 verdict”与“有实质性审查”是两个不同控制。

### 2.3 Coordinator 内部存在受控旁路

“只有 orchestration 可派发”描述的是路由授权边界，不代表 Coordinator 内部所有创建点都严格经过同一函数：

- stalled-domain 强制 specialist 以 `orchestration` 身份重入 `_handle_intent`：`src/hyperloom/orchestrator/phases/explore.py:540`。
- fan-out wave 可直接调用 `_handle_delegate`：`src/hyperloom/orchestrator/phases/explore.py:685`。
- framework phase 若干位置直接创建任务，再通过 `validate_dispatched_task` 重放部分验证，例如 `src/hyperloom/orchestrator/phases/framework.py:922,1557,2108,2498,5322`。

这些是 Coordinator 内部调度路径，不是普通 agent 可调用的公共旁路；但它们说明 PolicyGate 保护的是**路由入口**，并不构成对所有直接 executor/registry 调用的全局 capability sandbox。

---

## 3. Hyperloom 的模型间通信

Hyperloom 不是一个所有模型共享同一聊天记录、彼此直接对话的系统，而是多种通信通道并存。

### 3.1 Orchestration：provider 侧持续会话

Orchestration 使用 Claude SDK resumed conversation：

- 第一轮发送完整 SEED；
- 后续轮次发送较薄的 DELTA；
- 大块上下文通过 pull-on-demand 获取。

位置：

- `src/hyperloom/orchestrator/roles/claude.py:284`
- `src/hyperloom/orchestrator/roles/claude.py:482`

历史主要存在 provider 侧会话中，而不是本地维护一个每轮全量重发的无界 `messages[]`。

### 3.2 Specialist：任务/提示下行，done/artifact 上行

生产路径的 specialist 是独立 `claude` CLI 子进程。父侧把 prompt、任务、工作目录和资源交给子进程；子侧通过 heartbeat、partial、done 和 patch/artifact 文件回传结果。

合法 specialist intent 被限制为：

- `SEND_MESSAGE`
- `ALERT`
- `SPECIALIST_DONE`

见 `src/hyperloom/orchestrator/policy/gate.py:2025-2070`。

需区分两条退出路径：

1. **in-process / emit_intent**：经过 PolicyGate 的 specialist payload 验证。
2. **生产 subprocess 文件路径**：读取 `specialist_done.json`，依赖 `SpecialistRunner._finalize` 的清洗、task/gap/domain 重盖章和 patch provenance，而不是再次完整通过同一 R3 validator。

相关位置：

- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py:1737-1748`
- `src/hyperloom/orchestrator/specialists/subprocess_.py:676-677`
- `src/hyperloom/orchestrator/specialists/runner.py:1198-1328`

### 3.3 Critic / Robustness：JSON IPC

Critic 和 Robustness 通过 CLI/子进程 JSON request/response bridge 通信：

- `src/hyperloom/orchestrator/roles/_runtime_bridge.py:53-90`

因此不能说所有实时模型通信都走 message bus。

### 3.4 SQLite message bus：持久化事实层，不是唯一实时线路

Message bus 是 append-only SQLite 事件系统：

- topic 和 priority 在写入时经过 allowlist；
- 事件有序列；
- consumer 有 cursor；
- 支持 replay/resume。

位置：

- `src/hyperloom/orchestrator/bus/message_bus.py:189-192`
- `src/hyperloom/orchestrator/bus/message_bus.py:236-250`
- `src/hyperloom/orchestrator/bus/storage/schema.py:66-89`

更准确的模型是：

```text
实时执行通道：SDK resumed session / JSON IPC / task+done files
持久化审计层：SQLite message bus
模型当前视图：从持久化状态投影出的有界 prompt
```

Prompt inbox 渲染本 tick 的**全部**未读消息（`feat/zgong/explore-opt-11` 起）。此前有一个「只渲染最近约 20 条」的尾窗，已删除——信息完整性优先于 prompt 体积：

- `src/hyperloom/orchestrator/loop/conversation.py:549-555`（`cursor.load` → `bus.replay_for(after_seq=...)` → `rendered = list(msgs)`）

Critic 通过 pending proposals 获得额外补投：

- `src/hyperloom/orchestrator/loop/conversation.py:560`、`571`（`_augment_critic_inbox_with_pending`）

所以“系统记得多少”与“模型这一轮看到多少”仍然是分离的，但分界线变了：

- **同一 tick 内**：不再有截断，一批未读消息全量进入 prompt。
- **跨 tick**：cursor 已消费的历史不会重放。这是删除尾窗后**遗留的**信息缺口——`intent_router.py:100` 在每个 intent 之后都会 `_cursor_advance_to_latest`，所以尚未裁决的 proposal 依然可能被 cursor 跳过。这正是 `_augment_critic_inbox_with_pending` 必须保留的原因：只有它能从 durable `pending_proposals` 重新投递。orchestration/robustness 没有等价补投。

---

## 4. Hyperloom 的上下文长度管理

### 4.1 SEED + DELTA + pull-on-demand

Hyperloom 不在每轮重复发送完整运行状态，而采用：

```text
首次：完整 SEED
后续：轻量 DELTA
大内容：按需读取
```

这样减少重复 token，稳定 prompt cache，并提高变化信息的信噪比。

### 4.2 水位驱动 compaction

默认上下文窗口按 200K 估算。`feat/zgong/explore-opt-11` 删除了所有 hard/bypass 层，只剩单一软水位加节奏触发：

- soft threshold：70%（`DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION`）
- cadence：每 20 ticks / 30 分钟 / 400K 本地字符，或 phase 切换
- ~~hard threshold 85%~~、~~anti-thrashing 3 ticks 下限~~、~~emergency bypass 98%~~ —— 已删除

随之删除的符号：`is_hard_compaction`、`context_token_hard`、`DEFAULT_CONTEXT_TOKEN_HARD_FRACTION`、`deterministic_memory_fallback`、`_checkpoint_min_tick_gap`，以及 `INFERENCE_OPTIMIZER_CTX_HARD_FRACTION` 环境变量。degenerate summary 现在一律跳过 compaction（不再有 hard 路径的确定性 fallback），连续 3 次 degenerate 才升级告警。

位置：

- `src/hyperloom/orchestrator/state/orchestration_memory.py:22-32`（阈值常量）、`:76-99`（`should_checkpoint`）
- `src/hyperloom/orchestrator/loop/maintenance.py:195-213`（触发判定）、`:225-258`（degenerate 处理）

Compaction 不是简单删除旧消息，而是：

```text
provider transcript
    ↓ 模型自总结
持久化 working-memory / SharedState
    ↓ 结束旧会话并重新 SEED
new provider transcript
```

这把长期事实放入 durable state，把短期推理保留在会话上下文。

### 4.3 Token 水位的依据和局限

真实水位主要依赖 provider 返回的 input/cache usage。仓库内没有精确 tokenizer。本地字符数 fallback 只能累计本地构造的 delta 与 reply，无法完整看见 provider 保存的 resumed transcript，因此可能低估。

此外，maintenance 对没有 `conversational` 属性的 CodexBackend 会提前返回：

- `src/hyperloom/orchestrator/loop/maintenance.py:180-183`

所以成熟的 compaction 生命周期目前主要属于 Claude conversational backend，不是所有 backend 的完全统一能力。

---

## 5. Arbor 的两套 agent 机制

Arbor 内部至少有两套结构不同的 agent 系统，不能一概而论。

### 5.1 DFS / Domain Specialist

这一层最接近 Hyperloom specialist：

```text
Arbor orchestrator
    ↓ python3 -m arbor.cli_dispatch_tools dispatch
flock GPU pool acquire
    ↓
claude --print --output-format stream-json
    ↓
agents/<id>/
    ├─ task.json
    ├─ heartbeat
    ├─ results.jsonl
    ├─ patches/
    ├─ new_knowledge.md
    └─ done.json
```

位置：

- `/wekafs/zgong/Arbor/src/arbor/dispatch.py:234-297`
- `/wekafs/zgong/Arbor/src/arbor/comms.py:41`

它与 Hyperloom 一样使用独立 Claude CLI、GPU 分配、`ROCR_VISIBLE_DEVICES`、heartbeat 和 done 文件，但输出合同更开放：

- `results.jsonl` 可承载 patch、config change、分析等自由类别；
- 可写 `patches/`；
- 可写 `new_knowledge.md`；
- 最终由 orchestrator LLM 解释产物。

协议见 `/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:68-116`。

### 5.2 Kernel-agents

Kernel phase 使用 SDK-native `AgentDefinition` 和 fellow agents：

- fellow 不拥有 Agent/Task 递归能力；
- shape provenance 在 dispatch 前验证；
- KB 按需加载；
- history 紧凑化；
- 有 flock 序列化、最多约 200 条消息的共享 board。

位置：

- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/fellows/base.py:31`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/orchestrator/agent.py:173`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/workspace/coordinator.py:105-170`

Board 不是死代码：`post()` 有真实调用点：

- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/workspace/coordinator.py:234`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/workspace/coordinator.py:254`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/workspace/coordinator.py:300`

Fellows 根据 prompt 自主读写 board；orchestrator 本身并不集中驱动所有 A2A 消息。

---

## 6. Arbor 的上下文管理

### 6.1 DFS orchestration：本地无界 messages

DFS SDK orchestrator 在本地持续 append `messages`，下一轮重发该列表：

- `/wekafs/zgong/Arbor/src/arbor/cli.py:577`
- `/wekafs/zgong/Arbor/src/arbor/cli.py:603`

`max_turns` 高达 10000：

- `/wekafs/zgong/Arbor/src/arbor/cli.py:391`

模型未产生工具调用时还会注入 `[CONTINUE]`：

- `/wekafs/zgong/Arbor/src/arbor/cli.py:614-621`

主要保护是工具/文件输出截断：

- 约 50,000：`/wekafs/zgong/Arbor/src/arbor/cli.py:655`
- 约 100,000：`/wekafs/zgong/Arbor/src/arbor/cli.py:660`

没有看到 Hyperloom 风格的 provider-usage 水位、self-summary、session reseed。（注：Hyperloom 侧的 anti-thrashing 也已在 `feat/zgong/explore-opt-11` 删除，见 4.2。）DFS KB 还会整文件读入：

- `/wekafs/zgong/Arbor/src/arbor/kb.py:107-108`

因此 DFS 的自由探索伴随明显的上下文膨胀风险。

### 6.2 Kernel phase：上下文反而很节制

Kernel-agents 使用：

- KB 索引与按需加载：`/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/knowledge/loader.py:86-107`
- 约 5 项紧凑 history：`/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/loop/runner.py:416`
- 工具输出过滤和 cache-stable prompt。

因此不能笼统说 Arbor 不管理上下文：**DFS 较粗放，kernel-agents 较节制；Hyperloom 则把上下文生命周期提升到总编排 runtime。**

---

## 7. 为什么 Arbor specialist 更自由

### 7.1 Prompt 治理多，代码治理少

Arbor DFS 的关键原则包括：

- “Trust your agents”
- “No human approval is needed”
- 除少量 ground rules 外 “no restrictions”

位置：

- `/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:141-149`
- `/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:223`
- `/wekafs/zgong/Arbor/skills/SKILL.md:556`

这些是概率性的行为指导。Hyperloom 则把相同担忧实现成 PolicyGate、状态机和资源控制。

### 7.2 开放结果合同

Arbor specialist 可提交任意合理的 `results.jsonl` 类别、patch 和知识，由 orchestrator 解释。Hyperloom 把退出收敛为结构化 `specialist_done`，然后通过确定性代码持久化、清洗、送审和集成。

### 7.3 没有代码级 Critic 集成门

Arbor 失败后的主要策略是：

```text
classify failure -> escalation prompt -> launch fresh agent
```

位置：

- `/wekafs/zgong/Arbor/src/arbor/dispatch.py:488-500`
- `/wekafs/zgong/Arbor/src/arbor/dispatch.py:561-575`

Hyperloom 则要求 persisted Critic verdict 后才能执行 specialist patch integration。

### 7.4 “Arbor 更自由”只在部分轴成立

Arbor general specialists 也能：

- 获取 GPU lease：`/wekafs/zgong/Arbor/src/arbor/dispatch.py:421-439`
- 跑 micro-benchmark：`/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:144-145,170-173`
- 编写 patch：`/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:79-83`

它主要禁止自己运行完整 E2E serving benchmark：`/wekafs/zgong/Arbor/src/arbor/prompt_builder.py:146`。

Hyperloom specialist 在租用 GPU 上可以启动自己的非 8888 真实 server：

- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py:916-943`
- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py:954-983`

所以从**原始实验能力**看，Hyperloom 未必更弱；Arbor 更自由主要体现在治理和表达空间。并且 Arbor 的 kernel-agents 有 shape provenance、单层 fan-out 和 benchmark gate，也不是自由放任。

另一个细节是 Arbor orchestrator 是否拥有 Task 工具取决于启动路径：

- `src/arbor/cli.py` 路径排除 Task：`/wekafs/zgong/Arbor/src/arbor/cli.py:289-308`
- launcher 路径授予完整工具：`/wekafs/zgong/Arbor/launcher/run.sh:263`、`/wekafs/zgong/Arbor/launcher/orchestrator.md:3`

---

## 8. Hyperloom safeguard 全量清单

### 8.1 计数方法

计数原则：

1. 同一检查在多个位置重复执行只算一个控制，例如 persisted Critic verdict 在 PolicyGate 和 executor 重查仍算一个。
2. 同一函数中服务于同一准入边界的多个 matcher 合并，例如 freeform destructive-text tripwire 合入 dispatch payload gate。
3. 不按 `rule="..."` 错误字符串数量计数；整个 PolicyGate 虽可找到约 38 个不同规则名，但不少不属于 specialist 生命周期，或只是同一边界的不同失败原因。
4. Prompt 指导不算硬执行。
5. 通用 integrate/benchmark 正确性控制单独列为 B，不用来膨胀 specialist 专属数量。

### 8.2 A：Specialist 生命周期专属控制（39 项）

| # | 控制 | 强度 | 核心行为与证据 | 重要限制 |
|---:|---|---|---|---|
| 1 | Specialist dispatch ownership | 硬 | 仅 orchestration 可 delegate specialist；`policy/gate.py:1606-1635` | 不覆盖直接 executor 调用 |
| 2 | Specialist emitter confinement | 硬 | `specialist:<task_id>` 仅能发 done/message/alert；`gate.py:2025-2070` | task suffix 未与真实 worker/task 做认证绑定 |
| 3 | Dispatch payload/red-line gate | 硬 | 校验 anchored/freeform payload、gap、scope、waves、部分危险文本；`gate.py:1636-1741,1928-2023` | freeform 不校验 max_turns；文本扫描不是 Bash sandbox |
| 4 | GPU request admission | 硬 | GPU 必须是正整数、pool 可用且不超容量；`gate.py:1743-1845` | whole-machine profile 走不同 pool |
| 5 | Completion schema | 硬 | 校验 gap/domain/proposal_set/summary/confidence；`gate.py:2072-2169` | 无 proposal 数量上限、唯一名、closed schema 或 task 对照 |
| 6 | Persisted Critic verdict gate | 硬 | patch 集成前需 specialist_task_id 与 approve/advise；`gate.py:1530-1604`、`integrate_patch.py:1515-1527` | mock critic、advise permissive、部分豁免 |
| 7 | Persistent task state machine | 硬 | SQLite 状态机、合法迁移、attempt、唯一幂等键；`state/task_registry.py:28-46,145-218,276-329` | `kill_task` 主要改 DB/广播，不直接杀 asyncio/subprocess |
| 8 | Delegate idempotency/rollover | 硬 | 活跃重复拒绝，terminal 后最多五个 retry suffix；`loop/intent_router.py:383-499` | SELECT-then-INSERT 跨 writer 仍可能竞态 |
| 9 | Per-pump duplicate guard | 硬 | 单次 pump 内 task id 不重复 spawn；`loop/dispatcher.py:167-178,302-350` | 跨 pump 依赖持久状态 |
| 10 | Subprocess liveness envelope | 硬 | 调度/transport timeout、heartbeat/log、wall cap、reap；`specialists/subprocess_.py:453-579,701-844` | malformed done 可能先触发 reap；partial 可掩盖强制终止 |
| 11 | Structured lease release | 硬 | finally 中释放 GPU/Ray/lane；`loop/dispatcher.py:576-630`、`loop/sub_agent_runner.py:256-307` | cancellation 可释放 lease 却绕过 terminal bookkeeping |
| 12 | Orphan/expiry watchdog | 条件 | maintenance 识别 dead/expired row 并回收；`dispatcher.py:134-166`、`task_registry.py:367-500` | 依赖 watchdog 执行和 PID/TTL 证据，失败任务可能滞留 |
| 13 | Single-exit failure synthesis | 硬 | 各类失败规范化为空 `specialist_done`；`specialists/runner.py:331-362,1198-1220` | 需看 status/error 才能保留原始失败语义 |
| 14 | Partial-result recovery | 条件 | final 缺失时采用有效 partial；`runner.py:950-968`、`subprocess_.py:595-606` | 仅对 parseable partial 生效，可能把强杀表现成成功 |
| 15 | Bounded transient retry | 条件 | timeout/stale/crash 创建有限 fresh retry；`phases/explore.py:689-813` | 可由环境关闭，只覆盖进入 reaper 的 SubAgentResult |
| 16 | Atomic artifacts/state | 硬 | 父侧原子替换 heartbeat/partial/final/state；`runner.py:1521-1593`、`state/shared_state.py:1081-1167` | 生产 child 受提示使用 helper，并非 OS 强制 |
| 17 | Idempotent round ledger | 硬 | task id 作为 round id，已有 round 做 replace；`state/_shared_state/explore_state.py:57-97`、`phases/explore.py:1705-1759` | 相邻 counter/last bookkeeping 仍可能重复 |
| 18 | Serving-GPU carve-off | 条件 | 默认 pool 排除 serving process 的前 `serving_tp` 卡；`bus/gpu_pool.py:99-146`、`dispatcher.py:439` | `serving_tp=0`、显式 pool、whole-machine profile 可绕开 |
| 19 | GPU visibility pinning | 硬 | 本地 child 绑定 lease GPU，CPU specialist 清空可见 GPU；`subprocess_.py:458-477` | 环境 mask 不是 device cgroup |
| 20 | SQLite lease atomicity/TTL | 硬 | immediate transaction、unique key、过期清理；`bus/gpu_pool.py:261-264`、`bus/resource_lock.py:164-236` | 过期 row 需下次 acquire/maintenance 清理 |
| 21 | Lane serialization/concurrency | 硬 | research lane 与 serving/benchmark/profile 冲突表和容量限制；`bus/resource_lock.py:56-76`、`bus/storage/schema.py:30-35`、`loop/dispatcher.py:302-395,938-957` | capacity 可由 operator 配置 |
| 22 | Per-task git worktree | 条件 | 尝试创建 task branch/worktree 并作为 cwd；`subprocess_.py:223-296,391-395` | **fail-open**：无 git root 或 setup 失败会在非隔离 workspace 继续 |
| 23 | CLI tool allowlist/KB denylist | 硬 | 显式 allowedTools、移除无效 MCP/emit_intent、固定 KB-write denylist；`runner.py:95-123,423-457`、`subprocess_.py:633-699` | 仍允许 Bash/Edit/Write，非文件/进程 sandbox |
| 24 | Claude permission classifier | 条件 | restrictive permission mode 可恢复 per-tool classifier；`subprocess_.py:100-126`、`cli/executors.py:171-190` | 默认 `bypassPermissions`，通常只剩 tool allowlist |
| 25 | Single-layer leaf fan-out | 硬 | leaf 无 Task 工具，递归深度最多一层；`specialists/leaf.py:20-33`、`subprocess_.py:676-684` | Bash 仍可创建普通子进程 |
| 26 | Child environment allowlist | 硬 | 从固定普通变量/secret 名单重建环境；`subprocess_.py:40-97` | provider credential 默认继承，需显式关闭 |
| 27 | Preload/startup-hook scrub | 硬 | 移除 loader/preload 和语言启动 hook；`common/env_safety.py:41-65,203-220` | PATH/library paths 在其他表面管理 |
| 28 | Untrusted env override filter | 硬 | 拒绝危险 loader/path/shell/非法 env 名；`common/env_safety.py:22-40,164-200` | 不移除 base env 已接纳 credential |
| 29 | MCP config protection | 条件 | 尝试 chmod 0600，可不持久化 auth header；`specialists/mcp_config.py:38-115` | chmod 失败非致命，header 默认仍持久化 |
| 30 | Rebench helper port rejection | 硬 | helper 拒绝 8888 并选其他端口；`specialists/rebench.py:41-73` | 任意 Bash 启动 server 可绕过 helper |
| 31 | Raw server port guidance | 软 | prompt 要求不用 8888；`prompts/specialist_prompt_builder.py:930-943` | 无 firewall/namespace/Bash interception |
| 32 | Process-group cleanup | 硬 | 新 session，TERM 后 KILL 整个 process group；`subprocess_.py:547-553,805-844` | 是清理，不是对宿主进程的隔离 |
| 33 | Global kill prohibition | 软 | prompt 禁止 broad kill，只管理自有 PID；`specialist_prompt_builder.py:44-61`、`leaf.py:33` | 共享 PID namespace + Bash，纯建议 |
| 34 | Output/patch provenance | 硬 | 重盖章 gap/domain，限制 patch path，拒绝无效/逃逸/non-diff，添加 grounding；`runner.py:1222-1328`、`patch_safety.py:294-366` | 只检查被报告 artifact；直接 Bash/Edit/Write 不在该门内 |
| 35 | Automatic patch-to-Critic | 硬 | 仅真实 patch/resolvable artifact autosubmit，并抑制重复 review；`phases/explore.py:1381-1451,1495-1515` | 去重按 task/path，不按 patch content |
| 36 | Subprocess-to-inprocess fallback | 条件 | 优先 subprocess，缺 claude 或显式配置可回退 legacy in-process；`cli/executors.py:131-146,199-209` | 回退会失去 worktree、env scrub、CLI permission、process-group 隔离，且仅 warning |
| 37 | Substantive Critic quality | 条件 | 默认 critic-agent 提供实质评审；`roles/mock_backend.py:236-256`、`roles/__init__.py:37` | `--critic-mock` 自动批准，但 persisted-verdict 机制仍会通过 |
| 38 | Proposal-count target | 软 | prompt 要求至多 12 项；`runner.py:1234-1238`、`specialist_prompt_builder.py:1824` | runner 不截断，PolicyGate 不计数 |
| 39 | Specialist self-check guidance | 软 | 提示先做 accuracy/import/compile 自检；`specialist_prompt_builder.py:198-200,938` | 可选，不能替代 Coordinator KEEP evidence |

A 组计数：**26 硬、9 条件、4 软，共 39 项。**

### 8.3 B：通用下游优化正确性控制（8 项）

这些控制会保护 specialist 产物，但也服务于非 specialist 候选，因此单独计数。

| # | 控制 | 强度 | 核心行为与证据 | 重要限制 |
|---:|---|---|---|---|
| 40 | Integration containment/rollback | 硬 | 限制 root/target/workspace，stash dirty state，校验 apply，备份 artifact，普通失败回滚；`integrate_patch.py:882-1032,1911-2113` | 直接 child 写入绕过 executor；`apply_only` 变更后直接返回 |
| 41 | Crash-window sentinel | 条件 | mutation 前 best-effort 持久化 pending_integrate，resume 时 replay/rollback/clear；`integrate_patch.py:1986-2008`、`writeback.py:3376-3484` | save 失败仍继续；rollback 不完整恢复 artifact/config change |
| 42 | Real-workload integration benchmark | 条件 | 非 apply_only 时运行 bounded Magpie candidate，失败/异常/非正吞吐回滚；`integrate_patch.py:2150-2228,3051-3165` | apply_only 绕过；无 baseline config 时 fidelity 可降级 |
| 43 | Throughput/accuracy/confirmation KEEP gate | 条件 | 功能启用且有证据时要求阈值、阻止准确率回退、confirmation rebench；`integrate_patch.py:2550-2695`、`_accuracy_gate.py:75-115` | accuracy/confirmation 可关闭或缺失，KEEP 持久化 best-effort |
| 44 | Framework enablement branch | 条件 | 用 boot/timeout/progress/可用 accuracy 拒绝不可运行 candidate；`integrate_patch.py:2292-2467` | boot 成功但无 accuracy 可 provisional KEEP |
| 45 | Canonical config dedup | 硬 | 规范化 args/env/removal/mode/runtime override 后 fingerprint；`_canonical_fingerprint.py:4-120`、`explore.py:738-887` | patch content 不 fingerprint；KEEP bar 变化时可重试旧 sub-threshold |
| 46 | Baseline workload forwarding | 软 | 保存并向 explore/sweep 转发 materialized baseline config；`baseline.py:2586-2589`、`prelude.py:348` | 缺失时不 fail closed，可能静默回 smoke workload |
| 47 | RUN_EVAL accuracy gate | 条件 | 启用后解析 serving eval/script quality gate；`baseline.py:2594-2614` | `RUN_EVAL=false` 是合法 opt-out，可仅看吞吐 |

B 组计数：**2 硬、5 条件、1 软，共 8 项。**

### 8.4 主要 bypass 与缺口

1. PolicyGate 保护 routed intent，不保护任意 direct executor 调用。
2. `specialist:<task_id>` suffix 没有强认证到活跃 task/worker；terminal intent cardinality、task 与 gap/domain 一致性也未完全强制。
3. Freeform 不校验 `max_turns`；anchored `max_turns=0` 表示无界。
4. `kill_task` 主要更新 DB 和广播，不必然直接停止 asyncio task/subprocess。
5. Watchdog 回收的失败 specialist 可能滞留，除非其他 retry 路径接手。
6. Worktree 创建失败会 fail-open。
7. 缺少 `claude` 时可静默回退 in-process，失去大量 subprocess 隔离。
8. 默认 `bypassPermissions`，且允许 Bash/Edit/Write；`--add-dir`、verbatim `extra_claude_args`、共享 PID namespace 意味着不存在宿主级 sandbox。
9. Provider credentials 默认传给 specialist；GPU visibility mask 不是 device cgroup。
10. MCP chmod 是 best-effort，authorization header 默认可能落盘。
11. Arbitrary Bash server 可绕过 port helper；broad kill 禁令只是 prompt。
12. Patch provenance 只覆盖被报告 artifact，无法捕获直接 mutation。
13. Critic 可 mock；`advise` 是 permissive。
14. `max_proposals=12` 未硬执行；patch content 不做 canonical dedup。
15. `pending_integrate` 与 KEEP persistence 是 best-effort；crash rollback 不完整。
16. `apply_only` 可绕过 benchmark/KEEP gate。
17. 缺 baseline config 时可能回落到 smoke workload；accuracy 和 confirmation 可选。
18. 显式 `--specialist-model` 不重新经过 orchestration model allowlist。

---

## 9. Hyperloom 与 Arbor 对照

| 维度 | Hyperloom | Arbor DFS |
|---|---|---|
| 派发权限 | orchestration 身份 allowlist，代码强制 | 主要由 prompt 和 launcher 路径治理 |
| 结果协议 | 单一结构化 `specialist_done` | 开放 `results.jsonl`、patch、knowledge |
| 实时通信 | resumed SDK、JSON IPC、task/done files 多通道 | 主要是单向文件任务/结果 |
| 持久化通信 | SQLite append-only bus + cursor/replay | DFS 无统一 durable bus；kernel phase 有 200-msg board |
| 上下文 | SEED/DELTA、provider usage、单一软水位 compaction/reseed（inbox 本 tick 全量渲染） | DFS 本地 messages 无界增长，主要做输出截断 |
| Patch 准入 | persisted Critic verdict + executor 重查 | 无同等代码级 Critic verdict gate |
| 文件隔离 | 尝试每任务 worktree，但失败时 fail-open | 每 agent 目录、`--add-dir` 范围 |
| GPU 能力 | 可在租用卡上运行自己的非 8888 server | 可 lease GPU、写 patch、跑 micro-bench；禁止 E2E serving |
| 递归 | leaf 无 Task，固定一层 | 路径相关；kernel fellow 固定一层 |
| 超时 | wall cap + heartbeat/log + process reap | `timeout_minutes` 在部分 CLI dispatch 路径未形成完整 watchdog |
| 失败 | 合成合法空结果、partial recovery、有限 transient retry | classify failure 后用 escalation prompt 启动 fresh agent |
| 集成 | deterministic apply/benchmark/rollback/KEEP | 更多由 orchestrator 解释结果和驱动后续 |

Arbor CLI dispatch 的具体稳定性缺口：`timeout_minutes` 会进入 manifest，但 `_dispatch_via_cli` 没有完整 kill watchdog；`cmd_check` 也不总是调用 handle reap，GPU 回收更依赖显式 cleanup：

- `/wekafs/zgong/Arbor/src/arbor/dispatch.py:223,361,698`
- `/wekafs/zgong/Arbor/src/arbor/cli_dispatch_tools.py:148-176`
- `/wekafs/zgong/Arbor/src/arbor/gpu_pool.py:135,150`

---

## 10. 其他源码偏差与风险

### 10.1 Hyperloom `max_proposals` 文档与实现偏差

`DEFAULT_SPECIALIST_MAX_PROPOSALS=12` 只进入 prompt。`_finalize` 的旧 docstring 声称会 truncate，但实现明确把它当 prompt-side target，PolicyGate validator 也不检查 proposal 数量：

- `src/hyperloom/orchestrator/policy/gate.py:285`
- `src/hyperloom/orchestrator/specialists/runner.py:1175,1236`
- `src/hyperloom/orchestrator/policy/gate.py:2072-2169`

因此不能把它算作硬 safeguard。

### 10.2 Hyperloom orchestration model allowlist 默认不是 fail-closed

虽然允许列表为 opus-4-8/4-7/4-6，但 custom model 默认允许；只有设置 `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0` 后才成为真正 fail-closed gate：

- `src/hyperloom/inference_optimizer/cli/credentials.py:29-33`
- `src/hyperloom/inference_optimizer/cli/__init__.py:625-637,686-692`

并且显式 `--specialist-model` 不重新使用该 allowlist。

### 10.3 Arbor kernel-agents thinking API 已过时

Kernel-agents 默认模型为 `claude-opus-4-7`，但三个调用点仍使用固定 `thinking={type:'enabled', budget_tokens:N}` 形态：

- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/config.py:22,49`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/orchestrator/agent.py:101,234`
- `/wekafs/zgong/Arbor/kernel-agents/src/kernel_agents/orchestrator/trio.py:333`

对当前 4.7+ 模型，该固定 budget 形态已不兼容。Hyperloom 对应实现使用 adaptive thinking + separate effort：

- `src/hyperloom/orchestrator/roles/claude.py:551-553`

---

## 11. 最终判断

Hyperloom 并不是通过削弱 specialist 来换稳定性，而是把不可接受的不确定性从 agent 层移到 runtime 层：

```text
模型负责：
    研究、判断、提出候选、编写修改、解释证据

Runtime 负责：
    谁能派发
    派发参数是否合法
    何时和在哪个阶段运行
    使用哪些 GPU/lane
    子进程能看到哪些环境和工具
    写入和产物应位于哪里
    何时超时或回收
    输出必须是什么形状
    patch 如何送 Critic
    谁能批准
    何时可以集成
    如何 benchmark/correctness gate
    如何 KEEP/REVERT
    失败后状态机如何继续
```

Arbor DFS 更依赖“相信 agent + 开放结果 + 失败后重派”，因此更容易涌现出不在预设 schema 中的研究路径和产物；代价是上下文、超时、输出解释、资源回收和集成安全更多依赖 prompt compliance 与 orchestrator 判断。

Hyperloom 的代价则是：

- runtime 和状态机显著更复杂；
- 控制之间存在条件开关、fail-open 和旁路；
- 协议变窄，涌现式模型协作空间减少；
- 软约束容易被文档误写成硬保证；
- 每个新能力都需要同时维护 PolicyGate、schema、任务状态、资源、writeback 和 resume 语义。

最准确的一句话总结：

> **Arbor 的 specialist 更像被信任的研究员；Hyperloom 的 specialist 更像在生产调度器、资源管理器、审查门和事务式 benchmark 管线中工作的高权限承包者。**
