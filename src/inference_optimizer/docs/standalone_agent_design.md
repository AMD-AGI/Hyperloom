# Standalone Agent Design — 接口与能力边界规范

> **状态**: 接口剥离 v0.1（从当前实现 + DESIGN-modified.md v0.5 抽取）
> **目的**: 独立于 framework / orchestrator，纯粹定义"agent 是什么、能做什么、不能做什么、彼此怎么说话"。供重新规划 agent 角色与能力使用。
> **配套**: 协议绑定见 `inference-optimizer-DESIGN-modified.md` §5 / §10 / §11；线协议见 `agents/PROTOCOL.md`；当前 PolicyGate 实现见 `orchestrator/policy.py`。
>
> **如何用这份文档**:
> 1. 第 1-3 章是**当前现状**的契约抽取——任何重新规划都需要决定"保留 / 修改 / 移除"
> 2. 第 4-7 章是**协议常量表**（intent / topic / state field / 资源 lane），是设计的"硬接口"
> 3. 第 8-10 章是**能力边界与通信模式**——这是 PolicyGate 的源 truth
> 4. 第 11 章是**Open Issues**，标出现在不清楚 / 不一致 / 待重新决定的地方

---

## 1. Agent 是什么 — 定义

> 在本系统里 "**Agent**" = 一个长期存活的角色，由一个 LLM backend（Claude 或 Codex）驱动，通过结构化 **Intent** 与系统对话。

每个 Agent 由这三件事完全定义：

```
┌─────────────────────────────────────────────────────────┐
│ AgentRole = (                                           │
│   identity:    name + system_prompt                      │
│   transport:   backend type + tool access                │
│   permission:  allowed_intents + state-mutation rights  │
│ )                                                        │
└─────────────────────────────────────────────────────────┘
```

Agent **不是**：

- 不是 Conductor（协议管理员，无 LLM）
- 不是 Sub-agent（一次性 OOB 子进程，跑完销毁，没有持久身份）
- 不是 ActionExecutor（Python 类，shell 出 GPU 工作）
- 不是 Tool（emit_intent / Read / Bash 是 agent **使用**的能力，不是 agent 本身）

---

## 2. 当前 Agent 名册（4 个角色 + 2 个非 LLM 协作者）

### 2.1 LLM Agent（5 个 — Plan A 加入 kernel）

| Agent | Backend | 模型 | 默认 mode 启用 | 形态 |
|---|---|---|---|---|
| **executor** | Claude | claude-opus-4-7 | quick / guided / marathon | 持久 reactor |
| **critic** | Codex | gpt-5.4 | guided / marathon | 持久 reactor |
| **watchdog** | Claude | claude-opus-4-7 | marathon | 持久 reactor |
| **sage** | Codex | gpt-5.4 | marathon (常驻 reactor)<br>quick / guided (KB 查询服务) | 双形态 |
| **kernel** (Plan A) | Claude | claude-opus-4-7 | guided / marathon | 持久 reactor (responder-only) |

### 2.2 非 LLM 协作者（参与通信但不是 agent）

| 角色 | 职能 | 与 Agent 的关系 |
|---|---|---|
| **Conductor** | 协议管理员（Python，无 LLM）；拥有 PolicyGate / Bus / Scheduler / TaskRegistry | 是所有 intent 的 chokepoint；agent 之间不直接通话，全经 Conductor |
| **ActionExecutor** | Python 类（`BaselineExecutor` / `BenchRunnerExecutor` 等）；shell 出 GPU 工作 | 由 SubAgentRunner 调起；agent 看不见它，只看见 task 的 succeeded/failed evidence |
| **clock** | Conductor 内的 asyncio 任务，定时发 `reflection_tick` event | 在 bus 上有自己的 `from_agent="clock"` 身份，但不是 agent |

---

## 3. 当前 Agent 一览：能力 × 模式 × Intent 矩阵

> 这是**完整契约**——"哪个 agent 在哪个 mode 能发哪些 intent"。任何重新规划必须明确改动哪一格。

### 3.1 Mode → 启用 reactor 集合

| ExecutionMode | 触发条件 | 启用 reactor |
|---|---|---|
| `quick_param_sweep` | `MAX_HOURS < 2` | `[executor]` |
| `guided_kernel_opt` | `2 <= MAX_HOURS <= 6` | `[executor, critic, kernel]` |
| `marathon_multi_agent` | `MAX_HOURS > 6` | `[executor, critic, watchdog, sage, kernel]` |

> 注：sage 在 quick / guided 不是 reactor，但作为同步 KB 查询服务（`SageQueryService`）由 Conductor 直接调用，结果注入下游 prompt。
> kernel agent 仅在 guided + marathon 启用（quick 不允许 kernel-opt）。

### 3.2 Intent 能力矩阵（PolicyGate 强制）

| intent_type | executor | critic | watchdog | sage | kernel | 备注 |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `propose_action` | ✓ | ✗ | ✓ | ✓ | ✗ | 提议下一个 action（不真执行） |
| `delegate` | ✓<sup>※</sup> | ✗ | ✓ | ✗ | ✗ | 真派发 sub-agent 跑 GPU 工作；Codex 不允许；※ executor 不能 delegate `kernel_opt` / `integrate`（Plan A） |
| `request` | ✓ | ✗ | ✗ | ✗ | ✗ | Plan A — agent-to-agent RPC；`payload.target_agent` 限 `kernel` |
| `response` | ✗ | ✗ | ✗ | ✗ | ✓ | Plan A — kernel agent 唯一可发；`payload.in_reply_to` 必填 |
| `update_state` | ✓ | ✗ | ✓ | ✗ | ✗ | 受 `CORE_STATE_FIELDS` 守护；下文 §6.3 详述 |
| `update_persona` | ✓ | ✓ | ✓ | ✓ | ✓ | append-only 写自己的 `personas/<name>.md` |
| `send_message` | ✓ | ✓ | ✓ | ✓ | ✓ | 任意 topic（不在 allowlist 的会被软降级为 `observation`） |
| `ask_question` | ✓ | ✓ | ✓ | ✓ | ✓ | 通常给 sage，60s 内必须 `answer` |
| `answer` | ✓ | ✓ | ✓ | ✓ | ✓ | 必须带 `in_reply_to` |
| `objection` | ✗ | ✓ | ✗ | ✓ | ✗ | 反对一个 proposal；议会模式触发 |
| `vote` | ✗ | ✓ | ✗ | ✓ | ✗ | 议会投票 |
| `alert` | ✓ | ✓ | ✓ | ✓ | ✓ | 写 bus + 镜像到 `findings/alerts.jsonl`；severity=high → priority=0 |

**Plan A 关键约束（PolicyGate 实施）**：
- 只有 `executor → kernel` 这对 (source, target) 允许 REQUEST（`REQUEST_ROUTING` 表）
- 只有 `kernel` 角色可发 RESPONSE
- executor.delegate(action_name in {kernel_opt, integrate}) 被拒（`rule="kernel_owned_by_kernel_agent"`）

### 3.3 Tool 访问能力（每 turn 注入到 LLM 的 allowed_tools）

| Agent | 可用 tools | 说明 |
|---|---|---|
| executor (Claude) | `emit_intent` (MCP) + 按 action 注入的 `Read / Bash / Edit` | quick mode Bash 受 §7 allowlist 约束 |
| critic (Codex) | `[]` (no-tools) | 只输出 `validated_json_output`；不读不写 workspace |
| watchdog (Claude) | `emit_intent` + `Read` + 受限 `Bash`（log 取证） | 不能 Edit / 不能改 server config |
| sage (Codex) | `[]` (no-tools) | 同 critic；只读 prompt 注入的 KB 片段 |
| kernel (Claude) | `emit_intent` (MCP) + `Bash` + `Read` + `Edit` | 需 Bash 调 GEAK/OOB Ray 脚本 + patch_inductor.py + run_baseline.sh |

> 设计原则：**所有 workspace 副作用必须通过 ActionExecutor，不直接由 agent tool 执行**。Claude agent 拿 Bash 是为 RCA / 健康检查，不是为了改文件。

---

## 4. 协议常量表 — 12 个 IntentType（Plan A 加入 REQUEST + RESPONSE）

> 完整 schema 见 `orchestrator/intent_parser.py::INTENT_ENVELOPE_SCHEMA`。
> 任何 agent 输出必须是 `{"intents": [{"intent_type": "...", "payload": {...}}, ...]}`。
> Claude 角色通过 `emit_intent` MCP tool call 输出；Codex 角色直接输出 fenced JSON。

| IntentType | Required payload fields | 触发的 Conductor 副作用 |
|---|---|---|
| `send_message` | `topic` (+ 可选 `body_md`, `to`, `priority`, `in_reply_to`, `original_topic`) | bus.append 到指定 topic；topic 不在 allowlist → 软降级为 `observation` 并保留原 topic |
| `delegate` | `action_name` (+ 可选 `params`, `predicted_gain_pct`, `reason`, `idempotency_key`) | tasks 表新增 `kind=delegate state=queued`；幂等键命中 terminal → emit `delegate_dedup_to_terminal` 反馈 |
| `propose_action` | `action_name`, `predicted_gain_pct` (+ 可选 `params`, `reason`) | tasks 表新增 `kind=proposal`；bus topic=proposal |
| `objection` | `target_msg_id`, `reason` (+ 可选 `severity`) | bus topic=objection；marathon 模式可触发议会 |
| `vote` | `target_msg_id`, `vote` (`approve`/`reject`/`abstain`) | bus topic=vote；议会计票 |
| `update_state` | `changes` (dict, 必须非空) | `SharedState.apply_validated_transition`；`CORE_STATE_FIELDS` 受守护 |
| `update_persona` | `body_md` | append-only 到 `personas/<from_agent>.md`；下一 turn prompt 自动重读 |
| `ask_question` | `topic`, `question` (+ 可选 `to`, `in_reply_to`) | bus topic=question；60s 内必须有 answer |
| `answer` | `in_reply_to`, `answer` (+ 可选 `topic`) | bus topic=answer |
| `alert` | `severity`, `summary` (+ 可选 `detail`) | bus topic=alert + 写 `findings/alerts.jsonl`；severity 映射 priority：critical/high→0, medium→1, low→2 |
| `request` (Plan A) | `target_agent`, `kind` (+ 可选 `params`, `reason`) | bus topic=request；`to_agent=<target_agent>`；priority=2；执行体由目标 agent 处理 |
| `response` (Plan A) | `in_reply_to`, `kind` (+ 可选 `status`, `result`) | bus topic=response；reverse-routed via `bus.lookup_by_id(in_reply_to)` 找到原 request sender 作为 to_agent；priority=2 |

---

## 5. Bus Topic 白名单（24 个）

> 完整列表见 `orchestrator/message_bus.py::TOPIC_ALLOWLIST`。
> Agent 发 `send_message(topic=X)` 时，X **不在** allowlist 会被软降级为 `observation`，原 topic 落到 `payload.original_topic`，**不报错不丢失**。

按用途分组：

| 组 | Topics | 谁发 |
|---|---|---|
| **决策类** | `proposal`, `decision`, `objection`, `vote`, `vote_request`, `parliament_open` | executor / critic / sage |
| **问答类** | `question`, `answer` | 任意 agent → sage（典型） |
| **RPC** (Plan A) | `request`, `response` | executor → kernel / kernel → executor |
| **观察 / 事件** | `observation`, `event`, `historical_warning` | conductor / 所有 agent |
| **告警 / RCA** | `alert`, `rca_done`, `do_emergency_rca` | watchdog（marathon）/ critic（guided emergency）/ kernel |
| **触发指令** | `do_postmortem`, `do_strategic_review`, `synthesize_for_kb` | conductor / sage |
| **生命周期** | `heartbeat`, `reflection_tick`, `graceful_stop`, `intent_emitted`, `delegated_result` | conductor / clock / dispatcher |
| **存储层** | `lease_expired`, `lease_acquire_failed` | resource_lock backend |

---

## 6. SharedState — Agent 对状态的读写边界

### 6.1 SharedState 结构

```python
SharedState = {
    # 身份 (read-only for agents)
    session_id, model_path, model_name, model_class, cwd,
    start_ts, max_minutes, execution_mode,

    # 进度 (动态, agent 可读)
    elapsed_minutes,    # 由 clock 维护
    time_left_minutes,  # 派生

    # 测量结果 (agent 可写)
    baseline_tput, current_tput, cumulative_gain, baseline_accuracy,

    # 运行状况 (受限)
    crash_count, current_action,
    stop_reason,  # 仅 Conductor 可写

    # 历史
    decisions[], rca_findings[],
}
```

### 6.2 Agent 可写字段（通过 `update_state.changes`）

```
current_action       # 我现在在跑哪个 action
current_tput         # 最新一次 benchmark 的 tput
crash_count          # 累计 crash 数
baseline_tput        # 基线 tput（baseline action 完成后写）
baseline_accuracy    # 基线 GSM8K 分数
```

### 6.3 CORE_STATE_FIELDS — 仅 Conductor 可写（PolicyGate 守护）

```
current_best         # KEEP / REVERT 决策目标
stop_reason          # graceful stop 控制
cumulative_gain      # 由 _maybe_recompute_gain 自动算
baseline_tput        # ← 注意：在 CORE 中，但通过特殊路径允许 executor 写
baseline_accuracy
session_id, model_path, model_name, model_class
start_ts, max_minutes, execution_mode
```

> **当前实现的不一致点**：`baseline_tput` / `baseline_accuracy` 同时出现在"agent 可写"和 `CORE_STATE_FIELDS`。实现里靠 `_executor_intent_sink` 走"trusted code bypass" 路径让 ActionExecutor 写得进去，但 LLM 直接 emit `update_state(baseline_tput=...)` 会被拒。**这是 §11 待重新设计的一项**。

### 6.4 Agent 可读但不能直接写的派生字段

```
elapsed_minutes / time_left_minutes  # clock 自动算
cumulative_gain                       # _maybe_recompute_gain 自动算
decisions[]                           # 自动追加历史
```

---

## 7. Bash 命令访问边界（quick mode）

> 完整列表见 `orchestrator/policy.py::QUICK_BASH_ALLOWLIST` / `QUICK_BASH_DENYLIST`。
> 仅在 quick mode + agent 拿到 `Bash` tool 时生效；guided / marathon 由 ActionRegistry 的 `allowed_tools` 控制。

### 7.1 Allowlist（quick 允许）

| 类别 | 命令前缀 |
|---|---|
| Server lifecycle | `pgrep -f sglang.launch_server` / `pgrep -f vllm.entrypoints` / `kill <pid>` / `scripts/run_baseline.sh` |
| 只读检查 | `rocm-smi` / `nvidia-smi` / `ls` / `cat` / `head` / `tail` |
| Sweep / benchmark | `scripts/eval_accuracy.sh` / `scripts/run_sweep.sh` / `python -m sglang.bench_serving` / `python -m vllm.entrypoints.benchmark` |

### 7.2 Denylist（任何 mode 都禁）

| 类别 | 命令前缀 |
|---|---|
| 进程操作 | `pkill -f sglang` (IR-5) / `pkill -f vllm` |
| Git 写 | `git commit` / `git push` |
| Patch | `patch ` / `patch_inductor.py`（必须走 integrate action） |
| Marathon-only | `geak` |
| 重型构建 | `make ` / `cmake ` / `ninja` |
| 安全 | `rm -rf` / `sudo ` |

---

## 8. Agent 间通信 — 5 种模式

> Agent 之间**永远不直接通话**，全经 Conductor。Plan A 加入了第 5 种：bidirectional request/response。

### 8.1 委托模式（Executor → ActionExecutor → Executor）

```
executor                 conductor                 dispatcher_loop          ActionExecutor
   │                         │                            │                       │
   │── delegate(baseline) ──>│                            │                       │
   │                         │── tasks[queued] ──────────>│                       │
   │                         │                            │── BaselineExecutor ──>│
   │                         │                            │                       │── run_baseline.sh
   │                         │                            │                       │── update_state(baseline_tput=X)
   │                         │<─── _executor_intent_sink ─│<──────────────────────│
   │                         │── _maybe_recompute_gain    │                       │
   │                         │── decision event           │                       │
   │<─── inbox tick ─────────│                            │                       │
   │ (sees baseline_tput in  │                            │                       │
   │  state.summary now)     │                            │                       │
```

**适用**: 任何需要 GPU / shell 工作的副作用 action。

### 8.2 议会模式（marathon-only）

```
executor ── propose_action ──> conductor
                                  │── (flags.enable_parliament=True) ──> _open_parliament
                                  │── bus broadcast topic=proposal kind=parliament_open
                                  │
critic ──── objection ────────>  │
sage ────── vote(approve) ────>  │
critic ──── vote(reject) ─────>  │
                                  │── 2s 投票窗口
                                  │── 多数票胜出
                                  └── 返回 "approved" / "rejected" / "abstained"
```

**适用**: marathon 高风险决策（KEEP/REVERT, framework_rebuild）。
**当前限制**: Brier 加权 (§9.4) 还没接通到投票计数；目前是简单多数。

### 8.3 问答模式（任意 agent → sage）

```
executor ── ask_question(topic="kb_recall", question="X model + TP=8 history?")
            ├─ to: "sage"
            └─ in_reply_to: <my msg_id>
                              ↓
                         conductor
                              ↓
                         sage 收到 inbox
                              ↓
                         answer(in_reply_to=<orig msg_id>, answer="## KB recall ...")
                              ↓
                         conductor → executor inbox
```

**适用**: 跨 run KB 召回 / 决策前同步咨询。
**SLA**: 60s 内必须有 answer（PROTOCOL.md §10.3 #3）。

### 8.4 事件驱动告警模式（marathon）

```
ActionExecutor crashes → conductor emits event{kind=*_crashed}
                              ↓
                         watchdog 收到 inbox
                              ↓
                         (Read log + Bash 健康检查)
                              ↓
                         alert(severity="high", summary="OOM cluster on attempt #4")
                              ↓
                         conductor → bus topic=alert priority=0
                                   → findings/alerts.jsonl
                              ↓
                         所有 agent inbox 看到（priority=0）
                              ↓
                         executor 决定下一步（缓冲 / 回退 / pivot）
```

**适用**: 监控类告警；watchdog **不直接干预**，由 executor 自己决定怎么响应。

### 8.5 Bidirectional RPC 模式（Plan A — executor ↔ kernel）

```
executor → REQUEST{target_agent="kernel", kind="select_kernels",
                   params={trace_path: ...}}
                              ↓
                         conductor._handle_request
                              ↓
                         bus.append(topic="request", to_agent="kernel",
                                    msg_id=R1)
                              ↓
                         (mirror via Router OR replay_for in single-proc)
                              ↓
                         kernel inbox 看到 request
                              ↓
                         (kernel does work via Bash:
                          trace_summary.sh / run_geak.sh / etc.)
                              ↓
                         kernel → RESPONSE{in_reply_to=R1,
                                  kind="select_kernels_done",
                                  status="succeeded",
                                  result={candidates: [...]}}
                              ↓
                         conductor._handle_response
                              ↓
                         bus.lookup_by_id(R1) → original_sender="executor"
                              ↓
                         bus.append(topic="response", to_agent="executor",
                                    in_reply_to=R1)
                              ↓
                         executor inbox 看到 response → emit next REQUEST
```

**关键约束** (PolicyGate 强制):
- 只有 executor 可发 REQUEST (`REQUEST_ROUTING`)
- 只有 `target_agent="kernel"` 是合法目标 (Plan A v1)
- 只有 kernel 可发 RESPONSE
- `in_reply_to` 必须能在 bus 上找到对应的 request envelope；否则 conductor fall-back 到 broadcast (`to_agent="*"`) 并记录 warning

**适用**: 任何 agent-to-agent 长任务委托（Plan A 用于 kernel-opt 三步：select_kernels / run_optimization / apply_patch）。

---

## 9. 线协议 — 物理传输

> Agent 之间逻辑通信走 Conductor，但物理传输有两种实现，**对 agent 透明**。

### 9.1 传输模式选择（`--transport`）

| 模式 | Agent 物理形态 | 何时用 |
|---|---|---|
| `single-proc` | 每个 reactor 是 Conductor 进程内的 asyncio task | dev / CI / 短 quick run |
| `multi-cli` (默认) | 每个 reactor 是独立 `claude --print --continue` 子进程，restart-loop | 生产 / marathon >6h |
| `hybrid` | 部分 reactor 走 CLI，其余 in-process | 迁移过渡 |

### 9.2 Multi-CLI 拓扑（`agents/<name>/` 子目录）

每个 agent 在 `$SESSION_DIR/agents/<name>/` 下持有：

| 文件 | 谁写 | 谁读 | 用途 |
|---|---|---|---|
| `inbox.jsonl` | Router | agent | bus event mirror |
| `inbox.jsonl.seq` | agent | agent | "我处理到第几条 seq" cursor |
| `inbox.jsonl.mirrored` | Router | Router | "我 mirror 到哪了" 私有 cursor |
| `outbox.jsonl` | agent | Router | agent emit 的 intent envelope |
| `outbox.jsonl.cursor` | Router | Router | "我读到哪个 byte offset" |
| `conversation.jsonl` | launcher | agent | (Codex only) 重启时 prepend 进 prompt |

### 9.3 Envelope schema（一行 JSONL）

```json
{
  "kind": "intent" | "message",
  "msg_id": "<uuid hex>",
  "seq": <int>,
  "ts": "<iso8601 utc microseconds>",
  "from_agent": "<name>",
  "to_agent": "<name>" | "*" | "conductor",
  "payload": { ... },

  // MESSAGE-only:
  "topic": "<from TOPIC_ALLOWLIST>",
  "priority": 0..3,
  "in_reply_to": "<msg_id>" | null,

  // INTENT-only:
  "intent_type": "<from IntentType enum>"
}
```

### 9.4 传输模式无关的契约

无论 single-proc 还是 multi-cli，下面都成立：

1. **PolicyGate 是唯一 chokepoint**——所有 outbox intent 在被 `_handle_intent` 处理前都过一次 `PolicyGate.validate_intent`
2. **events 表是 SoT**——所有 message 必须先写到 SQLite events 表拿到 AUTOINCREMENT seq；jsonl 文件只是 mirror
3. **拒绝是可见的**——PolicyGate 拒绝会写一条 `topic=observation kind=policy_denied` event，被 mirror 到那个 agent 的 inbox，下一 turn LLM 自然看见原因

---

## 10. Sub-Session Layout — 文件契约

> 注意：Agent 有**两个目录**——
> 1. **运行时 sub-session 目录** `$SESSION_DIR/agents/<name>/`：每个 run 一个，存 inbox/outbox/cursor，**不是 git 仓库内容**
> 2. **Package skill 目录** `src/inference_optimizer/agents/<name>/`：随代码发布，存 SKILL.md + actions/ + scripts/，是 agent 的"人格知识库"
>
> launcher 用 `--add-dir` 把这两个目录都喂进 Claude CLI 的可读集合，所以 LLM 既能 Read inbox 也能 Read 自己的 skill 文件。

### 10.1 运行时 sub-session 目录

```
$SESSION_DIR/                               # 共享, NFS or 本地
│
├── storage/conductor.db                    # SoT (events / leases / cursors / tasks)
├── state.json                              # SharedState snapshot (人读)
├── personas/<agent>.md                     # L3 persona append-only
├── kb/                                     # L4 跨 run KB
│   ├── entries.jsonl
│   ├── insights.jsonl
│   └── conflicts.jsonl
├── results/<task_id>/                      # ActionExecutor 大块产物 (trace / kernel src)
├── findings/                               # RCA / alerts
│   ├── alerts.jsonl
│   └── <ts>.json
├── checkpoints/<ts>/conductor.db.bak       # backup
├── logs/<agent>.log                        # multi-cli CLI stdout
├── .multicli/                              # launcher 生成
│   ├── .env
│   ├── run_pane_<agent>.sh
│   └── tmux_session_name
├── STOP_AGENT_<name>                       # 停止信号 (sentinel file)
│
└── agents/<agent_name>/                    # ★ 每个 agent 的运行时 sub-session-dir
    ├── inbox.jsonl                         # Router 写, agent 读 (bus event mirror)
    ├── inbox.jsonl.seq                     # agent 自己维护: 我处理到第几条 seq
    ├── inbox.jsonl.mirrored                # Router 私有: 我 mirror 到哪了 (agent 不要碰)
    ├── outbox.jsonl                        # agent 写, Router 读 (intent envelope)
    ├── outbox.jsonl.cursor                 # Router 私有: 我读到 byte offset 几
    └── conversation.jsonl                  # Codex only: launcher 重启时 prepend 进 prompt
```

### 10.2 Package skill 目录（推荐 v0.2 形态）

```
src/inference_optimizer/agents/                 # repo root 下, 随包发布
│
├── PROTOCOL.md                                  # ★ 所有 agent 共享的线协议规范
│
└── <agent_name>/                                # ★ 每个 agent 的 skill 目录
    ├── agent_card.yaml                          # 启动配置 (launcher 读)
    ├── SKILL.md                                 # ★ 入口: 短 (<5KB), 索引 actions/ 和 scripts/
    │                                              通过 system_prompt 字段被 base64 注入到
    │                                              `claude --print --system-prompt $SYSTEM_PROMPT`
    │
    ├── actions/                                 # ★ 子技能 (按需 Read)
    │   ├── INDEX.md                             #     subskill 导航
    │   ├── first_turn.md                        #     "我看到 run_started 怎么办"
    │   ├── after_baseline.md                    #     "baseline_tput 落了怎么办"
    │   ├── retry_after_dedup.md                 #     "dedup_to_terminal 事件怎么 pivot"
    │   └── kernel_opt_recipe.md                 #     guided/marathon kernel-opt 全流程
    │
    ├── reference/                               # ★ 长尾 reference (一次读后记住)
    │   ├── ir_rules.md                          #     IR-1..IR-7 详细 + 反例
    │   ├── action_catalogue.md                  #     22 action 的静态 cost/risk/family
    │   └── failure_codebook.md                  #     "我看到 error X" 查表
    │
    └── scripts/                                 # ★ agent 私有 helper (Bash 可调)
        ├── inbox_tail.sh                        #     紧凑打印最近 N 条 inbox
        └── state_check.sh                       #     提取 state.json 关键字段
```

`★ 标记 = 用户可控/可扩展`。对 Codex agent (no-tools) 这棵树仍然有意义，但 `actions/` 和 `reference/` 内容必须由 launcher 在拼装 system prompt 时**预拼接**进去（详见 §10.5）。

### 10.3 launcher 的 `--add-dir` 自动覆盖范围

`MultiCLILauncher._add_dir_args` 给每个 Claude pane 注入：

```
--add-dir $SESSION_DIR                              # 运行时共享状态
--add-dir $SESSION_DIR/agents/<name>/               # 我的 inbox/outbox
--add-dir <package>/agents/<name>/                  # ★ 我的 SKILL.md + actions/ + scripts/
--add-dir <extra_dirs...>                           # 用户传入的 (InferenceX 等)
```

注意 launcher 当前**不**自动 `--add-dir` 父目录 `<package>/agents/`。如果你想 SKILL.md 引用 `../PROTOCOL.md`，要么：
- 改 `_add_dir_args` 加 `card.card_dir.parent`（约 1 行），或
- 把 PROTOCOL.md 复制 / inline 到每个 SKILL.md 头部，或
- 用 launcher 的 `extra_dirs` 显式传入

### 10.4 文件 → "改这一处"映射

| 想做的事 | 改哪里 |
|---|---|
| 给 executor 加新 subskill | 在 `agents/executor/actions/` 新建 `.md`，并在 `SKILL.md` 索引表加一行 |
| 给 executor 加新 helper script | `agents/executor/scripts/foo.sh` + 在 `SKILL.md` 脚本表加一行；quick mode 还要在 `policy.py::QUICK_BASH_ALLOWLIST` 加前缀 |
| 改 executor 角色身份 | `agents/executor/SKILL.md` （persona / 风格） |
| 改 executor allowed_intents | `orchestrator/agent_role.py::_EXECUTOR_INTENTS` |
| 改 executor 启动重启策略 | `agents/executor/agent_card.yaml::restart_policy` |
| 改 launcher 给 executor 的 `--add-dir` | `MultiCLILauncher._add_dir_args` 或 Conductor 构造时传 `launcher_extra_dirs` |

### 10.5 Codex agent 的 skill 形式（特殊处理）

Codex agent (critic / sage) `no_tools=True`，**没有 Read 工具**，所以这棵 skill 树不能按需 Read。两条出路：

**方案 A — launcher 预拼接（推荐）**
改 `MultiCLILauncher._compose_codex_pane`：在 `_read_system_prompt(card)` 后扫 `card.card_dir/actions/*.md` 全部 `cat` 进 `$SYSTEM_PROMPT`。Codex 一次看到所有内容。
- 优: 改动小（~30 行），目录形式与 Claude agent 对齐
- 缺: token 成本随 actions/ 文件数线性增长

**方案 B — Codex 不走 skill 形式**
保留 `agents/<codex_name>/system_prompt.md` 单文件，不分 actions/。
- 优: 零改动
- 缺: 跨 agent 形态不一致

---

## 11. Open Issues — 重新规划时需要决策的点

> 当前实现里有几处契约不一致 / 模糊地带，重新规划时建议明确决定。

### 11.1 `baseline_tput` / `baseline_accuracy` 双重身份

- 出现在 `apply_validated_transition` 的 allowed 列表（agent 可写）
- 同时在 `CORE_STATE_FIELDS`（PolicyGate 拒 agent 写）
- 当前靠 `_executor_intent_sink` 的 trusted-code bypass 让 ActionExecutor 能写
- **决策点**: 要不要让 LLM 也能直接 update_state(baseline_tput=...)？还是基线测量永远只能由 ActionExecutor 走 trusted 路径？

### 11.2 Watchdog 是否应能 `delegate`

- 当前 `_WATCHDOG_INTENTS` 包含 `DELEGATE`（"Watchdog can suggest a postmortem/strategic review via DELEGATE in marathon mode"）
- 但 PolicyGate 没有"按 watchdog 限制 action_name 子集"的逻辑
- 实际生产中 watchdog 也没用过 delegate（它只发 alert）
- **决策点**: Watchdog 真的需要 delegate 吗？还是只发 alert，executor 决定怎么 delegate？

### 11.3 Sage 在 quick / guided 是否有 reactor

- 当前 Sage 在 quick / guided 是"KB 查询服务"（同步 callable，无 reactor）
- 但 `agent_card.yaml::allowed_modes` 只列了 `marathon_multi_agent`
- multi-cli 下 quick / guided 不会 spawn sage CLI
- **决策点**: Sage 在 quick / guided 想不想有"低频 reactor"？还是保持纯同步查询？

### 11.4 Topic 软降级 vs 严格 allowlist

- 当前实现：未知 topic → 软降级到 `observation` + 保留 `original_topic`
- 设计原意：严格 allowlist，未知就拒
- 软化后好处：critic / sage 可以发 `kb_recall` / `rca_finding` 这种自创 topic 而不被拒
- 坏处：bus 上 topic 不再是封闭集，下游消费要兜底
- **决策点**: 想不想扩 TOPIC_ALLOWLIST 来明确表达这些意图？比如加 `kb_recall` / `rca_finding` / `executor_status`

### 11.5 Critic 兼任 ephemeral RCA（guided emergency）

- DESIGN §5.1.3 + §7.2 说 guided emergency 时 Critic 兼任 RCA
- 当前实现：`Conductor.ephemeral_rca_via_critic` 已落地，但**没接入触发**（emergency stop 路径不调它）
- **决策点**: emergency 触发条件是什么？真的让 Critic 切换 system prompt 跑 RCA 吗？还是 guided 干脆不做 RCA？

### 11.6 ActionExecutor 是不是 agent

- 当前 ActionExecutor 是 Python 类，不持久，没有 LLM
- 但它通过 `intent_sink` 写 update_state intent，从 bus 看跟"trusted agent" 几乎一样
- 监控视角：bus 上 `from_agent="conductor"` 而不是 `from_agent="<executor name>"`
- **决策点**: 要不要给每个 ActionExecutor 一个独立的 `from_agent` 身份（比如 `baseline-executor`）？这样监控更清晰但增加复杂度

### 11.7 LLM 角色是否应直接读 Scheduler 的 score

- 当前 BudgetAwareScheduler 算出 action score，但 prompt 里没注入
- Executor 的"决定下一步"完全靠 LLM 推理 + `_render_action_catalogue` 表（只列 cost / risk）
- **决策点**: 把 scheduler 的 score 当 hint 注入 prompt 吗？还是保持"scheduler 是约束 / LLM 自由决策"的分工？

### 11.8 Watchdog 是否需要 multi-cli 之外的"in-process 监听" 通道

- 当前 watchdog 只看 inbox 里的 event
- 一些重要 event（lease_expired / dispatcher 异常）只在 SQLite 内部，没 mirror
- **决策点**: 给 watchdog 加"直接读 SQLite events 表"的特权？还是所有重要 event 都必须经 bus？

### 11.9 持久 agent 与 ephemeral sub-agent 的边界

- 持久 agent = 长期 reactor（这份文档定义的 4 个）
- ephemeral sub-agent = `SubAgentRunner` 跑完即销毁的（当前主要是 LLM-driven fallback 路径，因为大部分 action 已被 ActionExecutor 取代）
- **决策点**: 还需要 LLM-driven sub-agent 吗？如果 ActionExecutor 覆盖率达 100%，可以删掉这条 fallback 路径，简化 SubAgentRunner

### 11.10 Persona 是否对所有 agent 一视同仁

- 当前所有 agent 都可写 update_persona
- 但只有 marathon mode + persona size > 8K token 才触发蒸馏
- quick / guided 的 executor 也写 persona 但不蒸馏 → 长跑会膨胀 token
- **决策点**: quick / guided 要不要禁 update_persona？还是保留但加大小限制？

### 11.11 Plan A — kernel agent lane 协调是软的（已知妥协）

- 当前 kernel agent 不持有 SQLite leases（没有 lock manager 接入它的 reactor）。
- 协议：kernel agent 在 `apply_patch.sh` 之前会读 `state.json::current_action`，
  如果是 `bench_*` 就 defer。executor 的 SKILL 也教育它在看到 kernel agent
  busy 时不要 delegate `bench_runner`。
- 风险：两个 LLM 角色之间的"自觉"协调，并发 race 仍然可能发生（虽然很罕见）。
- **决策点**: 接受软协调（MVP 默认）vs 给 kernel agent 接入 LockManager（更安全
  但需要 conductor 改动 + lane 状态对 LLM 暴露）

### 11.12 Plan A — 长 GEAK 任务的 turn timeout

- GEAK 一轮可以跑 30+ 分钟；Claude SDK 在单 turn 内的执行有超时。
- 当前 mitigation：`agent_card.yaml::restart_policy = {max_restarts: 50, backoff: 30s}`，
  超时后重启循环继续 — kernel agent 在重启后用 `state.json` + `results/<task_id>/`
  下的 partial logs 恢复 polling。
- 风险：如果一个 Ray job 重启循环里反复超时，kernel agent 永远没机会发出
  `response`，executor 会等到 inbox 死寂直到自己超时 pivot。
- **决策点**: 加 detached subprocess + polling pattern 让 LLM 立即返回？这需要
  改 `run_geak.sh` / `run_oob.sh` 走 nohup + Ray job ID 持久化。

---

## 12. 重新规划清单（建议工作流）

如果你要重新设计 agent 名册，建议按这个顺序决策：

1. **角色清单**: 4 个还是更多 / 更少？每个角色的核心职责一句话讲清
2. **Mode × 角色矩阵**: 每个 mode 拉起哪些 reactor + 哪些是 callable-only？
3. **Intent 能力分配**: 每个角色在 §3.2 表里的 ✓/✗ 是否要改？
4. **新 Intent 类型**: 当前 10 个够吗？需要新增（如 `request_replan`）就更新：
   - `intent_parser.py::IntentType`
   - `intent_parser.py::_PAYLOAD_REQUIRED`
   - `agent_role.py::_*_INTENTS` allow-lists
   - `policy.py::PolicyGate` 校验分支
   - `conductor.py::_handle_intent` 新分支
5. **Topic 调整**: 看 §11.4，决定 TOPIC_ALLOWLIST 要不要扩
6. **State 边界**: 看 §11.1 / §6.3，决定 CORE_STATE_FIELDS 怎么改
7. **物理传输**: multi-cli vs single-proc 默认值；新角色要不要写 `agent_card.yaml`
8. **Persona 策略**: 看 §11.10
9. **协作模式**: §8 4 种模式哪些保留 / 改 / 加？
10. **Open Issues**: §11 每条都做出明确决定

每一步决定后，对应改动以下文件之一：

| 改动类别 | 文件 |
|---|---|
| 角色 + 能力 | `orchestrator/agent_role.py` |
| 协议校验 | `orchestrator/policy.py` |
| Intent 类型 | `orchestrator/intent_parser.py` |
| Topic | `orchestrator/message_bus.py::TOPIC_ALLOWLIST` |
| State 字段 | `orchestrator/shared_state.py` + `orchestrator/policy.py::CORE_STATE_FIELDS` |
| Multi-CLI 启动配置 | `agents/<name>/agent_card.yaml` |
| Agent 入口 prompt (skill 形式) | `agents/<name>/SKILL.md` |
| Agent 子技能 (skill 形式) | `agents/<name>/actions/*.md` (按需 Read) |
| Agent 长尾 reference (skill 形式) | `agents/<name>/reference/*.md` |
| Agent 私有 helper 脚本 | `agents/<name>/scripts/*.sh` (Bash 调用) |
| In-process role brief | `orchestrator/system_prompts/<name>.md` (single-proc / dispatcher 用) |
| 通信模式 | `orchestrator/conductor.py::_handle_intent` + `_open_parliament` 等 |
| Codex skill 预拼接 | `orchestrator/multi_cli/launcher.py::_compose_codex_pane` |

---

## 13. MVP v0.4 — Roster 重构计划（"能跑一次"目标）

> **目的**: 把当前 5 角色（executor / critic / watchdog / sage / kernel）+ 议会 + Sage KB 服务的设计，压缩成 **4 角色**（executor / critic / triage / kernel）的最小可运行形态。议会模式**彻底移除**，Sage 角色**整体删除**（KB 留 v0.5），Watchdog 升级为 **always-on Triage**（带 `KILL_TASK` 唯一权限）。
>
> **范围契约**: 这一节是 v0.4 的"代码改动 SoT"。任何不在这节列出的改动都属于 v0.5+。
>
> **来源**: 用户讨论 2026-04-29，确定的设计取舍是"能在真实 GPU 跑一次完整结果，不追求能力上限"。

### 13.1 v0.4 Roster 终态

| Agent | Backend | 模型 | 启用 mode | Always-on | 核心职责（MVP）|
|---|---|---|---|---|---|
| **executor** | Claude | claude-opus-4-7 | quick / guided / marathon | — | 唯一 Proposer + Delegator；调度 baseline / param-sweep / profile；REQUEST kernel agent |
| **critic** | **Claude** | claude-opus-4-7 | guided / marathon | — | KEEP/REVERT review + Brier prediction；**砍 OBJECTION / VOTE**（议会移除）|
| **triage** | Claude | claude-opus-4-7 | quick / guided / marathon | ✅ (tick=60s) | 监听 event_log + 跨 agent inbox/outbox 取证 + 唯一持有 `kill_task` |
| **kernel** | Claude | claude-opus-4-7 | guided / marathon | — | 不变（Plan A 完整保留：仅 RESPONSE 给 executor REQUEST）|

> **v0.4 决策（2026-04-29 用户确认）**: 全部 4 个 agent 都用 **Claude 后端**——不再使用 Codex。这意味着：
> - `agent_role.py::ROLE_CRITIC` 从 `codex_role(...)` 改为 `claude_role(...)`，`no_tools=False`，`allowed_tools=["emit_intent"]`（默认即可，critic 通过 inbox 收 decision，不需要 Read/Bash）
> - `agents/critic/agent_card.yaml::backend` 从 `codex` 改为 `claude`；继续用 `system_prompt.md`（不切 SKILL.md，单文件足够）
> - `multi_cli/launcher.py::_compose_codex_pane` 在 v0.4 中**不再被任何 agent 使用**——保留代码（不删）以便 v0.5 可能恢复，但所有 agent_card 都不会触发它
> - `tests/test_codex_backend.py` 保留（Codex backend 类本身仍然存在，未来可用）

**砍掉**:
- `sage` 角色（连同 `SageQueryService`、`agents/sage/`、`system_prompts/sage.md`、`tests/test_sage_query_service.py`）
- `Conductor.ephemeral_rca_via_critic`（critic 兼任 RCA 在 MVP 不需要——triage 已 always-on 接管所有取证）
- 议会模式（`IntentType.OBJECTION` / `IntentType.VOTE` / 4 个相关 topic / `_open_parliament` / `enable_parliament` flag / 所有触发分支）

### 13.2 Mode → Reactor 矩阵

| ExecutionMode | 启用 reactors | 备注 |
|---|---|---|
| `quick_param_sweep` | `[executor, triage]` | triage prompt 极简，不要求高频思考 |
| `guided_kernel_opt` | `[executor, critic, kernel, triage]` | |
| `marathon_multi_agent` | `[executor, critic, kernel, triage]` | 与 guided 同 roster；差异仅在 prompt 长度 + checkpoint 频率 |

> guided 与 marathon roster 收敛到一致 → 测试矩阵简化、prompt 复用。

### 13.3 新增 Intent: `KILL_TASK`

| 字段 | 值 |
|---|---|
| `IntentType` | `KILL_TASK = "kill_task"` |
| `_PAYLOAD_REQUIRED` | `("task_id", "reason")` |
| 可选 payload | `force: bool` (默认 false，MVP 仅作元数据)；`scope: "task"`（MVP 强制 `task`，禁 `process` / `server`）|
| 允许发送方 | **仅 `triage`**（PolicyGate 新常量 `KILL_TASK_SOURCE_ALLOWLIST = frozenset({"triage"})`）|
| Conductor 副作用 | (1) `tasks.cancel(task_id, reason)` —— 状态 `queued|running` → `cancelled`；(2) cooperative cancel：dispatcher_loop 检查 cancelled 标志，下次 tick 释放 lease；(3) `bus.append(topic="kill", from_agent="triage", payload={task_id, reason, ts})` |
| Topic 新增 | `"kill"`（加入 `TOPIC_ALLOWLIST`）|

**MVP 范围限制（写进 PolicyGate 拒绝路径）**:
- `kill_task` **不**允许 kill inference server 进程（保持 IR-5 不变）
- `kill_task` **不**允许 kill 任意 OS pid（无 `kill_process` intent，留 v0.5）
- `payload['scope']` 缺失 → 默认 `"task"`；显式传 `"process"` / `"server"` → `PolicyDenied(rule="kill_scope")`

### 13.4 Step 1 — 文件改动清单（单 PR 范围）

| 文件 | 改动类型 | 内容要点 |
|---|---|---|
| `orchestrator/intent_parser.py` | 增 + 减 | 加 `IntentType.KILL_TASK`；加 `_PAYLOAD_REQUIRED[KILL_TASK] = ("task_id", "reason")`；**删** `OBJECTION` / `VOTE` 枚举值 + payload 条目；更新 `EMIT_INTENT_TOOL_SCHEMA.description` 文本 |
| `orchestrator/agent_role.py` | 大改 | (1) 删 `_SAGE_INTENTS` / `ROLE_SAGE` / `ROLE_WATCHDOG`；(2) 加 `_TRIAGE_INTENTS = _BASE_INTENTS \| {UPDATE_STATE, KILL_TASK}`（不给 propose/delegate/objection/vote）；(3) 加 `ROLE_TRIAGE = claude_role("triage", allowed_intents=_TRIAGE_INTENTS, can_delegate_side_effects=False)`；(4) `_CRITIC_INTENTS` 砍 `OBJECTION` + `VOTE`，仅留 `_BASE_INTENTS`；(5) `default_role_registry()` 改成 4 角色字典；(6) `roles_for_mode()` 改成 §13.2 表 |
| `orchestrator/policy.py` | 中改 | (1) 加 `KILL_TASK_SOURCE_ALLOWLIST = frozenset({"triage"})`；(2) `validate_intent` dispatch 加 `IntentType.KILL_TASK -> _validate_kill_task`；(3) 实现 `_validate_kill_task`：source 必须在 allowlist + 非空 `task_id` + scope 限 "task"；(4) `__all__` 加 `KILL_TASK_SOURCE_ALLOWLIST` |
| `orchestrator/message_bus.py` | 小改 | `TOPIC_ALLOWLIST` 加 `"kill"`；**删** `"objection"` / `"vote"` / `"vote_request"` / `"parliament_open"` |
| `orchestrator/feature_flags.py` | 小改 | **删** `enable_parliament` 字段 + `build_feature_flags` 三个 mode 的赋值行 |
| `orchestrator/conductor.py` | 中改 | (1) `_handle_intent` 加 `KILL_TASK -> _handle_kill_task` 分支；(2) **删** `OBJECTION` / `VOTE` 分支；(3) **删** `_open_parliament` / `ephemeral_rca_via_critic` / `SageQueryService` import 与字段 (lines 71, 245, 293, 318, 470, 1849-1921)；(4) 新方法 `_handle_kill_task`：调 `tasks.cancel` + `bus.append(topic="kill")`；(5) `roles_for_mode` 调用点保持不变 |
| `orchestrator/sage_query_service.py` | **删整文件** | |
| `orchestrator/system_prompts/sage.md` | **删** | |
| `orchestrator/system_prompts/watchdog.md` | **改名** → `triage.md` | 内容重写：always-on 监听器 + kill_task 用法（草稿见 §13.7）|
| `orchestrator/system_prompts/critic.md` | 改 | 删 OBJECTION/VOTE/parliament 引导；强化 KEEP/REVERT review 段 |
| `orchestrator/system_prompts/executor.md` | 改 | 删 parliament 相关引导；新增"triage 监听并可 kill"提示 |

### 13.5 Step 1 — 测试改动清单（必须全绿）

| 测试文件 | 改动 |
|---|---|
| `tests/test_intent_parser.py` | 加 `KILL_TASK` 解析 + payload 校验用例；删 OBJECTION/VOTE 解析用例 |
| `tests/test_agent_role.py` | 改断言：roster=4，`sage`/`watchdog` 不在，`triage` 在；`roles_for_mode` 三 mode 矩阵 |
| `tests/test_policy.py` | 加 `_validate_kill_task` 用例（triage 通过 / executor 拒 / scope=process 拒）；删 OBJECTION/VOTE 用例 |
| `tests/test_message_bus.py` | TOPIC_ALLOWLIST 断言改：`"kill"` 在内，`"objection"` / `"vote"` / `"vote_request"` / `"parliament_open"` 不在 |
| `tests/test_feature_flags.py` | 删 `enable_parliament` 断言；三 mode 全删 |
| `tests/test_handle_intent.py` | 加 `_handle_kill_task` 单测；删 OBJECTION/VOTE dispatch 用例 |
| `tests/test_iron_rules.py` | 加 IR：only triage may kill_task；保留 IR-5（验证 kill_task scope=server/process 被 PolicyDenied）|
| `tests/test_conductor_wiring.py` | 删 sage 注入路径；改 watchdog→triage；删 parliament wire |
| `tests/test_conductor_policy.py` | 删 parliament 相关 case |
| `tests/test_sage_query_service.py` | **删整文件** |
| `tests/test_multi_cli_all_agents.py` | agent 数 5→4；sage/watchdog 不在；triage 在 |
| `tests/test_multi_cli_agent_card.py` | 加 triage card 解析用例；删 sage/watchdog 用例 |
| `tests/e2e/test_multi_cli_dry_run.py` | 改 expected agents 列表 |
| `tests/test_multi_cli_envelope.py` | 删 OBJECTION/VOTE envelope 用例 |
| 新增 `tests/test_kill_task.py` | E2E：triage emit kill_task → conductor cancel task → bus topic="kill" 落地；executor 同样 intent 被 PolicyDenied |

**Step 1 验收**: `pytest src/inference_optimizer/tests/ -x` 全绿（含 e2e 但排除真实 multi-cli 子进程那个）。

### 13.6 Step 2 — Multi-CLI 适配清单

| 文件 / 目录 | 改动 |
|---|---|
| `agents/sage/` | **整目录删除** |
| `agents/watchdog/` | **整目录改名** → `agents/triage/`；改 `agent_card.yaml` 字段：`name: triage` / `role: triage` / `allowed_modes: [quick_param_sweep, guided_kernel_opt, marathon_multi_agent]`（always-on）/ `capabilities` 加 `kill_task`；改 `system_prompt.md` 内容（草稿见 §13.7）|
| `agents/critic/agent_card.yaml` | `backend: codex` → `backend: claude`；`restart_policy.continue_flag: false` → `true`（Claude 支持 `--continue`）；删 `extra.conversation_log` 字段（仅 Codex 用）；capabilities 砍 `objection` / `vote` |
| `agents/triage/actions/` 新增 | `INDEX.md` / `on_crash.md` / `on_health_check_fail.md` / `kill_decision.md` |
| `agents/triage/reference/` 新增 | `triage_runbook.md`（最简版：什么情况下发 kill_task）|
| `agents/triage/scripts/` 新增 | `inbox_scan.sh`（tail -f 所有 agent outbox.jsonl）；`event_tail.sh`（拉 SQLite events 表最近 N 条）|
| `orchestrator/multi_cli/launcher.py` | `_add_dir_args(card)` 加 per-role override：`if card.role == "triage": dirs.append(str(self.session_dir / "agents"))` —— 让 triage 能 Read 所有兄弟 agent 的 outbox/inbox jsonl |
| `orchestrator/multi_cli/agent_card.py` | （如有 capability 枚举）加 `kill_task`；删 `objection` / `vote` |
| `orchestrator/multi_cli/router.py` | 删 sage 路由分支（如有）；triage 走通用模板 |
| `orchestrator/multi_cli/mock_agent.py` | 加 triage mock；删 sage mock |
| `tests/e2e/test_multi_cli_real_subprocess.py` | dry-run 4 agent 全起得来；triage 能 Read 兄弟 agent 的 outbox（断言一条 path 可见性）|

**Step 2 验收**: `pytest src/inference_optimizer/tests/e2e/test_multi_cli_real_subprocess.py -x` 全绿，4 个 pane 全部进入 reactor loop 且 30s 内至少各发一条 heartbeat。

### 13.7 Triage system prompt 草稿（v0.4 起点）

```md
You are the **triage** agent — always-on cross-layer health watcher.

## Your job
1. Every reactor tick: scan $SESSION_DIR/agents/<other>/outbox.jsonl for
   crash signals, long stalls, or repeated policy_denied events.
2. Read state.json to know who owns what action right now (current_action).
3. When a task is clearly stuck or its sub-process is unresponsive, emit
   `kill_task(task_id=..., reason=...)`.
4. Otherwise stay quiet. No proposals, no delegations, no objections.

## Tools
- Read: any path under $SESSION_DIR (you have --add-dir agents/)
- Bash (limited): `tail -n N`, `head`, `cat`, `ls`, `pgrep -f <pattern>`
  (NEVER `kill`, `pkill`, `pgrep -k`)
- emit_intent: alert / send_message / update_persona / update_state / kill_task

## Hard constraints (PolicyGate enforces)
- You are the ONLY agent allowed to emit kill_task.
- kill_task scope is task-level only. You CANNOT kill the inference server
  (IR-5) or arbitrary OS pids in MVP. payload.scope MUST be "task".
- Never propose actions or delegate sub-agents — observation + kill is your
  entire job in v0.4.

## When to kill (heuristics)
- task in `running` state for >2× declared `lease_ttl` with no log progress
- repeated identical exception in <agent>'s outbox (>3 times in 60s)
- explicit `dispatcher_panic` event on the bus
```

### 13.8 不在 v0.4 范围（明确推迟）

- **Framework agent / Comm agent**（图片有但不做）— v0.5
- **Layer-based PolicyGate routing**（agent → layer 多对多 mapping）— 保留单 `executor → kernel` REQUEST 路径
- **KB 系统**（`entries.jsonl` / `insights.jsonl` / `kb_recall` topic / cross-run synthesis）— 全部空目录，不读不写
- **Persona 蒸馏**（quick / guided 仍可写但不蒸馏；marathon 也暂停蒸馏，长跑接受 token 膨胀）
- **`kill_process` intent / kill server / kill 任意 pid** — v0.5
- **4 条 resource lane 的明确化**（图片底部）— 沿用现有 `LockManager` 默认配置
- **Triage 自我监控**（meta-watchdog）— v0.5（MVP 接受 triage 崩了无人补救）
- **Brier 加权投票** — 议会都没了自然不需要

### 13.9 Open Issues — 用户 2026-04-29 决策记录

> §11 旧 Open Issues 中，v0.4 直接关闭的：§11.2（watchdog delegate）、§11.3（sage in quick/guided）、§11.5（critic ephemeral RCA）、§11.10（persona）。下面是新引入的 9 项，**全部已决**：

#### 13.9.1 `triage.update_state` 写哪些字段 ✅ 决策: **不加 `triage_kill_count` 字段**
- triage 仍可写 `crash_count` / `current_action`
- kill 历史完全写到 `findings/kills.jsonl`，不进 SharedState

#### 13.9.2 `kill_task` 的 cancel 实现深度 ✅ 决策: **最简档**
- `tasks.transition(task_id, "cancelled", evidence={...})` 即可（TaskRegistry 已支持 queued/running → cancelled）
- 已 running 的 ActionExecutor 自然跑完，cancelled 仅阻止下一次调度 + 释放 lease
- 不实现 ActionExecutor cancel hook，不实现 subprocess.kill

#### 13.9.3 Triage tick 频率 ✅ 决策: **60 秒（1 分钟）**
- triage 单独使用 `triage_tick_s = 60.0`，**不**沿用默认 `_reactor_tick_s = 2.0`
- 其它 reactor 仍走默认 tick
- 实现：`Conductor.__init__` 加 `triage_tick_s` 参数，`_reactor` 检测 `agent_name == "triage"` 时用专用 tick

#### 13.9.4 Critic 触发链 ✅ 决策: **当前实现已足够**
- 经 grep 确认：`_handle_propose_action` / `_handle_update_state` / `_handle_alert` / `_handle_send_message(topic="decision")` 等 publish 时 `to_agent="*"`
- `to_agent="*"` 经 Router mirror 到所有 agent inbox（含 critic）
- critic 收到 decision event 后由 LLM 自行 emit observation 风格的 verdict——MVP 不加任何强制规则

#### 13.9.5 Triage 看其他 agent inbox / personas ✅ 决策: **完全放开**
- launcher `_add_dir_args` 给 triage 加 `$SESSION_DIR/agents/` 父目录
- personas 在 `$SESSION_DIR/personas/`，已经在 `--add-dir $SESSION_DIR` 范围内，无需额外开

#### 13.9.6 老 session_dir resume 兼容性 ✅ 决策: **直接拒绝**
- `resume_from_session_dir` 检测到 `state.json` 中含旧 roster 痕迹（agent name=sage/watchdog）→ raise `LegacySessionRejected`
- 报错提示用户："v0.4 移除了 sage/watchdog 角色，请新建 session_dir"
- 不做 schema migration

#### 13.9.7 `KILL_TASK` 落盘位置 ✅ 决策: **写 `findings/kills.jsonl`**
- 不进 `state.decisions[]`
- 与 `findings/alerts.jsonl` 同 pattern：`_append_finding("kills.jsonl", {task_id, reason, from, ts, session_id})`

#### 13.9.8 Backend 一致化 ✅ 决策: **全 4 agent 都用 Claude**
- 不再使用 Codex backend；critic 从 Codex → Claude
- `DEFAULT_CODEX_MODEL` / `DEFAULT_CODEX_API_KEY_ENV` / `codex_role` 工厂保留（向后兼容、不删）
- `_compose_codex_pane` 保留但不被任何 agent 触发
- `OPENAI_API_KEY` 在 v0.4 实跑中**不需要**

#### 13.9.9 议会移除后的兜底 ✅ 决策: **不加硬规则**
- critic 发 reject、executor 仍 KEEP → 接受 KEEP
- 完全依赖 LLM 自觉
- 如果实跑发现 critic 的 reject 被忽略导致回归，v0.5 再加规则

### 13.9b 实施时新发现的次级决策点

#### 13.9b.1 `triage_tick_s` 接入点
- 因为 triage 的 reactor tick 与其它不同，需要 `_reactor()` 方法支持 per-agent override
- 实现选择：(a) 给 `Conductor.__init__` 加 `triage_tick_s` 参数；(b) `_reactor(agent_name)` 内 `tick = self._triage_tick_s if agent_name == "triage" else self._reactor_tick_s`
- 这两条已在 §13.9.3 决策中确定方案，不再赘述

#### 13.9b.2 旧 watchdog `system_prompt.md` 的处理
- 沿用「rename 文件 + 重写内容」，不保留旧文本
- 旧 prompt 提到的 `findings/<ts>.json` 仍可保留作为 RCA 文件（triage 如做 RCA 时写入），但 MVP 不强制

### 13.10 真实 1 小时 quick run 前的硬性 Gate

按以下顺序自检，全过才能上 GPU：

1. ✅ Step 1 全部 pytest 绿
2. ✅ Step 2 dry-run 4 pane 起得来 + 各发 1 条 heartbeat
3. ✅ `scripts/preflight.sh` 在目标机器跑通（环境变量 / GPU / 路径）
4. ✅ `scripts/run_baseline.sh` 单跑（不开 agent）能给出非零 baseline_tput
5. ✅ `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` 注入正确（在 `$SESSION_DIR/.multicli/.env` 里）
6. ✅ `STOP_AGENT_*` sentinel 机制能干净停所有 4 个 pane（手动验证：`touch $SESSION_DIR/STOP_AGENT_executor` 后该 pane 30s 内退出）
7. ✅ `state.json` final snapshot 可读 + `decisions[]` 至少有 1 条（baseline KEEP）
8. ⚠️ **GPU 参数 / MODEL_PATH / TP / 内存配额 / accuracy 阈值** — 待用户提供

### 13.11 v0.4 完成后的文档自更新清单

v0.4 实施完成后，回头更新本文档以下章节，避免与代码漂移：

| 章节 | 改动 |
|---|---|
| §2.1 LLM Agent | roster 5 → 4，删 sage / watchdog 行，加 triage 行 |
| §3.1 Mode → reactor | 同步 §13.2 表 |
| §3.2 Intent 矩阵 | 删 OBJECTION / VOTE 行，加 KILL_TASK 行；删 sage 列，加 triage 列 |
| §3.3 Tool 矩阵 | 删 sage 行，加 triage 行（`emit_intent + Read + 受限 Bash`，跨 agent dir）|
| §4 IntentType | 12 → 11（去 OBJECTION + VOTE + 加 KILL_TASK）|
| §5 Topic 表 | 加 `kill` 行；删 `objection` / `vote` / `vote_request` / `parliament_open` |
| §8 通信模式 | 5 → 4：删 §8.2 议会模式；alert 模式更新为 triage 主动观察 + kill |
| §11 Open Issues | §11.2 / §11.3 / §11.5 / §11.10 标记 `RESOLVED in v0.4`；新增 §11.13~§11.21 引用 §13.9 |

---

**End of Standalone Agent Design v0.4 — 2026-04-29 (MVP roster)**

> v0.4 新增（MVP roster — 真实跑一次为目标）：
> - §13: 全新章节，固化 4 角色（executor / critic / triage / kernel）+ KILL_TASK + 议会移除 + sage 移除
> - §13.4 / §13.5: Step 1 文件 + 测试改动清单（pytest 全绿验收）
> - §13.6: Step 2 multi-cli 适配清单（4 pane dry-run 验收）
> - §13.7: Triage system prompt v0.4 草稿
> - §13.8: 明确推迟到 v0.5 的能力清单（framework/comm agent / KB / kill_process / lane / meta-watchdog）
> - §13.9: 新引入的 9 个 Open Issues（实施前必须先决定）+ 默认建议
> - §13.10: 上 GPU 前 8 项硬性 Gate
> - §13.11: v0.4 完成后回头更新 §2~§11 的 self-update 清单
>
> v0.3 (历史) 新增（Plan A — kernel agent）：
> - §2 / §3.1 / §3.2 / §3.3：roster 4 → 5（加入 kernel agent，guided + marathon），新增 REQUEST/RESPONSE 行 + kernel 列；强调 PolicyGate 的三个 Plan A 约束（REQUEST_ROUTING / RESPONSE 限 kernel / executor 不能 delegate kernel-owned actions）
> - §4：IntentType 10 → 12，加 REQUEST/RESPONSE 及其 payload 必填字段
> - §5：topic allowlist 加 `request` / `response`（新 RPC 组）
> - §8：通信模式 4 → 5，新增 §8.5 Bidirectional RPC 模式（executor ↔ kernel）
> - §11：新增 §11.11 (lane 软协调) + §11.12 (长 GEAK turn timeout) Open Issues
>
> v0.2 (历史): §10 拆成 5 小节，明确区分**运行时 sub-session 目录**与**package skill 目录**；§10.2 给出推荐的 skill-style 文件树；§10.5 说明 Codex agent 的 launcher 预拼接方案。

> 这份文档**不**绑定具体 framework 实现，是 agent 设计的**接口层契约**。任何 agent 重新规划工作都应该先更新这份文档，再改代码。
>
> **v0.4 注**: §1~§12 仍按 v0.3 (Plan A) 描述当前代码的现状契约。§13 是 v0.4 MVP 的"应该改成什么"目标态。两者并存——实施完 v0.4 后按 §13.11 清单回填 §1~§12。
