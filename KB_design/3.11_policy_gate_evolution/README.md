# §3.11 PolicyGate 演进 — 5 条新规则

## 1. 设计目标

PolicyGate 是 Hyperloom 唯一仍由代码硬性把守的边界。v0.8 在不放松 v0.6
任何已有规则的前提下, 增加 5 条与新管线 + 新 sub-agent 形态对齐的规
则, 同时保留"拒绝即写 policy_denied 让 LLM 自纠"的标准否决流程。

成功标准:

- 每条新规则有: 触发条件 + 拒绝理由 (rule 标识符) + hint (给 LLM 的
  自纠提示) + 测试可复现路径。
- 5 条规则相互正交, 不会出现"同一个 intent 因为多条规则冲突而拒绝
  歧义" 的情况。
- 老 v0.6 规则无任何放松或修改 (除非是 §3.9 的评分字段相关 ENUM 校验
  和 §3.10 的 CORE_STATE_FIELDS 扩展)。

## 2. 现状回顾

v0.6 PolicyGate (`orchestrator/policy.py`) 的现有规则:

- 角色 × intent 兼容矩阵 (`_*_INTENTS`)
- REVIEW_VERDICT 仅 critic
- KILL_TASK / FORCE_DISPATCH / PRUNE_BRANCH / ESCALATE_STRATEGY_CHANGE
  仅 robustness
- REQUEST 路由 (orchestration → kernel)
- KERNEL_OWNED_ACTIONS 不可被 delegate
- CORE_STATE_FIELDS 只允 Coordinator 写
- 路径 containment (PATH_LIKE_FIELDS 必须在 SESSION_DIR 或
  SOURCE_FILE_ALLOWLIST)
- topic 白名单 / 优先级范围

否决路径: 拒绝时抛 `PolicyDenied(reason, rule, hint)`, Coordinator
catch 后写一条 `policy_denied` 观察事件到对应 LLM 角色的 inbox, LLM
下一 tick 看到后自纠。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节核心:

### Inv-11.1 — PolicyGate 仍是 pure validator

PolicyGate 不持有副作用 (不改 SharedState / 不写 KB / 不动 lease);
它只接受 (intent, source_role, current_state) 输入, 返回
allow / `PolicyDenied`。新增规则保持这一性质。

### Inv-11.2 — 拒绝总是可恢复

任何拒绝都伴随 `rule` 标识 + `hint`, hint 必须告诉 LLM **怎么改 intent
能让下次通过**。"无解的拒绝"必须直接抛硬错而不是 PolicyDenied (例如
配置不一致就该 fail-fast, 不该让 LLM 自己绕)。

### Inv-11.3 — 规则正交

任何 intent 在被拒绝时, 只命中**唯一一条**规则。规则间设计成检查顺
序无关或有显式优先级 (优先级表见 §6)。

## 4. 5 条新规则

### 4.1 规则 R1 — phase 兼容性

**触发**: 任意 `propose_action` / `delegate` / `request` intent, 且
`action_name` 或 `request.kind` 不在当前 `SharedState.phase` 允许集合
内 (依据 §3.2 §5)。

**rule 标识**: `phase_incompatible`

**hint 范例**:
```
You are currently in phase=EXPLORE. Allowed actions: explore,
specialist, recover. Action 'kernel_opt' is only allowed in KERNEL
phase. Either propose 'explore' / 'specialist', or wait for the
phase transition (your current explore round summary suggests
plateau is approaching).
```

**作用域**: 所有发出此类 intent 的角色, 但实际只有 Orchestration 会
被这条规则命中 (其它角色按角色 × intent 矩阵已经被前置规则拦)。

**白名单**: `recover` action (任何 phase 允许), `report` /
`session_breakdown` (CLOSE phase 允许), `target_analysis` /
`baseline` (PRELUDE phase 允许)。

### 4.2 规则 R2 — specialist 派发来源

**触发**: `delegate{action_name='specialist', ...}` 但 source_role ≠
`orchestration`。

**rule 标识**: `specialist_dispatch_source`

**hint 范例**:
```
Only the Orchestration role may dispatch specialists. If you
(role=robustness) want to escalate, use escalate_strategy_change
with hint='need_specialist:<domain>'; the orchestration tick will
pick it up.
```

**校验补充**:

- `params.domain` 必须 ∈ §3.5 §5 的已知 specialist domain 集合; 否则
  rule = `specialist_unknown_domain`, hint 列出已知 domain。
- `params.gap_canonical_id` 必填且非空; 否则 rule =
  `specialist_missing_gap`。
- `params.max_turns` 在合理上限内 (默认 ≤ 8); 否则 rule =
  `specialist_max_turns_excess`。

(以上 specialist 子规则归并到 R2, 但拒绝时 hint 要明确指示哪一项。)

### 4.3 规则 R3 — specialist_done 来源

**触发**: `specialist_done` intent 但
`from_agent` 不以 `specialist:<task_id>` 前缀开头, 或 `task_id` 与
TaskRegistry 中的 specialist 任务不匹配。

**rule 标识**: `specialist_done_source`

**hint 范例**:
```
specialist_done can only be emitted by an active specialist
sub-agent whose from_agent='specialist:<task_id>' matches a
running TaskRegistry entry of kind=specialist. Did you intend
'send_message{topic="advice"}'?
```

**额外校验** (子规则归并到 R3):

- `payload.gap_canonical_id` 必须与派发时 `task.params.gap` 完全一致;
  否则 rule = `specialist_done_gap_mismatch`。
- `payload.domain` 必须与派发时 `task.params.domain` 一致; 否则 rule
  = `specialist_done_domain_mismatch`。
- `payload.proposal_set` schema 校验 (每个 variant 的字段); 失败 rule
  = `specialist_done_schema`。

### 4.4 规则 R4 — KB 写权角色矩阵

**触发**: 任何 LLM 角色 (orchestration / kernel / critic / robustness /
specialist) 的 intent 试图直接调用 cortex-kb 写端点
(propose-point / propose-edge / hypothesize / ingest-attempt / verify
/ commit) 或试图通过 Bash 工具 spawn `cortex-kb` 子进程做写操作。

**rule 标识**: `kb_write_unauthorized`

**hint 范例**:
```
Direct KB writes are not allowed. The Coordinator owns all KB
writes; you express your intent through propose_action / delegate
/ specialist_done.proposal_set / review_verdict / kb_writes
(critic-agent commit-review). Please remove the cortex-kb write
call.
```

**实施层**:

- specialist 工具白名单不包含 cortex-kb 写端点 (只允许 traverse /
  find-recipe / query 等只读)。
- Bash 工具白名单显式排除 `cortex-kb propose-*` / `... hypothesize`
  / `... ingest-attempt` / `... verify` / `... commit` 等写命令。
- critic-agent 的 `commit-review` 例外: 它通过自己的 prepare-review /
  commit-review 协议产出 `kb_writes` 数组, Coordinator 代发, 不算 LLM
  直接写。

### 4.5 规则 R5 — Web / PR / Cortex MCP 工具白名单分级

**触发**: 任何 sub-agent / 反应器试图调用工具:

- `WebSearch` / `WebFetch`
- `mcp__pr_monitor__*`
- `mcp__cortex_kb__*` (即使是只读)

按角色 × phase 表校验:

| 工具集 | 谁能用 | 何时能用 |
|---|---|---|
| WebSearch / WebFetch | specialist only | EXPLORE phase only |
| mcp__pr_monitor__* (只读) | specialist only | 任何 phase (实际 EXPLORE 主用) |
| mcp__cortex_kb__traverse / find_recipe / query (只读) | specialist only | 任何 phase |
| mcp__cortex_kb__propose-* / hypothesize / verify / ... (写) | 无 (R4 已禁) | — |

**rule 标识**:

- 工具调用越权 → `tool_whitelist_role`
- phase 不对 → `tool_whitelist_phase`

**hint 范例**:
```
WebFetch is restricted to specialist sub-agents during EXPLORE
phase. As Orchestration in PRELUDE, you cannot invoke external
URLs. Use the warm_start_recipe and gaps lists in your prompt.
```

**实施层**:

- 工具白名单在 `sub_agent_runner` 启动 LLM backend 时显式传入, LLM
  backend 自身拒绝调白名单外工具 (无需走到 PolicyGate)。
- PolicyGate 仅作 *intent 层面*的二次校验 — 即如果 LLM 通过别的方式
  (例如以 Bash 包装) 试图调禁用工具, intent 层面拦下。

## 5. 否决流程 (与 v0.6 一致, 强化点)

```
   1. LLM 角色 emit intent
   2. PolicyGate.validate(intent, source_role, state)
      a. 角色 × intent 类型 (v0.6)
      b. 角色 source 白名单 (v0.6: REVIEW_VERDICT/KILL_TASK/...)
      c. REQUEST 路由 (v0.6)
      d. KERNEL_OWNED_ACTIONS (v0.6)
      e. R1 phase_incompatible        (v0.8 新)
      f. R2 specialist_dispatch_source (v0.8 新)
      g. R3 specialist_done_source     (v0.8 新)
      h. R4 kb_write_unauthorized      (v0.8 新)
      i. R5 tool_whitelist_role/phase  (v0.8 新)
      j. CORE_STATE_FIELDS             (v0.6, 已扩展见 §3.10)
      k. 路径 containment              (v0.6)
   3. 任一规则触发 → 抛 PolicyDenied(rule, hint)
   4. Coordinator catch:
      - 写 policy_denial_history
      - 累加 policy_denial_streak[(action, rule)]
      - 写一条 policy_denied event 到该 LLM 角色 inbox
      - 若 streak ≥ 10 同 (action, rule) → 写 stop_reason=policy_loop
   5. LLM 下 tick 看到 inbox 中 policy_denied + hint, 自纠
```

强化点:

- 规则的 *优先级*: 顺序按 §5 a–k, 越靠前越早判; 一旦命中即返回, 不再
  评估后续规则。
- `policy_denial_history` 中每条记录: timestamp / rule / hint / source
  / intent (redacted) / decision (denied)。breakdown 段可回溯 (§3.12)。

## 6. 规则正交性 (Inv-11.3 检验)

| 规则 | 触发面 | 与其它规则可能重叠? |
|---|---|---|
| R1 phase_incompatible | propose_action/delegate/request 在错 phase | 与角色矩阵正交 (角色矩阵先判) |
| R2 specialist_dispatch_source | delegate{action='specialist'} 来源错 | 与 R1 正交 (R1 先判 phase, R2 再判 source) |
| R3 specialist_done_source | specialist_done 来源错 | 与角色矩阵正交 (specialist_done 不属于已有角色 intent 集合, 单独走 R3) |
| R4 kb_write_unauthorized | 调 cortex-kb 写或 spawn 写命令 | 与 R5 重叠? 否 — R5 是工具白名单, R4 是 *写权* (R5 直接禁 cortex_kb 写工具, 即使用 Bash 也走 R4) |
| R5 tool_whitelist_role/phase | 用 Web/PR/Cortex MCP | 与 R4 正交 (R4 仅写, R5 含读) |

## 7. 接口/契约

PolicyGate 入口签名 (概念层):

```
validate(intent, source_role, shared_state) -> None | raises PolicyDenied
PolicyDenied: {reason, rule, hint}
```

`shared_state` 注入是 v0.8 新增 (v0.6 PolicyGate 已经接 SharedState
做 baseline_self_loop 校验, v0.8 在此基础上加 phase 字段读取)。

## 8. 实施步骤

1. **schema 锁定**: 把 R1–R5 的 rule 标识 / hint 模板 / 触发条件写
   入一份 contract 文档, 加测试覆盖每条规则的 deny+hint 路径。
2. **优先级表**: 在 PolicyGate 入口处把规则顺序固化 (§5 a–k)。
3. **specialist 子规则合并**: R2/R3 内部子规则在 hint 中区分, 但都用
   主 rule 标识, 减少 ENUM 爆炸。
4. **CORE_STATE_FIELDS 扩展**: 同步在 PolicyGate 校验 `update_state`
   时加入 §3.10 §6.1 列出的新字段。
5. **stop_reason ENUM 校验**: 在 update_state 路径上拦截非词表值
   (§3.8 §6 词表)。
6. **policy_denial_history 升级**: 加 hint 字段, breakdown 段读取 (§3.12)。

## 9. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| LLM 同时违反 R1 + R4 | R1 先命中 (优先级高), 返回 phase_incompatible; R4 不再评估 |
| specialist 在自己 prompt 中表达"我已经写了 KB"但实际没写 | 没有 intent 触发, 不算违规; specialist 的 self-report 仅供日志 |
| critic-agent 的 commit-review 输出 kb_writes, 被误判 R4 | critic-agent 的 commit-review 走的是 *Coordinator 代发*, 不是 LLM 直接写; R4 不命中 |
| Bash 工具调 `cortex-kb traverse` (只读) | R4 不拦 (R4 仅拦写); R5 检查工具白名单, specialist 角色允许 → pass; 其它角色拒, rule = `tool_whitelist_role` |
| LLM 试图 emit phase update_state | CORE_STATE_FIELDS 校验拦下, rule = `core_state_field_unauthorized` (沿用 v0.6) |
| Phase 转移期间 (Coordinator 正在写 phase) 同一 tick LLM 的 propose_action 看到旧 phase | PolicyGate 拿 in-memory state, Coordinator 串行写; 即使有 race, LLM 看到 phase=A propose A-action → 通过 → 此时 phase 已切到 B, 任务在执行时被发现不对; 由 dispatcher 二次校验拒 (即 dispatcher 也看 phase) |

## 10. 验收标准

- [ ] R1–R5 各能独立触发 + 写出 hint, 测试场景可复现。
- [ ] policy_denied event 进 inbox, LLM 下 tick 看到并自纠。
- [ ] policy_denial_streak 在同 (action, rule) 累计到 10 时, stop_reason
      = policy_loop。
- [ ] 规则间不重叠 (任一 deny 唯一命中一条规则)。
- [ ] CORE_STATE_FIELDS 扩展后 LLM 无法写 phase / cortex_session_id 等。

## 11. 依赖与影响面

- **上游**: §3.2 (phase), §3.5 (specialist), §3.6 (KB 写权 boundary),
  §3.7 (lane 不归 PolicyGate 管, 但 specialist 派发归 R2), §3.10
  (CORE_STATE_FIELDS 扩展)。
- **下游**:
  - §3.12 breakdown 中的 `policy_denials` 段。
  - §3.13 milestone M2 (phase 校验) + M5 (specialist 校验) 实施。

## 12. 哲学回引

本节是**Inv-1 / Inv-2** 的执行层守卫: PolicyGate 是写入边界。R1–R5
分别守住 phase 单调性、specialist 信息单向流、KB 写经中转、工具白名
单分级。**Inv-11.1 (PolicyGate pure validator)** 保持 v0.6 设计的纯
粹性, 不向 PolicyGate 引入副作用。
