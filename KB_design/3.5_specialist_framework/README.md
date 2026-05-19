# §3.5 Specialist Sub-Agent 框架

## 1. 设计目标

为 Hyperloom 引入第二种 sub-agent 形态 (LLM specialist), 与现有的
deterministic Python executor 共存。specialist 的唯一使命是
**产出"提案 + 引用证据"**, 不跑 E2E bench, 不应用 patch, 不抢
serving GPU。

成功标准:

- 在 EXPLORE phase 里, Orchestration 可以并行派发 N 个 specialist,
  每个 specialist 在受限工具集 + 注入式 prompt 下跑若干轮 LLM 对话,
  最终输出一条 `specialist_done` intent。
- specialist 失败 / 超时 / 空提议 都有明确语义, 不阻塞主 4-agent
  loop。
- specialist 的引入不破坏 v0.6 的 PolicyGate / Critic Review / Cortex
  写入中转 / serving GPU 单租户 等核心约束。

## 2. 现状回顾

v0.6 的 sub-agent 现状 (`orchestrator/sub_agent_runner.py`):

- 一种形态: deterministic Python executor。`SubAgentRunner.executor_registry[kind]`
  必为 `async def fn(ctx) -> dict`。
- 触发: `delegate{action_name, params}` → PolicyGate → TaskRegistry 入队
  → SubAgentRunner 串行 dispatch (`_pump_dispatcher_once` 内 for loop)。
- 资源: `requires_lanes` 由 yaml meta 给出, 默认抢 benchmark_lane /
  server_lifecycle。

v0.6 缺什么 (相对于 TBO `arbor.dispatch.dispatch_batch`):

- 没有 LLM driver 形态的 sub-agent。
- 没有 sub-agent 在自己工作目录中产出 done.json / patches / heartbeat
  / new_knowledge.md 的协议 (TBO 的 `comms.py`)。
- 没有 sub-agent 的 stale 检测 / 自动 GPU 回收。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节额外引入:

### Inv-5.1 — specialist 不出 patch

specialist 输出**只有提案**: 由文字描述 + 候选 variant 列表 + 引用
证据组成。它不写 git patch 文件, 不直接修改 framework 源码, 不调用
build/编译/micro-bench。

理由: Hyperloom 的 KEEP/REVERT 闸 + Critic 评审 + accuracy gate 都
绑定在 deterministic executor 的 E2E bench 上, specialist 自带 patch
会出现"micro speedup ≠ E2E speedup" 的 TBO 已踩过的坑。

### Inv-5.2 — specialist 输出经 Coordinator 中转

specialist 不直接写 SharedState, 不直接写 Cortex, 不直接给其他角色
inbox 发消息。它只能通过 `specialist_done` intent 交付结果, 由
Coordinator 解析并代发后续动作 (T2 hypothesize / propose_action /
ledger 增量等)。

### Inv-5.3 — specialist 单一退出协议

无论成功 / 失败 / 超时 / 空提议, specialist 都必须以一条
`specialist_done` intent 收尾。沉默 = stale, 由 robustness 杀掉并视
为空提议。

## 4. 双形态 sub-agent runner

v0.8 把 `SubAgentRunner` 概念上扩展为两个形态共存的 dispatcher:

| 维度 | deterministic executor | LLM specialist |
|---|---|---|
| 触发 | `delegate{action ∈ deterministic 集合}` | `delegate{action='specialist', params.domain=...}` |
| 执行 | 调用 `executor_registry[action]` (async fn) | 启动一个 LLM 后端调用 (Claude / Codex / Cursor), 多 turn |
| 工具集 | subprocess.run (Magpie / GEAK / OOB) | Read / Grep / Glob / Bash(白名单) / WebSearch / WebFetch / Cortex MCP / PR Monitor MCP |
| Lane | 默认 benchmark_lane / server_lifecycle | research_lane (新增, §3.7) |
| 输出 | 直接 dict | `specialist_done` intent (内含 dict) |
| 失败 | exception / timeout 直接 fail | stale / token overflow / tool error 由 robustness 杀 |
| Lifecycle 文件 | 任务 workspace 下的 `benchmark_report.json` 等 | 任务 workspace 下的 `prompt.md` / `transcript.jsonl` / `heartbeat.json` / `specialist_done.json` |

两个形态共享 task 队列 / TaskRegistry / lease 机制 / idempotency_key
约定, 区别只在 dispatch 时根据 action_name 路由到不同 runner。

## 5. specialist 类型 (运行时清单)

清单是**运行时配置**, 不写死在 yaml meta 里。Coordinator 维护一份
`specialist_domains` 字典:

| domain key | 关心的层 | 默认工具集附加 | 默认 KB 子图锚点 |
|---|---|---|---|
| `kernel_specialist` | aiter / sglang kernels / triton | Read aiter source, PR Monitor (ROCm/aiter, triton-lang/triton) | `kernel.*` 域 traverse |
| `framework_specialist` | sglang / vllm scheduler / cuda_graph / kv_cache | Read sglang/vllm source, PR Monitor (sgl-project/sglang, ROCm/vllm) | `framework.*` 域 |
| `comm_specialist` | RCCL / NCCL / QuickReduce / AllReduce | PR Monitor (相关 repo) | `communication.*` 域 |
| `compiler_specialist` | torch.compile / inductor / triton | PR Monitor (triton-lang/triton, pytorch) | `compiler.*` 域 |
| `system_specialist` | KFD / driver / 内存 / dispatch overhead | Read /proc, rocm-smi, dmesg (read-only) | `systems.*` 域 |
| `pr_intel_specialist` | 跨仓库 PR 检索, 给其他 specialist 提引用 | PR Monitor 全开 | `pr_intelligence.*` 域 |

特点:

- domain 是 **prompt 装配维度**, 不是新 IntentType, 不是新 Role; 增删
  一个 domain 不需要改 PolicyGate 矩阵 (只要 `params.domain` 在已知
  集合内)。
- 每个 domain 都对应 Cortex 的某个 KB 顶层域 + 某些 PR Monitor 仓库,
  避免 specialist prompt 拼到一团乱麻。

domain 选择策略 (Orchestration 一侧):

- gap 描述里出现 attention / MoE / GEMM → kernel_specialist
- gap 描述里出现 scheduler / cuda_graph / kv_cache → framework_specialist
- gap 描述里出现 allreduce / collective → comm_specialist
- gap 描述里出现 inductor / torch.compile → compiler_specialist
- gap 描述里出现 dispatch overhead / launch latency → system_specialist
- 想做 PR-only 的横向研究 → pr_intel_specialist

允许同一 gap 派多个 domain (例如同时派 kernel + pr_intel 让后者给前
者提 PR 参考), 由 Orchestration 自由组合。

## 6. specialist prompt 装配契约

由 Coordinator 现场拼装, 9 段固定:

1. **Identity & autonomy** — "你是 X 类 specialist; 不要碰 serving
   GPU; 不要写 patch; 不要直接调 Cortex 写入。"
2. **Hardware context** — gpu_type / TP / HBM / 计算峰值 / 已知架构
   特性 (例如 MI300X 的 8 XCD / 256MB Infinity Cache)。
3. **Gap statement** — 由 Orchestration 给出: gap 的 canonical_id /
   symptom / layer / 最近一次复现的 evidence。
4. **Cortex KB subgraph** — 以本次 gap canonical_id 为锚 traverse 3 步
   / 4 分支的子图, 含历史 attempt + 验证状态 + negation 边。
5. **Recipe summary** — `find-recipe` 拿到的 best_config + what_worked
   + what_failed + remaining_gaps + pitfalls (T0 缓存)。
6. **PR feed** — pr-monitor 上当前 domain 关心仓库的近期 N 条 PR
   摘要 (标题 + 链接 + 标签)。
7. **Local source navigation hint** — `framework_source_roots` 列表
   + 重点目录 + "只读不可写"提示。
8. **Output protocol** — 必须以一条 `specialist_done` 收尾, 内含字段
   契约 (见 §7)。
9. **Iron rules** — 不动 serving GPU / 不跑 E2E / 不直接写 KB / 不修
   framework 源码 / 必须在 max_specialist_turns 内收尾 / 写日志到
   workspace 而非 stdout。

每段都允许为空 (Coordinator 在某段无内容时, 写 `(none)` 占位以保持
prompt 结构稳定)。

## 7. specialist_done intent 契约

新增 IntentType: `specialist_done`。

payload 必填字段:

- `gap_canonical_id`: 与派发时一致 (双向校验)。
- `domain`: 与派发时一致。
- `proposal_set`: list of variant; 每个 variant 字段 ⊆ §3.4 explore
  variant 字段 (`name`, `extra_args`, `extra_envs`, `kb_evidence`,
  `pr_evidence`, `source_evidence`, `reason`)。
- `confidence`: 0.0–1.0 自评。
- `empty`: 布尔, true 表示无提议; 此时 proposal_set = [], reason 必填。
- `summary`: 一段简短自然语言总结 (≤ 500 字符), 给后续轮的 prompt
  作"上轮发生了什么"。

可选:

- `new_findings`: list of "我学到了但没用到的"知识点; Coordinator 可
  在 commit 时把这些写入 Cortex 的 `optimization_node` (HYPOTHESIZED)
  + investigation 边, 即使本次没成 KEEP, 也保留为 KB 资产。
- `residual_questions`: 给下一轮 specialist 留的 followup。

PolicyGate 校验:

- `from_agent` 必须以 `specialist:<task_id>` 前缀, 否则拒绝。
- intent 内字段 schema 必须满足上述契约。
- `gap_canonical_id` 必须匹配派发时 task.params.gap (防止串话)。

## 8. specialist 任务生命周期

```
   queued                           running                            terminal
   ──────                           ───────                            ────────
   created by Orchestration's       specialist_runner picks up,         specialist_done received
   delegate intent + PolicyGate     acquires research_lane slot,        → SubAgentResult.state =
   approval                         spawns LLM backend with assembled    "succeeded"
                                    prompt + restricted toolset
                                                                        OR
                                    LLM runs N turns, calling tools
                                    (Read/Grep/Cortex/PR Monitor),       robustness kill_task
                                    writes transcript to workspace       (stale heartbeat) →
                                                                        SubAgentResult.state =
                                    on each turn: heartbeat update      "cancelled" + synthesise
                                                                        empty specialist_done

                                    final turn must emit
                                    specialist_done intent (else
                                    counts as stale)
```

每个 specialist 任务的 workspace 在 `<session_dir>/runs/specialist/<task_id>/`:

- `prompt.md` — 装配后的完整 system+user prompt
- `transcript.jsonl` — 每 turn 的 LLM 交互记录
- `heartbeat.json` — 最后一次更新时间 + 当前 turn (≤ max_turns)
- `tool_calls.jsonl` — Cortex / PR Monitor / Read / Bash 等工具调用日
  志 (谁、何时、参数)
- `specialist_done.json` — 退出时由 specialist 自己写入的最终 payload
  副本 (与 inbox 中的 intent 等价, 仅作审计)

## 9. stale 检测与失败模式

由 robustness 在每个 tick 扫:

| 信号 | 阈值 | 行为 |
|---|---|---|
| `heartbeat.json` 超 X 分钟未更新 | X = max_turns × per_turn_max_min × 1.5 (默认 ≈ 10 分钟) | `kill_task`, robustness alert (medium) |
| LLM backend 返回 token overflow / quota error | 不依赖 robustness, specialist runner 自报 | task → failed; 合成空 specialist_done |
| LLM backend 连续 3 turn 未调用任何工具 (空回复) | specialist runner 自检 | 视为收敛失败, 强制退出, 合成空 specialist_done |
| transcript 中检测到尝试调用禁用工具 (Edit / Write / git apply) | specialist runner 拦截 (工具白名单) | 工具调用直接拒绝, 不杀 task; 多次拒绝则 task → failed |
| `specialist_done` payload 校验失败 | PolicyGate 拒收 | 视为空 specialist_done; 不重派 (避免循环) |

## 10. 与 deterministic executor 的并存边界

specialist 与 deterministic executor 在以下方面**严格分离**:

- yaml meta: deterministic action 各有 yaml; specialist 没有 yaml,
  由 `domain` 字段做参数化。
- workspace 路径: deterministic action 在 `runs/<action>/<task_id>/`,
  specialist 在 `runs/specialist/<task_id>/` (扁平, 不分 domain)。
- ledger: deterministic action 写自己的 `<action>_attempts` /
  `last_action_failures`; specialist 写 `specialist_rounds` 集合, 不
  污染 deterministic action 的 audit 路径。
- KEEP/REVERT 概念: 不适用于 specialist (它没跑 bench)。

## 11. 实施步骤

1. **新 IntentType `specialist_done`** + PolicyGate 校验
   (`from_agent` 前缀, payload schema)。
2. **runner 路由**: `SubAgentRunner.run_task` 概念上做 if/elif 分流:
   action_name == 'specialist' → 走 SpecialistRunner; 否则原 path。
3. **prompt 装配**: 在 prompt_builder 旁新增 specialist_prompt 装配
   逻辑, 9 段固定 + domain 模板表 (kernel/framework/...)。
4. **工具白名单**: 在 specialist runner 启动 LLM 时显式传入
   `allowed_tools = [Read, Grep, Glob, Bash(白名单), WebSearch,
   WebFetch, mcp__pr_monitor__*, mcp__cortex_kb__traverse,
   mcp__cortex_kb__find_recipe]`。
5. **heartbeat 写入**: specialist runner 每个 turn 后更新
   heartbeat.json; 由 robustness 扫描。
6. **stale 杀任务**: robustness 的 stale 扫描成为新增 tick step。
7. **空 specialist_done 合成**: kill 后 Coordinator 必须合成一条 empty
   specialist_done 写入 Orchestration inbox, 否则 EXPLORE 轮逻辑会
   永远等。
8. **resume 兼容**: 老 v0.6 session 没有 specialist 任务记录, resume
   后第一次 EXPLORE 派 specialist 即可。

## 12. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| Orchestration 派 N > research_lane.capacity | TaskRegistry 入队等候, 后续 lane 释放即开跑 |
| specialist 工作中 phase 已切走 (Coordinator 因别的原因决定 EXPLORE 退出) | Coordinator 在 phase 退出时主动 kill_task 全部 specialist; 合成 empty done |
| specialist 在 prompt 中收到一份 PR feed 但 PR Monitor 此刻不可达 | prompt 中 PR feed 段写 `(empty: pr_monitor unavailable)`, specialist 仍跑, 不挂任务 |
| 同一 gap 同时被多个 specialist 提议同一 variant | 由 §3.4 explore_search.fingerprint 去重处理, specialist 自身不需要避免 |
| specialist 在 transcript 中泄漏认证 token | tool 调用日志在写入 workspace 前 redact (token / api_key 等) |

## 13. 验收标准

- [ ] EXPLORE 中可见 1+ 个并行 specialist 任务, 各 transcript 完整, 各
      heartbeat 更新; 全部以 specialist_done 收尾。
- [ ] kill 一个 specialist (人工或自动), 主 reactor loop 收到一条空
      specialist_done; round 不卡住。
- [ ] specialist 不能成功调 Edit / Write / git apply (尝试即被拒)。
- [ ] specialist 不能直接写 SharedState / 调 cortex-kb hypothesize (
      尝试 → PolicyGate 拒)。
- [ ] new_findings 里的非 KEEP 提议被作为 HYPOTHESIZED 边写入 Cortex
      (在 §3.6 commit 时点)。
- [ ] specialist round 的 metadata 落到 SharedState.specialist_rounds
      与 breakdown.specialist_runs。

## 14. 依赖与影响面

- **上游**: §3.1 (主轴 C 双形态), §3.3 (Orchestration 决策), §3.6
  (KB / PR / 源码三源), §3.7 (research_lane)。
- **下游**:
  - §3.4 explore 的 grid 上游就是 specialist proposal_set。
  - §3.10 SharedState 新增 `specialist_rounds`。
  - §3.11 PolicyGate 校验 specialist_done / domain / 工具白名单。
  - §3.12 breakdown.specialist_runs 段。
  - §3.13 milestone M5 / M6 实施本节。

## 15. 哲学回引

本节是**主轴 C** 的核心落地 (执行体一分为二);
**Inv-5.1 / Inv-5.2 / Inv-5.3** 守住"specialist 只产出提案、Coordinator
中转、单一退出协议" 三个边界;
**Inv-3 (serving GPU 单租户)** 通过 research_lane 与 benchmark_lane
互不冲突保证。
