# §3.4 EXPLORE 阶段合并 — backends + params → explore

## 1. 设计目标

把 v0.6 中重复度极高的 `backends` 和 `params` 两个 action 合并为单一
`explore` action, 同时把"specialist 提案 → Critic 评审 → explore
executor 串行 bench"装配成 EXPLORE phase 的标准节奏。

成功标准:

- 一个 yaml meta + 一个 ledger + 一个执行体, 没有 backends/params 的
  二分。
- LLM (Orchestration) 在 EXPLORE 阶段只需要思考 "派几个 specialist"
  和 "选哪批 variant 让 explore executor 跑", 不需要思考"这个 flag
  是 backend 还是 param"。
- v0.6 已经积累的 `backends_search` / `params_search` ledger 在 resume
  时无损迁移到新 ledger。

## 2. 现状回顾

v0.6 的 `backends` 和 `params` action 在 `actions/_meta/`:

| 维度 | backends | params |
|---|---|---|
| 调度 lane | server_lifecycle + benchmark_lane | server_lifecycle + benchmark_lane |
| 执行体 | `BackendsExecutor` → `_grid_runner._run_magpie` | `ParamsExecutor` → `_grid_runner._run_magpie` |
| 渲染目标 | `EXTRA_SGLANG_ARGS` 或 backend flags | `EXTRA_SGLANG_ARGS` 或 sglang 运行时 param |
| 默认 grid | `DEFAULT_BACKENDS_GRID` / `DEFAULT_VLLM_BACKENDS_GRID` | `DEFAULT_PARAMS_GRID` / `DEFAULT_VLLM_PARAMS_GRID` / `DEFAULT_NCCL_GRID` |
| ledger | `backends_search` (tested/accepted/rejected/winners_history) | `params_search` (同结构) |
| LLM grid 注入 | 支持 (LLM 可填 `params.grid`) | 支持 |

观察: 两者只在"默认 grid 内容" + "渲染时落到的 EXTRA_*_ARGS env 名"
有差异, 其余几乎相同。`_grid_runner` 已经把这两部分都参数化, 合并几
乎是把两个 entry point 折叠为一个。

`validate_stack` 的本质是 "重新 bench 当前 optimization_stack 的全
组合配置", 在 v0.8 被吸收进 EXPLORE 串行 integrate 内 (每次 KEEP 后
都自动 stack-rebench), 不再独立 action。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量, §3.2 phase 不变量。本节额外引入:

### Inv-4.1 — 单 ledger

EXPLORE 阶段所有 variant 的"已 tested / 已 accept / 已 reject / 各轮
winner"信息只存于 `explore_search` 一份 ledger。不允许出现"backends
专属字段" / "params 专属字段"。

### Inv-4.2 — variant 的 canonical_fingerprint 唯一

每个 variant 由 `canonical_fingerprint = sha1(sorted(extra_args) +
sorted(extra_envs) + framework + tp + workload_signature)` 决定唯一
身份。同一 fingerprint 在 ledger 中只能出现一次, 重复 propose 直接
被去重 (无论 propose 来自 specialist / LLM / 默认 grid)。

## 4. 核心机制

### 4.1 explore action 的语义

`explore.yaml` (新增 meta):

| 字段 | 取值 / 说明 |
|---|---|
| family | shallow |
| pipeline_phase | explore |
| requires_lanes | server_lifecycle, benchmark_lane |
| allowed_tools | emit_intent, Read, Bash |
| side_effects | launches_server, reads_server, writes_results |
| description | "Apply a batch of N candidate variants serially; KEEP/REVERT each, stack onto optimization_stack" |
| typical_runtime_min | per-variant base × N (与 v0.6 backends p50 一致) |

输入 `task.params` 中的核心字段:

- `grid`: 必填, list of variant; 每个 variant 至少含
  `name`, `extra_args`, `extra_envs`, `provenance`(specialist domain
  / `default_grid` / `llm_direct`), `kb_evidence`(可选), `pr_evidence`(可选)。
- `base_extra_args`: 当前 stack 累积的 baseline args (来自
  `current_best.extra_sglang_args`)。
- `config_path` / `result_dir` / `benchmark_script`: 与 baseline 的
  workload contract 完全一致 (沿用 v0.6 `baseline_config_path` 机制)。

执行体(逻辑层面):

1. 接到 grid → 用 explore_search.fingerprint 集合去重, 标记重复 variant
   为 SKIPPED。
2. 串行 (不并行) 跑剩余 variant, 每个 variant:
   - 渲染一份 Magpie YAML, 重启 server。
   - 跑 E2E bench (沿用 baseline executor 的 timeout / leak salvage)。
   - 解析 benchmark_report.json → 单轮 result。
   - **立刻**判断 KEEP/REVERT (复用 v0.6 promotion 规则: 0.2% 阈值 +
     accuracy gate)。
   - KEEP 即把 variant 推入 `optimization_stack`, 更新 `current_best`,
     重新计算 `base_extra_args` 给后续 variant; REVERT 即写
     `last_action_failures` + ledger.rejected。
3. 整批跑完返回 result, 给 Coordinator 走 `_promote_to_shared_state` /
   `_handle_unpromotable_result` 的统一记账逻辑。

注意: v0.6 backends/params 是"先全跑完再选 best winner"的 batch 思路,
**v0.8 explore 是"每跑完一个就 KEEP/REVERT 决定"的 stack 思路**。
后者更接近 TBO Iron Rule "one change at a time", 也消除了 v0.6
"winner 没动过 stack 但 ledger 显示 KEEP" 的歧义。

### 4.2 specialist → orchestration → explore 三段流水

```
                       ┌────────────────────────── one EXPLORE round ──────────────────────────┐
                       │                                                                        │
   T-A  Coordinator    │   把 gap_list partition 成 D 个 domain (kernel/framework/comm/...)     │
                       │   决定 N (≤ research_lane.capacity) 个 specialist 派发, 每个绑 1 domain│
                       │                                                                        │
   T-B  Orchestration  │   for each specialist:                                                 │
                       │       emit delegate{action='specialist',                                │
                       │             params={                                                    │
                       │                domain=...,                                              │
                       │                gap=<canonical_id>,                                      │
                       │                kb_subgraph=<traverse 结果>,                            │
                       │                pr_feed=<预热 PR 摘要>,                                  │
                       │                source_roots=[...],                                      │
                       │                max_turns=K                                              │
                       │             },                                                          │
                       │             idempotency_key='spec-<round>-<domain>'                    │
                       │       }                                                                  │
                       │                                                                        │
   T-C  并行 specialist│   N specialists 在 research_lane 上同时跑, 每个返回 specialist_done   │
                       │   {                                                                    │
                       │       gap: <canonical_id>,                                              │
                       │       proposal_set: [variant1, variant2, ...],                          │
                       │       confidence: 0.0-1.0,                                              │
                       │       kb_evidence: [...], pr_evidence: [...], source_evidence: [...]   │
                       │   }                                                                    │
                       │                                                                        │
   T-D  Orchestration  │   汇总所有 specialist_done, 去重 (canonical_fingerprint),              │
                       │   排序 (specialist confidence 优先, ties 按 proposal_set 内顺序)        │
                       │   选 top-M 个 variant (M = explore_round_batch_size, 默认 5)            │
                       │   emit propose_action{                                                  │
                       │       action_name='explore',                                            │
                       │       payload={                                                         │
                       │          grid=<top-M variants>,                                         │
                       │          base_extra_args=<current stack>,                               │
                       │          provenance_summary=<specialist 出处>                          │
                       │       },                                                                │
                       │       idempotency_key='explore-round-<N>'                              │
                       │   }                                                                    │
                       │                                                                        │
   T-E  Critic Review  │   一组 M 个 variant 一次评审, verdict 可整组 approve/reject/redirect, │
                       │   也可对单个 variant 给 verdict (新增"分项评审"语义)                  │
                       │                                                                        │
   T-F  explore exec   │   approved 的 variant 串行 bench, 每个立刻 KEEP/REVERT                  │
                       │   全部跑完后返回 result + ledger update                                │
                       │                                                                        │
   T-G  Coordinator    │   _promote_to_shared_state / _handle_unpromotable_result 沿用,          │
                       │   每个 variant 触发 Cortex verify confirmed/refuted (T3)              │
                       │                                                                        │
   loop until plateau  └────────────────────────────────────────────────────────────────────────┘
```

### 4.3 explore_search ledger

合并后的统一 ledger 概念字段 (具体 schema 留实施稿):

| 字段 | 含义 |
|---|---|
| `tested[]` | 所有跑过的 variant fingerprint + outcome (KEEP/REVERT/SKIPPED/FAILED) |
| `accepted[]` | KEEP 的 variant 列表 (含 stack index, gain 历史) |
| `rejected[]` | REVERT 的 variant 列表 (含 reason, accuracy_drop 等) |
| `winners_history[]` | 每轮入 stack 的 variant 元数据 (round_id, variant_name, gain_pct, kb_edge_id) |
| `discovered_flags` | 来自 specialist source_evidence 的"我在源码里看到这个 flag 但还没试过"列表 (供下轮种子) |
| `synergy_attempted` | 已尝试过的 flag 组合 (避免反复尝试同一组合) |
| `domains_round_summary` | 按 domain 维度的派发-提议-KEEP 统计 |

**resume 迁移**: v0.6 `backends_search` 与 `params_search` 在 SharedState
加载时合并:

- `tested`/`accepted`/`rejected`: 直接 union, 同 fingerprint 优先取 KEEP
  > REVERT > SKIPPED 的最高级别。
- `winners_history`: 时序 merge 后按 round_id 排序。
- 新字段 `discovered_flags` / `synergy_attempted` / `domains_round_summary`
  在迁移时填空 (老 session 没有 specialist 概念)。

迁移是**一次性写入**, 写完后旧字段在 SharedState 中保留至少一个版本
(打 deprecated 标), 便于回滚验证, 之后才正式清除。

### 4.4 与 validate_stack 的合并

`validate_stack` 在 v0.6 的语义: "把 optimization_stack 的所有
KEEP'd 项重新 bench 一遍, 验证它们组合后的稳定性"。v0.8 把这个语义
**自动嵌入 explore executor 的每个 KEEP 之后**:

- explore executor 跑完一个 variant 且判定 KEEP 后, **不立即返回**,
  而是触发一次 "stack rebench" — 用当前累积的 stack(含刚 KEEP 的
  variant)再跑一次 E2E。
- 如果 stack rebench 的 tput < `base_tput * (1 + threshold)`, 即认为
  组合不稳定, 把刚刚 KEEP 的 variant 标 `keep_unstable_in_stack`, 从
  `optimization_stack` 弹出, 视同 REVERT。
- 这样 validate_stack 不再需要独立 action; 它的"稳定性闸门"语义内嵌
  到 KEEP 的判定中。

**额外好处**: 不会再出现 v0.6 中"explore KEEP 完, 后来 validate_stack
跑一次发现整组组合反而劣化, 只能整组撤回"的复杂回退路径。

资源:stack rebench 仍占 benchmark_lane, 整段操作仍在同一份
`task.lease`, 不需要新 lane。

## 5. 接口/契约

### 5.1 explore action 的输入契约

| 字段 | 强制 | 说明 |
|---|---|---|
| `grid` | 是 | list of variant dict |
| `grid[].name` | 是 | 唯一名 (round 内不重复) |
| `grid[].extra_args` | 否 | 字符串 (sglang/vllm CLI 风格) |
| `grid[].extra_envs` | 否 | dict[str,str] |
| `grid[].provenance` | 是 | "specialist:<domain>" / "llm_direct" / "default_grid" |
| `grid[].kb_edge_id` | 否 | T2 hypothesize 创建的 edge_id (有则在 T3 用作 verify 锚点) |
| `grid[].kb_evidence` | 否 | KB 引用 list |
| `grid[].pr_evidence` | 否 | PR url + sha list |
| `grid[].source_evidence` | 否 | 框架源码引用 list |
| `base_extra_args` | 是 | 当前 current_best 的累积 args |
| `config_path` | 是 | 来自 baseline 的 workload contract |
| `accuracy_baseline` | 否 | 用于精度 gate, 缺省走 baseline_accuracy |

### 5.2 explore action 的输出契约

返回 dict 至少包含:

- `status`: succeeded / failed
- `output_throughput`: 整批 KEEP'd variant 累积后的 stack tput
- `best_variant`: 整批中最好的单 variant 描述
- `winners[]`: 这批被 KEEP 的 variant
- `losers[]`: 这批被 REVERT 的 variant
- `keep_unstable_in_stack[]`: 跑了 stack-rebench 后被弹出的 variant
- `explore_search_update`: ledger 增量更新
- `discovered_flags_update`: specialist source_evidence 中提到但本次
  未试的 flag (写到 ledger.discovered_flags)

### 5.3 与 Critic Review 的契约

- Critic 收到的 propose_action 中 payload.grid 是 M 个 variant。
- Critic 评审反馈是 dict { variant_name → verdict }, 每 verdict ∈
  REVIEW_VERDICTS (`approve / reject / redirect / advise / needs_review`)。
- 整组 reject / 全部 needs_review 时, propose_action 视为 reject, 不
  跑 explore。
- 部分 redirect 时, redirected variant 被替换为 Critic 给出的等价
  variant (通过 `redirect_to.variant` 字段) 后再跑。

## 6. 实施步骤

1. **新 yaml meta**: 创建 `actions/_meta/explore.yaml`, 字段如 §4.1。
2. **执行体逻辑**: 设计 `ExploreExecutor` 的概念 = "BackendsExecutor +
   ParamsExecutor 的并集 + 每 variant KEEP/REVERT 即时判定 + stack
   rebench"。
3. **ledger 数据模型**: SharedState 新增 `explore_search` 字段; 老
   `backends_search` / `params_search` 标 deprecated, 在 load 时一次
   性迁移。
4. **fingerprint 函数**: 单一 canonical_fingerprint 函数, 取代 v0.6
   两份 ledger 各自的 fingerprint 实现。
5. **prompt builder 升级**: Orchestration prompt 中 "ACTIONS YOU MAY
   USE" 列出 `explore` 而非 backends / params; 提到合并语义。
6. **default_grid 兜底**: 在 specialist 不可用 / proposal_set 全空时,
   提供 SKILL.md 默认 grid 作为 LLM 第一轮种子, 防止 EXPLORE 死等。
7. **validate_stack 内嵌**: explore executor KEEP 后 stack rebench 的
   逻辑独立成内部步骤, 共享 lease。
8. **resume 迁移**: 一次性脚本, SharedState load 时调用; 不放在 hot
   path。

## 7. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| grid 为空 | 视同 round 空 proposal_set, Orchestration 计 specialist_empty_streak |
| grid 内某个 variant 渲染失败 | 标 SKIPPED, 不计入 KEEP/REVERT, 继续下一个 |
| 第 k 个 variant 跑挂 (server crash) | KEEP/REVERT 视为 REVERT, robustness 触发 server cleanup; 后续 variant 仍跑, 不打断整 batch |
| stack rebench 跑挂 | 视为 keep_unstable_in_stack, 上一个 KEEP 弹出, 不影响后续 |
| Critic 把整组 reject | 不跑 explore, ledger 不更新; 计入 specialist_round_summary 为 "rejected_by_critic" |
| Critic 全 needs_review | 视为本轮 abort; Orchestration 下轮重新派 specialist, 不重复同样 grid |
| LLM 直接绕过 specialist 用 default_grid 反复 propose 同 fingerprint | fingerprint 去重直接 SKIPPED, ledger 写 dedup_count 反映 |

## 8. 验收标准

- [ ] 一份 `explore.yaml` + 一份 ledger + 一份执行体, 旧 yaml 已删除。
- [ ] 一个 v0.6 的 backends_search ledger 在 v0.8 启动后被无损迁移到
      explore_search; backends/params 各自的 winners_history 顺序保留。
- [ ] specialist + Orchestration + Critic + explore executor 的 4 步
      闭环在一个完整 EXPLORE 轮内可见 4 类 event (delegate 派 specialist
      / specialist_done / propose_action / delegated_result)。
- [ ] 每次 KEEP 都触发 stack rebench, 不稳定的 KEEP 自动弹出且
      `optimization_stack` 不残留。
- [ ] breakdown.capability_summary 新格式中只见 `explore` 行 (不再有
      backends / params / validate_stack)。
- [ ] 重复 propose 同 fingerprint 时, 第二次进 ledger 标 SKIPPED 而
      不是 REVERT。

## 9. 依赖与影响面

- **上游**: §3.2 (EXPLORE phase 内才允许), §3.3 (Orchestration 的
  EXPLORE 决策), §3.5 (specialist 提案上游), §3.6 (KB / PR 给 variant
  打证据)。
- **下游**:
  - §3.7 research_lane 容量决定 N 上限。
  - §3.8 plateau_explore 信号依赖 winners_history 与 specialist 派发统计。
  - §3.10 SharedState 新字段 `explore_search` / `specialist_rounds`。
  - §3.11 PolicyGate 把 backends/params/validate_stack 标记为非法
    action_name。
  - §3.12 breakdown 中 backends/params 行合并。
  - §3.13 milestone M3 实施本节; M5/M6 把 specialist 接到 T-B/T-C 的位置。

## 10. 哲学回引

本节是**主轴 A** 的 EXPLORE 段落落地: phase 内自由组合, 不靠 scoreboard;
**主轴 C** 的执行体分工: specialist 出提案, explore executor 跑 bench;
**Inv-3 (serving GPU 单租户)**: explore 全程占 benchmark_lane, 串行验证。
