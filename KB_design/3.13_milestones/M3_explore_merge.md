# M3 — explore action 合并 (backends + params + validate_stack)

## 1. 设计目标

把 v0.6 的 `backends` / `params` 两个 action 折叠为单一 `explore`,
`validate_stack` 的语义内嵌进 KEEP 之后的 stack rebench 步骤; ledger
统一为 `explore_search`。

落地后用户应当看到: prompt 中只有一个 `explore` 候选 action; breakdown
的 capability_summary 只见一行 `explore` (兼容 alias 仍保留 backends /
params 行); 老 session resume 后 ledger 无损迁移。

## 2. 范围

**包含**:

- 新 yaml meta `actions/_meta/explore.yaml`。
- 删除 `backends.yaml` / `params.yaml` / `validate_stack.yaml`。
- 新执行体 ExploreExecutor (概念上 = BackendsExecutor + ParamsExecutor
  + 每 variant 即时 KEEP/REVERT + KEEP 后内嵌 stack rebench)。
- SharedState 新 ledger `explore_search` (合并 backends_search +
  params_search + backend_winners_history)。
- 老 ledger 迁移逻辑 (resume 一次性)。
- canonical_fingerprint 函数 (单点)。
- prompt_builder 中 actions 列表 / DECISION FRAMEWORK 文本同步更新。
- breakdown.capability_summary 改写 + 兼容 alias。
- M2 phase 状态机的 plateau_explore proxy 改用 explore_search
  winners_history (替换 v0.6 暂用的 params_no_promote_streak)。

**不包含**:

- specialist 接入 (M5/M6); 本里程碑 EXPLORE 仍由 LLM 直接 propose
  variant grid (default_grid 或 LLM-picked grid)。
- explore variant 的 KB hypothesize per-variant 路径 (留 M5; 本里程
  碑仍按 M1 的"per propose_action 一条 hypothetical 边" 简化)。

## 3. 与 M1/M2 的关系

- M1 提供 KB 写通路; M3 在每个 variant KEEP/REVERT 时仍只触发 *propose
  action 级别*的 ingest_attempt + verify, 不细化到 variant (M5 才细化)。
- M2 提供 phase 状态机; 本里程碑 EXPLORE phase 内允许 actions 集合从
  `{backends, params, sweep, validate_stack}` 改为 `{explore, sweep}`
  (sweep 暂留, M3 不动 sweep)。
- PolicyGate 角色 × intent 矩阵补一条: `propose_action` /
  `delegate{action_name='backends'/'params'/'validate_stack'}` 在 M3
  落地后即返回 `policy_denied: action_deprecated`, hint 引导改用
  `explore`。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| `actions/_meta/explore.yaml` | family=shallow, lanes=[server_lifecycle, benchmark_lane], pipeline_phase=explore |
| ExploreExecutor 概念 | 输入 grid, 串行跑每 variant + 即时 KEEP/REVERT + KEEP 后内嵌 stack rebench, 返回 explore_search update |
| SharedState.explore_search | tested / accepted / rejected / winners_history / discovered_flags / synergy_attempted / domains_round_summary |
| canonical_fingerprint | sha1(sorted(extra_args) + sorted(extra_envs) + framework + tp + workload_signature) |
| ledger 迁移函数 | union/merge backends_search + params_search → explore_search; backend_winners_history 时序 merge |
| prompt_builder | actions 列表只列 explore + sweep; DECISION 段提"variant 来自 LLM 自填或下个里程碑的 specialist" |
| breakdown.capability_summary | 删 backends/params/validate_stack; 加 explore; 兼容 alias 见 §3.12 §4.2 |
| PolicyGate 'action_deprecated' | 拒绝旧 action 名的 propose / delegate, hint 给替换路径 |

## 5. variant 契约

每个 variant 字典字段 (M3 layer):

- `name`: 唯一名 (round 内不重)
- `extra_args`: sglang/vllm CLI 风格字符串
- `extra_envs`: dict[str,str]
- `provenance`: M3 阶段固定为 `'llm_direct'` 或 `'default_grid'` (
  `'specialist:<domain>'` 留 M5)
- `kb_evidence` / `pr_evidence` / `source_evidence`: 可空 (M5/M6 填)

ExploreExecutor 内部:

1. 对每个 variant 用 canonical_fingerprint 去重, 命中 ledger 即标
   SKIPPED, 不跑。
2. 串行跑剩余 variant: render YAML → restart server → run E2E bench →
   解析 result → 立刻 KEEP/REVERT (沿用 v0.6 0.2% 阈值 + accuracy gate)。
3. 每个 KEEP 后, *内嵌* 一次 stack rebench: 用累积的 stack (含刚 KEEP
   的 variant) 再跑一次 E2E。
4. 若 stack rebench tput < base_tput × (1 + threshold_stable), 标
   `keep_unstable_in_stack`, 弹出刚 KEEP 的 variant, 视同 REVERT。
5. 全 batch 跑完返回 result + ledger update。

## 6. ledger 字段 (统一)

`SharedState.explore_search`:

```
tested: list[
  {
    fingerprint: str,
    name: str,
    outcome: 'KEEP' | 'REVERT' | 'SKIPPED' | 'FAILED' | 'KEEP_UNSTABLE',
    round_id: int,
    ts: iso_str,
  }
]

accepted: list[
  {
    fingerprint: str,
    variant: {name, extra_args, extra_envs, provenance, ...},
    gain_pct: float,
    stack_index: int,    # 入 optimization_stack 的索引
    accepted_at_round: int,
    ts: iso_str,
  }
]

rejected: list[
  {
    fingerprint: str,
    variant: {...},
    reason: 'gain_below_threshold' | 'accuracy_drop' | 'stack_unstable' | 'failed' | ...,
    round_id: int,
    ts: iso_str,
  }
]

winners_history: list[
  {
    round_id: int,
    variant_name: str,
    fingerprint: str,
    gain_pct: float,
    extra_args: str,
    extra_envs: dict,
  }
]

discovered_flags: list[
  {flag: str, source: 'specialist:<domain>' | 'default_grid', first_seen_round: int}
]

synergy_attempted: list[list[str]]   # 已尝试过的 flag 组合

domains_round_summary: list[
  # M3 层为空; M5/M6 填 specialist 派发统计
]
```

## 7. 迁移逻辑 (一次性, resume 时跑)

```
backends_search ∪ params_search → explore_search:
  - tested = union (按 fingerprint 去重, 同 fingerprint 取最高 outcome:
             KEEP > REVERT > SKIPPED)
  - accepted = union, 按 ts 排序
  - rejected = union
  - winners_history = merge (源 = backend_winners_history +
             params_search.winners_history) sort by (round_id, ts)
  - discovered_flags = union
  - synergy_attempted = union
  - domains_round_summary = []

backend_winners_history 字段保留为 alias (软删除), 至少一个版本周期。
```

resume 后老 ledger 字段在 SharedState 类中标 deprecated, 写时跳过持
久化, 读时仍返回内存值 (避免老 reader 崩)。

## 8. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | 新 explore.yaml 落地; ActionRegistry 加载支持; PolicyGate 默认接收 |
| 2 | SharedState.explore_search 字段 + 迁移函数 (drop 老字段写入路径但保留读) |
| 3 | canonical_fingerprint 函数 (统一); 替换 v0.6 两个 fingerprint |
| 4 | ExploreExecutor 概念落地 (沿用 _grid_runner 大部分内部逻辑, 新 wrapper) |
| 5 | KEEP 后内嵌 stack rebench |
| 6 | prompt_builder 改写 actions 列表 + DECISION 文字 |
| 7 | PolicyGate `action_deprecated` 拒 backends/params/validate_stack |
| 8 | breakdown.capability_summary 改写 + 兼容 alias |
| 9 | 删除 backends.yaml / params.yaml / validate_stack.yaml + 删除 BackendsExecutor / ParamsExecutor / ValidateStackExecutor 注册路径 (保留 _grid_runner / accuracy_gate 等共享代码) |

## 9. 验收清单

- [ ] 冷启 session, EXPLORE 内可见 1+ 个 `explore` 任务; 每个任务跑 N
      variant, 每个 KEEP 后立即 stack rebench。
- [ ] breakdown.capability_summary 见 explore 行 + 兼容 alias backends /
      params / validate_stack 行 (从 explore 派生)。
- [ ] resume v0.6 session: backends_search / params_search 数据进
      explore_search; 后续 propose 不重复跑同 fingerprint。
- [ ] 老 yaml (backends/params/validate_stack) 删除后, action_registry
      不报错 (因为已经从 enable list 移除)。
- [ ] PolicyGate 拒绝 propose_action='backends' / 'params' /
      'validate_stack' 并给出 `action_deprecated` hint。
- [ ] phase 状态机的 plateau_explore proxy 改用 explore_search
      winners_history 计算, M2 注释中的 "provisional" 标记可清理。

## 10. 风险与回退

主风险:

- **stack rebench 把原本 KEEP 的 variant 弹出, 导致 cumulative_gain
  抖动**. 缓解: stack rebench 阈值给保守值 (默认 0.5%), 灰度调小。
- **fingerprint 函数边界 case**: workload_signature 派生不一致 →
  不同 round 同 variant 被认为不同 fingerprint. 缓解: workload_signature
  派生唯一 (CONC/ISL/OSL/precision/TP 五字段哈希), 加 unit 测试。
- **resume 老 ledger 数据丢失**: 缓解: 迁移函数前对老字段做 backup
  copy 到 `state.json.backup` 文件, 出错可手工恢复。

回退:

- PR9 删除 yaml 是不可回滚的; 之前 PR 都可回。回退建议: 先 revert
  PR9 (恢复老 yaml), PolicyGate 拒绝路径放开, prompt 加回 backends /
  params action, ledger 切回老字段名。
- 紧急 degrade 模式: `--legacy-explore-split` flag (M3 文档承诺), 强
  制走老 backends / params 路径, 关闭 explore action。

## 11. 哲学回引

本里程碑直接落地 §3.4 概念; **Inv-4.1 (单 ledger)** + **Inv-4.2 (variant
fingerprint 唯一)** 是本里程碑的内部强约束; **主轴 C** (执行体一分为
二) 在本里程碑 *尚未发力*, 但通过 yaml + ledger 的提前合并为 M5
specialist 接入腾出了空间。
