# Dead-A — 已废止 action 工件链 (yaml + executor + 注册)

> 风险等级: **HIGH-misleading**
> 体检报告: `../KB_design_gaps.MD` §12.2
> 关联功能 gap: `Gap-10` (legacy actions still allowed)

## 1. 问题描述

KB_design §3.4 / §3.15 §2.3 / M3 明确: `backends` / `params` /
`validate_stack` 三个 action 合并入 `explore`; `dream` / `re_explore` /
`comm_optimization` / `compiler_tuning` 四个动作由 specialist domain
替代.

当前 tree 中: 7 个 deprecated action 中 **3 个全链 active** (yaml +
executor + cli 注册 + phase allowlist + prompt + critic.md + coordinator
promote 分支 + 测试). 4 个仅 yaml 残留.

对 LLM 阅读器的误导:
- 看 `_REAL_EXECUTORS_FULL` 含 `backends` / `params` / `validate_stack` →
  以为这是 canonical 路径
- orchestration.md 示例 `delegate{action_name='backends'}` → LLM 模仿
- prompt FULL_ENABLED_ACTIONS 含 legacy → LLM 视野里 legacy 优先

## 2. 详细位置清单

### A.1 backends 全链 (HIGH)

| 件 | 路径 | 行 |
|---|---|---|
| yaml | `actions/_meta/backends.yaml` | 全文 |
| executor 模块 | `orchestrator/action_executors/backends.py` | ~500 行 |
| public package | `action_executors/__init__.py` | 14-24 |
| cli 注册 | `cli.py` `_REAL_EXECUTORS_FULL` | 627 |
| phase 允许集 | `phase_state.py` `PHASE_ALLOWED_ACTIONS[EXPLORE]` | 96 |
| prompt | `prompt_builder.py` `FULL_ENABLED_ACTIONS` | 59-65 |
| orchestration.md | EXPLORE allowed actions + example | 32-42, 95-96 |
| critic.md | EXPLORE 列 backends | 15-18 |
| coordinator promote | `_promote_to_shared_state` `task_kind == "backends"` 分支 | ~3585+ |
| 测试 | `tests/test_backends_search.py` | 全文 |
| 测试 | `tests/test_prompt_builder.py` (含断言 backends 在 enabled) | 部分 |

### A.2 params 全链 (HIGH)

同 A.1 结构, 文件:
- `actions/_meta/params.yaml`
- `action_executors/params.py`
- cli.py:628
- coordinator promote `task_kind == "params"` 分支 (~3703+)
- `tests/test_params_search.py`, `tests/test_p2_3_param_executors.py`

### A.3 validate_stack 全链 (HIGH)

见 `Dead-C` (单独文件, 与 A.3 同链但 v0.8 设计争议更大).

### A.4 dream / re_explore / comm_optimization / compiler_tuning yaml (MEDIUM)

仅 yaml 残留, 无 executor / 无 cli 注册. 但 ActionRegistry 加载它们,
`tests/test_p1_2_full_action_catalogue.py` 锁住 deprecation.

### A.5 explore 入口存在但 promote 缺失 (HIGH — 最讽刺)

`explore` 是 v0.8 canonical action, executor 已注册 (cli.py:635), **但**:

- `coordinator._promote_to_shared_state` 全文**无** `task_kind == "explore"`
  分支 (~3356-3886). 落到 generic fallback
- `SharedState.apply_explore_search_update` (1997-2053) +
  `record_explore_accepted` (2055+): **生产代码 0 调用点** (仅 tests)
- `_sequence_denial_for_action` 的 `sequence_actions` 集合 (1757-1764)
  **不含** explore, 跳过所有顺序门

验证:

```
$ rg "apply_explore_search_update|record_explore_accepted" inference_optimizer/orchestrator
inference_optimizer/orchestrator/shared_state.py  # 定义
$ rg "apply_explore_search_update|record_explore_accepted" inference_optimizer/tests
inference_optimizer/tests/test_v08_m3_explore.py  # 仅测试
```

LLM 读 KB_design 会以为 explore 是 GA. 实际看代码: M3 PR 只接了
*executor 入口*, *promote 出口* 没接.

## 3. 根本原因

M3 设计 (§3.13 M3 §PR9 "delete legacy yaml + executor") 标记为 final
PR. 实际:
- PR1–PR8 (ledger / executor / promote / migration) 都合了
- **PR9 没合**, reviewer 推迟"等 explore 端到端 GA 验证"
- explore 自身因 A.5 没 GA, 形成死循环: legacy 不能删 (因为 explore 没
  接全) → explore 没人 push 接全 (因为 legacy 兜底)

§3.13 设计的 `--legacy-explore-split` flag 用来管理灰度, 但**没实现**.

## 4. 修复路径

### Phase 0 — 修 A.5 (explore promote)

**前提**, 否则 Phase 1 删 legacy 会导致 EXPLORE 阶段无可用 grid runner.

#### PR 0.1 — `_promote_to_shared_state` 加 explore 分支

```text
async def _promote_to_shared_state(self, task_kind, result, ...):
    ...
    if task_kind == "explore":
        await self._promote_explore_result(result, task, ...)
        return
    if task_kind == "backends":
        ...
```

#### PR 0.2 — `_promote_explore_result` 实现

```text
async def _promote_explore_result(self, result, task, ...):
    """v0.8 M3 — promote explore round outcomes to SharedState.

    Routes per-variant outcomes through:
    - apply_explore_search_update (ledger)
    - record_explore_accepted (winners_history)
    - _lift_to_current_best (KEEP variant promote)
    """
    per_variant = result.get("per_variant_outcomes") or []
    accepted = [v for v in per_variant if v.get("outcome") == "KEEP"]
    rejected = [v for v in per_variant if v.get("outcome") == "REVERT"]

    # Update ledger
    self.shared_state.apply_explore_search_update({
        "tested": {v["variant_name"]: v for v in per_variant},
        "accepted": [v["variant_name"] for v in accepted],
        "rejected": [{"variant_name": v["variant_name"], "reason": v.get("reason", "")} for v in rejected],
    })

    # Update winners_history + current_best
    for v in accepted:
        self.shared_state.record_explore_accepted(v)
        # Compare to current_best, lift if better
        if self._explore_variant_beats_current_best(v):
            self._lift_to_current_best(v, source="explore")
```

#### PR 0.3 — `_sequence_denial_for_action` 加 explore

`coordinator.py:1757-1764`:

```text
sequence_actions = frozenset({
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "operator_tuning", "vendor_kernel_config",
    "explore",  # ← NEW: explore also subject to baseline + profile prereqs
})
```

但 explore 实际不需要 profile 前置 (EXPLORE phase 内), 应当只对 baseline
有要求. 细节调整待 PR 落地确认.

### Phase 1 — 删除 legacy (A.1 / A.2)

依赖 Gap-10 PR. 见 Gap-10 文档.

### Phase 2 — 清理 A.4 stub yamls

依赖 Gap-13.

## 5. 验收口径

- [x] explore action 端到端工作: `_promote_to_shared_state` 走 explore
      分支, `apply_explore_search_update` 调用计数 ≥ 1 per round
      (Gap-10 落地了 A.5)
- [x] breakdown.capability_summary.explore.tested > 0 per fresh session
- [x] M3 验收清单第 1 条 (`breakdown.capability_summary` 只见 explore 行
      作主行) 成立 — `_AUDIT_ACTIONS` 已收紧到
      ``{baseline, profile, sweep, explore}``
- [x] backends / params / validate_stack yaml 物理删除
- [x] action_executors/backends.py / params.py / validate_stack.py 物理删除
- [x] `cli.py:_REAL_EXECUTORS_FULL` 不含三者 (Gap-10 移除了注册)

## 6. 实际落地 (2026-05-20, 合并 Dead-A.5 + Phase 1 + Phase 2)

* **A.5 (explore promote)** 由 Gap-10 落地, 本次复用.
* **物理删除**: 3 yaml (`actions/_meta/{backends,params,validate_stack}.yaml`)
  + 3 executor 模块 (`action_executors/{backends,params,validate_stack}.py`)
  + 1 agent doc (`actions/validate_stack.md`) + 6 legacy 测试文件
  (`test_{backends_search,params_search,p2_3_param_executors,
  validate_stack,p2_5_grid_promotion,p3_search_space_expansion}.py`).
* **coordinator.py 收紧**: `_promote_to_shared_state` 删除
  `validate_stack` + 合并 ``("backends","params","sweep")`` 块为
  sweep-only ~25 行 (原 ~280 行); `_materialize_approved_proposal` /
  `_handle_delegate` 删除 backends/params 专属 plumbing;
  `_params_grid_exhausted` / `_backends_grid_exhausted` 帮手 +
  legacy imports 删除; `_has_no_more_leverage` 简化; `_AUDIT_ACTIONS`
  收紧到 ``{baseline, profile, sweep, explore}``.
* **action_executors/__init__.py** 删除 3 个 executor + 13 个
  legacy 常量导出.
* **prompt_builder.py** `_format_grid_injection_hint` 删除 backends /
  params 分支; `GRID_INJECTABLE_ACTIONS` 收紧到
  ``{explore, sweep}``; `DEPRECATED_ACTIONS` map 收紧为一行 hint
  per 名 (PolicyGate `action_deprecated` 规则消费).
* **shared_state.py** 收紧字段 docstring: `params_search` /
  `backends_search` 说明改为 "v0.6 resume parity"; `_AUDIT_ACTIONS`
  + `_KEY_METRIC_MAP` 收紧到 v0.8 set; `to_prompt_summary` 渲染
  改 `last_explore=` / `last_sweep=` (不再渲染 last_backends 等).
* **session_paths.py** `_RUNS_ACTIONS_FALLBACK` 删除 3 个 legacy 名.
* **examples/p2_full_optimize_demo.py** 改注册 explore 取代
  backends + params.
* **tests** 调整: `test_p1_2_full_action_catalogue.py` 将 3 个名移到
  `_REMOVED_LEGACY_ACTIONS` (反向锁); `test_p1_1_action_registry.py`
  pin explore 取代 backends; `test_coordinator_audit_wiring.py` /
  `test_coordinator_baseline_fingerprint.py` pivot 到 explore;
  `test_p5_decision.py` 删除 A/B 段 (v0.6 阈值测试不再适用);
  `test_shared_state_action_attempts.py` parametrise 改 v0.8 audit set.

## 7. 风险 / 回退

- **回退**: revert 本 commit. v0.6 dataclass 字段
  (``backends_search`` / ``params_search`` / ``backends_attempts``
  等) 仍保留在 SharedState 上, 所以 v0.6 state.json 仍能 load —
  只是 fresh session 不会再有写入这些字段的代码路径.

## 8. 关联

- `Gap-10` (legacy actions still allowed) — 入口已关 (Gap-10 commit)
- `Dead-C` (validate_stack 死路径) — 同链, 已在本次清理
- `Dead-G` (测试反向锁死 deprecation) — 同链, 6 个 legacy 测试文件
  已删除
