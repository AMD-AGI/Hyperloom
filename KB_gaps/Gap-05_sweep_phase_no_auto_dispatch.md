# Gap-05 — SWEEP phase 入口不自动 sweep + 不读 `recipe.sweep_grid`

> 严重度: **P1 主要** (degraded behavior)
> 主轴影响: **主轴 A (流程固化)** + **主轴 B (Cortex 跨 session 知识)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-5

## 1. 问题描述

KB_design §3.2 §5.4 明确: 进入 SWEEP phase 应当 **自动构造 sweep grid**
(来自 SKILL.md 默认 grid + Cortex `recipe.sweep_grid` 字段, 后者优先),
**自动 enqueue `sweep` action**.

实际:
- phase 切换 EXPLORE/KERNEL → SWEEP **自动** (compute_next_phase ✅), 但
  *不 enqueue sweep task*. LLM 必须主动 `propose_action='sweep'`.
- sweep executor `_build_grid` (sweep.py:59-94) **只读 task params**
  (CONC / ISL / OSL); 完全不读 `state.warm_start_recipe`.

后果:
- SWEEP phase 退化为"LLM 自觉发 sweep". 若 max_minutes 用尽前 LLM 没
  propose, `_enter_closing_phase` 强制 enqueue report 跳过 sweep.
- 跨 session 学到的 sweep grid (主轴 B 的关键收益之一) 永远不被复用.
  warm_start_recipe 字段在 §3.6 / M1 落地, 但 SWEEP 阶段读不到.

## 2. 现状代码 trace

### 2.1 phase 切换无 enqueue

(同 Gap-04 §2.1) `_advance_phase_if_needed` 切到 PHASE_SWEEP 后不 enqueue
任何任务.

### 2.2 sweep executor 只读 task params

`action_executors/sweep.py:59-94`:

```text
def _build_grid(params: dict) -> list[dict]:
    """Build sweep grid from task params only."""
    concs = params.get("conc_values") or DEFAULT_CONC_VALUES
    isl_osls = params.get("isl_osl_configs") or DEFAULT_ISL_OSL_CONFIGS
    ...
    return [{"conc": c, "isl": i, "osl": o} for c in concs for (i, o) in isl_osls]
```

无 `state.warm_start_recipe` 读取; 无 `recipe.sweep_grid` 覆盖路径.

### 2.3 phase_state.py SWEEP allowlist

`phase_state.py:109-111` 允许 `{sweep, recover}`. ✅ 但 LLM 不发就没人发.

## 3. 设计意图

§3.2 §5.4 SWEEP 阶段节奏:

```
1. 自动构造 sweep grid (来自 SKILL.md 默认 grid + Cortex
   `recipe.sweep_grid` 字段, 后者优先).
2. `sweep` action 串行跑每个组合, 每个组合都是 1 次 E2E bench.
3. 失败 ≤ ε 个组合时仍标 SUCCESS; > ε 标 PARTIAL.
```

设计目的:
- 跨 session 复用 sweep grid: 上次发现 (CONC=128, ISL=8192, OSL=512)
  是 sweet spot, 下次直接复用, 不再瞎试.
- 操作员看到"自动 sweep 已跑完"作为 v0.8 端到端正确性指标.

## 4. 根本原因

同 Gap-04: phase 切换函数没挂副作用 hook. M2 / M7 / M3 各自 PR 都没人
负责 "SWEEP 入口自动化". M1 / §3.6 落了 warm_start_recipe 字段但没人
接到 sweep executor.

附加成因:
- sweep executor 的接口设计偏 "外部 params 驱动" (从 LLM propose 派
  发的 params 取). 改为"从 state 读 fallback"需要轻微 API 变化.

## 5. 修复路径

### PR 5.1 — phase-entry hook (复用 Gap-04 框架)

Gap-04 PR 5.1 引入 `_on_phase_entered` 框架; 这里加 `_on_enter_sweep`.

### PR 5.2 — `_on_enter_sweep` 自动 enqueue

```text
async def _on_enter_sweep(self, from_phase: str) -> None:
    state = self.shared_state
    # Read sweep grid: recipe.sweep_grid > defaults
    grid_params = self._build_sweep_params_from_recipe(state)
    task = Task(
        task_id=f"internal-sweep-{uuid.uuid4().hex[:8]}",
        kind="sweep",
        state="queued",
        params=grid_params,
        idempotency_key="internal-sweep-phase-entry",
        from_agent="coordinator",
        priority=10,
    )
    await self.task_registry.enqueue(task)
    log.info("SWEEP entered → auto-enqueued sweep task %s with %d combos",
             task.task_id, len(grid_params.get("isl_osl_configs", [])))
```

### PR 5.3 — `_build_sweep_params_from_recipe` helper

```text
def _build_sweep_params_from_recipe(self, state) -> dict:
    recipe = state.warm_start_recipe or {}
    sweep_grid = recipe.get("sweep_grid") if isinstance(recipe, dict) else None

    if sweep_grid:
        # Cortex recipe overrides defaults
        return {
            "conc_values": sweep_grid.get("conc_values") or DEFAULT_CONC_VALUES,
            "isl_osl_configs": sweep_grid.get("isl_osl_configs") or DEFAULT_ISL_OSL_CONFIGS,
            "max_combos": sweep_grid.get("max_combos") or DEFAULT_MAX_COMBOS,
            "source": "cortex_recipe",
        }

    # Fallback: SKILL.md default grid
    return {
        "conc_values": DEFAULT_CONC_VALUES,
        "isl_osl_configs": DEFAULT_ISL_OSL_CONFIGS,
        "max_combos": DEFAULT_MAX_COMBOS,
        "source": "skill_md_default",
    }
```

### PR 5.4 — sweep executor 兼容旧 path

sweep.py `_build_grid` 不需要改 — 接口已经接收 params dict. 唯一改动
是 *who supplies the params* (LLM vs Coordinator). 现有 LLM propose 路
径继续工作 (Coordinator hook 之前 LLM 主动 propose 仍合法), 但优先级
低于 internal task.

### PR 5.5 — 测试

`tests/test_v08_sweep_phase_auto.py`:

- mock SWEEP entry (set phase to KERNEL with plateau_kernel signal)
- 断言 task_registry 含 1 个 kind='sweep' task
- 断言 task.params.source ∈ {'cortex_recipe', 'skill_md_default'}
- mock state.warm_start_recipe.sweep_grid = {...}; 验证用 cortex 路径
- mock state.warm_start_recipe = {} (Cortex 不可达时); 验证用 default

## 6. 验收口径

- [ ] fresh session SWEEP phase 入口 ≤ 1 tick 内 task_registry 有 1 个
      kind='sweep' task
- [ ] 该 task 完成后 breakdown.sweep.grid_source 字段标 `cortex_recipe`
      或 `skill_md_default`
- [ ] phase_history[entered=SWEEP] evidence 含 `auto_sweep_enqueued=true`
- [ ] 若 Cortex T0 拉到 sweep_grid recipe, sweep task params 来自 recipe
- [ ] LLM 仍可主动 propose sweep (back-compat)

## 7. 风险 / 回退

- **recipe.sweep_grid 格式错误**: 用 schema 验证 (`isl_osl_configs`
  必须是 list of tuples); 验证失败 fallback 到 default + warning.
- **--no-kernel mode SWEEP**: EXPLORE → SWEEP 直接转, 同样应当自动
  enqueue. PR 5.2 不区分 from_phase, 全部 KERNEL/EXPLORE 进入 SWEEP 都
  enqueue.
- **回退**: 删除 `_on_enter_sweep` 即退到当前 (LLM-driven sweep).

## 8. 关联 gap

- **同框架**: Gap-04 (KERNEL profile), Gap-06 (CLOSE 5 步顺序器) — 三
  个 phase-entry hook 共享 `_on_phase_entered` 调度
- **关联**: warm_start_recipe 字段已落 (§3.6 / M1), 但 sweep 是首个 *消费方*.
  若 Cortex T0 没拉到 recipe (e.g. fresh model 没历史), 走 default 路径,
  端到端仍 OK.
