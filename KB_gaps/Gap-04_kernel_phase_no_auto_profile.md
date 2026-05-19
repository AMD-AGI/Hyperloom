# Gap-04 — KERNEL phase 入口不自动 profile

> 严重度: **P1 主要** (degraded behavior)
> 主轴影响: **主轴 A (流程固化)** — phase 切换不带行为
> 体检报告: `../KB_design_gaps.MD` §5 Gap-4

## 1. 问题描述

KB_design §3.2 §5.3 step 1 明确: "进入 KERNEL 即跑一次 `profile` (**固
定动作, 不需要 LLM propose**). 该次 profile 写入 `last_profile_trace`,
锚定 KERNEL 阶段所有 `select_kernels` 的 trace_input."

实际: phase 切换由 `_advance_phase_if_needed` 自动 (EXPLORE → KERNEL),
但**只更新 phase 字段, 不 enqueue 任何 task**. KERNEL 入口时
`last_profile_trace == ""`, LLM 必须自觉 propose profile. 若 LLM 漏掉,
后续 `select_kernels` 没有 trace, `kernel_opt` pipeline 受阻.

## 2. 现状代码 trace

### 2.1 phase 切换不带 hook

`coordinator.py:697-772`:

```text
def _advance_phase_if_needed(self) -> None:
    """Compute the next phase and write phase_history. No side effects."""
    state = self.shared_state
    current = state.phase or PHASE_PRELUDE
    nxt = compute_next_phase(state, ...)
    if nxt == current:
        return
    self._append_phase_history(from_=current, to=nxt, reason=..., evidence=...)
    state.phase = nxt
    state.phase_started_ts = _now_iso()
    state.phase_started_unix = time.time()
    # NO enqueue task here
```

### 2.2 `_required_next_step` 仅 prompt-level nag

`coordinator.py:1631-1635`:

```text
if "kernel" in self.role_registry and not state.last_profile_trace:
    return "TODO: propose `profile` to obtain a trace before kernel_opt"
```

这是字符串拼接进 Orchestration prompt, 不强制 enqueue. LLM 看到 TODO
可以选择忽略.

### 2.3 sequence gate

`_sequence_denial_for_action` (~1757-1764) 检测 `kernel_opt` 等 sequence
action 时, 如 `last_profile_trace==""` 会 deny. 这是 *deny path*, 不是
auto-enqueue path.

## 3. 设计意图

§3.2 §5.3:
- step 1: "进入即跑一次 `profile`". 设计明确 *固定动作*, 不让 LLM 选择.
- "**该次 profile 写入 `last_profile_trace`, 锚定 KERNEL 阶段所有
  `select_kernels` 的 trace_input**." — 没 profile 就没有 trace 锚点.

设计目的:
- 防止 LLM 跳过 profile 导致 kernel_opt 没有 trace
- 防止重复 profile (KERNEL 阶段 1 次, EXPLORE 阶段不可发 profile)

## 4. 根本原因

§3.2 设计是"phase 转移由 Coordinator 持有", 但实施层面 phase 切换函
数 (`compute_next_phase` + `_advance_phase_if_needed`) 是从 plateau
判定衍生的, 没人在切换点挂 *副作用 hook*. M2 PR 关注 phase 字段 +
PolicyGate R1, M7 PR 关注 plateau 计算, 中间 "phase 切换的副作用"
没人接.

类似 gap (Gap-05 SWEEP, Gap-06 CLOSE) 出于相同原因, 三者宜一起修.

## 5. 修复路径

### PR 5.1 — 引入 phase-entry hook 框架

`coordinator.py` 加:

```text
async def _on_phase_entered(self, from_phase: str, to_phase: str) -> None:
    """Side-effect hook fired after _advance_phase_if_needed commits."""
    if to_phase == PHASE_EXPLORE:
        await self._on_enter_explore(from_phase)
    elif to_phase == PHASE_KERNEL:
        await self._on_enter_kernel(from_phase)
    elif to_phase == PHASE_SWEEP:
        await self._on_enter_sweep(from_phase)
    elif to_phase == PHASE_CLOSE:
        await self._on_enter_close(from_phase)
```

### PR 5.2 — KERNEL 入口自动 enqueue profile

```text
async def _on_enter_kernel(self, from_phase: str) -> None:
    state = self.shared_state
    if state.kernel_enabled is False:
        # Should not happen — compute_next_phase routes no-kernel runs to SWEEP
        log.warning("entered KERNEL with kernel_enabled=False")
        return
    if state.last_profile_trace:
        # Already have a trace (resume case); skip
        log.info("KERNEL entered with existing last_profile_trace; "
                 "skipping auto-profile")
        return
    # Build a synthetic task identical to what an LLM propose_action='profile'
    # would build; route through TaskRegistry so it gets standard audit.
    task = self._build_internal_profile_task(reason="kernel_phase_entry")
    await self.task_registry.enqueue(task)
    log.info("KERNEL entered → auto-enqueued profile task %s", task.task_id)
```

### PR 5.3 — `_build_internal_profile_task` helper

```text
def _build_internal_profile_task(self, *, reason: str) -> Task:
    return Task(
        task_id=f"internal-profile-{uuid.uuid4().hex[:8]}",
        kind="profile",
        state="queued",
        params={
            "source": "coordinator_internal",
            "reason": reason,
            "benchmark_script": self.shared_state.baseline_benchmark_script,
            ...
        },
        idempotency_key=f"internal-profile-{reason}",
        from_agent="coordinator",
        priority=10,  # high — KERNEL phase 依赖
    )
```

特性:
- 用专门的 idempotency_key, 防止 phase 反复进入 (实际 phase 单调 Inv-2.1)
- `from_agent="coordinator"` 表明非 LLM 发起, 跳过 sequence gate
- 路径走标准 dispatcher, breakdown 中显示为 profile task

### PR 5.4 — `_advance_phase_if_needed` 触发 hook

```text
async def _advance_phase_if_needed(self) -> None:
    ...
    if nxt == current:
        return
    self._append_phase_history(...)
    state.phase = nxt
    ...
    # v0.8 §3.2 — phase-entry side effects
    await self._on_phase_entered(from_phase=current, to_phase=nxt)
```

### PR 5.5 — 测试

`tests/test_v08_kernel_phase_auto_profile.py` (新增):

- mock EXPLORE → KERNEL 切换 (写 plateau_explore signal)
- 验证 task_registry 内 ≥ 1 个 `kind='profile'` task
- 验证该 profile task 的 idempotency_key 包含 'kernel_phase_entry'
- resume 场景: `last_profile_trace != ""` 时不重复 enqueue
- `--no-kernel` 场景: 跳过 KERNEL → SWEEP, 不 enqueue profile

## 6. 验收口径

- [ ] fresh session: 进入 KERNEL phase 后 ≤ 1 tick 内, task_registry 有
      1 个 kind='profile' task (status=running 或 succeeded)
- [ ] 该 profile task 完成后, `state.last_profile_trace != ""`
- [ ] phase_history[entered=KERNEL] 的 evidence 含 `auto_profile_enqueued=true`
- [ ] resume KERNEL phase session (已有 last_profile_trace) 不重复 enqueue
- [ ] `--no-kernel` mode 跳过 KERNEL, 不 enqueue profile

## 7. 风险 / 回退

- **重复 profile**: 如果 phase 状态机 bug 让 KERNEL 进入两次, idempotency_key
  防重复 (TaskRegistry 已有去重).
- **profile 失败**: 现有 `_required_next_step` TODO 仍兜底; KERNEL phase
  内 LLM 提议 kernel_opt 时被 sequence gate 拒, 提示先 profile.
- **回退**: 删除 `_on_enter_kernel` 内的 auto-enqueue 即退到当前行为.
  整个 phase-entry hook 框架不影响其他 phase (Gap-05/06 也用此框架).

## 8. 关联 gap

- **同框架**: Gap-05 (SWEEP), Gap-06 (CLOSE) — 3 个 phase-entry hook
  同期上线
- **独立**: Gap-04 不依赖 specialist 链 (Gap-01/02/03)
