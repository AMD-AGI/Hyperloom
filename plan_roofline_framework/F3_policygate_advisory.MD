# F3 — PolicyGate 关键规则 + Soft Advisory

> 工作日: 1-2 天
> 风险: 低-中 (主要是 PolicyGate 规则,可独立 toggle)
> 依赖: F2 完成 (tag `f2-done` 在位)
> 完成节点: tag `f3-done` + ~5 个 commit
>
> **完成 F3 之后:**
> - 引入三条 PolicyGate 规则 (N9, N19c, N31),全部 toggle 控制
> - 新增 saturated_layers soft advisory:在 prompt 注入 hint,**不**硬拒 specialist dispatch
> - 替代 v0.6 backends_attempts / params_attempts sequence_denial 规则的 v0.8 等价物
> - 绝**不**引入 'roofline 屏蔽到顶方向 dispatch 的硬规则' (用户问询时已论证: 风险大于收益)

## 0. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export MAIN_ROOT="/wekafs/zgong/Hyperloom"

cd "$FEAT_ROOT"
git log --oneline f2-done -1
git status --short
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3   # >= 1626
bash docs/integration/check_preserved.sh
```

## 1. 设计回顾 — 用户问询的关键判断

> **用户原始问题**: 是否让 Orchestration 基于 roofline 饱和度屏蔽对应方向的 specialist dispatch?
>
> **结论**: **不引入硬规则**;以 soft advisory + retry budget 实现等效效果。
>
> 理由 (在 docs/notes_lossen_explore/新增PR报告roofline_framework.MD §6 详述):
> - 硬屏蔽会被 noisy roofline (单次 trace) 误触发,导致整个方向永久关闭
> - LLM 已经能从 analysis.md 看到饱和度,自主放弃比强制更稳
> - retry budget (每 specialist domain 5 次空 proposal 上限) 已经覆盖了"无效尝试"的常见情形
> - 真要硬屏蔽,加 escape hatch 的成本 > soft advisory 本身

## 2. 步骤清单

```
F3-1   N9 PolicyGate: deny direct profile when composite is on  -> 1 commit
F3-2   N19c PolicyGate: gain-driven kernel_opt lock              -> 1 commit
F3-3   N31 Coordinator: auto-enqueue final roofline in CLOSE     -> 1 commit
F3-4   Roofline saturation soft advisory (prompt injection)      -> 1 commit
F3-5   Replace v0.6 backends/params sequence_denial -> v0.8 form -> 1 commit
F3-6   smoke session + 打 tag f3-done                            -> 0/1 commit
```

---

## 3. F3-1: N9 — Deny direct profile when composite is on

### 设计

当 `--use-roofline-composite=True` 时,**禁止** `propose_action{action_name='profile'}`
(以及 `pmc_roofline`)。强制 LLM 通过 `roofline` 复合 action 拿数据。

### 编辑 `inference_optimizer/orchestrator/policy.py`

```python
@policy_rule
def n9_deny_direct_profile_when_composite_on(intent, state):
    """N9 (F3-1): Deny propose_action{profile|pmc_roofline} when
    --use-roofline-composite is on. Forces atomic roofline use to
    keep the snapshot_id counter monotonic + analysis.md cache
    aligned with the trace.

    Toggle: shared_state.deny_direct_profile (off by default in F3-1
    to allow operator A/B test; flipped on by F3-6 if smoke is green)."""
    if intent.get("kind") != "propose_action":
        return ALLOW
    action = intent.get("action_name")
    if action not in ("profile", "pmc_roofline"):
        return ALLOW
    if not state.use_roofline_composite:
        return ALLOW
    if not state.deny_direct_profile:
        return ALLOW   # F3-1 default — operator hasn't flipped yet
    return DENY(
        rule="n9_deny_direct_profile_when_composite_on",
        message=(
            f"action={action} is denied while --use-roofline-composite is on; "
            f"propose action_name='roofline' instead to get atomic "
            f"profile + trace_analyze + analysis.md snapshot."
        ),
        recovery_hint="propose_action{action_name='roofline'}",
    )
```

### 测试

`inference_optimizer/tests/test_f3_policy_rules.py` (新建):

```python
import pytest


@pytest.fixture
def state_composite_on_n9_on():
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.use_roofline_composite = True
    s.deny_direct_profile = True
    return s


def test_n9_denies_profile_when_composite_and_n9_on(state_composite_on_n9_on):
    from inference_optimizer.orchestrator.policy import (
        n9_deny_direct_profile_when_composite_on, DENY, ALLOW,
    )
    intent = {"kind": "propose_action", "action_name": "profile"}
    decision = n9_deny_direct_profile_when_composite_on(intent, state_composite_on_n9_on)
    assert decision.is_deny()
    assert "roofline" in decision.recovery_hint


def test_n9_denies_pmc_roofline_when_composite_and_n9_on(state_composite_on_n9_on):
    from inference_optimizer.orchestrator.policy import (
        n9_deny_direct_profile_when_composite_on,
    )
    intent = {"kind": "propose_action", "action_name": "pmc_roofline"}
    decision = n9_deny_direct_profile_when_composite_on(intent, state_composite_on_n9_on)
    assert decision.is_deny()


def test_n9_allows_roofline_when_composite_and_n9_on(state_composite_on_n9_on):
    from inference_optimizer.orchestrator.policy import (
        n9_deny_direct_profile_when_composite_on,
    )
    intent = {"kind": "propose_action", "action_name": "roofline"}
    decision = n9_deny_direct_profile_when_composite_on(intent, state_composite_on_n9_on)
    assert decision.is_allow()


def test_n9_allows_profile_when_composite_off():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import (
        n9_deny_direct_profile_when_composite_on,
    )
    s = SharedState()
    s.use_roofline_composite = False
    s.deny_direct_profile = True
    intent = {"kind": "propose_action", "action_name": "profile"}
    decision = n9_deny_direct_profile_when_composite_on(intent, s)
    assert decision.is_allow()


def test_n9_allows_profile_when_n9_toggle_off():
    """Even if composite is on, until operator flips deny_direct_profile,
    profile/pmc_roofline remain allowed (escape hatch for debugging)."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import (
        n9_deny_direct_profile_when_composite_on,
    )
    s = SharedState()
    s.use_roofline_composite = True
    s.deny_direct_profile = False
    intent = {"kind": "propose_action", "action_name": "profile"}
    decision = n9_deny_direct_profile_when_composite_on(intent, s)
    assert decision.is_allow()
```

### 命令

```bash
pytest inference_optimizer/tests/test_f3_policy_rules.py -v
# 期望 5 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1631 passed (1626 + 5)

git add inference_optimizer/orchestrator/policy.py
git add inference_optimizer/tests/test_f3_policy_rules.py
git commit -m "feat(F3-1): PolicyGate rule n9_deny_direct_profile_when_composite_on

Forces LLM to use atomic 'roofline' action instead of 'profile' /
'pmc_roofline' when --use-roofline-composite is on. Keeps
analysis.md cache and snapshot_id counter aligned with the trace
(see roofline.py docstring §5/§8.5 stale-cache invariant).

Two-toggle gating:
- --use-roofline-composite (F1) must be on
- --deny-direct-profile (F3) must be on (default off — operator can
  A/B test before flipping)

Tests: 5 new. Total: 1631 passed."
```


---

## 4. F3-2: N19c — Gain-driven kernel_opt lock

### 设计

`kernel_opt` 是 GPU 重活 (~10-30 min/run). **只有当**便宜的 EXPLORE/specialist 提案
不再产生增益时,才允许进入 kernel_opt。具体形式: `last_cheap_delta_gain_pct < epsilon`
连续 N 次 → 解锁。

> 对 main 上的对应规则(N19c gain-driven)做了两点本 branch 适配:
> 1. main 用 `validate_stack_succeeded` 作前提;v0.8 没这个,改为 `phase==KERNEL` 作前提
> 2. main 用 `last_cheap_explore_gain_pct`;v0.8 现有 `last_explore_delta_gain_pct` (PR-A4)

### 编辑 `inference_optimizer/orchestrator/policy.py`

```python
@policy_rule
def n19c_gain_driven_kernel_opt(intent, state):
    """N19c (F3-2): kernel_opt locked until cheap exploration converges.

    'Cheap' = explore + specialist + integrate_patch (no kernel agent
    invocation). When the running average of last 3 cheap rounds yields
    < epsilon (default 0.5%), unlock kernel_opt. Otherwise deny.

    Toggle: shared_state.gain_driven_kernel_opt (F0-10 added; default off).

    Adapted from main's N19c rule:
    - Uses phase==KERNEL as precondition (v0.6's validate_stack_succeeded
      is gone in v0.8 — see docs/integration/MAIN_FEATURES_DROPPED.md §1.3)
    - Reads last_explore_delta_gain_pct (PR-A4) instead of v0.6 backends
      gain field"""
    if intent.get("kind") != "propose_action":
        return ALLOW
    if intent.get("action_name") != "kernel_opt":
        return ALLOW
    if not state.gain_driven_kernel_opt:
        return ALLOW

    # Hard precondition: must be in KERNEL phase
    if state.phase != "KERNEL":
        # Phase allowlist already handles this; defer to that rule
        return ALLOW

    # Read recent cheap deltas
    history = list(getattr(state, "explore_delta_gain_history", []))[-3:]
    if len(history) < 3:
        return DENY(
            rule="n19c_gain_driven_kernel_opt",
            message=(
                f"kernel_opt is locked: only {len(history)} cheap rounds "
                f"so far (need 3 for gain-trend window). Run more "
                f"explore / specialist rounds first."
            ),
            recovery_hint="propose_action{action_name='explore'} or dispatch_specialist{...}",
        )
    avg = sum(history) / 3.0
    epsilon = 0.5   # tunable
    if avg >= epsilon:
        return DENY(
            rule="n19c_gain_driven_kernel_opt",
            message=(
                f"kernel_opt is locked: cheap exploration is still "
                f"earning gain (last 3 rounds avg = {avg:.2f}% >= {epsilon}%). "
                f"Continue cheap rounds until avg drops below threshold."
            ),
            recovery_hint="propose_action{action_name='explore'} or dispatch_specialist{...}",
        )
    return ALLOW
```

### 测试

继续追加到 `test_f3_policy_rules.py`:

```python
def test_n19c_off_by_default_allows():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import n19c_gain_driven_kernel_opt
    s = SharedState()
    s.gain_driven_kernel_opt = False
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    assert n19c_gain_driven_kernel_opt(intent, s).is_allow()


def test_n19c_denies_when_window_short():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import n19c_gain_driven_kernel_opt
    s = SharedState()
    s.gain_driven_kernel_opt = True
    s.phase = "KERNEL"
    s.explore_delta_gain_history = [3.0, 1.0]   # only 2
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    decision = n19c_gain_driven_kernel_opt(intent, s)
    assert decision.is_deny()
    assert "cheap rounds" in decision.message


def test_n19c_denies_when_average_above_epsilon():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import n19c_gain_driven_kernel_opt
    s = SharedState()
    s.gain_driven_kernel_opt = True
    s.phase = "KERNEL"
    s.explore_delta_gain_history = [2.0, 1.5, 1.2]   # avg 1.57 > 0.5
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    decision = n19c_gain_driven_kernel_opt(intent, s)
    assert decision.is_deny()
    assert "still earning gain" in decision.message


def test_n19c_allows_when_average_below_epsilon():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import n19c_gain_driven_kernel_opt
    s = SharedState()
    s.gain_driven_kernel_opt = True
    s.phase = "KERNEL"
    s.explore_delta_gain_history = [0.4, 0.3, 0.2]   # avg 0.3 < 0.5
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    assert n19c_gain_driven_kernel_opt(intent, s).is_allow()


def test_n19c_allows_when_phase_not_kernel():
    """Phase allowlist handles non-KERNEL kernel_opt; n19c defers."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import n19c_gain_driven_kernel_opt
    s = SharedState()
    s.gain_driven_kernel_opt = True
    s.phase = "EXPLORE"
    s.explore_delta_gain_history = [3, 3, 3]
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    assert n19c_gain_driven_kernel_opt(intent, s).is_allow()
```

### 命令

```bash
pytest inference_optimizer/tests/test_f3_policy_rules.py -v -k "n19c"
# 期望 5 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1636 passed (1631 + 5)

git add inference_optimizer/orchestrator/policy.py
git add inference_optimizer/tests/test_f3_policy_rules.py
git commit -m "feat(F3-2): PolicyGate rule n19c_gain_driven_kernel_opt

Locks kernel_opt action until cheap exploration converges (3-round
moving average of explore_delta_gain_pct < 0.5%).

Adapted from main's N19c rule with two v0.8 changes:
- precondition: phase==KERNEL (instead of validate_stack_succeeded
  which is v0.6 only)
- gain field: last_explore_delta_gain_pct (PR-A4) instead of v0.6
  backends gain field

Toggle: --gain-driven-kernel-opt (F0-10, default off).

Tests: 5 new. Total: 1636 passed."
```


---

## 5. F3-3: N31 — Auto-enqueue final roofline in CLOSE

### 设计

进入 CLOSE phase 时,Coordinator 自动 enqueue 一个 `roofline` task,得到 final 快照对比
PRELUDE / 早期快照,放进 final report。LLM 不需要主动 propose。

### 编辑 `inference_optimizer/orchestrator/coordinator.py`

找到 phase 跳转处 (一般是 `_advance_phase` 或类似):

```python
async def _advance_phase(self, target_phase: str) -> None:
    prev_phase = self.shared_state.phase
    self.shared_state.phase = target_phase
    # ... existing transitions logging ...

    # F3-3 (N31): on CLOSE entry, auto-enqueue a final roofline if the
    # composite is on AND the last snapshot is stale (>5 min old or
    # cumulative_gain_validated has moved >= 3% since snapshot).
    if (
        target_phase == "CLOSE"
        and self.shared_state.use_roofline_composite
        and prev_phase != "CLOSE"   # idempotent — only on first entry
    ):
        await self._maybe_enqueue_final_roofline()


async def _maybe_enqueue_final_roofline(self) -> None:
    last_ta = self.shared_state.last_trace_analyze or {}
    if not isinstance(last_ta, dict):
        return
    snapshot_ts = float(last_ta.get("ts_unix", 0.0))
    age_s = time.time() - snapshot_ts if snapshot_ts else 1e9
    snapshot_gain = float(last_ta.get("roofline_baseline_gain_at_snapshot", 0.0))
    current_gain = float(self.shared_state.cumulative_gain_validated)
    delta = abs(current_gain - snapshot_gain)
    stale_due_to_age = age_s > 300.0       # 5 min
    stale_due_to_gain = delta > 3.0        # 3 %
    if not (stale_due_to_age or stale_due_to_gain or snapshot_ts == 0.0):
        return   # snapshot is fresh enough — reuse for final report

    self._enqueue_internal_task(
        kind="roofline",
        params={"_provenance": "n31_final_roofline_close_entry"},
        priority=0,        # next tick
    )
    log.info(
        "n31_final_roofline enqueued: age=%.1fs delta=%.2f%% (gain %.2f -> %.2f)",
        age_s, delta, snapshot_gain, current_gain,
    )
```

### 测试

`inference_optimizer/tests/test_f3_n31_final_roofline.py` (新建):

```python
import pytest
import time


@pytest.mark.asyncio
async def test_n31_enqueues_when_no_snapshot():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.use_roofline_composite = True
    s.phase = "KERNEL"
    s.last_trace_analyze = {}   # never ran roofline
    coord = Coordinator(s)   # NOTE: adjust ctor to your branch's signature
    enqueued = []
    coord._enqueue_internal_task = lambda **kw: enqueued.append(kw)
    await coord._advance_phase("CLOSE")
    assert len(enqueued) == 1
    assert enqueued[0]["kind"] == "roofline"
    assert enqueued[0]["params"]["_provenance"] == "n31_final_roofline_close_entry"


@pytest.mark.asyncio
async def test_n31_no_enqueue_when_snapshot_fresh():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.use_roofline_composite = True
    s.phase = "KERNEL"
    s.cumulative_gain_validated = 12.0
    s.last_trace_analyze = {
        "ts_unix": time.time() - 60,    # 1 min ago — fresh
        "roofline_baseline_gain_at_snapshot": 11.5,   # delta < 3%
    }
    coord = Coordinator(s)
    enqueued = []
    coord._enqueue_internal_task = lambda **kw: enqueued.append(kw)
    await coord._advance_phase("CLOSE")
    assert len(enqueued) == 0


@pytest.mark.asyncio
async def test_n31_no_enqueue_when_composite_off():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.use_roofline_composite = False
    s.phase = "KERNEL"
    coord = Coordinator(s)
    enqueued = []
    coord._enqueue_internal_task = lambda **kw: enqueued.append(kw)
    await coord._advance_phase("CLOSE")
    assert len(enqueued) == 0


@pytest.mark.asyncio
async def test_n31_enqueues_when_gain_drift_large():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.use_roofline_composite = True
    s.phase = "KERNEL"
    s.cumulative_gain_validated = 18.0
    s.last_trace_analyze = {
        "ts_unix": time.time() - 60,    # fresh in time
        "roofline_baseline_gain_at_snapshot": 11.0,   # delta = 7% > 3%
    }
    coord = Coordinator(s)
    enqueued = []
    coord._enqueue_internal_task = lambda **kw: enqueued.append(kw)
    await coord._advance_phase("CLOSE")
    assert len(enqueued) == 1
```

### 命令

```bash
pytest inference_optimizer/tests/test_f3_n31_final_roofline.py -v
# 期望 4 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1640 passed (1636 + 4)

git add inference_optimizer/orchestrator/coordinator.py
git add inference_optimizer/tests/test_f3_n31_final_roofline.py
git commit -m "feat(F3-3): N31 auto-enqueue final roofline on CLOSE entry

When phase transitions PRELUDE/EXPLORE/KERNEL/SWEEP -> CLOSE and
--use-roofline-composite is on, Coordinator auto-enqueues a roofline
task IF either:
- no snapshot exists, OR
- last snapshot is older than 5 minutes, OR
- cumulative_gain_validated has drifted >= 3% since snapshot.

Final report's analysis.md cell will reference this fresh snapshot.

Tests: 4 new. Total: 1640 passed."
```


---

## 6. F3-4: Roofline saturation soft advisory

### 设计

> **重申:绝不引入"硬屏蔽某方向 dispatch"的规则。**
>
> 用 prompt 注入的形式给 LLM 提供"此方向已饱和"的 hint,
> LLM 仍可自主选择 dispatch — 这正是用户问询时讨论的 soft advisory 方案。

### 数据流

1. RooflineExecutor (F1) 写完 `last_trace_analyze` 后,F3-4 的新代码路径
   计算每方向饱和度并塞进 `state.roofline_saturation_history` (F0-8 已加字段)
2. prompt_builder 在渲染下一 tick prompt 时,把饱和度 ≥ 80% 的方向作为 hint 注入

### 编辑 `inference_optimizer/orchestrator/roofline_snapshot.py`

(F0-7 已 cp 来了,这一步只做扩展)

```python
def derive_saturation_per_direction(analysis_md_text: str) -> dict[str, float]:
    """Parse analysis.md and return {direction: saturation_pct}.

    Direction → bottleneck mapping:
        compute        ↔ # Compute Kernel Optimizations section
        memory         ↔ # Memory-Bound Kernels section
        host_overhead  ↔ # Host overhead / Idle %
        comm           ↔ # AllReduce / RCCL section
    Saturation (%) is read from the Executive Summary table.
    Missing sections → 0.0.
    """
    # Implementation: regex over analysis.md headers + table cells.
    # See `_analysis_keyword_map.py` (F0-7 cp) for the keyword dictionary.
    # ... (impl ~30 lines) ...
```

### 编辑 RooflineExecutor 以记录 history

在 `RooflineExecutor.__call__` 的 step 5 之后(成功 record_trace_analyze 之后),加:

```python
# F3-4: append saturation history
if getattr(self.shared_state, "roofline_saturation_advisory", False):
    sat = derive_saturation_per_direction(
        self.shared_state.last_trace_analyze.get("analysis_md_text", "")
    )
    snap_id = self.shared_state.last_trace_analyze.get("roofline_snapshot_id", 0)
    record = {"snapshot_id": snap_id, **sat}
    self.shared_state.roofline_saturation_history.append(record)
    # cap history at 10
    if len(self.shared_state.roofline_saturation_history) > 10:
        self.shared_state.roofline_saturation_history.pop(0)
```

### 编辑 prompt_builder

新增 `_format_roofline_saturation_advisory(state)`:

```python
def _format_roofline_saturation_advisory(state) -> str:
    """F3-4 — render saturated-direction hints into orchestration prompt.

    Soft advisory: lists directions with saturation >= 80% as 'hint:
    likely diminishing returns'. Does NOT recommend dropping the
    direction — only flags the cost vs reward trade-off."""
    if not getattr(state, "roofline_saturation_advisory", False):
        return ""
    history = list(getattr(state, "roofline_saturation_history", []))
    if not history:
        return ""
    last = history[-1]
    sat_lines = []
    for direction in ("compute", "memory", "host_overhead", "comm"):
        pct = float(last.get(direction, 0.0))
        if pct >= 80.0:
            sat_lines.append(f"  - {direction}: {pct:.1f}% saturated")
    if not sat_lines:
        return ""
    return (
        "=== Roofline Saturation Advisory (snapshot #"
        f"{last.get('snapshot_id', '?')}) ===\n"
        "The following directions are saturated; further work in these\n"
        "areas typically yields diminishing returns. You MAY still\n"
        "dispatch into them if you have a specific hypothesis (e.g. a\n"
        "PR #N or a known refactor); otherwise prefer cheaper directions:\n"
        + "\n".join(sat_lines) + "\n"
        "=== End Saturation Advisory ==="
    )
```

调用点:

```python
# 在 build_orchestration_prompt 里:
sections.append(_format_analysis_md_full(state))   # F1-4
sections.append(_format_roofline_saturation_advisory(state))   # F3-4
```

### 测试

继续追加到 `test_f3_policy_rules.py`:

```python
def test_saturation_advisory_off_by_default_returns_empty():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _format_roofline_saturation_advisory,
    )
    s = SharedState()
    s.roofline_saturation_advisory = False
    assert _format_roofline_saturation_advisory(s) == ""


def test_saturation_advisory_renders_when_above_threshold():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _format_roofline_saturation_advisory,
    )
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 3, "compute": 92.0, "memory": 35.0,
         "host_overhead": 12.0, "comm": 5.0},
    ]
    out = _format_roofline_saturation_advisory(s)
    assert "compute: 92.0% saturated" in out
    assert "memory" not in out
    assert "host_overhead" not in out
    assert out.startswith("=== Roofline Saturation Advisory")
    assert out.rstrip().endswith("=== End Saturation Advisory ===")


def test_saturation_advisory_empty_when_no_direction_saturated():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _format_roofline_saturation_advisory,
    )
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 1, "compute": 50.0, "memory": 60.0,
         "host_overhead": 30.0, "comm": 10.0},
    ]
    assert _format_roofline_saturation_advisory(s) == ""
```

### 命令

```bash
pytest inference_optimizer/tests/test_f3_policy_rules.py -v -k "saturation"
# 期望 3 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1643 passed (1640 + 3)

git add inference_optimizer/orchestrator/roofline_snapshot.py
git add inference_optimizer/orchestrator/action_executors/roofline.py
git add inference_optimizer/orchestrator/system_prompts/prompt_builder.py
git add inference_optimizer/tests/test_f3_policy_rules.py
git commit -m "feat(F3-4): roofline saturation soft advisory

After every successful roofline run, derive saturation per direction
(compute / memory / host_overhead / comm) from analysis.md and append
to shared_state.roofline_saturation_history (capped at 10).

Orchestration prompt gets a new 'Roofline Saturation Advisory' section
listing directions with >= 80% saturation, flagging the cost-vs-reward
trade-off WITHOUT hard-rejecting any dispatch.

Critically: this does NOT add a PolicyGate rule that denies
dispatch_specialist toward saturated directions. (See decision in
docs/notes_lossen_explore/新增PR报告roofline_framework.MD §6 — soft
advisory beats hard gating because (a) noisy single-trace saturation
shouldn't permanently close a direction; (b) LLM can already see
the analysis.md and self-select; (c) retry budget already covers
runaway empty proposals.)

Toggle: --roofline-saturation-advisory (F0-10, default off).

Tests: 3 new. Total: 1643 passed."
```


---

## 7. F3-5: 替换 v0.6 backends/params sequence_denial

### 背景

main 上 PolicyGate 有两条 v0.6 残留规则:
- `backends_attempts < 1` → deny kernel_opt
- `params_attempts < 1` → deny kernel_opt

v0.8 没有 backends/params 字段写者 → 这两条 denial 永真 → kernel_opt 永久锁死。

F0-6 (MAIN_FEATURES_DROPPED.md) 决定**主动放弃** main 上这两条规则。F3-5 引入 v0.8 等价版:

### 新规则: `explore_attempts_minimum_before_kernel_opt`

```python
@policy_rule
def explore_attempts_minimum_before_kernel_opt(intent, state):
    """F3-5 (replaces v0.6 backends_attempts/params_attempts sequence_denial):
    Require at least one successful explore round before kernel_opt.

    v0.8 explore action subsumes v0.6 backends + params + validate_stack.
    'Successful' means produced at least one variant proposal that was
    integrated (i.e., explore_attempts in SharedState >= 1).

    Toggle: always on once F3-5 is committed (no escape hatch — this is
    a baseline correctness rule)."""
    if intent.get("kind") != "propose_action":
        return ALLOW
    if intent.get("action_name") != "kernel_opt":
        return ALLOW
    n = int(getattr(state, "explore_attempts_succeeded", 0) or 0)
    if n >= 1:
        return ALLOW
    return DENY(
        rule="explore_attempts_minimum_before_kernel_opt",
        message=(
            f"kernel_opt requires at least 1 successful explore round "
            f"first (current explore_attempts_succeeded = {n}). "
            f"Run explore + integrate_patch first."
        ),
        recovery_hint="propose_action{action_name='explore'}",
    )
```

### 同时清理 (如果存在) 老的 backends/params denial

在 `policy.py` 找:

```bash
grep -n "backends_attempts\|params_attempts" inference_optimizer/orchestrator/policy.py
```

如果有,**删除**这些规则的整段定义(或者用 docstring 注释为 "removed in F3-5"):

```python
# REMOVED in F3-5 — superseded by explore_attempts_minimum_before_kernel_opt
# (this branch has no v0.6 backends/params writer, so the old rule was
# permanently true and locked kernel_opt forever)
```

### 检查 SharedState 上是否已有 `explore_attempts_succeeded`

```bash
grep -n "explore_attempts_succeeded\|explore_attempts" \
   inference_optimizer/orchestrator/shared_state.py | head
```

若不存在,加:

```python
# inference_optimizer/orchestrator/shared_state.py
explore_attempts_succeeded: int = 0   # incremented by Coordinator on integrate_patch ok
```

并在 Coordinator 里 `_promote_to_shared_state` 处加一行:

```python
# F3-5: increment explore counter on successful integrate_patch
if task_kind == "integrate_patch" and result.get("status") == "succeeded":
    self.shared_state.explore_attempts_succeeded += 1
```

### 测试

继续追加到 `test_f3_policy_rules.py`:

```python
def test_explore_minimum_denies_when_zero():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import (
        explore_attempts_minimum_before_kernel_opt,
    )
    s = SharedState()
    s.explore_attempts_succeeded = 0
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    decision = explore_attempts_minimum_before_kernel_opt(intent, s)
    assert decision.is_deny()


def test_explore_minimum_allows_when_at_least_one():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import (
        explore_attempts_minimum_before_kernel_opt,
    )
    s = SharedState()
    s.explore_attempts_succeeded = 1
    intent = {"kind": "propose_action", "action_name": "kernel_opt"}
    assert explore_attempts_minimum_before_kernel_opt(intent, s).is_allow()


def test_explore_minimum_does_not_deny_other_actions():
    """Rule only targets kernel_opt; other actions pass through."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.policy import (
        explore_attempts_minimum_before_kernel_opt,
    )
    s = SharedState()
    s.explore_attempts_succeeded = 0
    for action in ("explore", "specialist", "roofline", "integrate_patch"):
        intent = {"kind": "propose_action", "action_name": action}
        assert explore_attempts_minimum_before_kernel_opt(intent, s).is_allow(), \
            f"action={action} should not be denied by explore_minimum rule"


def test_no_legacy_backends_attempts_rule():
    """F3-5: ensure the v0.6 backends_attempts denial does not exist on
    this branch (would lock kernel_opt forever)."""
    from inference_optimizer.orchestrator import policy as policy_mod
    src = open(policy_mod.__file__).read()
    # active rule defs (not commented-out documentation):
    import re
    # Catches `def XXX_backends_attempts...` style or  `rule_id="...backends_attempts..."` strings
    matches = re.findall(r"^(?!\s*#).*backends_attempts", src, flags=re.MULTILINE)
    matches = [m for m in matches if "REMOVED" not in m and "F3-5" not in m]
    assert matches == [], (
        f"Legacy v0.6 'backends_attempts' rule still present:\n" +
        "\n".join(matches)
    )


def test_no_legacy_params_attempts_rule():
    from inference_optimizer.orchestrator import policy as policy_mod
    import re
    src = open(policy_mod.__file__).read()
    matches = re.findall(r"^(?!\s*#).*params_attempts", src, flags=re.MULTILINE)
    matches = [m for m in matches if "REMOVED" not in m and "F3-5" not in m]
    assert matches == [], f"Legacy v0.6 'params_attempts' rule still present"
```

### 命令

```bash
pytest inference_optimizer/tests/test_f3_policy_rules.py -v -k "explore_minimum or legacy"
# 期望 5 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1648 passed (1643 + 5)

# 关键 smoke: 跑一段 EXPLORE → KERNEL 流程,确认 kernel_opt 不会永久锁死
INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1 \
   python -m inference_optimizer.cli optimize \
   --model deepseek-ai/DeepSeek-R1-0528 --max-hours 0.5 --dry-run

git add inference_optimizer/orchestrator/policy.py
git add inference_optimizer/orchestrator/shared_state.py
git add inference_optimizer/orchestrator/coordinator.py
git add inference_optimizer/tests/test_f3_policy_rules.py
git commit -m "feat(F3-5): replace v0.6 backends/params sequence_denial with v0.8 explore equivalent

Removes (or asserts absence of) v0.6 sequence_denial rules
'backends_attempts < 1' / 'params_attempts < 1' which would lock
kernel_opt forever on this branch (no writers exist for those fields).

Replaces with explore_attempts_minimum_before_kernel_opt: at least
1 successful integrate_patch (== one full cheap explore round
landed) is required before propose_action{kernel_opt}.

Adds shared_state.explore_attempts_succeeded counter; Coordinator
bumps it on integrate_patch success.

Tests: 5 new (3 behavioral + 2 'no-legacy-rule' guards).
Total: 1648 passed."
```


---

## 8. F3-6: Smoke session + 打 tag f3-done

### 命令 (操作员人工)

```bash
cd "$FEAT_ROOT"

# 找空闲 GPU,跑一次"全开"smoke (3 个 toggle 全开):
INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1 \
INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=1 \
INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE=1 \
INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT=1 \
INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY=1 \
python -m inference_optimizer.cli optimize \
    --model Qwen/Qwen3-30B-A3B-Base \
    --workload "isl=2048,osl=512,bs=4" \
    --max-hours 4.0 \
    2>&1 | tee /tmp/f3_smoke_$(date +%Y%m%d_%H%M).log

# 关键 grep:
grep "n9_deny_direct_profile" /tmp/f3_smoke_*.log | head
grep "n19c_gain_driven_kernel_opt" /tmp/f3_smoke_*.log | head
grep "n31_final_roofline" /tmp/f3_smoke_*.log | head
grep "Roofline Saturation Advisory" /tmp/f3_smoke_*.log | head
grep "explore_attempts_minimum" /tmp/f3_smoke_*.log | head

# 应该:
# - n9 命中至少 1 次 (LLM 误提 profile,被改写为 roofline 的 recovery_hint)
# - n19c 在 KERNEL phase 命中 1+ 次 (kernel_opt 被锁直到 explore avg 收敛)
# - n31 在 phase 进入 CLOSE 时 enqueue final roofline
# - Saturation Advisory 在 prompt 里至少出现 1 次 (任意方向 >=80%)
# - explore_attempts_minimum 命中 0 次 (因为 EXPLORE 先于 KERNEL,正常流不触发)

# **不应该**有的 grep 命中:
grep "phase_incompatible.*roofline" /tmp/f3_smoke_*.log    # 应空 — F0-9 加了 allowlist
grep "no_executor.*roofline"          /tmp/f3_smoke_*.log    # 应空 — F1-3 注册了
grep "framework_pr.*top_level"        /tmp/f3_smoke_*.log    # 应空 — F2 没引入 top-level action

# 归档:
mkdir -p docs/integration
cp /tmp/f3_smoke_*.log docs/integration/

git add docs/integration/f3_smoke_*.log
git commit -m "chore(F3-6): archive F3 'all-toggles-on' smoke session log

4h Qwen3-30B smoke run on MI300X with all 5 F1/F2/F3 toggles enabled.

Outcome:
- $(grep -c n9_deny_direct_profile /tmp/f3_smoke_*.log) N9 denials of direct profile
- $(grep -c n19c_gain_driven_kernel_opt /tmp/f3_smoke_*.log) N19c kernel_opt locks
- $(grep -c n31_final_roofline /tmp/f3_smoke_*.log) N31 final-roofline enqueues
- $(grep -c 'Roofline Saturation Advisory' /tmp/f3_smoke_*.log) saturation advisory injections

Negative checks (must be 0):
- $(grep -c 'phase_incompatible.*roofline' /tmp/f3_smoke_*.log) phase_incompatible roofline
- $(grep -c 'no_executor.*roofline' /tmp/f3_smoke_*.log) no_executor roofline
- $(grep -c 'framework_pr.*top_level' /tmp/f3_smoke_*.log) framework_pr as top-level action"

# 打 tag
git tag -a f3-done -m "F3 (PolicyGate rules + soft advisory) complete

Commits since f2-done:
$(git log --oneline f2-done..HEAD)

State:
- 1648+ passing tests
- 3 new PolicyGate rules: n9, n19c, explore_attempts_minimum
- 1 Coordinator auto-enqueue rule: n31 final roofline
- 1 prompt injection: roofline saturation soft advisory
- 4h Qwen3-30B smoke session green with ALL toggles on
- v0.6 backends_attempts/params_attempts denials confirmed absent

Toggle state (defaults):
- --use-roofline-composite=False
- --framework-agent-enabled=False
- --deny-direct-profile=False (inert until --use-roofline-composite=True)
- --gain-driven-kernel-opt=False
- --roofline-saturation-advisory=False

To enable in production:
  export INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1
  # rest are recommended but optional toggles

Integration freeze (SKILL.md) can now be lifted; remove the
'## Integration Freeze' section in a follow-up commit if desired.

This concludes the roofline + framework-agent integration arc."

git push origin f3-done
git push origin "$FEAT_BRANCH"
```

### 验收

- [ ] `git log --oneline f2-done..f3-done | wc -l` ≈ 6
- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` >= 1648 passed
- [ ] `bash docs/integration/check_preserved.sh` rc=0
- [ ] 4h all-toggles-on smoke session 完成,无 phase 卡死
- [ ] log 中 n9 / n19c / n31 / saturation advisory 各命中 ≥ 0 次(命中数依模型而异,但**没有任何 stack trace**)
- [ ] log 中**未**出现 phase_incompatible.*roofline / no_executor.*roofline / framework_pr.*top_level

### 后续 (post-F3)

可选清理:
1. 删除 `inference_optimizer/SKILL.md` 的 `## Integration Freeze` 段(F0-11 加的)
2. `--use-roofline-composite` 翻默认为 True (前面 F1-6 跳过的)
3. `--deny-direct-profile` 翻默认为 True
4. 把 `docs/notes_lossen_explore/` 下的会议笔记打包归档
5. 关 `pre-roofline-merge` tag (留作历史)
6. (如果整个分支要合 main) 发 PR 引用 `docs/integration/MAIN_FEATURES_DROPPED.md`
   解释为什么不接受某些 main 改动

---

## 9. F3 整体回滚

### 回到 F2 状态

```bash
git reset --hard f2-done
```

### 单步回滚

```bash
# 例如 N31 在某 phase 跳转里出 race:
git revert <F3-3 commit sha>
# 退化路径:CLOSE 不自动跑 final roofline,但 LLM 仍然可以手动提
```

### 紧急 kill switch (不需要 commit)

```bash
# 任何一条规则都可单独关:
unset INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE       # N9 关
unset INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT    # N19c 关
unset INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY   # advisory 关

# 注:N31 final roofline 没有专门 toggle (它跟随 --use-roofline-composite)。
#     如要关:--use-roofline-composite=False
```

---

## 10. F3 完成后的 commit 拓扑

```
* (tag: f3-done)
* chore(F3-6): archive F3 all-toggles-on smoke session log
* feat(F3-5): replace v0.6 backends/params sequence_denial
* feat(F3-4): roofline saturation soft advisory
* feat(F3-3): N31 auto-enqueue final roofline on CLOSE entry
* feat(F3-2): PolicyGate rule n19c_gain_driven_kernel_opt
* feat(F3-1): PolicyGate rule n9_deny_direct_profile_when_composite_on
* (tag: f2-done)
```

测试数走势: `1626 → 1631 → 1636 → 1640 → 1643 → 1648 → 1648`.

---

## 11. F0 → F3 全程总览

| 阶段 | 工作日 | commit | 测试增量 | 风险 | 默认行为变化 |
|---|---|---|---|---|---|
| F0 | 1 | 11 | +10 (1584→1594) | 极低 | 无 |
| F1 | 2-3 | 7 | +24 (1594→1618) | 中 | 无 (toggle off) |
| F2 | 2-3 | 6 | +8 (1618→1626) | 中-高 | 无 (toggle off) |
| F3 | 1-2 | 6 | +22 (1626→1648) | 低-中 | 无 (toggle off) |
| **合计** | **6-9** | **30** | **+64** | — | **无** |

**关键不变量**:
- 任何 F-步骤 commit **完成后**,关全部 toggle → 行为等同 `pre-roofline-merge` tag
- 测试数单调不降,never红色
- v0.8 specialist-first 契约 (IR-4) 全程保持
- 没有 v0.6 backends/params/scoring 复活

**关键能力新增**(toggle on 后):
- ✓ PR #288 roofline composite action: profile + trace_analyze 原子化,LLM 看到完整 analysis.md
- ✓ PR #280 framework-agent: serving_specialist 能搜 + 拉 + propose 上游 PR (走 v0.8 contract)
- ✓ N9 / N19c / N31 PolicyGate 规则
- ✓ Roofline saturation soft advisory (用户问询的"屏蔽"问题的稳健形态)
- ✓ KB writeback to lessons.jsonl

完成。
