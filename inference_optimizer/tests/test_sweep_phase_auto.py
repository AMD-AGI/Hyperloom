# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.2 §5.4 / KB_gaps/Gap-05 — SWEEP phase auto-dispatch tests.

KB_gaps/Gap-05: SWEEP entry must auto-construct a grid and enqueue the
``sweep`` action (recipe grid wins over SKILL.md defaults). Covers the
grid-builder helper, the enqueue + ``_on_enter_sweep`` hook, the e2e
Coordinator path, and the PolicyGate ``sweep_phase_singleton`` rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend, MockTurn, ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


# Fixtures
@dataclass
class _BareState:
    """SharedState stand-in covering every attribute the SWEEP hook + helper read."""

    warm_start_recipe: dict | None = None
    baseline_config_path: str = ""
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


class _StubTaskRegistry:
    """create_or_return_existing double, keyed by idempotency_key."""

    def __init__(self):
        self._tasks: dict[str, Any] = {}

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list | None = None,
        allowed_tools: list | None = None,
        side_effects: list | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ):
        from inference_optimizer.orchestrator.task_registry import Task

        existing = self._tasks.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid

        task = Task(
            task_id=task_id or _uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[idempotency_key] = task
        return task, False


@pytest.fixture
def coord(tmp_path: Path):
    """Lean Coordinator stub for hook unit tests."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c.role_registry = {"kernel": object()}
    return c


@pytest.mark.asyncio
async def test_drain_pending_keep_integrates_records_result_once(
    tmp_path: Path, monkeypatch,
):
    """SWEEP entry drain must record integrate results so the same KEEP is not retried until cap."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    c.shared_state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/kernel.cu",
        },
    }
    calls: list[str] = []

    async def _fake_integrate_handler(payload, *, session_dir):
        calls.append(payload["kernel_id"])
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": payload["kernel_id"],
            "patch_path": "/tmp/optimized.cu",
            "target_file": "/tmp/kernel.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-k004"),
        }

    async def _noop_roofline(*, reason: str):
        return None

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.kernel_request_handlers.integrate_handler",
        _fake_integrate_handler,
    )
    c._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._drain_pending_keep_integrates()

    assert calls == ["k004"]
    assert c.shared_state.kernel_integrate_attempts
    assert c.shared_state.next_pending_keep_kernel_id() == ""
    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["kernel_id"] == "k004"


def test_pending_keep_kernel_ids_prioritize_trace_impact_over_micro():
    """E2E integrate order should prefer trace impact over isolated micro speedup."""
    state = SharedState()
    state.last_trace_analyze = {
        "hot_kernels_top15": [
            {"kernel_id": "k001", "gpu_pct": 60.0},
            {"kernel_id": "k004", "gpu_pct": 10.0},
        ],
    }
    state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/rmsnorm.cu",
        },
        "k001": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.51,
            "last_source_file": "/tmp/moe.cu",
        },
    }

    assert state.pending_keep_kernel_ids() == ["k001", "k004"]
    assert state.next_pending_keep_kernel_id() == "k001"


def test_pending_keep_kernel_ids_do_not_retry_needs_review():
    """A recorded NEEDS_REVIEW attempt should not auto-rerun the same patch."""
    state = SharedState()
    state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/rmsnorm.cu",
        },
        "k001": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.51,
            "last_source_file": "/tmp/moe.cu",
        },
    }
    state.record_kernel_integrate_result({
        "status": "ok",
        "decision": "NEEDS_REVIEW",
        "kernel_id": "k004",
        "patch_path": "/tmp/k004_opt.cu",
        "target_file": "/tmp/rmsnorm.cu",
        "new_tput": 100.8,
        "gain_pct": 0.8,
        "workspace": "/tmp/integrate-k004",
    })

    assert state.pending_keep_kernel_ids() == ["k001"]
    assert state.next_pending_keep_kernel_id() == "k001"


@pytest.mark.asyncio
async def test_positive_needs_review_stack_validation_promotes_combo(tmp_path: Path):
    """Two positive sub-threshold kernel patches should get one combined E2E validation."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result({
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "kernel_id": kid,
            "patch_path": f"/tmp/{kid}_opt.cu",
            "target_file": f"/tmp/{kid}.cu",
            "new_tput": 100.0 + gain,
            "gain_pct": gain,
            "workspace": f"/tmp/integrate-{kid}",
        })

    validation_calls = 0

    async def _fake_stack_validation(entries):
        nonlocal validation_calls
        validation_calls += 1
        assert {e["kernel_id"] for e in entries} == {"k001", "k004"}
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k001+k004",
            "patch_path": "/tmp/k001_opt.cu+/tmp/k004_opt.cu",
            "target_file": "/tmp/k001.cu+/tmp/k004.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-stack"),
            "apply_result": {"status": "ok"},
            "stack_kernel_ids": ["k001", "k004"],
            "stack_validation": True,
        }

    async def _noop_roofline(*, reason: str):
        return None

    c._run_kernel_stack_validation_e2e = _fake_stack_validation
    c._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._maybe_validate_positive_needs_review_stack()

    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["kernel_id"] == "k001+k004"
    assert c.shared_state.cumulative_gain_validated == pytest.approx(2.0)
    assert validation_calls == 1
    resolved_entries = [
        entry for entry in c.shared_state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    ]
    assert all(entry["stack_resolved"] is True for entry in resolved_entries)
    assert {
        entry["stack_validation_kernel_id"] for entry in resolved_entries
    } == {"k001+k004"}

    await c._maybe_validate_positive_needs_review_stack()

    assert validation_calls == 1
    stack_entries = [
        item for item in c.shared_state.optimization_stack
        if isinstance(item, dict) and item.get("kernel_id") == "k001+k004"
    ]
    assert stack_entries
    assert stack_entries[0].get("stack_kernel_ids") == ["k001", "k004"]


@pytest.mark.asyncio
async def test_recovers_pending_stack_validation_after_crash(tmp_path: Path):
    """A saved pending stack result should finish promotion without re-applying."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result({
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "kernel_id": kid,
            "patch_path": f"/tmp/{kid}_opt.cu",
            "target_file": f"/tmp/{kid}.cu",
            "new_tput": 100.0 + gain,
            "gain_pct": gain,
            "workspace": f"/tmp/integrate-{kid}",
        })
    stack = c._stack_entries_for_validation(["k001", "k004"])
    c._mark_stack_validation_in_progress(stack, "k001+k004")
    c.shared_state.pending_stack_validation_result = {
        "status": "ok",
        "decision": "KEEP",
        "kernel_id": "k001+k004",
        "patch_path": "/tmp/k001_opt.cu+/tmp/k004_opt.cu",
        "target_file": "/tmp/k001.cu+/tmp/k004.cu",
        "base_tput": 100.0,
        "new_tput": 102.0,
        "gain_pct": 2.0,
        "workspace": str(tmp_path / "integrate-stack"),
        "apply_result": {"status": "ok"},
        "stack_kernel_ids": ["k001", "k004"],
        "stack_validation": True,
    }
    c.shared_state.save(tmp_path)

    validation_calls = 0

    async def _should_not_run(entries):
        nonlocal validation_calls
        validation_calls += 1
        raise AssertionError("stack validation should not re-run during recovery")

    async def _noop_roofline(*, reason: str):
        return None

    c._run_kernel_stack_validation_e2e = _should_not_run
    c._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._recover_interrupted_stack_validation()

    assert validation_calls == 0
    assert c.shared_state.current_best["kernel_id"] == "k001+k004"
    assert not c.shared_state.pending_stack_validation_result
    resolved = [
        entry for entry in c.shared_state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    ]
    assert all(entry.get("stack_resolved") for entry in resolved)


def test_positive_needs_review_integrates_skip_in_progress_entries():
    """In-flight stack members must not be re-selected for another validation."""
    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState()
    c.shared_state.kernel_integrate_attempts = {
        "k001": {
            "kernel_id": "k001",
            "patch_path": "/tmp/k001_opt.cu",
            "target_file": "/tmp/k001.cu",
            "last_decision": "NEEDS_REVIEW",
            "best_gain_pct": 0.6,
            "stack_validation_in_progress": True,
        },
        "k004": {
            "kernel_id": "k004",
            "patch_path": "/tmp/k004_opt.cu",
            "target_file": "/tmp/k004.cu",
            "last_decision": "NEEDS_REVIEW",
            "best_gain_pct": 0.8,
        },
    }

    eligible = c._positive_needs_review_integrates()
    assert len(eligible) == 1
    assert eligible[0]["kernel_id"] == "k004"


# 1. _build_sweep_params_from_recipe — pure static helper
def test_build_sweep_params_defaults_when_no_recipe():
    """No warm_start_recipe → SKILL.md defaults + source='skill_md_default'."""
    from inference_optimizer.orchestrator.action_executors.sweep import (
        DEFAULT_CONC_VALUES, DEFAULT_ISL_OSL, DEFAULT_NUM_PROMPTS_FACTOR,
    )

    state = _BareState()
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["source"] == "skill_md_default"
    assert out["conc_values"] == DEFAULT_CONC_VALUES
    assert out["isl_osl_configs"] == DEFAULT_ISL_OSL
    assert out["num_prompts_factor"] == DEFAULT_NUM_PROMPTS_FACTOR


def test_build_sweep_params_full_recipe_override():
    """Recipe with all three fields → all three overridden + source=cortex_recipe."""
    state = _BareState(
        warm_start_recipe={
            "sweep_grid": {
                "conc_values":      [8, 32, 128],
                "isl_osl_configs":  ["1024:1024", "4096:4096"],
                "num_prompts_factor": 7,
            },
        },
    )
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["source"] == "cortex_recipe"
    assert out["conc_values"] == [8, 32, 128]
    assert out["isl_osl_configs"] == ["1024:1024", "4096:4096"]
    assert out["num_prompts_factor"] == 7


def test_build_sweep_params_partial_recipe_per_field_fallback():
    """Recipe overriding only conc_values → that field from recipe, the rest from defaults, source=cortex_recipe."""
    from inference_optimizer.orchestrator.action_executors.sweep import (
        DEFAULT_ISL_OSL, DEFAULT_NUM_PROMPTS_FACTOR,
    )

    state = _BareState(
        warm_start_recipe={"sweep_grid": {"conc_values": [256]}},
    )
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["source"] == "cortex_recipe"
    assert out["conc_values"] == [256]
    assert out["isl_osl_configs"] == DEFAULT_ISL_OSL
    assert out["num_prompts_factor"] == DEFAULT_NUM_PROMPTS_FACTOR


def test_build_sweep_params_accepts_isl_osl_as_pair_lists():
    """[[isl, osl], [isl, osl]] form converts to ['isl:osl', ...]."""
    state = _BareState(
        warm_start_recipe={
            "sweep_grid": {"isl_osl_configs": [[2048, 512], [8192, 1024]]},
        },
    )
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["isl_osl_configs"] == ["2048:512", "8192:1024"]


@pytest.mark.parametrize(
    "bad",
    [None, [], "foo", [None], ["a", "b"], "1,2,3"],
    ids=["none", "empty", "string", "list_of_none", "non_int_strings", "csv_string"],
)
def test_build_sweep_params_rejects_malformed_conc_values(bad):
    """conc_values must be a non-empty list of int-coercible values;
    anything else → fallback to default."""
    from inference_optimizer.orchestrator.action_executors.sweep import (
        DEFAULT_CONC_VALUES,
    )

    state = _BareState(warm_start_recipe={"sweep_grid": {"conc_values": bad}})
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["conc_values"] == DEFAULT_CONC_VALUES


@pytest.mark.parametrize(
    "bad",
    [None, [], "1024:1024", [None], [{"isl": 1024}], [[1024]]],
    ids=["none", "empty", "string", "list_of_none", "dict_inside", "list_wrong_arity"],
)
def test_build_sweep_params_rejects_malformed_isl_osl(bad):
    """Non-list / wrong-shape isl_osl_configs → default fallback."""
    from inference_optimizer.orchestrator.action_executors.sweep import (
        DEFAULT_ISL_OSL,
    )

    state = _BareState(warm_start_recipe={"sweep_grid": {"isl_osl_configs": bad}})
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["isl_osl_configs"] == DEFAULT_ISL_OSL


@pytest.mark.parametrize(
    "bad",
    [0, -1, "x", None],
    ids=["zero", "negative", "string", "none"],
)
def test_build_sweep_params_rejects_non_positive_num_prompts_factor(bad):
    """num_prompts_factor must be a positive int; zero / negative / non-int
    → default fallback."""
    from inference_optimizer.orchestrator.action_executors.sweep import (
        DEFAULT_NUM_PROMPTS_FACTOR,
    )

    state = _BareState(
        warm_start_recipe={"sweep_grid": {"num_prompts_factor": bad}},
    )
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["num_prompts_factor"] == DEFAULT_NUM_PROMPTS_FACTOR


@pytest.mark.parametrize(
    "bad",
    [None, "raw text", 42, [], {"sweep_grid": "not a dict"}],
    ids=["none", "string", "int", "list", "sweep_grid_not_dict"],
)
def test_build_sweep_params_non_dict_recipe_falls_back(bad):
    """A non-dict recipe (or non-dict sweep_grid) → defaults."""
    state = _BareState(warm_start_recipe=bad)  # type: ignore[arg-type]
    out = Coordinator._build_sweep_params_from_recipe(state)
    assert out["source"] == "skill_md_default"


# 2. _enqueue_internal_sweep_task — params inheritance
@pytest.mark.asyncio
async def test_enqueue_internal_sweep_task_inherits_baseline_config(coord):
    coord.shared_state.baseline_config_path = "/tmp/baseline.yaml"
    coord.shared_state.current_best = {"extra_server_args": "--mla 1"}
    coord.shared_state.last_baseline = {"benchmark_script": "sglang_mi300x.sh"}
    task = await coord._enqueue_internal_sweep_task(reason="phase_entry")
    assert task.kind == "sweep"
    assert task.idempotency_key == "internal-sweep-phase_entry"
    assert task.params["source"] == "skill_md_default"
    assert task.params["reason"] == "phase_entry"
    assert task.params["config_path"] == "/tmp/baseline.yaml"
    assert task.params["base_extra_args"] == "--mla 1"
    assert task.params["benchmark_script"] == "sglang_mi300x.sh"
    # Grid params present so executor doesn't fall back to its own defaults.
    assert isinstance(task.params["conc_values"], list)
    assert isinstance(task.params["isl_osl_configs"], list)
    assert isinstance(task.params["num_prompts_factor"], int)


@pytest.mark.asyncio
async def test_enqueue_internal_sweep_task_omits_empty_strings(coord):
    """Empty extra_server_args / benchmark_script must not land in params."""
    coord.shared_state.current_best = {"extra_server_args": ""}
    coord.shared_state.last_baseline = {"benchmark_script": ""}
    task = await coord._enqueue_internal_sweep_task(reason="phase_entry")
    assert "base_extra_args" not in task.params
    assert "benchmark_script" not in task.params


@pytest.mark.asyncio
async def test_enqueue_internal_sweep_task_cortex_recipe_propagates(coord):
    """Recipe-driven grid surfaces as source='cortex_recipe' on the task."""
    coord.shared_state.warm_start_recipe = {
        "sweep_grid": {
            "conc_values":     [128],
            "isl_osl_configs": ["1024:1024"],
        },
    }
    task = await coord._enqueue_internal_sweep_task(reason="phase_entry")
    assert task.params["source"] == "cortex_recipe"
    assert task.params["conc_values"] == [128]
    assert task.params["isl_osl_configs"] == ["1024:1024"]


# 3. _on_enter_sweep hook
@pytest.mark.asyncio
async def test_on_enter_sweep_enqueues_and_stamps_evidence(coord):
    """Happy path: the hook enqueues a sweep task and stamps the phase_history evidence."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-sweep-phase_entry" in coord.tasks._tasks
    task = coord.tasks._tasks["internal-sweep-phase_entry"]
    assert task.kind == "sweep"

    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_sweep_enqueued"] is True
    assert evidence["auto_sweep_task_id"] == task.task_id
    assert evidence["auto_sweep_grid_source"] == "skill_md_default"
    # combos = |conc_values| × |isl_osl_configs| = 3 × 3 (SKILL.md defaults)
    assert evidence["auto_sweep_combos"] == 9


@pytest.mark.asyncio
async def test_on_enter_sweep_cortex_recipe_evidence(coord):
    """Recipe-driven grid surfaces grid_source='cortex_recipe' + a recipe-derived combos count."""
    coord.shared_state.warm_start_recipe = {
        "sweep_grid": {
            "conc_values":     [8, 32],
            "isl_osl_configs": ["1024:1024", "4096:4096", "8192:1024"],
        },
    }
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_sweep_grid_source"] == "cortex_recipe"
    assert evidence["auto_sweep_combos"] == 6  # 2 × 3


@pytest.mark.asyncio
async def test_on_enter_sweep_idempotent_on_reentry(coord):
    """Re-entering SWEEP twice hits the same idempotency_key and reuses the task."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    task1 = coord.tasks._tasks["internal-sweep-phase_entry"]
    coord.shared_state.phase_history.append(
        {"to_phase": "SWEEP", "reason": "re_entry_test", "evidence": {}},
    )
    await coord._on_enter_sweep(from_phase="SWEEP")
    task2 = coord.tasks._tasks["internal-sweep-phase_entry"]
    assert task1 is task2
    assert len(coord.tasks._tasks) == 1


@pytest.mark.asyncio
async def test_on_enter_sweep_failure_records_evidence(coord, monkeypatch):
    """If the enqueue raises, the hook records ``auto_sweep_error`` and returns without propagating."""
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr(coord, "_enqueue_internal_sweep_task", _boom)
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    # Should not raise:
    await coord._on_enter_sweep(from_phase="KERNEL")
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert "auto_sweep_error" in evidence
    assert "simulated DB outage" in evidence["auto_sweep_error"]
    # No task was enqueued.
    assert coord.tasks._tasks == {}


# 4. End-to-end via real Coordinator
@pytest.mark.asyncio
async def test_phase_transition_into_sweep_enqueues_sweep_e2e(tmp_path: Path):
    """End-to-end: a SWEEP transition + hook dispatcher persists the sweep task under the v0.8 idempotency_key."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "kernel":        MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
        knowledge_plane=None,
    )
    # Seed state at KERNEL boundary as if a plateau_kernel just fired.
    coord.shared_state.phase = "KERNEL"
    coord.shared_state.kernel_enabled = True
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain = 12.0
    coord.shared_state.last_profile_trace = "/tmp/dummy.trace.json.gz"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
        {"to_phase": "KERNEL",  "evidence": {}, "reason": "plateau_explore"},
    ]

    # Simulate a real KERNEL → SWEEP transition.
    coord.shared_state.record_phase_transition(
        to_phase="SWEEP",
        reason="plateau_kernel",
        evidence={"trigger": "test_e2e"},
    )
    await coord._on_phase_entered(from_phase="KERNEL", to_phase="SWEEP")

    rows = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-sweep-phase_entry",),
    )
    assert len(rows) == 1
    assert rows[0]["kind"] == "sweep"
    assert rows[0]["state"] == "queued"

    last_history = coord.shared_state.phase_history[-1]
    assert last_history["to_phase"] == "SWEEP"
    evidence = last_history.get("evidence") or {}
    assert evidence.get("auto_sweep_enqueued") is True
    assert evidence.get("auto_sweep_task_id")
    assert evidence.get("auto_sweep_grid_source") in (
        "cortex_recipe", "skill_md_default",
    )


@pytest.mark.asyncio
async def test_phase_transition_explore_to_sweep_no_kernel_mode(tmp_path: Path):
    """``--no-kernel`` runs go EXPLORE → SWEEP directly; the SWEEP entry hook must still enqueue."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    role_registry = {
        k: v for k, v in default_role_registry().items() if k != "kernel"
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=role_registry,
        cortex_kb=None,
        knowledge_plane=None,
    )
    coord.shared_state.kernel_enabled = False
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "evidence": {}, "reason": "test_forced"},
    ]
    await coord._on_phase_entered(from_phase="EXPLORE", to_phase="SWEEP")
    rows = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-sweep-phase_entry",),
    )
    assert len(rows) == 1, "SWEEP auto-enqueue must run in --no-kernel mode too"


# 5. Idempotency key structural cross-check
def test_internal_sweep_idempotency_key_does_not_collide_with_llm_path():
    """The internal hook's ``internal-sweep-<reason>`` key must never collide with the LLM's ``approved-<msg_id>`` key."""
    internal_key = "internal-sweep-phase_entry"
    # Mirror the format _materialize_approved_proposal builds.
    llm_key = "approved-msg_abc123"
    assert internal_key != llm_key
    assert not llm_key.startswith("internal-")
    assert not internal_key.startswith("approved-")


# 6. PolicyGate sweep_phase_singleton rule
class _SweepSingletonState:
    """SharedState stand-in carrying just the fields the ``sweep_phase_singleton`` rule reads."""

    def __init__(self, phase_history=None):
        self.phase_history = list(phase_history or [])


def _sweep_phase_row(*, auto_sweep_task_id: str = "") -> dict:
    """Build a phase_history row mirroring what ``record_phase_transition`` produces on SWEEP entry."""
    evidence: dict = {}
    if auto_sweep_task_id:
        evidence["auto_sweep_task_id"] = auto_sweep_task_id
        evidence["auto_sweep_enqueued"] = True
    return {
        "to_phase": "SWEEP",
        "from_phase": "EXPLORE",
        "reason": "explore_done",
        "evidence": evidence,
    }


def _make_policy_gate(*, shared_state):
    """Plain PolicyGate wired to the role registry + the test's SharedState double."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.orchestrator.policy import PolicyGate
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=shared_state,
    )


def test_sweep_singleton_denies_delegate_after_auto_enqueue_stamped():
    """Once the SWEEP row carries ``evidence.auto_sweep_task_id``, an LLM ``delegate{action='sweep'}`` is denied via ``sweep_phase_singleton``."""
    from inference_optimizer.orchestrator.policy import PolicyDenied

    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="auto-sweep-abc123")],
    )
    gate = _make_policy_gate(shared_state=state)

    with pytest.raises(PolicyDenied) as excinfo:
        gate._validate_sweep_singleton(
            payload={"action_name": "sweep", "params": {}},
            intent_kind="delegate",
        )
    assert excinfo.value.rule == "sweep_phase_singleton"
    # Hint must mention the bypass switch so the operator-debug path is discoverable.
    assert "bypass_sweep_singleton" in (excinfo.value.hint or "")


def test_sweep_singleton_denies_propose_action_after_auto_enqueue_stamped():
    """Same shape on the propose_action channel — defense in depth."""
    from inference_optimizer.orchestrator.policy import PolicyDenied

    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="auto-sweep-xyz")],
    )
    gate = _make_policy_gate(shared_state=state)

    with pytest.raises(PolicyDenied) as excinfo:
        gate._validate_sweep_singleton(
            payload={"action_name": "sweep", "params": {}},
            intent_kind="propose_action",
        )
    assert excinfo.value.rule == "sweep_phase_singleton"


def test_sweep_singleton_inert_before_auto_enqueue_stamps_evidence():
    """Race-window: a SWEEP row without a stamped ``auto_sweep_task_id`` keeps the rule inert so the Coordinator's auto-enqueue isn't falsely blocked."""
    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="")],
    )
    gate = _make_policy_gate(shared_state=state)

    # Must NOT raise.
    gate._validate_sweep_singleton(
        payload={"action_name": "sweep", "params": {}},
        intent_kind="delegate",
    )


def test_sweep_singleton_inert_outside_sweep_phase():
    """When the latest phase row isn't SWEEP, the rule stays silent (``_validate_phase_action`` fires instead)."""
    explore_row = {
        "to_phase": "EXPLORE",
        "from_phase": "PRELUDE",
        "reason": "prelude_done",
        "evidence": {"auto_sweep_task_id": "stale"},
    }
    state = _SweepSingletonState(phase_history=[explore_row])
    gate = _make_policy_gate(shared_state=state)
    # The rule keys on phase_history[-1].to_phase=="SWEEP", so the stale id is inert.
    gate._validate_sweep_singleton(
        payload={"action_name": "sweep"},
        intent_kind="delegate",
    )


def test_sweep_singleton_inert_when_phase_history_empty():
    """Defensive: an empty phase_history must not raise."""
    state = _SweepSingletonState(phase_history=[])
    gate = _make_policy_gate(shared_state=state)
    gate._validate_sweep_singleton(
        payload={"action_name": "sweep"},
        intent_kind="delegate",
    )


def test_sweep_singleton_inert_when_shared_state_is_none():
    """PolicyGate without a SharedState reference self-defends with an early return."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.orchestrator.policy import PolicyGate

    gate = PolicyGate(role_registry=default_role_registry())
    assert gate.shared_state is None
    gate._validate_sweep_singleton(
        payload={"action_name": "sweep"},
        intent_kind="delegate",
    )


def test_sweep_singleton_self_clears_at_sweep_to_close_transition():
    """Once SWEEP→CLOSE happens, the latest row is CLOSE so the singleton rule stops firing."""
    state = _SweepSingletonState(
        phase_history=[
            _sweep_phase_row(auto_sweep_task_id="auto-sweep-abc"),
            {
                "to_phase": "CLOSE",
                "from_phase": "SWEEP",
                "reason": "sweep_done",
                "evidence": {},
            },
        ],
    )
    gate = _make_policy_gate(shared_state=state)
    # Must NOT raise — the singleton rule looks at phase_history[-1],
    # which is now CLOSE.
    gate._validate_sweep_singleton(
        payload={"action_name": "sweep"},
        intent_kind="delegate",
    )


def test_sweep_singleton_bypass_flag_lets_operator_force_second_sweep():
    """Operator escape hatch: ``params.bypass_sweep_singleton=True`` silences the rule for a second sweep."""
    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="auto-sweep-abc")],
    )
    gate = _make_policy_gate(shared_state=state)
    # Must NOT raise.
    gate._validate_sweep_singleton(
        payload={
            "action_name": "sweep",
            "params": {
                "bypass_sweep_singleton": True,
                "grid": {"conc_values": [128]},
            },
        },
        intent_kind="delegate",
    )


# 6b. End-to-end through full validate_intent (delegate / propose_action)
def test_validate_intent_denies_llm_sweep_delegate_in_active_sweep_phase():
    """Through full ``validate_intent``: a sweep delegate in active SWEEP fires the singleton rule before ``_validate_phase_action``."""
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import PolicyDenied

    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="auto-sweep-abc")],
    )
    gate = _make_policy_gate(shared_state=state)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "sweep",
            "predicted_gain_pct": 1.0,
            "params": {"grid": {"conc_values": [64]}},
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "sweep_phase_singleton"


def test_validate_intent_denies_llm_sweep_propose_in_active_sweep_phase():
    """Same shape on propose_action."""
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import PolicyDenied

    state = _SweepSingletonState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="auto-sweep-abc")],
    )
    gate = _make_policy_gate(shared_state=state)
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "sweep"},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "sweep_phase_singleton"
