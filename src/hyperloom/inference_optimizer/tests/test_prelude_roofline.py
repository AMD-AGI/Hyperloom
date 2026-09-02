# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE-bootstrap analysis-task enqueue tests (kind/idempotency/benchmark-script wiring of ``_enqueue_internal_analysis_task``)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


# Minimal SharedState + TaskRegistry doubles.
@dataclass
class _BareState:
    baseline_tput: float = 100.0
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 0.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    roofline_snapshots: list[dict[str, Any]] = field(default_factory=list)
    kernel_optimizer: str = "native"
    phase_history: list[dict[str, Any]] = field(default_factory=list)

    def save(self, _session_dir: Path | None) -> None:
        pass


class _StubTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Any] = {}
        self._by_idem: dict[str, Any] = {}

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        **_extras: Any,
    ):
        existing = self._by_idem.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid
        from hyperloom.orchestrator.state.task_registry import Task

        task = Task(
            task_id=_uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[task.task_id] = task  # type: ignore[assignment]
        self._by_idem[idempotency_key] = task  # type: ignore[assignment]
        return task, False

    async def get(self, task_id: str):
        return self._tasks[task_id]


@pytest.fixture
def coord(tmp_path: Path, monkeypatch) -> Coordinator:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        kernel_optimizer="native",
    )
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c._run_deadline = None
    c._run_started_monotonic = None
    return c


def test_prelude_initial_roofline_task_contract(coord: Coordinator):
    """The PRELUDE-bootstrap roofline represents baseline, not current_best."""
    coord.shared_state.current_best = {
        "extra_server_args": "--tp 8 --enable-mla",
        "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
        "remove_args": ["--old-backend"],
        "unset_envs": ["OLD_BACKEND"],
        "args_mode": "replace",
    }
    coord.shared_state.last_baseline = {
        "benchmark_script": "magpie_serving_bench.sh",
    }

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="prelude_initial"),
    )

    assert task.kind == "roofline"
    assert task.idempotency_key == "internal-analysis-prelude_initial"
    assert task.params["reason"] == "prelude_initial"
    assert task.params["source"] == "coordinator_internal"
    assert "base_extra_args" not in task.params
    assert task.params["benchmark_script"] == "magpie_serving_bench.sh"


def test_prelude_initial_roofline_uses_baseline_server_args(
    coord: Coordinator,
    monkeypatch,
):
    """PRELUDE roofline injects baseline's own server args, never current_best's."""
    coord.shared_state.current_best = {
        "extra_server_args": "--enable-torch-compile --quantization fp8",
    }
    import hyperloom.orchestrator.kernel.roofline_ceiling as rc

    monkeypatch.setattr(
        rc,
        "_read_baseline_yaml_server_args",
        lambda _state: "--attention-backend AITER",
    )

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="prelude_initial"),
    )

    assert task.params["base_extra_args"] == "--attention-backend AITER"
    assert "--enable-torch-compile" not in task.params["base_extra_args"]
    assert "fp8" not in task.params["base_extra_args"]


@pytest.mark.asyncio
async def test_prelude_initial_roofline_is_idempotent(coord: Coordinator):
    """A second call with the same reason returns the same task (no double-enqueue on resume)."""
    first = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    second = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert first.task_id == second.task_id
    assert len(coord.tasks._tasks) == 1


@pytest.mark.asyncio
async def test_distinct_reasons_produce_distinct_tasks(coord: Coordinator):
    """A watermark-driven roofline is a separate task; the idempotency key is reason-scoped."""
    prelude = await coord._enqueue_internal_analysis_task(
        reason="prelude_initial",
    )
    watermark = await coord._enqueue_internal_analysis_task(
        reason="explore_keep_watermark",
    )
    assert prelude.task_id != watermark.task_id
    assert "internal-analysis-prelude_initial" in coord.tasks._by_idem
    assert "internal-analysis-explore_keep_watermark" in coord.tasks._by_idem


@pytest.mark.asyncio
async def test_failed_roofline_does_not_dedup_away_the_retry(coord: Coordinator):
    """The whole blackout, stated directly.

    ``_needs_roofline_for_watermark`` re-arms once a roofline has failed, so the
    system asks for a retry on purpose. It used to re-ask under a per-cycle
    singleton key, so the registry handed back the attempt that had already
    failed and nothing ran. Four sessions went by with no GPU evidence at all
    while the log reported "enqueued" each time.
    """
    first = await coord._enqueue_internal_analysis_task(
        reason="integrate_keep_watermark",
    )

    coord.shared_state.roofline_failure_streak = 1
    retry = await coord._enqueue_internal_analysis_task(
        reason="integrate_keep_watermark",
    )

    assert retry.task_id != first.task_id
    assert len(coord.tasks._tasks) == 2


@pytest.mark.asyncio
async def test_each_further_failure_earns_its_own_attempt(coord: Coordinator):
    """Two failures in a row are two distinct retries, not one repeated."""
    seen = set()
    for streak in (0, 1, 2):
        coord.shared_state.roofline_failure_streak = streak
        task = await coord._enqueue_internal_analysis_task(
            reason="integrate_keep_watermark",
        )
        seen.add(task.task_id)

    assert len(seen) == 3


@pytest.mark.asyncio
async def test_a_roofline_that_worked_is_never_re_run(coord: Coordinator):
    """The streak resets to zero on a successful snapshot, so success collapses
    back onto the original key and stays idempotent across resumes."""
    coord.shared_state.roofline_failure_streak = 0
    first = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    second = await coord._enqueue_internal_analysis_task(reason="prelude_initial")

    assert first.task_id == second.task_id
    assert first.idempotency_key == "internal-analysis-prelude_initial"


@pytest.mark.asyncio
async def test_profile_kind_keeps_the_plain_key(coord: Coordinator):
    """Only roofline retries; the profile kind has no failure streak to spend."""
    coord.shared_state.enable_roofline = False
    coord.shared_state.roofline_failure_streak = 2
    task = await coord._enqueue_internal_analysis_task(reason="prelude_initial")

    assert task.kind == "profile"
    assert task.idempotency_key == "internal-analysis-prelude_initial"


@pytest.mark.parametrize(
    "named_state,reopens",
    [
        ("succeeded", True),
        ("cancelled", True),
        # A watchdog-reclaimed roofline reports no result, so the gate release
        # is the only thing that can ever clear the marker it left.
        ("failed", True),
        ("running", False),
        ("queued", False),
    ],
)
@pytest.mark.asyncio
async def test_watermark_gate_reopens_exactly_when_the_roofline_it_names_finished(
    coord: Coordinator,
    named_state: str,
    reopens: bool,
):
    """The gate exists so two rooflines never run at once, so it must hold for
    every live state and release for every finished one. It is persisted state:
    a marker left on a finished task is a wedge the next resume inherits."""
    state = coord.shared_state
    state.baseline_tput = 100.0
    state.cumulative_gain_validated = 50.0
    state.last_roofline_tput = 0.0

    named = await coord._enqueue_internal_analysis_task(
        reason="integrate_keep_watermark",
    )
    coord.tasks._tasks[named.task_id].state = named_state
    state.auto_roofline_pending_task_id = named.task_id
    state.roofline_failure_streak = 1  # its trace analysis failed
    assert coord._needs_roofline_for_watermark() is False

    enqueued = await coord._maybe_enqueue_watermark_roofline(
        reason="integrate_keep_watermark",
    )

    assert enqueued is reopens
    assert (state.auto_roofline_pending_task_id != named.task_id) is reopens


def test_watermark_stops_re_arming_once_retries_are_spent(coord: Coordinator):
    """A roofline leg costs the better part of an hour, so a collector that is
    broken rather than flaky must not be allowed to spend the session on it."""
    from hyperloom.orchestrator.loop.coordinator_helpers import (
        _MAX_ROOFLINE_FAILURE_RETRIES,
    )

    state = coord.shared_state
    state.baseline_tput = 100.0
    state.cumulative_gain_validated = 50.0  # cur = 150, well over the watermark
    state.last_roofline_tput = 0.0
    state.auto_roofline_pending_task_id = ""

    state.roofline_failure_streak = _MAX_ROOFLINE_FAILURE_RETRIES
    assert coord._needs_roofline_for_watermark() is True

    state.roofline_failure_streak = _MAX_ROOFLINE_FAILURE_RETRIES + 1
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_roofline_inherits_current_best_args(coord: Coordinator):
    """Watermark roofline still profiles the optimized current_best config."""
    coord.shared_state.current_best = {
        "extra_server_args": "--tp 8 --enable-mla",
        "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
        "remove_args": ["--old-backend"],
        "unset_envs": ["OLD_BACKEND"],
        "args_mode": "replace",
    }

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="explore_keep_watermark"),
    )

    assert task.params["reason"] == "explore_keep_watermark"
    assert task.params["base_extra_args"] == "--tp 8 --enable-mla"
    assert task.params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER_LINEAR": "1"}
    assert task.params["base_remove_args"] == ["--old-backend"]
    assert task.params["base_unset_envs"] == ["OLD_BACKEND"]
    assert task.params["base_args_mode"] == "replace"


@pytest.mark.asyncio
async def test_enable_roofline_false_picks_profile_kind(coord: Coordinator):
    """When ``enable_roofline`` is False, the task switches kind to ``profile`` keeping the reason-scoped key."""
    coord.shared_state.enable_roofline = False
    task = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert task.kind == "profile"
    assert task.idempotency_key == "internal-analysis-prelude_initial"


class _StubSub:
    """Records tasks handed to ``run_task``; optionally lands a fresh snapshot to simulate a completed reprofile."""

    def __init__(self, state: Any = None, landed_tput: float | None = None) -> None:
        self.tasks_run: list[Any] = []
        self._state = state
        self._landed_tput = landed_tput

    async def run_task(self, task: Any, **_kwargs: Any) -> None:
        self.tasks_run.append(task)
        if self._state is not None and self._landed_tput is not None:
            self._state.roofline_snapshots.append(
                {"achieved_tok_per_sec": self._landed_tput},
            )


@pytest.mark.asyncio
async def test_on_enter_kernel_reprofiles_on_change(coord: Coordinator, monkeypatch):
    """KERNEL entry (no-GEMM path) reprofiles inline when projected tput (120) diverges from the last measured trace (100), anchoring on the new snapshot."""
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.sub = _StubSub(coord.shared_state, landed_tput=120.0)
    monkeypatch.setattr(coord.phase_machine, "_kernel_enabled", lambda: True)
    monkeypatch.setattr(coord.dispatcher, "_gemm_tuning_required_before_kernel_opt", lambda: False)
    coord.shared_state.cumulative_gain_validated = 20.0  # cur = 100 * 1.20 = 120

    await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

    assert len(coord.sub.tasks_run) == 1
    # The reason carries a profile fingerprint suffix so repeated kernel entries
    # at the same gain stack are distinguishable in the task log.
    assert coord.sub.tasks_run[0].params["reason"].startswith("kernel_entry_g0_")
    assert coord.shared_state.last_roofline_tput == 120.0


@pytest.mark.asyncio
async def test_on_enter_kernel_skips_gemm_but_still_runs_fusion(coord: Coordinator, monkeypatch):
    """Disabling GEMM tuning must not disable the independently gated fusion stage."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "1")
    monkeypatch.setattr(coord.phase_machine, "_kernel_enabled", lambda: True)
    monkeypatch.setattr(coord.phase_kernel, "_geak_enabled", lambda: False)
    monkeypatch.setattr(coord.phase_kernel, "_fusion_required_before_kernel_opt", lambda: True)
    assert coord._gemm_tuning_required_before_kernel_opt() is False

    fusion_calls = 0

    async def _run_fusion() -> None:
        nonlocal fusion_calls
        fusion_calls += 1

    async def _skip_reprofile() -> None:
        return None

    monkeypatch.setattr(coord.phase_kernel, "_run_forge_fusion", _run_fusion)
    monkeypatch.setattr(coord.phase_kernel, "_maybe_reprofile_for_kernel", _skip_reprofile)

    await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

    assert fusion_calls == 1


@pytest.mark.asyncio
async def test_on_enter_kernel_skips_gemm_but_still_dispatches_kernel_opt(coord: Coordinator, monkeypatch):
    """The phase dispatches its own kernel_opt on both entry routes.

    The dispatch sat on the GEMM route alone, so skipping GEMM tuning removed
    the phase's source-level kernel work too -- two unrelated settings, with
    nothing in the log connecting them. A run then held eight routable
    candidates, cleared the dispatch floor, and reached SWEEP having optimized
    nothing, because the only remaining path was an orchestration request that
    was never made.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "1")
    monkeypatch.setattr(coord.phase_machine, "_kernel_enabled", lambda: True)
    monkeypatch.setattr(coord.phase_kernel, "_geak_enabled", lambda: False)
    monkeypatch.setattr(coord.phase_kernel, "_fusion_required_before_kernel_opt", lambda: False)
    assert coord._gemm_tuning_required_before_kernel_opt() is False

    dispatched = 0

    async def _skip_reprofile() -> None:
        return None

    async def _dispatch() -> None:
        nonlocal dispatched
        dispatched += 1

    monkeypatch.setattr(coord.phase_kernel, "_maybe_reprofile_for_kernel", _skip_reprofile)
    monkeypatch.setattr(coord.phase_kernel, "_kernel_opt_work_remains", lambda: True)
    monkeypatch.setattr(coord.phase_kernel, "_run_kernel_opt_nomination", _dispatch)

    await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

    assert dispatched == 1


@pytest.mark.asyncio
async def test_kernel_entry_does_not_dispatch_without_untried_candidates(coord: Coordinator, monkeypatch):
    """Nothing routable left is the one reason to hand the phase back."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "1")
    monkeypatch.setattr(coord.phase_machine, "_kernel_enabled", lambda: True)
    monkeypatch.setattr(coord.phase_kernel, "_geak_enabled", lambda: False)
    monkeypatch.setattr(coord.phase_kernel, "_fusion_required_before_kernel_opt", lambda: False)

    dispatched = 0

    async def _skip_reprofile() -> None:
        return None

    async def _dispatch() -> None:
        nonlocal dispatched
        dispatched += 1

    monkeypatch.setattr(coord.phase_kernel, "_maybe_reprofile_for_kernel", _skip_reprofile)
    monkeypatch.setattr(coord.phase_kernel, "_kernel_opt_work_remains", lambda: False)
    monkeypatch.setattr(coord.phase_kernel, "_run_kernel_opt_nomination", _dispatch)

    await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

    assert dispatched == 0


def test_a_trace_recorded_with_task_params_is_not_stale(coord: Coordinator):
    """The two writers of ``last_profile_workload`` disagree by construction.

    The roofline path records through ``record_profile_workload(task_params)``
    and fills ``server_args`` / ``extra_envs``; the kernel-entry path records
    through ``profile_workload_context()`` and leaves them empty. Comparing the
    whole dict therefore reported a change on every first KERNEL entry -- a full
    re-profile plus a second TraceLens pass, with the serving configuration
    provably unchanged -- and then stopped, because the re-profile it forced had
    rewritten the record in the other writer's shape.
    """
    state = coord.shared_state
    state.current_best = {
        "extra_server_args": "--block-size 128 --enable-expert-parallel",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    state.last_profile_status = "succeeded"
    state.last_profile_workload = state.profile_workload_context(
        {
            "base_extra_args": "--block-size 128 --enable-expert-parallel",
            "base_extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        }
    )
    # The record and a freshly built context differ, exactly as in production.
    assert state.last_profile_workload != state.profile_workload_context()

    assert coord.phase_kernel._profile_workload_changed() is False


def test_a_trace_of_a_different_workload_is_still_stale(coord: Coordinator):
    """Only the parameterization is forgiven; the workload itself still counts."""
    state = coord.shared_state
    state.last_profile_status = "succeeded"
    state.last_profile_workload = state.profile_workload_context()
    assert coord.phase_kernel._profile_workload_changed() is False

    state.isl = int(state.isl or 0) + 4096

    assert coord.phase_kernel._profile_workload_changed() is True


def _recorded_under(state, *, server_args: str, envs: dict) -> None:
    """Record a profile the way the roofline path does: with the task params."""
    state.current_best = {"extra_server_args": server_args, "extra_envs": dict(envs)}
    state.last_profile_status = "succeeded"
    state.last_profile_workload = state.profile_workload_context(
        {"base_extra_args": server_args, "base_extra_envs": dict(envs)}
    )


def _reprofiles(coord: Coordinator) -> bool:
    """Whether the two staleness checks together call for a re-profile."""
    phase = coord.phase_kernel
    signature = phase._current_profile_config_signature()
    return phase._profile_config_changed(signature) or phase._profile_workload_changed()


_BASE_ARGS = "--block-size 128 --enable-expert-parallel"
_BASE_ENVS = {"VLLM_ROCM_USE_AITER": "1"}


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        ("nothing moved", lambda s: None, False),
        (
            "explore added a server arg",
            lambda s: s.current_best.__setitem__("extra_server_args", _BASE_ARGS + " --max-num-batched-tokens 16384"),
            True,
        ),
        (
            "explore added an env",
            lambda s: s.current_best.__setitem__("extra_envs", {**_BASE_ENVS, "VLLM_ROCM_USE_AITER_MOE": "1"}),
            True,
        ),
        (
            "an env changed value",
            lambda s: s.current_best.__setitem__("extra_envs", {**_BASE_ENVS, "VLLM_ROCM_USE_AITER": "0"}),
            True,
        ),
        ("the workload changed", lambda s: setattr(s, "isl", int(s.isl or 0) + 4096), True),
    ],
)
def test_serving_config_changes_still_force_a_reprofile(coord: Coordinator, label, mutate, expected):
    """Forgiving the parameterization must not forgive a real config change.

    A configuration EXPLORE found and integrated changes which kernels run, so a
    trace taken before it is genuinely stale. Those changes reach
    ``_profile_config_changed``, which reads them from ``current_best`` on both
    sides; only the recording-shape mismatch was taken out of
    ``_profile_workload_changed``. This pins the boundary between the two.
    """
    state = coord.shared_state
    _recorded_under(state, server_args=_BASE_ARGS, envs=_BASE_ENVS)
    assert _reprofiles(coord) is False, "the recorded trace starts fresh"

    mutate(state)

    assert _reprofiles(coord) is expected, label


@pytest.mark.asyncio
async def test_kernel_entry_reprofile_skips_when_unchanged(coord: Coordinator):
    """Projected tput matching the last measured trace (cur == measured) skips the reprofile."""
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.sub = _StubSub(coord.shared_state)
    coord.shared_state.cumulative_gain_validated = 0.0  # cur = 100 == measured
    coord.shared_state.last_profile_workload = coord.shared_state.current_profile_workload_context()

    await coord._maybe_reprofile_for_kernel()

    assert coord.sub.tasks_run == []


@pytest.mark.asyncio
async def test_kernel_entry_reprofiles_legacy_trace_without_runtime_fingerprint(
    coord: Coordinator,
):
    """A pre-upgrade trace without runtime metadata is stale and must be refreshed."""
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.shared_state.cumulative_gain_validated = 0.0
    coord.shared_state.last_profile_workload = {}
    coord.sub = _StubSub(coord.shared_state, landed_tput=100.0)

    await coord._maybe_reprofile_for_kernel()

    assert len(coord.sub.tasks_run) == 1


@pytest.mark.asyncio
async def test_kernel_entry_reprofiles_when_backend_context_changes(coord: Coordinator):
    """A backend/env change invalidates the prior trace even when throughput is unchanged."""
    state = coord.shared_state
    state.framework = "vllm"
    state.precision = "fp8"
    state.model_path = "/models/qwen"
    state.tp = 1
    state.conc = 64
    state.isl = 1024
    state.osl = 1024
    state.max_model_len = 4096
    state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    state.cumulative_gain_validated = 0.0
    state.last_profile_workload = state.profile_workload_context(
        {"base_extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "0"}}
    )
    state.current_best = {
        "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
    }
    coord.sub = _StubSub(state, landed_tput=100.0)

    await coord._maybe_reprofile_for_kernel()

    assert len(coord.sub.tasks_run) == 1
    assert coord.sub.tasks_run[0].params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER_LINEAR": "1"}


@pytest.mark.asyncio
async def test_kernel_entry_reprofile_runs_without_measured_trace(coord: Coordinator):
    """No measured trace yet (no snapshot) but a non-zero projected gain still reprofiles so GEAK gets a real trace."""
    coord.shared_state.roofline_snapshots = []
    coord.sub = _StubSub(coord.shared_state, landed_tput=150.0)
    coord.shared_state.cumulative_gain_validated = 50.0  # cur = 150

    await coord._maybe_reprofile_for_kernel()

    assert len(coord.sub.tasks_run) == 1
    assert coord.shared_state.last_roofline_tput == 150.0


@pytest.mark.asyncio
async def test_kernel_entry_reprofile_swallows_failure(coord: Coordinator):
    """A reprofile failure is best-effort: it never propagates and the anchor is left untouched."""

    class _RaisingSub:
        async def run_task(self, _task: Any, **_kwargs: Any) -> None:
            raise RuntimeError("profile crashed")

    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.sub = _RaisingSub()
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 20.0  # cur = 120 != measured 100 → triggers

    await coord._maybe_reprofile_for_kernel()  # must not raise

    assert coord.shared_state.last_roofline_tput == 100.0


@pytest.mark.asyncio
async def test_kernel_entry_reprofiles_when_workload_changed_at_same_tput(
    coord: Coordinator,
):
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.shared_state.conc = 64
    coord.shared_state.last_profile_status = "succeeded"
    coord.shared_state.last_profile_workload = coord.shared_state.profile_workload_context()
    coord.shared_state.conc = 128
    coord.sub = _StubSub(coord.shared_state, landed_tput=100.0)

    await coord._maybe_reprofile_for_kernel()

    assert len(coord.sub.tasks_run) == 1
    assert coord.shared_state.last_profile_workload["conc"] == 128


@pytest.mark.asyncio
async def test_kernel_entry_reprofiles_when_backend_config_changed_at_same_tput(
    coord: Coordinator,
):
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    # The latest trace was profiled under a different backend (triton), recorded
    # in last_profile_workload['serving_config'] exactly as the roofline executor
    # writes it. last_profile_args stays the plain-args field it is elsewhere.
    coord.shared_state.last_profile_status = "succeeded"
    coord.shared_state.last_profile_workload = {
        "framework": "sglang",
        "precision": "fp8",
        "model_path": "/models/qwen",
        "tp": 1,
        "conc": 64,
        "isl": 1024,
        "osl": 1024,
        "max_model_len": 4096,
        "serving_config": {
            "engine": "",
            "extra_server_args": "--attention-backend triton",
            "extra_envs": {},
        },
    }
    coord.shared_state.current_best = {
        "extra_server_args": "--attention-backend aiter",
        "extra_envs": {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"},
    }
    coord.sub = _StubSub(coord.shared_state, landed_tput=100.0)

    await coord._maybe_reprofile_for_kernel()

    assert len(coord.sub.tasks_run) == 1
    # After the reprofile the recorded workload reflects the current config
    # (aiter + the new env), so the next entry sees no config change.
    assert (
        coord.shared_state.last_profile_workload["serving_config"]["extra_envs"]["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"]
        == "256"
    )

    coord.sub = _StubSub(coord.shared_state)
    await coord._maybe_reprofile_for_kernel()
    assert coord.sub.tasks_run == []
