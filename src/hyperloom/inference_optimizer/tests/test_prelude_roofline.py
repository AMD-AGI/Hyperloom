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

    async def run_task(self, task: Any) -> None:
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

    await coord._on_enter_kernel(from_phase="EXPLORE")

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

    await coord._on_enter_kernel(from_phase="EXPLORE")

    assert fusion_calls == 1


@pytest.mark.asyncio
async def test_kernel_entry_reprofile_skips_when_unchanged(coord: Coordinator):
    """Projected tput matching the last measured trace (cur == measured) skips the reprofile."""
    coord.shared_state.roofline_snapshots = [{"achieved_tok_per_sec": 100.0}]
    coord.sub = _StubSub(coord.shared_state)
    coord.shared_state.cumulative_gain_validated = 0.0  # cur = 100 == measured
    coord.shared_state.last_profile_workload = (
        coord.shared_state.current_profile_workload_context()
    )

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
    assert coord.sub.tasks_run[0].params["base_extra_envs"] == {
        "VLLM_ROCM_USE_AITER_LINEAR": "1"
    }


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
        async def run_task(self, _task: Any) -> None:
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
    coord.shared_state.last_profile_workload = (
        coord.shared_state.profile_workload_context()
    )
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
        coord.shared_state.last_profile_workload["serving_config"]["extra_envs"][
            "SGLANG_FP8_BLOCKSCALE_CK_MAX_M"
        ]
        == "256"
    )

    coord.sub = _StubSub(coord.shared_state)
    await coord._maybe_reprofile_for_kernel()
    assert coord.sub.tasks_run == []
