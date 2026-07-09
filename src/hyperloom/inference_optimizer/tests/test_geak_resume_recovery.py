# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for crash-safe GEAK handback recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.machine_state import ESCALATE_HINT_SKIP_TO_SWEEP
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task


class _TaskRegistry:
    def __init__(self) -> None:
        self.created: list[Task] = []

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
    ) -> tuple[Task, bool]:
        task = Task(
            task_id=task_id or f"task-{len(self.created)}",
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self.created.append(task)
        return task, False


@pytest.mark.asyncio
async def test_geak_kernel_phase_recovers_existing_ok_result_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated result written before a coordinator crash must be recovered."""
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "throughput_speedup": 1.16,
        "final_throughput_tok_s": 116.0,
        "ttft_ms": 10.0,
        "tpot_ms": 2.0,
        "eval_dir": str(geak_dir / "final"),
        "report_path": str(geak_dir / "final" / "architect_report.md"),
        "bench_script": str(geak_dir / "final" / "bench_e2e.sh"),
        "accepted_config": {"flags": "--max-num-batched-tokens 16384", "env": "E=1"},
        "accepted_kernels": ["fused_moe_kernel_gptq_awq"],
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
        model_path="/models/kimi",
        gpu_type="mi300x",
        isl=8192,
        osl=1024,
        conc=64,
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None

    def _runner_should_not_be_needed(_name: str) -> Path:
        raise RuntimeError("runner should not be resolved when result.json exists")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        _runner_should_not_be_needed,
    )

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    assert coord.shared_state.geak_result["status"] == "ok"
    assert coord.shared_state.current_best["action"] == "geak_e2e"
    assert coord.shared_state.current_best["tput"] == 116.0
    assert coord.shared_state.cumulative_gain == pytest.approx(16.0)
    assert coord.shared_state.optimization_stack[0]["action"] == "geak_e2e"
    assert coord.shared_state.pending_escalate_hint == ESCALATE_HINT_SKIP_TO_SWEEP

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["geak_result"]["status"] == "ok"
    assert saved["cumulative_gain"] == pytest.approx(16.0)

    coord.tasks = _TaskRegistry()
    task = await coord._enqueue_internal_sweep_task(reason="phase_entry")

    assert task.kind == "sweep"
    assert task.params["geak_result"]["status"] == "ok"
    assert task.params["geak_result"]["bench_script"].endswith("bench_e2e.sh")


@pytest.mark.asyncio
async def test_geak_kernel_phase_does_not_reuse_already_promoted_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new cycle must rerun GEAK, not promote a stale prior-cycle result.

    ``geak/`` is a fixed path, so a prior cycle's ``result.json`` survives
    into the next KERNEL entry. When state already recorded that win the recovery
    short-circuit must NOT fire, otherwise every later cycle silently reuses the
    first cycle's result.
    """
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "bench_script": str(geak_dir / "bench_e2e.sh"),
        "accepted_config": {"flags": "", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "geak_e2e", "tput": 116.0},
        model_path="/models/kimi",
        gpu_type="mi300x",
        isl=8192,
        osl=1024,
        conc=64,
    )
    # State already carries the prior cycle's promoted win.
    coord.shared_state.optimization_stack = [
        {"action": "geak_e2e", "variant_name": "geak_e2e", "tput": 116.0},
    ]
    coord.shared_state.geak_result = dict(result)

    resolved: list[str] = []

    def _runner_resolved(name: str) -> Path:
        resolved.append(name)
        raise RuntimeError("stop before launching subprocess")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        _runner_resolved,
    )

    await coord._run_geak_kernel_phase(from_phase="EXPLORE")

    # The recovery short-circuit must NOT have fired; the normal path resolves
    # the runner (and here aborts via the injected error).
    assert resolved, "new cycle must re-run GEAK, not reuse stale result.json"
