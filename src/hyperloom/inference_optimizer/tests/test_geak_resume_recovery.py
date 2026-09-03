# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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
    """A result written before a coordinator crash must be recovered on resume.

    Recovery records the win as an unvalidated candidate (into ``geak_result`` +
    ``geak_pending``) and enqueues the main-flow rebench; it does not promote the
    self-reported value into current_best / the gain ledger. The headline is only
    written once the rebench validates it.
    """
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
    coord.tasks = _TaskRegistry()
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

    # The result.json is recovered into state, but as an unvalidated candidate.
    assert coord.shared_state.geak_result["status"] == "ok"
    assert coord.shared_state.geak_pending["status"] == "awaiting_rebench"
    assert coord.shared_state.geak_pending["self_reported_tput"] == 116.0
    # No premature headline: current_best / gain / stack are untouched.
    assert coord.shared_state.current_best["action"] == "baseline"
    assert coord.shared_state.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in coord.shared_state.optimization_stack)
    assert coord.shared_state.pending_escalate_hint == ESCALATE_HINT_SKIP_TO_SWEEP

    # The main-flow rebench was enqueued to validate the recovered candidate.
    rebench = [t for t in coord.tasks.created if (t.params or {}).get("geak_fallback")]
    assert rebench, "recovery must enqueue a geak main-flow rebench"
    assert coord.shared_state.geak_pending["revalidation_task_id"] == rebench[0].task_id

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["geak_result"]["status"] == "ok"
    assert saved["geak_pending"]["status"] == "awaiting_rebench"
    assert saved["geak_pending"]["revalidation_task_id"] == rebench[0].task_id


@pytest.mark.asyncio
async def test_geak_kernel_phase_does_not_reuse_already_promoted_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new cycle must rerun GEAK, not promote a stale prior-cycle result.

    ``geak/`` is a fixed path, so a prior cycle's ``result.json`` survives into
    the next KERNEL entry. When state already recorded that win the recovery
    short-circuit must not fire, else every later cycle silently reuses the first
    cycle's result.
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

    await coord._run_geak_kernel_phase(from_phase="FRAMEWORK_AGENT")

    # The recovery short-circuit must not have fired; the normal path resolves
    # the runner (and here aborts via the injected error).
    assert resolved, "new cycle must re-run GEAK, not reuse stale result.json"


@pytest.mark.asyncio
async def test_geak_handoff_preserves_serving_fidelity_knobs_and_output_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEAK must baseline the same engine Hyperloom measured.

    A missing max-model-len or GPU memory-utilization handoff lets GEAK/e2e
    launch a subtly different vLLM server than the Hyperloom baseline, which
    turns kernel wins into non-reproducible E2E deltas.
    """
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={
            "action": "baseline",
            "tput": 100.0,
            "extra_server_args": "--no-enable-prefix-caching",
        },
        model_path="/models/gpt-oss-120b",
        gpu_type="mi355x",
        isl=1024,
        osl=1024,
        conc=64,
        max_model_len=2248,
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None

    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("GPU_MEMORY_UTILIZATION", "0.9")

    def _runner_resolved(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        _runner_resolved,
    )

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["max_model_len"] == 2248
    assert handoff["mem_fraction"] == pytest.approx(0.9)
    assert handoff["accepted_flags"] == "--no-enable-prefix-caching"
    assert handoff["raw_baseline_tput"] == 100.0
    assert handoff["e2e_metric"] == "output"
