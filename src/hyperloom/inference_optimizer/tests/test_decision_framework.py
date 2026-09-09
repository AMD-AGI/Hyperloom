# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision-framework regression tests.

Covers the Coordinator-owned kernel and gemm lanes mirroring their results into
``shared_state``, the native-source guards and batch selection in the
run_optimization handler, and the retirement of kernels stuck in
PARTIAL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    # Pin the kernel-agent root so request handlers resolve from disk.
    kernel_agent_root = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    # Stub the interpreter resolver to avoid a real Magpie import probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from hyperloom.orchestrator.actions.executors import _grid_runner

    monkeypatch.setattr(
        _grid_runner,
        "_resolve_magpie_python",
        lambda: "/usr/bin/python3",
    )
    return make_session_dir()


# C — kernel-opt response recorded to SharedState
@pytest.mark.asyncio
async def test_trace_analyze_does_not_record_kernel_opt(
    session_dir,
    monkeypatch,
):
    """Only run_optimization (not trace_analyze) writes to last_kernel_opt."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        async def fake(payload, *, session_dir):
            return {"status": "ok", "hot_kernels": [{"kernel_id": "k001"}]}

        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "trace_analyze",
            fake,
        )
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "trace_analyze", "params": {"trace_input": "/tmp/t.json"}},
        )
        await c._handle_intent("orchestration", intent)
        assert c.shared_state.last_kernel_opt == {}
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_gemm_tuning_response_records_to_shared_state(
    session_dir,
):
    """``run_gemm_tuning`` is a Coordinator-owned lane the model can no longer
    REQUEST (the intent path denies it), so the recording is exercised on the
    live entrypoint every dispatch converges on: ``_handle_gemm_tuning_result``
    records the result and persists the state."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.kernel_optimizer = "native"
        c.shared_state.precision = "fp8"
        c.shared_state.framework = "sglang"
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "candidates_path": "/tmp/candidates.json",
        }
        c.shared_state.save(session_dir)

        await c.phase_kernel._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
            }
        )

        # The E2E validator rewrites the stored result to its measured outcome,
        # and the history keeps exactly one row for the one dispatch.
        last = c.shared_state.last_gemm_tuning
        assert last["status"] == "complete"
        assert last["best_speedup"] == 1.2
        assert last["e2e_validated"] is True
        assert c.shared_state.gemm_tuning_attempts == [last]
        assert "last_gemm_tuning=" in c.shared_state.to_prompt_summary()
        # State persisted across reload.
        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.last_gemm_tuning["status"] == "complete"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_optimization_no_longer_gated_on_fp8_gemm_tuning(session_dir):
    """The gemm-before-run_optimization sequence deny was removed; the request-layer pre-deny no longer fires."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.kernel_optimizer = "native"
        c.shared_state.precision = "fp8"
        c.shared_state.framework = "sglang"
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "candidates_path": "/tmp/candidates.json",
        }

        assert c._sequence_denial_for_request("kernel_agent", "run_optimization") is None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_kernel_entry_auto_runs_gemm_tuning_for_fp8_sglang(
    session_dir,
    monkeypatch,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.kernel_optimizer = "native"
        c.shared_state.precision = "fp8"
        c.shared_state.framework = "sglang"
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "candidates_path": "/tmp/candidates.json",
        }
        c.shared_state.auto_kernel_opt_enabled = False
        calls: list[dict] = []
        tuned = session_dir / "tuned.csv"
        tuned.write_text("M,N,K,kernelId\n16,512,7168,3\n", encoding="utf-8")

        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        async def fake_handler(payload, *, session_dir):
            calls.append(dict(payload))
            return {
                "status": "complete",
                "decision": "KEEP",
                "best_speedup": 1.28,
                "tuned_file": str(tuned),
            }

        async def fake_integrate(payload, *, session_dir):
            return {"decision": "KEEP", "new_tput": 900.0, "gain_pct": 12.5}

        monkeypatch.setattr(
            kernel_request_handlers,
            "run_gemm_tuning_handler",
            fake_handler,
        )
        monkeypatch.setattr(kernel_request_handlers, "integrate_handler", fake_integrate)
        monkeypatch.setattr(
            KernelPhase,
            "_merge_gemm_candidate_with_runtime",
            lambda _self, _env_var, env_value: env_value,
        )

        await c._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

        assert calls
        assert c.shared_state.gemm_tuning_attempts
        assert c.shared_state.current_best["action"] == "gemm_tuning"
        # The measured rebench, not baseline_tput * best_speedup (1024.0).
        assert c.shared_state.current_best["tput"] == 900.0
        assert c.shared_state.cumulative_gain_validated == pytest.approx(12.5)
        assert c.shared_state.optimization_stack[-1]["action"] == "gemm_tuning"
        assert c._gemm_tuning_required_before_kernel_opt() is False
    finally:
        await c.stop()


# D — native-only guard for kernel optimization handler
def _partial_kernel_opt_result(kernel_id: str, decision: str = "PARTIAL") -> dict[str, Any]:
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "proposal": {"decision": decision, "reasons": ["no measurable speedup found"]},
        "verification": {
            "compile_passed": True,
            "correctness_passed": False,
            "micro_speedup": 1.0,
            "best_artifact_path": f"/tmp/{kernel_id}.hip",
        },
    }
