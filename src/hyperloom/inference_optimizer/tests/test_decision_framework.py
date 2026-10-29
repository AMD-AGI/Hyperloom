# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision-framework regression tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.kernel import _forge_lane_budget as forge_lane_budget


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Stub the interpreter resolver to avoid a real Magpie import probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from hyperloom.orchestrator.actions.executors import _benchmark_interpreter

    monkeypatch.setattr(
        _benchmark_interpreter,
        "_resolve_magpie_python",
        lambda _env=None: "/usr/bin/python3",
    )
    return make_session_dir()


@pytest.mark.asyncio
async def test_run_gemm_tuning_response_records_to_shared_state(
    session_dir,
):
    """``run_gemm_tuning`` is a Coordinator-owned lane the model can no longer REQUEST (the intent path denies it), so the recording is exercised on the live entrypoint every dispatch converges on: ``_handle_gemm_tuning_result`` records the result and persists the state."""
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
                "backend": "forge",
                "candidates": [
                    {
                        "tuner": "a8w8_blockscale_tuned_gemm",
                        "env": {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tmp/a8w8_blockscale_tuned_gemm.csv"},
                    }
                ],
            }
        )

        # The E2E validator rewrites the stored result to its measured outcome, and the history keeps exactly one row
        # for the one dispatch.
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
async def test_a_gemm_lane_the_phase_could_not_fund_is_no_attempt_and_stays_armed(session_dir):
    """The lane never started, so it neither counts as an attempt nor latches GEMM tuning off for a later entry."""
    from hyperloom.orchestrator.kernel.lane_budget import LANE_GEMM

    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.framework = "sglang"

        await c.phase_kernel._handle_gemm_tuning_result(forge_lane_budget._unfunded_lane_skip(LANE_GEMM))

        assert c.shared_state.last_gemm_tuning == {}
        assert c.shared_state.gemm_tuning_attempts == []
        assert c.phase_kernel._gemm_tuning_required_before_kernel_opt()
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

        assert c.dispatcher.admission_denial_for_request("kernel_agent", "run_optimization") is None
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
        c.shared_state.baseline_runtime_sec = 600.0
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
                "backend": "forge",
                "candidates": [
                    {
                        "tuner": "a8w8_blockscale_tuned_gemm",
                        "env": {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(tuned)},
                        "best_micro_speedup": 1.28,
                    }
                ],
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

        await c.phase_kernel.run_agent(SimpleNamespace(task=SimpleNamespace(params={"from_phase": "FRAMEWORK_AGENT"})))

        assert calls
        assert c.shared_state.gemm_tuning_attempts
        assert c.shared_state.current_best["action"] == "gemm_tuning"
        # The measured rebench, not baseline_tput * best_speedup (1024.0).
        assert c.shared_state.current_best["tput"] == 900.0
        assert c.shared_state.cumulative_gain_validated == pytest.approx(12.5)
        assert c.shared_state.optimization_stack[-1]["action"] == "gemm_tuning"
        assert c.phase_kernel._gemm_tuning_required_before_kernel_opt() is False
    finally:
        await c.stop()
