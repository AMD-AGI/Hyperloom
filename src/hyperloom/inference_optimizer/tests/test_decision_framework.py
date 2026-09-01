# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision-framework regression tests.

Covers ``_handle_request`` mirroring ``run_optimization`` / ``run_gemm_tuning``
results into ``shared_state``, the native-source guards and batch selection in
the run_optimization handler, and ``record_kernel_opt`` retiring kernels stuck
in PARTIAL.
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
async def test_run_optimization_response_records_to_shared_state(
    session_dir,
    monkeypatch,
):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k006"],
        }
        c.shared_state.save(session_dir)
        # Stub the handler so we don't shell out.
        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        async def fake(payload, *, session_dir, **kwargs):
            return {
                "status": "ok",
                "kernel_id": "k006",
                "proposal": {"decision": "PARTIAL", "reasons": ["correctness evidence missing"]},
                "verification": {
                    "compile_passed": True,
                    "correctness_passed": False,
                    "micro_speedup": 1.0,
                    "best_artifact_path": "/tmp/v3.hip",
                },
            }

        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "run_optimization",
            fake,
        )
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "run_optimization", "params": {"kernel_id": "k006"}},
        )
        await c._handle_intent("orchestration", intent)
        # SharedState gained a last_kernel_opt entry.
        ko = c.shared_state.last_kernel_opt
        assert ko["kernel_id"] == "k006"
        assert ko["decision"] == "PARTIAL"
        assert ko["compile_passed"] is True
        assert ko["correctness_passed"] is False
        assert ko["best_artifact_path"] == "/tmp/v3.hip"
        # to_prompt_summary surfaces it so Orch sees the outcome.
        summary = c.shared_state.to_prompt_summary()
        assert "kernel_id=k006" in summary and "decision=PARTIAL" in summary
        # State persisted across reload.
        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.last_kernel_opt["kernel_id"] == "k006"
    finally:
        await c.stop()


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
    monkeypatch,
):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
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

        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        async def fake(payload, *, session_dir):
            return {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
            }

        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "run_gemm_tuning",
            fake,
        )
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
            ),
        )

        # The E2E validator rewrites the stored result to its measured outcome.
        assert c.shared_state.last_gemm_tuning["status"] == "complete"
        assert c.shared_state.last_gemm_tuning["best_speedup"] == 1.2
        assert c.shared_state.gemm_tuning_attempts
        assert "last_gemm_tuning=" in c.shared_state.to_prompt_summary()
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


@pytest.mark.asyncio
async def test_kernel_entry_continues_to_kernel_opt_after_gemm(
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
        c.shared_state.auto_kernel_opt_enabled = True
        c.shared_state.untried_hot_reusable_kernels = lambda: ["k001"]  # type: ignore[method-assign]

        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        async def fake_gemm(payload, *, session_dir):
            return {
                "status": "complete",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "tuned_file": "/tmp/tuned.csv",
            }

        kernel_calls: list[dict] = []

        async def fake_kernel(payload, *, session_dir, record_partial=None):
            kernel_calls.append(dict(payload))
            return {"status": "failed", "batch_mode": True, "batch_results": []}

        monkeypatch.setattr(kernel_request_handlers, "run_gemm_tuning_handler", fake_gemm)
        monkeypatch.setattr(kernel_request_handlers, "run_optimization_handler", fake_kernel)

        await c._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

        assert kernel_calls
        assert kernel_calls[0]["candidates_path"] == "/tmp/candidates.json"
    finally:
        await c.stop()


# D — native-only guard for kernel optimization handler
@pytest.mark.asyncio
async def test_run_optimization_handler_rejects_compile_generated_source(session_dir):
    from hyperloom.orchestrator.kernel.request_handlers import (
        run_optimization_handler,
    )

    result = await run_optimization_handler(
        {
            "kernel_id": "k_inductor",
            "kernel_name": "triton_poi_fused_add_mul_0",
            "source_file": "/tmp/torchinductor_root/ab/cdef.py",
            "dry_run": True,
        },
        session_dir=session_dir,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "runtime_generated_kernel"
    assert "not be reusable" in result["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_rejects_missing_native_source(session_dir):
    from hyperloom.orchestrator.kernel.request_handlers import (
        run_optimization_handler,
    )

    result = await run_optimization_handler(
        {"kernel_id": "k_unknown", "dry_run": True},
        session_dir=session_dir,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_native_source"


@pytest.mark.asyncio
async def test_run_optimization_handler_uses_candidates_path_native_guard(
    session_dir,
    tmp_path,
):
    from hyperloom.orchestrator.kernel.request_handlers import (
        run_optimization_handler,
    )

    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text(
        """
        {
          "hot_kernels": [
            {
              "kernel_id": "k001",
              "name": "triton_red_fused_sum_0",
              "source_file": "/tmp/torchinductor_root/xy/generated.py",
              "reusable_native_kernel": false,
              "optimization_notes": "runtime-generated compile kernel"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    result = await run_optimization_handler(
        {
            "kernel_id": "k001",
            "candidates_path": str(candidates),
            "dry_run": True,
        },
        session_dir=session_dir,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "non_reusable_kernel"
    assert "runtime-generated" in result["reason"]


@pytest.mark.asyncio
async def test_run_optimization_handler_forwards_verification_evidence(
    session_dir,
    tmp_path,
    monkeypatch,
):
    from hyperloom.orchestrator.kernel import request_handlers as krh

    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text(
        """
        {
          "hot_kernels": [
            {
              "kernel_id": "k006",
              "name": "aiter_native_kernel",
              "source_file": "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
              "reusable_native_kernel": true,
              "gpu_pct": 12.5
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    seen = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        seen["cmd"] = cmd
        return 0, '{"status":"ok"}', ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    await krh.run_optimization_handler(
        {
            "kernel_id": "k006",
            "candidates_path": str(candidates),
            "micro_speedup": 1.25,
            "e2e_gain_pct": 0.7,
            "correctness_passed": True,
            "accuracy_passed": True,
            "dry_run": True,
        },
        session_dir=session_dir,
    )
    cmd = seen["cmd"]
    assert "--correctness-passed" in cmd
    assert cmd[cmd.index("--correctness-passed") + 1] == "true"
    assert "--accuracy-passed" in cmd
    assert cmd[cmd.index("--accuracy-passed") + 1] == "true"
    assert "--micro-speedup" in cmd
    assert "--e2e-gain-pct" in cmd


@pytest.mark.asyncio
async def test_run_optimization_handler_batches_reusable_kernels_and_selects_best(
    session_dir,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    from hyperloom.orchestrator.kernel import request_handlers as krh

    # Disable the default min_gpu_pct gate so the test focuses on the batch best-selection.
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "0.0")
    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text(
        """
        {
          "hot_kernels": [
            {
              "kernel_id": "k003",
              "name": "native_moe_gemm",
              "source_file": "/sgl-workspace/aiter/csrc/kernels/moe.cu",
              "reusable_native_kernel": true
            },
            {
              "kernel_id": "k006",
              "name": "native_quant",
              "source_file": "/sgl-workspace/aiter/csrc/kernels/quant_kernels.cu",
              "reusable_native_kernel": true
            }
          ],
          "reusable_native_kernel_ids": ["k003", "k006"]
        }
        """,
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    async def fake_single(payload, *, session_dir, timeout_override_sec=None):
        kernel_id = payload["kernel_id"]
        backend = payload["backends"]
        calls.append((kernel_id, backend))
        # k006 wins on forge (KEEP); k003 stays PARTIAL so batch best-selection picks k006.
        keep = kernel_id == "k006" and backend == "forge"
        speedup = 1.31 if keep else 1.0
        return {
            "status": "ok",
            "kernel_id": kernel_id,
            "selected_backends": [backend],
            "best_artifact_path": f"/tmp/{kernel_id}-{backend}.cu",
            "verification": {
                "compile_passed": keep,
                "correctness_passed": keep,
                "micro_speedup": speedup,
                "best_artifact_path": f"/tmp/{kernel_id}-{backend}.cu",
            },
            "proposal": {
                "decision": "KEEP" if keep else "PARTIAL",
                "reasons": [],
            },
        }

    monkeypatch.setattr(krh, "_run_optimization_single", fake_single)
    result = await krh.run_optimization_handler(
        {
            "candidates_path": str(candidates),
            "backend_order": "forge",
            "budget_minutes": 60,
            "max_parallel": 2,
        },
        session_dir=session_dir,
    )

    assert result["batch_mode"] is True
    assert result["batch_kernel_ids"] == ["k003", "k006"]
    assert result["backend_order"] == ["forge"]
    assert result["kernel_id"] == "k006"
    assert result["proposal"]["decision"] == "KEEP"
    assert result["verification"]["micro_speedup"] == pytest.approx(1.31)

    by_kernel: dict[str, list[str]] = {}
    for kernel_id, backend in calls:
        by_kernel.setdefault(kernel_id, []).append(backend)
    assert by_kernel["k003"] == ["forge"]
    assert by_kernel["k006"] == ["forge"]


# E — record_kernel_opt retires kernels stuck in PARTIAL.
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


def test_record_kernel_opt_first_partial_does_not_retire():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    assert "k001" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["partial_count"] == 1
    assert state.kernel_opt_attempts["k001"]["attempts"] == 1
    assert "rejected_reason" not in state.kernel_opt_attempts["k001"]


def test_record_kernel_opt_retires_after_max_partial_attempts():
    state = SharedState()
    # Default max_partial = 2 → second PARTIAL must retire.
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    assert "k001" in state.rejected_kernel_ids
    entry = state.kernel_opt_attempts["k001"]
    assert entry["partial_count"] == 2
    assert entry["attempts"] == 2
    assert "max_partial_attempts_2_without_keep" == entry["rejected_reason"]


def test_record_kernel_opt_revert_retires_immediately():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k002", decision="REVERT"))
    assert "k002" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k002"]["rejected_reason"] == "revert_decision"
    # partial_count is not bumped for REVERT (it's a different terminal state).
    assert state.kernel_opt_attempts["k002"].get("partial_count", 0) == 0


def test_record_kernel_opt_keep_resets_partial_streak():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k003"))
    assert state.kernel_opt_attempts["k003"]["partial_count"] == 1
    keep = _partial_kernel_opt_result("k003", decision="KEEP")
    state.record_kernel_opt(keep)
    assert state.kernel_opt_attempts["k003"]["partial_count"] == 0
    assert "k003" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k003"]["last_decision"] == "KEEP"


def test_record_kernel_opt_max_partial_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL", "3")
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    assert "k004" not in state.rejected_kernel_ids
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    assert "k004" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k004"]["rejected_reason"] == "max_partial_attempts_3_without_keep"


def test_record_kernel_opt_history_capped_at_ten():
    state = SharedState()
    for _ in range(15):
        state.record_kernel_opt(_partial_kernel_opt_result("k005"))
    history = state.kernel_opt_attempts["k005"]["history"]
    assert len(history) == 10
    assert state.kernel_opt_attempts["k005"]["attempts"] == 15
    assert state.kernel_opt_attempts["k005"]["partial_count"] == 15
    assert "k005" in state.rejected_kernel_ids


def test_record_kernel_opt_persists_attempts_across_reload(tmp_path):
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k006"))
    state.save(tmp_path)
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.kernel_opt_attempts["k006"]["partial_count"] == 1
    assert reloaded.kernel_opt_attempts["k006"]["attempts"] == 1


def test_record_kernel_opt_prompt_summary_surfaces_history():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k007"))
    state.record_kernel_opt(_partial_kernel_opt_result("k007"))
    summary = state.to_prompt_summary()
    assert "kernel_id=k007" in summary
    assert "history=attempts=2/partial=2" in summary
    assert "retired=max_partial_attempts_2_without_keep" in summary
