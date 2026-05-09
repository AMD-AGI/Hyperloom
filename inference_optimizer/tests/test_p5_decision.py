"""P5 decision-framework regression tests.

Locks the fixes to the resume5 9h finding: Orch was stuck in a 38-round
params loop emitting the same select_kernels REQUEST over and over,
never advancing to run_optimization, while 35/38 winning variants beat
the current_best by 0.3–0.84% but never crossed the 1.0% promote bar.

Three behaviours are guarded:

A. ``_promote_to_shared_state`` for grid actions now uses a 0.5%
   single-shot KEEP threshold (relaxed from 1.0%) AND additionally
   promotes any consistent winner that wins ≥ 2 of last 3 rounds
   with avg gain ≥ 0.3% — the cross-round signal-vs-noise check.
B. ``params_no_promote_streak`` increments on every grid round that
   doesn't promote and resets on promotion. The prompt summary
   surfaces the streak so Orch can switch to kernel-opt at >= 5.
C. ``_handle_request`` for ``run_optimization`` mirrors the handler's
   result into ``shared_state.last_kernel_opt`` so subsequent Orch
   turns see decision/speedup and don't re-dispatch the same kernel_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ---------------------------------------------------------------------------
def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path))
    return make_session_dir()


def _baseline_state(c: Coordinator, base: float = 800.0, current: float = 833.6):
    c.shared_state.baseline_tput = base
    c.shared_state.current_best = {"action": "backends", "tput": current}
    c.shared_state.cumulative_gain = (current - base) / base * 100.0
    c.shared_state.params_no_promote_streak = 0
    c.shared_state.params_winner_history = []


# ===========================================================================
# A1 — 0.5% threshold (relaxed from 1.0)
# ===========================================================================
@pytest.mark.asyncio
async def test_promote_at_half_pct_threshold(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        # +0.6% over current_best — would NOT have promoted at 1.0%, MUST at 0.5%.
        result = {
            "status": "succeeded",
            "output_throughput": 838.6,  # +0.6% over 833.6
            "best_variant": {"name": "decode_steps_8",
                             "extra_sglang_args": "--num-continuous-decode-steps 8"},
        }
        await c._promote_to_shared_state("params", result)
        cb = c.shared_state.current_best
        assert cb["action"] == "params"
        assert cb["variant_name"] == "decode_steps_8"
        assert c.shared_state.cumulative_gain == pytest.approx(
            (838.6 - 800.0) / 800.0 * 100.0, abs=0.01
        )
        # Streak resets on a real promotion.
        assert c.shared_state.params_no_promote_streak == 0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_no_promote_below_half_pct(session_dir):
    """+0.3% over current_best is below 0.5% AND not yet a consistent
    winner (only 1 round) — must NOT promote."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        result = {
            "status": "succeeded",
            "output_throughput": 836.1,  # +0.3% over 833.6
            "best_variant": {"name": "mem_fraction_0_85",
                             "extra_sglang_args": "--mem-fraction-static 0.85"},
        }
        await c._promote_to_shared_state("params", result)
        # current_best unchanged
        assert c.shared_state.current_best["action"] == "backends"
        # but the run was recorded to history + streak ticked
        assert len(c.shared_state.params_winner_history) == 1
        assert c.shared_state.params_no_promote_streak == 1
    finally:
        await c.stop()


# ===========================================================================
# A2 — Cross-round consistent winner (resume5 9h scenario)
# ===========================================================================
@pytest.mark.asyncio
async def test_consistent_winner_promoted_below_threshold(session_dir):
    """Decode_steps_8 wins 2 of last 3 rounds with avg gain ~0.6% but
    no individual round crosses 0.5%. The cross-round path catches it."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        # Round 1: decode_steps_8 +0.40% — sub-threshold, recorded
        await c._promote_to_shared_state("params", {
            "status": "succeeded",
            "output_throughput": 836.93,
            "best_variant": {"name": "decode_steps_8"},
        })
        # Round 2: chunked_prefill_64k +0.10% — sub-threshold, recorded
        await c._promote_to_shared_state("params", {
            "status": "succeeded",
            "output_throughput": 834.4,
            "best_variant": {"name": "chunked_prefill_64k"},
        })
        # Round 3: decode_steps_8 +0.45% again — still sub-threshold,
        # but NOW decode_steps_8 has 2 of last 3 rounds with avg ~0.42%.
        await c._promote_to_shared_state("params", {
            "status": "succeeded",
            "output_throughput": 837.4,
            "best_variant": {"name": "decode_steps_8"},
        })
        cb = c.shared_state.current_best
        assert cb["variant_name"] == "decode_steps_8"
        assert cb["tput"] == pytest.approx(837.4, abs=0.1)
        assert c.shared_state.params_no_promote_streak == 0
    finally:
        await c.stop()


# ===========================================================================
# B — Plateau counter
# ===========================================================================
@pytest.mark.asyncio
async def test_plateau_counter_increments_on_no_promote(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        for _ in range(7):
            # +0.05% gain — below threshold, alone with no consistent winner
            await c._promote_to_shared_state("params", {
                "status": "succeeded",
                "output_throughput": 834.0,
                "best_variant": {"name": f"variant_{_}"},  # different each time
            })
        assert c.shared_state.params_no_promote_streak == 7
        # And the prompt summary surfaces it so Orch sees the plateau.
        summary = c.shared_state.to_prompt_summary()
        assert "params_no_promote_streak=7" in summary
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_plateau_counter_resets_on_promote(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        # 4 sub-threshold rounds → streak = 4
        for i in range(4):
            await c._promote_to_shared_state("params", {
                "status": "succeeded",
                "output_throughput": 834.0,
                "best_variant": {"name": f"v{i}"},
            })
        assert c.shared_state.params_no_promote_streak == 4
        # Now a real +0.6% promotion → streak resets
        await c._promote_to_shared_state("params", {
            "status": "succeeded",
            "output_throughput": 838.6,
            "best_variant": {"name": "winner"},
        })
        assert c.shared_state.params_no_promote_streak == 0
    finally:
        await c.stop()


# ===========================================================================
# C — kernel-opt response recorded to SharedState
# ===========================================================================
@pytest.mark.asyncio
async def test_run_optimization_response_records_to_shared_state(
    session_dir, monkeypatch,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_select_kernels = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k006"],
        }
        c.shared_state.save(session_dir)
        # Stub the handler so we don't shell out.
        from inference_optimizer.orchestrator import kernel_request_handlers
        async def fake(payload, *, session_dir):
            return {
                "status": "ok",
                "kernel_id": "k006",
                "proposal": {"decision": "PARTIAL",
                             "reasons": ["correctness evidence missing"]},
                "verification": {"compile_passed": True,
                                 "correctness_passed": False,
                                 "micro_speedup": 1.0,
                                 "best_artifact_path": "/tmp/v3.hip"},
            }
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "run_optimization", fake,
        )
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "run_optimization",
                     "params": {"kernel_id": "k006"}},
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
async def test_select_kernels_does_not_record_kernel_opt(
    session_dir, monkeypatch,
):
    """Only run_optimization (not select_kernels) writes to last_kernel_opt."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import kernel_request_handlers
        async def fake(payload, *, session_dir):
            return {"status": "ok", "hot_kernels": [{"kernel_id": "k001"}]}
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "select_kernels", fake,
        )
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "select_kernels",
                     "params": {"trace_input": "/tmp/t.json"}},
        )
        await c._handle_intent("orchestration", intent)
        assert c.shared_state.last_kernel_opt == {}
    finally:
        await c.stop()


# ===========================================================================
# D — native-only guard for kernel optimization handler
# ===========================================================================
@pytest.mark.asyncio
async def test_run_optimization_handler_rejects_compile_generated_source(session_dir):
    from inference_optimizer.orchestrator.kernel_request_handlers import (
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
    from inference_optimizer.orchestrator.kernel_request_handlers import (
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
    session_dir, tmp_path,
):
    from inference_optimizer.orchestrator.kernel_request_handlers import (
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
    session_dir, tmp_path, monkeypatch,
):
    from inference_optimizer.orchestrator import kernel_request_handlers as krh

    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text(
        """
        {
          "hot_kernels": [
            {
              "kernel_id": "k006",
              "name": "aiter_native_kernel",
              "source_file": "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
              "reusable_native_kernel": true
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
async def test_run_optimization_handler_batches_reusable_kernels_with_backend_fallback(
    session_dir, tmp_path, monkeypatch,
):
    from inference_optimizer.orchestrator import kernel_request_handlers as krh

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

    async def fake_single(payload, *, session_dir):
        kernel_id = payload["kernel_id"]
        backend = payload["backends"]
        calls.append((kernel_id, backend))
        keep = kernel_id == "k006" and backend == "codex"
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
            "budget_minutes": 60,
            "max_parallel": 2,
        },
        session_dir=session_dir,
    )

    assert result["batch_mode"] is True
    assert result["batch_kernel_ids"] == ["k003", "k006"]
    assert result["backend_order"] == ["claude", "codex", "geak"]
    assert result["kernel_id"] == "k006"
    assert result["proposal"]["decision"] == "KEEP"
    assert result["verification"]["micro_speedup"] == pytest.approx(1.31)

    by_kernel: dict[str, list[str]] = {}
    for kernel_id, backend in calls:
        by_kernel.setdefault(kernel_id, []).append(backend)
    assert by_kernel["k003"] == ["claude", "codex", "geak"]
    assert by_kernel["k006"] == ["claude", "codex"]
