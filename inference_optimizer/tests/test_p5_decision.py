"""P5 decision-framework regression tests.

Locks the fixes to the resume5 9h finding: Orch was stuck in a 38-round
params loop emitting the same select_kernels REQUEST over and over,
never advancing to run_optimization, while 35/38 winning variants beat
the current_best by 0.3–0.84% but never crossed the 1.0% promote bar.

Three behaviours are guarded:

A. ``_promote_to_shared_state`` for grid actions uses a 0.1% single-shot
   KEEP threshold (relaxed from the original 1.0% bar through 0.5% to
   today's 0.1%). The 0.1% gate lets sub-noise winners enter the
   optimization stack so they can compound; ``validate_stack`` is the
   final filter. A separate ``consistent_winner`` detector still exists
   for the cross-round signal-vs-noise check (≥ 2 of last 3 rounds with
   avg gain ≥ 0.3%) but is mostly dormant under the current 0.1% bar.
B. ``params_no_promote_streak`` increments on every grid round that
   doesn't promote and resets on promotion. The prompt summary surfaces
   the streak so Orch can switch to kernel-opt at >= 5.
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
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Point HYPERLOOM_KERNEL_AGENT_ROOT at the repo's kernel-agent tree so
    # ``_run_optimization_single`` can resolve tool paths in tests that
    # exercise the handler dispatch instead of monkeypatching the subprocess
    # call. Same convention as test_p2_2_profile_and_handlers.py.
    from inference_optimizer.orchestrator import kernel_request_handlers as krh
    kernel_agent_root = Path(__file__).resolve().parents[2] / "kernel-agent"
    monkeypatch.setattr(krh, "HYPERLOOM_KERNEL_AGENT_ROOT", kernel_agent_root)
    return make_session_dir()


def _baseline_state(c: Coordinator, base: float = 800.0, current: float = 833.6):
    c.shared_state.baseline_tput = base
    c.shared_state.current_best = {"action": "backends", "tput": current}
    c.shared_state.cumulative_gain = (current - base) / base * 100.0
    c.shared_state.params_no_promote_streak = 0
    c.shared_state.params_winner_history = []


# ===========================================================================
# A1 — 0.1% single-shot KEEP threshold (relaxed from 1.0% → 0.5% → 0.1%).
# ===========================================================================
@pytest.mark.asyncio
async def test_promote_at_one_tenth_pct_threshold(session_dir):
    """+0.6% over current_best is far above the 0.1% bar — MUST promote.
    Same behaviour as the historical 0.5%-bar test; kept under the new
    threshold to lock that the bar didn't accidentally tighten."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
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
async def test_no_promote_below_one_tenth_pct(session_dir):
    """+0.04% over current_best is below the 0.1% 1-shot bar AND only one
    round of history exists, so the cross-round detector can't fire
    either — MUST NOT promote, but the streak/history must tick."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        _baseline_state(c, base=800.0, current=833.6)
        result = {
            "status": "succeeded",
            "output_throughput": 833.93,  # +0.04% over 833.6
            "best_variant": {"name": "mem_fraction_0_85",
                             "extra_sglang_args": "--mem-fraction-static 0.85"},
        }
        await c._promote_to_shared_state("params", result)
        # current_best unchanged (still the prior backends winner).
        assert c.shared_state.current_best["action"] == "backends"
        # but the run was recorded to history + streak ticked
        assert len(c.shared_state.params_winner_history) == 1
        assert c.shared_state.params_no_promote_streak == 1
    finally:
        await c.stop()


# ===========================================================================
# A2 — Cross-round consistent winner detector
# ===========================================================================
# The cross-round path lives in ``SharedState.consistent_winner`` and is
# wired into ``_promote_to_shared_state`` as a fallback when the 1-shot
# bar is missed. Under the current ``PROMOTE_THRESHOLD_PCT=0.1`` +
# ``CROSS_ROUND_MIN_AVG_GAIN_PCT=0.3`` combo it cannot trigger through
# the integrated path (any single round with gain >= 0.1% short-circuits
# into 1-shot promote, and no set of rounds with each gain < 0.1% can
# average >= 0.3%). Test the detector directly so its math stays locked
# even though the integration path is currently dormant — this keeps
# the algorithm correct should we ever re-tighten the 1-shot bar.
def test_consistent_winner_detector_returns_dominant_variant(session_dir):
    """3 rounds: decode_steps_8 wins 2 with avg gain 0.42%; the detector
    must return the latest decode_steps_8 record so the caller can lift
    its tput / variant_name into ``current_best``."""
    state = SharedState.load_or_init(session_dir)
    state.push_params_winner(
        action="params", variant_name="decode_steps_8",
        tput=836.93, gain_pct=0.40,
    )
    state.push_params_winner(
        action="params", variant_name="chunked_prefill_64k",
        tput=834.4, gain_pct=0.10,
    )
    state.push_params_winner(
        action="params", variant_name="decode_steps_8",
        tput=837.4, gain_pct=0.45,
    )
    winner = state.consistent_winner(
        lookback=3, min_appearances=2, min_avg_gain_pct=0.3,
    )
    assert winner is not None
    assert winner["variant_name"] == "decode_steps_8"
    # The detector returns the most-recent record for the winning variant
    # so the caller can lift its tput verbatim — not the average across
    # appearances.
    assert winner["tput"] == pytest.approx(837.4, abs=0.01)
    assert winner["gain_pct"] == pytest.approx(0.45, abs=0.01)


def test_consistent_winner_detector_rejects_insufficient_appearances(session_dir):
    """A variant appearing only once cannot win cross-round, even with a
    big single-round gain — that's exactly what the 1-shot bar already
    handles."""
    state = SharedState.load_or_init(session_dir)
    state.push_params_winner(
        action="params", variant_name="decode_steps_8",
        tput=836.93, gain_pct=0.40,
    )
    state.push_params_winner(
        action="params", variant_name="chunked_prefill_64k",
        tput=834.4, gain_pct=0.10,
    )
    assert state.consistent_winner(
        lookback=3, min_appearances=2, min_avg_gain_pct=0.3,
    ) is None


def test_consistent_winner_detector_rejects_below_avg_gain(session_dir):
    """The dominant variant appears twice but its average gain is below
    ``min_avg_gain_pct`` — must NOT promote. Locks the noise floor on
    the cross-round path so it doesn't lift pure jitter."""
    state = SharedState.load_or_init(session_dir)
    state.push_params_winner(
        action="params", variant_name="mem_fraction_0_85",
        tput=834.1, gain_pct=0.05,
    )
    state.push_params_winner(
        action="params", variant_name="mem_fraction_0_85",
        tput=834.3, gain_pct=0.08,
    )
    state.push_params_winner(
        action="params", variant_name="decode_steps_8",
        tput=834.0, gain_pct=0.04,
    )
    assert state.consistent_winner(
        lookback=3, min_appearances=2, min_avg_gain_pct=0.3,
    ) is None


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
        # k006 wins on Claude (second slot in the GEAK-first ladder), which
        # exercises the short-circuit-after-KEEP behaviour without skipping
        # GEAK. k003 keeps producing PARTIAL so it exhausts the full ladder.
        keep = kernel_id == "k006" and backend == "claude"
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
    assert result["backend_order"] == ["geak", "claude", "codex"]
    assert result["kernel_id"] == "k006"
    assert result["proposal"]["decision"] == "KEEP"
    assert result["verification"]["micro_speedup"] == pytest.approx(1.31)

    by_kernel: dict[str, list[str]] = {}
    for kernel_id, backend in calls:
        by_kernel.setdefault(kernel_id, []).append(backend)
    assert by_kernel["k003"] == ["geak", "claude", "codex"]
    assert by_kernel["k006"] == ["geak", "claude"]


# ===========================================================================
# E — record_kernel_opt retires kernels stuck in PARTIAL
# (regression for the r24 custom_allreduce inner-GEAK 401 retry-loop where
# every `kernel_opt` returned PARTIAL and Orch re-dispatched the same
# kernel_id every tick because the prior policy only retired on REVERT)
# ===========================================================================
def _partial_kernel_opt_result(kernel_id: str,
                                decision: str = "PARTIAL") -> dict[str, Any]:
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "proposal": {"decision": decision,
                     "reasons": ["no measurable speedup found"]},
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
    assert (
        state.kernel_opt_attempts["k004"]["rejected_reason"]
        == "max_partial_attempts_3_without_keep"
    )


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
