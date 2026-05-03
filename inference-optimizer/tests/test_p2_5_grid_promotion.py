"""P2-5 grid-runner promotion tests.

Locks the post-1h-validation fixes:

1. ``Coordinator._materialize_approved_proposal`` injects ``base_tput`` +
   ``base_extra_args`` into task.params for backends/params/sweep so the
   grid-runner runner can compute a real ``best_gain_pct`` against
   current_best (not against 0.0, which the 1h validation hit).
2. ``Coordinator._promote_to_shared_state`` lifts a winning grid result
   into ``SharedState.current_best`` + recomputes ``cumulative_gain``
   when the new tput beats the current best by ≥ 1% (per marathon's
   KEEP threshold).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator, PendingProposal
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
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
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_ROOT", str(tmp_path))
    return make_session_dir("p2-5-test")


# ===========================================================================
# _materialize_approved_proposal injects base_tput
# ===========================================================================
@pytest.mark.asyncio
async def test_materialize_injects_base_tput_for_backends(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 820.0,
                                         "extra_sglang_args": "--foo bar"}
        c.shared_state.save(session_dir)
        pending = PendingProposal(
            proposal_msg_id="prop-backends-1", from_agent="orchestration",
            action_name="backends", predicted_gain_pct=5.0,
            payload={"action_name": "backends",
                     "predicted_gain_pct": 5.0,
                     "params": {"variant_timeout_sec": 600}},
        )
        await c._materialize_approved_proposal(pending)
        # Find the created task by idempotency_key
        tasks = await c.tasks.queued()
        bench_tasks = [t for t in tasks if t.kind == "backends"]
        assert bench_tasks, "expected a queued backends task"
        params = bench_tasks[0].params
        assert params["base_tput"] == 820.0  # current_best wins over baseline
        assert params["base_extra_args"] == "--foo bar"
        assert params["variant_timeout_sec"] == 600  # original kept
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_materialize_falls_back_to_baseline_when_no_current_best(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 750.0
        c.shared_state.current_best = {}  # nothing yet
        c.shared_state.save(session_dir)
        pending = PendingProposal(
            proposal_msg_id="prop-params-1", from_agent="orchestration",
            action_name="params", predicted_gain_pct=3.0,
            payload={"action_name": "params"},
        )
        await c._materialize_approved_proposal(pending)
        tasks = [t for t in await c.tasks.queued() if t.kind == "params"]
        assert tasks
        assert tasks[0].params["base_tput"] == 750.0
        assert tasks[0].params["base_extra_args"] == ""
        assert tasks[0].params["max_candidates_per_round"] == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_materialize_does_not_inject_for_non_grid_actions(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.save(session_dir)
        pending = PendingProposal(
            proposal_msg_id="prop-baseline-1", from_agent="orchestration",
            action_name="baseline", predicted_gain_pct=0.0,
            payload={"action_name": "baseline"},
        )
        await c._materialize_approved_proposal(pending)
        tasks = [t for t in await c.tasks.queued() if t.kind == "baseline"]
        assert tasks
        # baseline isn't a grid runner — no base_tput injection
        assert "base_tput" not in tasks[0].params
    finally:
        await c.stop()


# ===========================================================================
# _promote_to_shared_state for backends/params/sweep wins
# ===========================================================================
@pytest.mark.asyncio
async def test_promote_backends_winner_updates_current_best_and_gain(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        c.shared_state.cumulative_gain = 0.0
        c.shared_state.save(session_dir)

        result = {
            "status": "succeeded",
            "output_throughput": 880.0,  # +10% over baseline
            "best_variant": {
                "name": "attn_triton",
                "extra_sglang_args": "--attention-backend triton",
                "ttft_mean_ms": 130.0,
                "e2el_mean_ms": 2300.0,
                "workspace": "/tmp/workspaces/winner",
            },
        }
        await c._promote_to_shared_state("backends", result)
        assert c.shared_state.cumulative_gain == pytest.approx(10.0)
        cb = c.shared_state.current_best
        assert cb["action"] == "backends"
        assert cb["tput"] == 880.0
        assert cb["variant_name"] == "attn_triton"
        assert cb["extra_sglang_args"] == "--attention-backend triton"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_backends_does_not_overwrite_when_below_threshold(session_dir):
    """Improvement < 0.5% (P5 relaxed threshold; marathon was 1.0%) and
    not yet a consistent winner across rounds → current_best stays."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        c.shared_state.cumulative_gain = 0.0
        c.shared_state.save(session_dir)

        # +0.25% — below the 0.5% single-round KEEP bar (P5)
        result = {
            "status": "succeeded",
            "output_throughput": 802.0,
            "best_variant": {"name": "attn_aiter"},
        }
        await c._promote_to_shared_state("backends", result)
        assert c.shared_state.cumulative_gain == 0.0
        assert c.shared_state.current_best["action"] == "baseline"  # unchanged
        # A single sub-threshold winner shouldn't trigger the cross-round
        # promote either (needs ≥ 2 of 3 rounds for the same variant).
        assert c.shared_state.params_no_promote_streak == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_params_chains_on_top_of_backends_winner(session_dir):
    """A 2nd winning grid round (params on top of backends) keeps stacking."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        # Pretend backends already won — current_best is now the +5% winner.
        c.shared_state.current_best = {"action": "backends", "tput": 840.0,
                                         "extra_sglang_args": "--attention-backend triton"}
        c.shared_state.cumulative_gain = 5.0
        c.shared_state.save(session_dir)

        result = {
            "status": "succeeded",
            "output_throughput": 880.0,  # +5% over backends winner = +10% over baseline
            "best_variant": {
                "name": "decode_steps_8",
                "extra_sglang_args": "--num-continuous-decode-steps 8",
            },
        }
        await c._promote_to_shared_state("params", result)
        # cumulative_gain is computed against ORIGINAL baseline_tput
        assert c.shared_state.cumulative_gain == pytest.approx(10.0)
        assert c.shared_state.current_best["action"] == "params"
        assert c.shared_state.current_best["tput"] == 880.0
        assert c.shared_state.current_best["extra_sglang_args"] == (
            "--attention-backend triton --num-continuous-decode-steps 8"
        )
        stack = c.shared_state.optimization_stack
        assert [e["variant_name"] for e in stack] == [
            "legacy_current_best",
            "decode_steps_8",
        ]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_handles_missing_best_variant_gracefully(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        c.shared_state.save(session_dir)
        result = {
            "status": "no_winners",
            "output_throughput": None,  # nothing to promote
            "best_variant": None,
        }
        await c._promote_to_shared_state("backends", result)
        assert c.shared_state.current_best["action"] == "baseline"
        assert c.shared_state.cumulative_gain == 0.0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_sweep_result_records_last_sweep_without_overwriting_best(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {
            "action": "params",
            "tput": 840.0,
            "extra_sglang_args": "--cuda-graph-max-bs 8",
        }
        result = {
            "status": "succeeded",
            "grid_size": 2,
            "sweep_grid": [
                {"name": "conc4_isl256_osl256", "status": "succeeded",
                 "conc": 4, "isl": 256, "osl": 256,
                 "output_throughput": 500.0, "e2el_mean_ms": 100.0},
                {"name": "conc16_isl1024_osl1024", "status": "succeeded",
                 "conc": 16, "isl": 1024, "osl": 1024,
                 "output_throughput": 900.0, "e2el_mean_ms": 300.0},
            ],
            "pareto_front": [],
            "best_for_each_conc": {},
            "workspace": "/tmp/sweep",
        }
        await c._promote_to_shared_state("sweep", result)
        assert c.shared_state.current_best["action"] == "params"
        assert c.shared_state.current_best["tput"] == 840.0
        assert c.shared_state.last_sweep["grid_size"] == 2
        assert c.shared_state.last_sweep["best_overall"]["name"] == \
            "conc16_isl1024_osl1024"
        assert "last_sweep=grid_size=2" in c.shared_state.to_prompt_summary()
    finally:
        await c.stop()
