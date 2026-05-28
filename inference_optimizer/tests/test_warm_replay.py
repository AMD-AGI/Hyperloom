"""GAP 1 — warm-recipe replay tests.

Coverage:

* ``_maybe_enqueue_warm_replay`` skip paths (disabled / already
  attempted / no recipe / low confidence / empty best_config).
* ``_maybe_enqueue_warm_replay`` enqueues with the right params shape
  when a high-confidence T1 / T2 hit is present.
* ``_promote_warm_replay`` decision logic: reproduced → stack push +
  cumulative_gain bump; drift → outcome only; failed → outcome only.
* Resume safety — ``warm_replay_attempted`` persists across the field
  layer so a robustness restart cannot double-spend the replay budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator


# ===========================================================================
# fixtures
# ===========================================================================
@dataclass
class _StubTask:
    task_id: str = "task-warm-1"
    kind: str = "replay_warm_recipe"
    params: dict = field(default_factory=dict)
    state: str = "succeeded"


@dataclass
class _StubSharedState:
    """Minimal SharedState surface ``_maybe_enqueue_warm_replay`` /
    ``_promote_warm_replay`` actually read / write."""

    framework: str = "sglang"
    model_name: str = "DeepSeek-R1"
    gpu_type: str = "MI300X"
    baseline_tput: float = 600.0
    baseline_config_path: str = "/tmp/baseline.yaml"
    warm_start_recipe: dict = field(default_factory=dict)
    warm_replay_attempted: bool = False
    warm_replay_outcome: dict = field(default_factory=dict)
    warm_history_injected: bool = False
    explore_search: dict = field(default_factory=dict)
    optimization_stack: list = field(default_factory=list)
    gain_per_stack_entry: list = field(default_factory=list)
    cumulative_gain: float = 0.0
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_stack_len: int = 0
    current_best: dict = field(default_factory=dict)
    tick: int = 0
    phase: str = "PRELUDE"

    def save(self, *args, **kwargs):  # noqa: D401 — stub
        pass


class _StubTaskRegistry:
    """Captures ``create_or_return_existing`` calls so tests can assert."""

    def __init__(self):
        self.calls: list[dict] = []

    async def create_or_return_existing(self, *, kind, params, idempotency_key):
        self.calls.append({
            "kind": kind, "params": dict(params),
            "idempotency_key": idempotency_key,
        })
        task = _StubTask(
            task_id=f"task-{idempotency_key}",
            kind=kind,
            params=dict(params),
        )
        return task, False


def _make_coord(
    tmp_path: Path,
    *,
    warm_start_recipe: dict | None = None,
    warm_replay_enabled: bool = True,
    warm_replay_min_confidence: float = 0.7,
    warm_replay_min_reproduce_pct: float = 0.8,
    warm_replay_attempted: bool = False,
) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = _StubSharedState(
        warm_start_recipe=warm_start_recipe or {},
        warm_replay_attempted=warm_replay_attempted,
    )
    coord.tasks = _StubTaskRegistry()
    coord._warm_replay_enabled = warm_replay_enabled
    coord._warm_replay_min_confidence = warm_replay_min_confidence
    coord._warm_replay_min_reproduce_pct = warm_replay_min_reproduce_pct
    coord._journal = None
    return coord


def _warm_recipe_t1(
    *,
    extra_sglang_args: str = "--attention-backend AITER",
    extra_envs: dict | None = None,
    expected_gain_pct: float = 25.0,
    confidence: float = 0.85,
    tier: str = "T1_exact",
    sessions: list | None = None,
    what_failed: list | None = None,
) -> dict:
    """Build a fake warm_start_recipe payload mirroring what
    ``find_recipe_with_fallback`` returns.

    ``expected_gain_pct`` lands inside ``attrs.sessions[0].gain_pct``
    because that's where ``_maybe_enqueue_warm_replay`` actually reads
    the historical gain from (see GAP 1 / FIX-2). For convenience
    callers can also pass a ``sessions`` list explicitly to test the
    multi-session max() path.
    """
    recipe_sessions = sessions if sessions is not None else [
        {"session_id": "prior-session-A", "gain_pct": expected_gain_pct, "stack_len": 1},
    ]
    attrs: dict = {
        "model":     "DeepSeek-R1",
        "hardware":  "MI300X",
        "framework": "sglang",
        "best_config": {
            "extra_sglang_args": extra_sglang_args,
            "extra_envs": dict(extra_envs or {}),
        },
        "sessions": recipe_sessions,
    }
    if what_failed is not None:
        attrs["what_failed"] = what_failed
    return {
        "tier": tier,
        "confidence": confidence,
        "recipe": {
            "id": 1,
            "canonical_id": "recipe:deepseek-r1:sglang:mi300x",
            "kind": "recipe",
            "attrs": attrs,
        },
    }


# ===========================================================================
# Skip paths
# ===========================================================================
@pytest.mark.asyncio
async def test_warm_replay_skips_when_disabled_by_flag(tmp_path):
    """``--no-warm-replay`` → skip + flip the one-shot guard so a
    resume without the flag (the common robustness_monitor.sh case)
    cannot retroactively trigger a replay against operator intent."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=False,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert "disabled_by_flag" in coord.shared_state.warm_replay_outcome["reason"]
    assert coord.tasks.calls == []
    # The one-shot guard MUST flip on disabled-skip so a robustness
    # resume that loses ``--no-warm-replay`` (which is the common
    # operator failure mode) does NOT belatedly trigger replay.
    assert coord.shared_state.warm_replay_attempted is True


@pytest.mark.asyncio
async def test_warm_replay_resume_with_lost_disable_flag_is_still_blocked(
    tmp_path,
):
    """End-to-end resume safety: launch 1 sets ``--no-warm-replay``
    (flipping warm_replay_attempted=True via the disabled-skip path);
    launch 2 (robustness resume) doesn't re-pass the flag; the second
    ``_maybe_enqueue_warm_replay`` short-circuits on
    warm_replay_attempted, not on warm_replay_enabled, so even with
    enabled=True the replay does NOT fire a second time."""
    # Launch 1 — operator disabled warm-replay.
    coord1 = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=False,
    )
    await coord1._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord1.shared_state.warm_replay_attempted is True
    # Launch 2 — robustness restarts, ``--no-warm-replay`` not re-passed
    # (warm_replay_enabled defaults to True). State.json restored,
    # so warm_replay_attempted persists.
    coord2 = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=True,
        warm_replay_attempted=True,  # restored from state.json
    )
    task = await coord2._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord2.tasks.calls == []


@pytest.mark.asyncio
async def test_warm_replay_skips_when_already_attempted(tmp_path):
    """Resume safety: a previous boot already ran the replay; no second
    enqueue."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_attempted=True,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.tasks.calls == []


@pytest.mark.asyncio
async def test_warm_replay_skips_when_no_warm_start_recipe(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe={})
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_attempted is True
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert coord.shared_state.warm_replay_outcome["reason"] == "no_warm_start_recipe"


@pytest.mark.asyncio
async def test_warm_replay_skips_when_confidence_below_threshold(tmp_path):
    """Only T1 / T2 fire by default. Lower-tier hits (T3 / T4 / T5 / T6)
    are too far from the workload to be worth a verify spend."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(
            confidence=0.55, tier="T3_same_family",
        ),
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "skipped"
    assert "below_threshold" in outcome["reason"]
    assert outcome["warm_recipe_tier"] == "T3_same_family"


@pytest.mark.asyncio
async def test_warm_replay_skips_when_best_config_empty(tmp_path):
    """A seed-only recipe (registered canonical_id but no actual args)
    isn't worth replaying — there's nothing to apply."""
    recipe = _warm_recipe_t1(extra_sglang_args="", extra_envs={})
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_outcome["reason"] == "best_config_empty"


# ===========================================================================
# Enqueue path
# ===========================================================================
@pytest.mark.asyncio
async def test_warm_replay_enqueues_with_warm_best_config_args_envs(tmp_path):
    """Happy path: high-confidence T1 hit with a real best_config →
    task created carrying the warm config in ``params``."""
    recipe = _warm_recipe_t1(
        extra_sglang_args="--attention-backend AITER --kv-cache-dtype fp8",
        extra_envs={"VLLM_ROCM_USE_AITER": "1"},
        expected_gain_pct=25.0,
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is not None
    assert task.kind == "replay_warm_recipe"
    assert len(coord.tasks.calls) == 1
    call = coord.tasks.calls[0]
    assert call["kind"] == "replay_warm_recipe"
    assert call["idempotency_key"] == "warm-replay-prelude"
    params = call["params"]
    assert params["extra_sglang_args"] == "--attention-backend AITER --kv-cache-dtype fp8"
    assert params["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert params["config_path"] == "/tmp/baseline.yaml"
    assert params["warm_expected_gain_pct"] == 25.0
    assert params["warm_recipe_tier"] == "T1_exact"
    assert params["warm_recipe_conf"] == 0.85
    assert params["baseline_tput_anchor"] == 600.0
    # ``warm_replay_attempted`` flipped True for resume safety.
    assert coord.shared_state.warm_replay_attempted is True
    assert coord.shared_state.warm_replay_outcome["status"] == "in_flight"
    assert coord.shared_state.warm_replay_outcome["replay_task_id"] == task.task_id


# ===========================================================================
# Promote — reproduced
# ===========================================================================
def test_promote_warm_replay_reproduced_pushes_stack_and_updates_gain(
    tmp_path,
):
    """When measured gain ≥ expected × min_reproduce, push the warm
    config onto the stack and bump cumulative_gain so the rest of the
    session inherits it as the starting point."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "warm_recipe_tier": "T1_exact",
        "warm_recipe_conf": 0.85,
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    task = _StubTask(params={
        "extra_sglang_args": "--attention-backend AITER",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    })
    # Measured 23% gain (600 → 738) — above 25% × 0.8 = 20% threshold.
    result = {"status": "succeeded", "output_throughput": 738.0}
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert outcome["actual_gain_pct"] == 23.0
    assert outcome["throughput_after"] == 738.0
    # Stack push.
    assert len(coord.shared_state.optimization_stack) == 1
    entry = coord.shared_state.optimization_stack[0]
    assert entry["action"] == "replay_warm_recipe"
    assert entry["extra_sglang_args"] == "--attention-backend AITER"
    assert entry["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert entry["tput"] == 738.0
    # Gain bookkeeping.
    assert coord.shared_state.gain_per_stack_entry == [23.0]
    assert coord.shared_state.cumulative_gain == 23.0
    assert coord.shared_state.cumulative_gain_validated == 23.0
    assert coord.shared_state.cumulative_gain_validated_stack_len == 1
    # current_best lifted.
    assert coord.shared_state.current_best["action"] == "warm_replay"
    assert coord.shared_state.current_best["tput"] == 738.0


def test_promote_warm_replay_drift_does_not_push_stack(tmp_path):
    """Measured gain below the reproduce threshold → tag as ``drift``
    but do NOT push onto stack (let EXPLORE start from clean baseline)."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "T1_exact",
    }
    task = _StubTask(params={
        "extra_sglang_args": "--attention-backend AITER",
    })
    # 10% gain — well below 25% × 0.8 = 20% threshold.
    result = {"status": "succeeded", "output_throughput": 660.0}
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "drift"
    assert outcome["actual_gain_pct"] == 10.0
    assert "below" in outcome["reason"]
    # NO stack push.
    assert coord.shared_state.optimization_stack == []
    assert coord.shared_state.cumulative_gain == 0.0


def test_promote_warm_replay_succeeded_but_zero_gain_is_drift(tmp_path):
    """``expected_gain_pct=0`` (recipe didn't carry historical gain) →
    any positive measurement counts as reproduced. Zero / negative
    measured gain falls to drift."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 0.0,
    }
    task = _StubTask(params={"extra_sglang_args": "--foo"})
    # Measured tput == baseline_tput → 0% gain → drift.
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.warm_replay_outcome["status"] == "drift"
    assert coord.shared_state.warm_replay_outcome["actual_gain_pct"] == 0.0


def test_promote_warm_replay_failed_records_outcome(tmp_path):
    """Subprocess failure (timeout / OOM / crash) → tag as ``failed``
    with the error_class verbatim so the report can render it."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }
    result = {
        "status": "failed",
        "error_class": "crash",
        "error": "GPU OOM during prefill",
    }
    coord._promote_warm_replay(result, task=_StubTask())

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "failed"
    assert outcome["error_class"] == "crash"
    assert "GPU OOM" in outcome["reason"]
    # No stack push on failure.
    assert coord.shared_state.optimization_stack == []


# ===========================================================================
# FIX-1 — warm_recipe.what_failed injection into explore_search.rejected
# ===========================================================================
def test_inject_warm_recipe_history_skips_when_no_recipe(tmp_path):
    """No warm_start_recipe → nothing to inject, flag still flipped to
    prevent retries."""
    coord = _make_coord(tmp_path, warm_start_recipe={})
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    assert coord.shared_state.warm_history_injected is True
    assert coord.shared_state.explore_search.get("rejected", []) == []


def test_inject_warm_recipe_history_adds_what_failed_rows(tmp_path):
    """Every what_failed row carries a canonical fingerprint into the
    rejected ledger, with ``source=warm_start_recipe``."""
    recipe = _warm_recipe_t1(
        what_failed=[
            {
                "name": "fp4_kv_cache",
                "extra_sglang_args": "--kv-cache-dtype fp4",
                "extra_envs": {},
                "gain_pct": -8.0,
                "error_class": "regress",
            },
            {
                "name": "tilelang_mla",
                "extra_sglang_args": "",
                "extra_envs": {"SGLANG_HACK_FLASHMLA_BACKEND": "tilelang"},
                "gain_pct": None,
                "error_class": "crash",
            },
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 2
    rejected = coord.shared_state.explore_search["rejected"]
    assert len(rejected) == 2
    # Both rows carry a 16-char canonical fingerprint and a marker.
    for row in rejected:
        assert isinstance(row.get("fingerprint"), str) and len(row["fingerprint"]) == 16
        assert row["reason"] == "warm_recipe_what_failed"
        assert row["source"] == "warm_start_recipe"
        assert row["source_tier"] == "T1_exact"
    # The marker fields preserve the original gain_pct / error_class.
    assert any(r["error_class"] == "regress" for r in rejected)
    assert any(r["error_class"] == "crash" for r in rejected)
    assert coord.shared_state.warm_history_injected is True


def test_inject_warm_recipe_history_is_idempotent(tmp_path):
    """Resume safety: re-invoking the injector after the one-shot flag
    is set must NOT re-append the same rows."""
    recipe = _warm_recipe_t1(
        what_failed=[{
            "name": "x", "extra_sglang_args": "--bad-flag",
            "extra_envs": {}, "gain_pct": -10.0,
        }],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    coord._inject_warm_recipe_history_into_ledger()
    first = list(coord.shared_state.explore_search["rejected"])
    # Second call: flag already set, must short-circuit.
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    assert coord.shared_state.explore_search["rejected"] == first


def test_inject_warm_recipe_history_dedupes_with_existing_ledger(tmp_path):
    """If the ledger already carries a row with the same fingerprint
    (e.g. from a prior in-session explore round that lost it on
    resume), we don't add a duplicate."""
    from inference_optimizer.orchestrator.action_executors._canonical_fingerprint import (
        canonical_fingerprint,
    )
    failed_args = "--kv-cache-dtype fp4"
    pre_existing_fp = canonical_fingerprint(failed_args, {})
    recipe = _warm_recipe_t1(
        what_failed=[{
            "name": "fp4", "extra_sglang_args": failed_args,
            "extra_envs": {}, "gain_pct": -8.0,
        }],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    # Simulate an in-session ledger that already records this fingerprint.
    coord.shared_state.explore_search = {
        "rejected": [{
            "name": "explore_round_1_X",
            "fingerprint": pre_existing_fp,
            "reason": "stack_unstable",
        }],
    }
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    # Existing row preserved verbatim.
    assert len(coord.shared_state.explore_search["rejected"]) == 1
    assert coord.shared_state.explore_search["rejected"][0]["reason"] == "stack_unstable"


def test_inject_warm_recipe_history_skips_empty_rows(tmp_path):
    """A what_failed row with neither args nor envs is unreplayable
    (nothing to dedup against). Skip silently."""
    recipe = _warm_recipe_t1(
        what_failed=[
            {"name": "bogus", "extra_sglang_args": "", "extra_envs": {}},
            {"name": "real", "extra_sglang_args": "--actual-flag", "extra_envs": {}},
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 1
    # ``bogus`` was skipped; ``real`` made it in.
    assert coord.shared_state.explore_search["rejected"][0]["name"] == "real"


# ===========================================================================
# FIX-2 — expected_gain_pct comes from sessions[].gain_pct
# ===========================================================================
@pytest.mark.asyncio
async def test_warm_replay_pulls_expected_gain_from_sessions_max(tmp_path):
    """The historical gain anchor is the MAX of ``attrs.sessions[].gain_pct``
    so a recipe validated by N sessions exposes the best a prior run
    ever achieved (not the most recent one, which could be a regress)."""
    recipe = _warm_recipe_t1(
        sessions=[
            {"session_id": "older",   "gain_pct": 12.0, "stack_len": 1},
            {"session_id": "best",    "gain_pct": 28.0, "stack_len": 4},
            {"session_id": "newer",   "gain_pct": 20.0, "stack_len": 2},
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 28.0


@pytest.mark.asyncio
async def test_warm_replay_zero_expected_when_no_sessions(tmp_path):
    """Recipes ingested from non-hyperloom sources may not carry
    sessions[]. The expected_gain falls to 0 and ``_promote`` will
    accept any positive measurement (see existing drift / zero
    coverage tests)."""
    recipe = _warm_recipe_t1(sessions=[])
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 0.0


@pytest.mark.asyncio
async def test_warm_replay_falls_back_to_flat_gain_pct_for_arbor_seed(tmp_path):
    """Arbor-style offline-ingest seeds carry a flat ``gain_pct``
    attr instead of sessions[]. We still read it as the expected
    anchor."""
    coord = _make_coord(tmp_path)
    # Manually build a recipe with the legacy flat ``gain_pct`` attr.
    coord.shared_state.warm_start_recipe = {
        "tier": "T2_same_shape",
        "confidence": 0.75,
        "recipe": {
            "attrs": {
                "best_config": {"extra_sglang_args": "--foo", "extra_envs": {}},
                "gain_pct": 18.0,   # flat — no sessions[]
            },
        },
    }
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 18.0


# ===========================================================================
# FIX-5 — cumulative_gain derived from baseline_tput, not summed
# ===========================================================================
def test_promote_warm_replay_cumulative_gain_uses_tput_ratio(tmp_path):
    """Cumulative gain after warm-replay = (tput / baseline_tput - 1)
    × 100, not measured_gain assignment. The two are identical when
    stack starts empty (current case) but the formula is the
    authoritative one for any future code paths that may push onto
    stack before warm-replay fires."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "T1_exact",
    }
    task = _StubTask(params={
        "extra_sglang_args": "--attention-backend AITER",
    })
    # baseline 600, measured 738 → gain = 23% exactly via tput ratio.
    result = {"status": "succeeded", "output_throughput": 738.0}
    coord._promote_warm_replay(result, task=task)
    # cumulative_gain == round((738/600 - 1) * 100, 3) == 23.0
    assert coord.shared_state.cumulative_gain == 23.0
    assert coord.shared_state.cumulative_gain_validated == 23.0


# ===========================================================================
# FIX-6 — warm-replay enqueue ordering (history-inject first, replay
# second, analysis third)
# ===========================================================================
# The PRELUDE ordering rule is enforced by ``_promote_to_shared_state``
# rather than by ``_maybe_enqueue_warm_replay`` directly; that path is
# tested through integration scenarios (test_cortex_t0_anchor /
# test_close_phase_sequencer cover the surrounding lifecycle). Adding
# an integration-level test for the ordering would require constructing
# a fully-wired coordinator, which is out of scope for this unit
# module. We rely on the manual code comment + log statement to make
# the intent unambiguous at the call site.


def test_promote_warm_replay_zero_baseline_tput_is_failure(tmp_path):
    """Defense in depth: an invalid baseline_tput (shouldn't happen
    given the call site only fires on a successful baseline) must
    not produce a divide-by-zero / nonsensical gain — tag as failed."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 0.0
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=_StubTask())
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"
    assert "invalid_tput" in coord.shared_state.warm_replay_outcome["reason"]
