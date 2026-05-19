"""Roofline-v2 C4c: Coordinator integration for the ``roofline`` action.

Two integration points pinned here:

1. **sequence_denial** — propose_action("roofline") is gated on a
   cached ``last_select_kernels.analysis_md_text`` (the executor has
   no useful input otherwise). The denial carries a concrete
   actionable hint pointing the main LLM to
   ``request{kind="select_kernels"}``.

2. **task completion** — when a ``roofline`` task succeeds (or returns
   the fallback dict), the Coordinator's ``_promote_to_shared_state``
   writes the result through C2's ``record_roofline_analysis`` so the
   structured decision lands in ``SharedState.last_roofline_analysis``
   for the C5 prompt renderer to pick up on subsequent ticks.

These tests use the same Coordinator construction pattern as
``test_required_step_gates.py`` so they share assumptions about what
state must be pre-seeded for the gate-under-test to be the only
denial in play.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


# ---------------------------------------------------------------------------
# Test fixtures (mirror test_required_step_gates.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _write_baseline_json(session_dir: Path) -> Path:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return path


def _seed_post_select_kernels(coord: Coordinator,
                               *, analysis_md_text: str = "FAKE_REPORT",
                               snapshot_id: int = 3) -> None:
    """Open every earlier gate so the roofline gate is what we are testing.

    Writes the target_analysis marker JSON, sets baseline_tput,
    last_profile_trace, last_profile_pmc_summary, and the
    last_select_kernels cache (including the C1 analysis_md_text
    / snapshot_id fields)."""
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
        "analysis_md_text": analysis_md_text,
        "analysis_md_path": "/tmp/analysis.md",
        "roofline_snapshot_id": snapshot_id,
    }


# ===========================================================================
# sequence_denial — roofline is in sequence_actions and gated on analysis_md
# ===========================================================================
def test_roofline_added_to_sequence_actions(session_dir):
    """``roofline`` must be recognised by ``_sequence_denial_for_action``
    so the pre-baseline / target_analysis gates fire for it like every
    other sequence action. Otherwise the gate function early-returns
    None for unknown actions and lets the propose through."""
    coord = Coordinator(session_dir, backends=_backends_full())
    # No baseline yet → target_analysis gate must fire (proving the
    # action was recognised — unknown actions get None, not a denial).
    denied = coord._sequence_denial_for_action("roofline")
    assert isinstance(denied, PolicyDenied)
    assert "target_analysis must run first" in str(denied)


def test_roofline_denied_when_no_cached_analysis_md(session_dir):
    """The roofline-specific gate fires when ``analysis_md_text`` is
    empty even though baseline / profile / select_kernels have all
    nominally run."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord, analysis_md_text="")

    denied = coord._sequence_denial_for_action("roofline")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "select_kernels must run first" in str(denied)
    assert denied.hint and "analysis.md" in denied.hint
    assert denied.hint and "trace_input" in denied.hint


def test_roofline_denied_when_select_kernels_cache_missing(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    s.last_select_kernels = {}  # not yet populated

    denied = coord._sequence_denial_for_action("roofline")
    assert isinstance(denied, PolicyDenied)
    assert "select_kernels must run first" in str(denied)


def test_roofline_allowed_when_analysis_md_cached(session_dir):
    """Happy path — every prerequisite gate is open AND analysis_md_text
    is cached → roofline propose passes through."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord)

    denied = coord._sequence_denial_for_action("roofline")
    assert denied is None


def test_roofline_gate_does_not_leak_to_other_actions(session_dir):
    """Removing the cache only blocks roofline; baseline / params /
    backends remain unaffected (those don't need a roofline-specific
    cache check)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord, analysis_md_text="")  # roofline gate would fire

    assert coord._sequence_denial_for_action("roofline") is not None
    # params / backends ride the standard prerequisites only (target_
    # analysis JSON written, baseline_tput > 0, last_profile_trace set);
    # they do NOT require analysis_md_text.
    assert coord._sequence_denial_for_action("params") is None
    assert coord._sequence_denial_for_action("backends") is None
    assert coord._sequence_denial_for_action("sweep") is None


# ===========================================================================
# task completion → record_roofline_analysis (C2 wiring point)
# ===========================================================================
@pytest.mark.asyncio
async def test_promote_writes_well_formed_roofline_result(session_dir):
    """When a roofline task succeeds, the structured result lands in
    ``last_roofline_analysis`` via C2's recorder."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord, snapshot_id=4)

    result = {
        "status": "succeeded",
        "snapshot_id": 4,
        "analyzed_at_iso": "2026-05-19T10:30:00+00:00",
        "analyzed_at_gain_pct": 3.2,
        "based_on_analysis_md": "/tmp/analysis.md",
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {"comm": 0.45, "compute": 0.30},
        "suggested_prunes": [
            {"family": "kernel_opt", "reason": "saturated", "confidence": "high"},
        ],
        "suggested_next_actions": [
            {"kind": "params", "rationale": "overlap", "priority": "high"},
        ],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "{...}",
    }

    await coord._promote_to_shared_state("roofline", result)

    cached = coord.shared_state.last_roofline_analysis
    assert cached["snapshot_id"] == 4
    assert cached["primary_bottleneck"] == "comm"
    assert cached["bottleneck_distribution"]["comm"] == 0.45
    assert len(cached["suggested_prunes"]) == 1
    assert cached["suggested_prunes"][0]["family"] == "kernel_opt"
    assert len(cached["suggested_next_actions"]) == 1
    assert cached["suggested_next_actions"][0]["kind"] == "params"


@pytest.mark.asyncio
async def test_promote_writes_degraded_fallback(session_dir):
    """A degraded fallback dict (sub-agent failed, timed out, or
    returned non-JSON) must still round-trip cleanly so the C5
    renderer can show "analysis unavailable for snapshot #N"."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord, snapshot_id=2)

    result = {
        "status": "succeeded",
        "degraded": True,
        "snapshot_id": 2,
        "analyzed_at_iso": "2026-05-19T10:30:00+00:00",
        "analyzed_at_gain_pct": 0.0,
        "based_on_analysis_md": "/tmp/analysis.md",
        "primary_bottleneck": "unknown",
        "bottleneck_distribution": {},
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "",
        "error": "sub_agent_timeout_after_60s",
    }

    await coord._promote_to_shared_state("roofline", result)

    cached = coord.shared_state.last_roofline_analysis
    assert cached["snapshot_id"] == 2
    assert cached["primary_bottleneck"] == "unknown"
    assert cached["suggested_prunes"] == []


@pytest.mark.asyncio
async def test_promote_handles_non_dict_result_safely(session_dir):
    """``_promote_to_shared_state`` short-circuits on non-dict input,
    and ``record_roofline_analysis`` silently no-ops on non-dict too.
    The combined path must never write garbage / never raise."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord)

    await coord._promote_to_shared_state("roofline", None)  # type: ignore[arg-type]
    assert coord.shared_state.last_roofline_analysis == {}


@pytest.mark.asyncio
async def test_promote_does_not_clobber_other_state(session_dir):
    """Roofline branch must only touch ``last_roofline_analysis`` —
    not optimization_stack / cumulative_gain / etc."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord, snapshot_id=1)
    coord.shared_state.cumulative_gain_validated = 4.7
    coord.shared_state.optimization_stack = [{"kind": "params"}]

    result = {
        "primary_bottleneck": "memory",
        "snapshot_id": 1,
        "based_on_analysis_md": "/tmp/analysis.md",
    }
    await coord._promote_to_shared_state("roofline", result)

    assert coord.shared_state.cumulative_gain_validated == 4.7
    assert coord.shared_state.optimization_stack == [{"kind": "params"}]
    assert coord.shared_state.last_roofline_analysis["primary_bottleneck"] == "memory"


# ===========================================================================
# Regression — pre-roofline gates still behave correctly
# ===========================================================================
def test_pre_existing_gates_still_fire_first_for_roofline(session_dir):
    """target_analysis / baseline / profile gates must still take
    precedence over the new roofline-specific gate — otherwise the
    main LLM would see "select_kernels first" before having a
    baseline, which would be confusing."""
    coord = Coordinator(session_dir, backends=_backends_full())
    # No baseline JSON → target_analysis gate
    assert "target_analysis" in str(
        coord._sequence_denial_for_action("roofline"),
    )

    # Write JSON but no baseline_tput → baseline gate
    _write_baseline_json(coord.session_dir)
    assert "baseline must run first" in str(
        coord._sequence_denial_for_action("roofline"),
    )

    # Baseline ok, no profile trace → profile gate
    coord.shared_state.baseline_tput = 100.0
    assert "profile must run" in str(
        coord._sequence_denial_for_action("roofline"),
    )

    # Profile ok, no select_kernels → roofline-specific gate
    coord.shared_state.last_profile_trace = "/tmp/profile.tar.gz"
    coord.shared_state.last_profile_pmc_summary = "/tmp/pmc.json"
    coord.shared_state.last_select_kernels = {}
    assert "select_kernels must run first" in str(
        coord._sequence_denial_for_action("roofline"),
    )

    # Cached but no analysis_md_text → still roofline-specific
    coord.shared_state.last_select_kernels = {"trace_input": "x"}
    assert "select_kernels must run first" in str(
        coord._sequence_denial_for_action("roofline"),
    )

    # analysis_md_text populated → all gates pass
    coord.shared_state.last_select_kernels = {
        "trace_input": "x",
        "analysis_md_text": "report",
        "roofline_snapshot_id": 1,
    }
    assert coord._sequence_denial_for_action("roofline") is None


def test_pre_existing_actions_still_passable_after_roofline_added(session_dir):
    """Regression: adding ``roofline`` to ``sequence_actions`` must not
    change the gate behaviour for the pre-existing actions."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_select_kernels(coord)

    for action in ("baseline", "profile", "params", "backends", "sweep"):
        assert coord._sequence_denial_for_action(action) is None, (
            f"{action} unexpectedly denied"
        )
