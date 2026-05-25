"""Roofline-v2 N3: sequence_denial gate for roofline + dependent actions.

Pins three contracts:

1. `roofline` itself is recognised by `_sequence_denial_for_action`
   (in the `sequence_actions` set) so the upstream target_analysis /
   baseline gates apply to it like every other sequence action.
2. `roofline` is exempt from the existing "profile must run before X"
   gate because it internally runs profile as its first sub-step.
3. The 4 optimisation actions (backends / params / kernel_opt /
   comm_optimization) are denied when `last_trace_analyze.analysis_md_text`
   is empty; the denial carries a concrete hint pointing the LLM to
   propose `roofline`. Other actions (sweep / validate_stack / report
   / integrate / sweep) are NOT subject to the gate.
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
# Test fixtures
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


def _write_baseline_json(sd: Path) -> Path:
    p = target_baseline_json(sd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return p


def _seed_post_baseline(coord: Coordinator) -> None:
    """target_analysis JSON + baseline_tput → both gates open."""
    _write_baseline_json(coord.session_dir)
    coord.shared_state.baseline_tput = 100.0


def _seed_post_roofline(coord: Coordinator, *,
                          analysis_md_text: str = "FAKE_REPORT") -> None:
    """target_analysis JSON + baseline_tput + last_profile_trace +
    last_trace_analyze with cached analysis_md_text — every prereq open."""
    _seed_post_baseline(coord)
    s = coord.shared_state
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "analysis_md_text": analysis_md_text,
        "analysis_md_path": "/tmp/analysis.md",
        "roofline_snapshot_id": 1,
    }


# ---------------------------------------------------------------------------
# roofline action recognition + own profile gate exemption
# ---------------------------------------------------------------------------
def test_roofline_in_sequence_actions(session_dir):
    """Without this `_sequence_denial_for_action` early-returns None
    for unknown actions, bypassing every gate."""
    coord = Coordinator(session_dir, backends=_backends_full())
    denied = coord._sequence_denial_for_action("roofline")
    assert isinstance(denied, PolicyDenied)
    assert "target_analysis must run first" in str(denied)


def test_roofline_passes_profile_gate(session_dir, monkeypatch):
    """`roofline` is exempt from the profile-must-run-first gate
    because it internally runs profile.

    Updated for N9: direct `profile` propose is now blocked. We test
    the escape-hatch path here to assert the underlying profile gate
    behaviour separately from the N9 hard-block (which is fully
    covered in test_n9_profile_blocked_from_propose.py).
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", "1")
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    # No last_profile_trace yet
    assert coord.shared_state.last_profile_trace == ""

    # roofline / profile (escape-hatch-on) / validate_stack pass
    assert coord._sequence_denial_for_action("roofline") is None
    assert coord._sequence_denial_for_action("profile") is None

    # other actions should be denied with profile-gate
    denied = coord._sequence_denial_for_action("backends")
    assert isinstance(denied, PolicyDenied)
    assert "profile must run" in str(denied)
    # N3+N9: hint now points at roofline as the only recommended path
    # (profile is escape-hatch-only, not advertised in the hint)
    assert denied.hint and "roofline" in denied.hint


def test_roofline_denied_when_baseline_not_run(session_dir):
    """baseline gate fires for roofline like every other sequence action."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(coord.session_dir)
    # baseline_tput still 0
    denied = coord._sequence_denial_for_action("roofline")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)


def test_roofline_allowed_after_baseline(session_dir):
    """baseline complete → roofline propose passes through."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    assert coord._sequence_denial_for_action("roofline") is None


# ---------------------------------------------------------------------------
# 3 propose-path optimization actions require fresh analysis_md_text.
# kernel_opt routes via REQUEST kind="run_optimization" and is covered
# by _sequence_denial_for_request (tested separately below).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", [
    "backends", "params", "comm_optimization",
])
def test_optimization_action_denied_without_analysis_md(session_dir, action):
    """3 propose-path optimization actions blocked when last_trace_analyze
    cache empty."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    # No last_trace_analyze cache
    coord.shared_state.last_trace_analyze = {}

    denied = coord._sequence_denial_for_action(action)
    assert isinstance(denied, PolicyDenied), f"{action} should be denied"
    assert denied.rule == "execution_order"
    assert "roofline must run first" in str(denied)
    # Hint must explicitly suggest roofline (the LLM's actionable next step)
    assert denied.hint and "roofline" in denied.hint
    assert denied.hint and "composite" in denied.hint


@pytest.mark.parametrize("action", [
    "backends", "params", "comm_optimization",
])
def test_optimization_action_denied_when_analysis_md_text_empty(session_dir, action):
    """Even if last_trace_analyze dict exists but analysis_md_text is empty,
    the gate still fires (e.g. failed trace_analyze that left a stub dict)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.gz",
        "analysis_md_text": "",  # empty — the failure case
    }
    denied = coord._sequence_denial_for_action(action)
    assert isinstance(denied, PolicyDenied)
    assert "roofline must run first" in str(denied)


@pytest.mark.parametrize("action", [
    "backends", "params", "comm_optimization",
])
def test_optimization_action_allowed_when_analysis_md_cached(session_dir, action):
    """Happy path — every prereq open + analysis_md_text populated."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_roofline(coord)
    assert coord._sequence_denial_for_action(action) is None


# ---------------------------------------------------------------------------
# kernel_opt: covered by REQUEST-path gate (run_optimization).
# ---------------------------------------------------------------------------
def test_run_optimization_request_denied_without_analysis_md(session_dir):
    """kernel_opt is dispatched via REQUEST kind="run_optimization" not
    propose_action; _sequence_denial_for_request enforces the
    last_trace_analyze.trace_input == last_profile_trace invariant."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    coord.shared_state.last_trace_analyze = {}  # missing

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    # Pre-existing message ("trace_analyze must run first") retained;
    # hint updated to mention roofline as preferred path (N3).
    assert "trace_analyze must run first" in str(denied)
    assert denied.hint and "roofline" in denied.hint


def test_run_optimization_request_allowed_with_matching_cache(
    session_dir, monkeypatch,
):
    """run_optimization passes the trace_analyze cache check when
    last_trace_analyze.trace_input matches last_profile_trace (the
    pre-N13 invariant).

    Roofline-v2 N13 adds a separate ordering gate (cheap actions +
    snapshot >= 2). Use the escape hatch to keep this test focused
    on the trace_analyze cache check; the N13 ordering gate has
    dedicated coverage in test_n13_kernel_opt_ordering.py.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "1")
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.gz",
        "analysis_md_text": "REPORT",
    }
    assert coord._sequence_denial_for_request(
        "kernel", "run_optimization",
    ) is None


# ---------------------------------------------------------------------------
# Non-gated actions stay unaffected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", [
    "sweep", "validate_stack", "report", "integrate",
])
def test_non_optimization_actions_not_subject_to_roofline_gate(
    session_dir, action,
):
    """sweep / validate_stack / report / integrate do NOT require
    analysis_md_text. Only the explicit propose-path optimization
    actions (backends / params / comm_optimization) do (per §6.5
    design doc).

    Note: `profile` is intentionally NOT in this parametrize set
    because N9 introduced a separate hard-block on profile-direct-
    propose with its own "roofline" hint (see
    test_n9_profile_blocked_from_propose.py); the roofline-gate
    presence-of-roofline-in-message check would incorrectly fire on
    the N9 denial.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    # No analysis_md_text
    coord.shared_state.last_trace_analyze = {}

    denied = coord._sequence_denial_for_action(action)
    # Some actions have their own additional gates (e.g. integrate needs
    # kernel_opt KEEP first); but the roofline gate specifically must
    # NOT fire. So if denied, the message must NOT mention "roofline".
    if denied is not None:
        assert "roofline" not in str(denied).lower(), (
            f"{action} unexpectedly denied by roofline gate: {denied!r}"
        )


# ---------------------------------------------------------------------------
# Gate ordering — pre-existing gates fire first
# ---------------------------------------------------------------------------
def test_roofline_gate_fires_after_baseline_gate(session_dir):
    """LLM must see baseline-gate before roofline-gate so it doesn't
    chase a roofline propose when baseline_tput is still 0."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(coord.session_dir)
    # No baseline_tput → baseline-gate fires for `backends`
    denied = coord._sequence_denial_for_action("backends")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)
    # Must NOT be the roofline gate
    assert "roofline must run first" not in str(denied)


def test_roofline_gate_fires_after_validate_stack_gate(session_dir):
    """validate_stack precedence is checked BEFORE roofline gate;
    otherwise the LLM would chase roofline when actually it owes
    a validate_stack rebench first."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.gz"
    coord.shared_state.last_trace_analyze = {}  # roofline-gate would fire

    # Now add an unvalidated KEEP — validate_stack-gate should fire first
    coord.shared_state.optimization_stack.append({
        "kind": "params",
        "variant_name": "vA",
        "gain_pct": 1.2,
    })
    # Don't bump cumulative_gain_validated_stack_len → stack has unvalidated entry

    denied = coord._sequence_denial_for_action("backends")
    assert isinstance(denied, PolicyDenied)
    assert "validate_stack required first" in str(denied)
    assert "roofline must run first" not in str(denied)


def test_gate_progression_complete_chain(session_dir):
    """Walk through every gate in order to confirm precedence."""
    coord = Coordinator(session_dir, backends=_backends_full())

    # Stage 1: no target_analysis JSON → target_analysis gate
    denied = coord._sequence_denial_for_action("backends")
    assert "target_analysis must run first" in str(denied)

    # Stage 2: target_analysis JSON written, no baseline → baseline gate
    _write_baseline_json(coord.session_dir)
    denied = coord._sequence_denial_for_action("backends")
    assert "baseline must run first" in str(denied)

    # Stage 3: baseline_tput set, no profile_trace → profile gate
    coord.shared_state.baseline_tput = 100.0
    denied = coord._sequence_denial_for_action("backends")
    assert "profile must run" in str(denied)

    # Stage 4: profile_trace set, no analysis_md → roofline gate
    coord.shared_state.last_profile_trace = "/tmp/p.gz"
    coord.shared_state.last_trace_analyze = {}
    denied = coord._sequence_denial_for_action("backends")
    assert "roofline must run first" in str(denied)

    # Stage 5: analysis_md_text populated → all gates pass
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "report",
        "roofline_snapshot_id": 1,
    }
    assert coord._sequence_denial_for_action("backends") is None
