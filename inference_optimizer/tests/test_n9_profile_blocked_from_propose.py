"""Roofline-v2 N9: profile direct-propose is hard-blocked.

Pins the GPU-empirical post-fix per design §6.5 N9 amendment.

Context: N3 originally allowed `profile` as a soft-deprecated direct
propose path with a hint pointing at `roofline`. Qwen3-32B GPU
session 14:36-15:03 showed the main LLM fell back to v0
training-distribution behaviour and proposed `profile` directly,
forcing `roofline` to redo the profile internally — wasting ~10 min
wall-clock per session. N9 hard-rejects the direct propose;
`profile_executor` remains accessible via `RooflineExecutor` (which
bypasses `_sequence_denial_for_action` by calling the executor as a
plain coroutine, not via `propose_action`).

Tests pin:

* `propose_action(profile)` denied by default with
  rule="execution_order" + hint mentioning `roofline` + design §6.5 N9.
* `INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE=1` (and the truthy
  variants accepted by the env-parse logic) restores N3 soft-hint
  behaviour as the documented escape hatch.
* Empty / unset / falsy env values keep the block in effect.
* `roofline` and other non-profile actions are unaffected by the gate.
* RooflineExecutor sub-step path (bypass) is NOT covered here — that's
  in test_roofline_executor.py (the executor directly invokes
  profile_executor without going through _sequence_denial_for_action).
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
# Test fixtures (mirror test_roofline_sequence_denial.py)
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


def _write_baseline_marker(sd: Path) -> Path:
    p = target_baseline_json(sd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return p


def _seed_post_baseline(coord: Coordinator) -> None:
    """target_analysis marker + baseline_tput open every gate upstream
    of the N9 profile gate."""
    _write_baseline_marker(coord.session_dir)
    coord.shared_state.baseline_tput = 100.0


# ---------------------------------------------------------------------------
# Default behaviour — propose(profile) is hard-blocked
# ---------------------------------------------------------------------------
def test_profile_propose_denied_by_default(session_dir, monkeypatch):
    """No env flag set → propose(profile) hard-rejected."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", raising=False)
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)

    denied = coord._sequence_denial_for_action("profile")
    assert isinstance(denied, PolicyDenied), \
        "profile must be denied by default per N9"
    assert denied.rule == "execution_order"
    assert "use `roofline` instead" in str(denied)
    assert "design/roofline-v2.md §6.5 N9" in str(denied)


def test_profile_denial_hint_mentions_roofline_and_escape_hatch(
    session_dir, monkeypatch,
):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", raising=False)
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    denied = coord._sequence_denial_for_action("profile")
    assert denied is not None
    assert denied.hint and "roofline" in denied.hint
    assert denied.hint and "INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE" in denied.hint
    # The hint must explain WHY profile is blocked (duplicates roofline's
    # internal work) so the LLM can self-correct rather than retry
    assert denied.hint and "atomic" in denied.hint.lower()


# ---------------------------------------------------------------------------
# Escape hatch — env flag restores N3 soft-hint behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "Yes", "on"])
def test_env_escape_hatch_allows_profile_propose(
    session_dir, monkeypatch, env_value,
):
    """All documented truthy env values restore N3 behaviour."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", env_value)
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    # With escape hatch on, profile passes the N9 gate. Other gates
    # (target_analysis JSON, baseline_tput) are already open via the
    # fixture, so the action passes the full chain.
    denied = coord._sequence_denial_for_action("profile")
    assert denied is None, (
        f"profile must be allowed with env={env_value!r}; "
        f"got denied={denied!r}"
    )


@pytest.mark.parametrize("env_value", ["", "0", "false", "no", "off", "garbage"])
def test_env_falsy_or_unknown_keeps_block(
    session_dir, monkeypatch, env_value,
):
    """Falsy / empty / unknown env values do NOT lift the block."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", env_value)
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    denied = coord._sequence_denial_for_action("profile")
    assert isinstance(denied, PolicyDenied), (
        f"profile must stay denied with env={env_value!r}"
    )


# ---------------------------------------------------------------------------
# N9 does NOT affect other actions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", [
    "roofline", "baseline", "sweep", "validate_stack",
    "report", "target_analysis",
])
def test_other_actions_unaffected_by_n9_gate(
    session_dir, monkeypatch, action,
):
    """The N9 gate fires ONLY for action == 'profile'. Every other
    sequence action either passes or fails on its own pre-N9 gate —
    never with the N9 'use roofline' message."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", raising=False)
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)

    denied = coord._sequence_denial_for_action(action)
    if denied is not None:
        # If denied at all, it must NOT be by the N9 gate.
        assert "use `roofline` instead" not in str(denied), (
            f"action={action!r} unexpectedly denied by N9 gate: {denied!r}"
        )


# ---------------------------------------------------------------------------
# Gate ordering — N9 fires before profile-gate but after baseline/target gates
# ---------------------------------------------------------------------------
def test_n9_fires_after_target_analysis_gate(session_dir, monkeypatch):
    """target_analysis JSON missing → target_analysis gate fires first,
    not N9. (Operator should see the more fundamental denial.)"""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", raising=False)
    coord = Coordinator(session_dir, backends=_backends_full())
    # NO target_baseline.json written
    denied = coord._sequence_denial_for_action("profile")
    assert isinstance(denied, PolicyDenied)
    assert "target_analysis must run first" in str(denied)
    # Must NOT be the N9 message
    assert "use `roofline`" not in str(denied)


def test_n9_fires_after_baseline_gate(session_dir, monkeypatch):
    """baseline_tput=0 → baseline gate fires first, not N9."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE", raising=False)
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_marker(coord.session_dir)
    # baseline_tput still 0
    denied = coord._sequence_denial_for_action("profile")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)
    assert "use `roofline`" not in str(denied)


# ---------------------------------------------------------------------------
# orchestration.md guidance section presence
# ---------------------------------------------------------------------------
def test_orchestration_md_includes_n9_hard_rule():
    """The orchestration system prompt must include the explicit
    'NEVER propose profile directly' rule so the LLM sees the constraint
    even before hitting PolicyGate."""
    from inference_optimizer.paths import asset_system_prompts_dir
    text = (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8",
    )
    assert "NEVER propose `profile` directly" in text
    assert "design §6.5 N9" in text
    # Must explain alternative
    assert "roofline" in text
