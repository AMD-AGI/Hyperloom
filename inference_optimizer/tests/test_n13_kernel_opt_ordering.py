"""Roofline-v2 N13: kernel_opt requires post-cheap-exploration snapshot.

GPU-empirical follow-up to N12 (DeepSeek-R1 session findings):
N12 made the orchestration system prompt tell the LLM "kernel_opt
last" as a HARD RULE, but prompt-layer guidance is not enough — the
LLM may still emit `request{kind="run_optimization"}` prematurely
(empirical N9 lesson: profile direct-propose was N3-soft-deprecated
and the LLM still did it; only the N9 PolicyGate hard-block worked).

N13 enforces the ordering in `_sequence_denial_for_request` for
`req_kind == "run_optimization"`: rejects unless
   backends_attempts >= 1 AND
   params_attempts   >= 1 AND
   snapshot_id       >= 2

The 3rd prerequisite is the most important: snapshot #1 is the
baseline trace; snapshot #2 (or later) must be a fresh roofline
re-run after cheap exploration, capturing the post-cheap kernel
distribution. Running kernel_opt against snapshot #1 after enabling
`--enable-torch-compile` or a different attention backend would
target kernels that may no longer be on the critical path.

Escape hatch: `INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1` restores
pre-N13 behaviour. Use cases: v0 baseline comparison, debug, unit
tests.
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
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
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


def _seed_through_roofline_snapshot_1(coord: Coordinator) -> None:
    """Open every prerequisite up to snapshot #1 — kernel_opt should
    still be denied (need snapshot >= 2)."""
    _write_baseline_marker(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.trace.json.gz"
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.trace.json.gz",
        "analysis_md_text": "FAKE",
        "roofline_snapshot_id": 1,
    }


def _seed_cheap_exploration_done(coord: Coordinator) -> None:
    """After backends + params each ran once (still snapshot #1).
    Should still be denied — snapshot must advance to >= 2 too."""
    _seed_through_roofline_snapshot_1(coord)
    s = coord.shared_state
    s.backends_attempts = [{"status": "succeeded"}]
    s.params_attempts = [{"status": "succeeded"}]


def _seed_post_re_roofline(coord: Coordinator) -> None:
    """All 3 prerequisites met — kernel_opt should be allowed."""
    _seed_cheap_exploration_done(coord)
    s = coord.shared_state
    s.last_trace_analyze["roofline_snapshot_id"] = 2


# ---------------------------------------------------------------------------
# N13 prerequisites — denied paths
# ---------------------------------------------------------------------------
def test_kernel_opt_denied_before_any_cheap_exploration(session_dir, monkeypatch):
    """Right after first roofline (snapshot=1), no backends/params yet
    → all 3 prereqs missing."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "kernel_opt requires post-cheap-exploration snapshot" in str(denied)
    # All 3 prereqs surfaced in the error
    assert "backends_attempts=0" in str(denied)
    assert "params_attempts=0" in str(denied)
    assert "snapshot_id=1" in str(denied)


def test_kernel_opt_denied_when_only_backends_done(session_dir, monkeypatch):
    """backends done, params missing, snapshot still 1."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)
    coord.shared_state.backends_attempts = [{"status": "succeeded"}]

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    # backends pre-req met, but params + snapshot still missing
    assert "backends_attempts" not in str(denied)
    assert "params_attempts=0" in str(denied)
    assert "snapshot_id=1" in str(denied)


def test_kernel_opt_denied_when_only_params_done(session_dir, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)
    coord.shared_state.params_attempts = [{"status": "succeeded"}]

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    assert "backends_attempts=0" in str(denied)
    assert "params_attempts" not in str(denied)
    assert "snapshot_id=1" in str(denied)


def test_kernel_opt_denied_when_snapshot_still_1_after_cheap_round(
    session_dir, monkeypatch,
):
    """backends + params done but didn't re-roofline → snapshot
    stays at 1; still denied. This is the central N13 insight: the
    LLM must re-propose roofline AFTER cheap actions, not just rely
    on the first snapshot."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_cheap_exploration_done(coord)
    # snapshot still 1
    assert coord.shared_state.last_trace_analyze["roofline_snapshot_id"] == 1

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    # Only snapshot prereq missing now
    assert "backends_attempts" not in str(denied)
    assert "params_attempts" not in str(denied)
    assert "snapshot_id=1" in str(denied)
    assert "re-propose roofline" in str(denied)


def test_kernel_opt_denial_hint_mentions_re_roofline_and_escape_hatch(
    session_dir, monkeypatch,
):
    """Hint must be actionable: tell the LLM to re-propose roofline +
    document the escape hatch for operators."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_cheap_exploration_done(coord)

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert denied is not None and denied.hint is not None
    hint = denied.hint
    assert "roofline" in hint
    assert "snapshot_id" in hint
    assert "INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT" in hint
    assert "§6.5.1" in hint  # design doc reference


# ---------------------------------------------------------------------------
# N13 prerequisites — allowed paths
# ---------------------------------------------------------------------------
def test_kernel_opt_allowed_after_all_prereqs_met(session_dir, monkeypatch):
    """Happy path — all 3 prereqs met → kernel_opt passes the gate."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_post_re_roofline(coord)

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert denied is None


@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "Yes", "on"])
def test_escape_hatch_allows_kernel_opt_with_snapshot_1(
    session_dir, monkeypatch, env_value,
):
    """Escape hatch restores pre-N13 behaviour: kernel_opt allowed at
    snapshot #1 even without backends/params attempts."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", env_value)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)
    # No backends/params attempts seeded — escape hatch should still allow

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert denied is None, (
        f"escape hatch env={env_value!r} must allow kernel_opt; got {denied!r}"
    )


@pytest.mark.parametrize("env_value", ["", "0", "false", "no", "off", "garbage"])
def test_escape_hatch_falsy_keeps_block(session_dir, monkeypatch, env_value):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", env_value)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)
    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)


# ---------------------------------------------------------------------------
# N13 does NOT affect other REQUEST kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["trace_analyze", "integrate", "apply_patch"])
def test_n13_does_not_affect_other_request_kinds(session_dir, monkeypatch, kind):
    """Only `run_optimization` (the kernel_opt REQUEST path) is
    subject to N13 prereqs. Other kinds still go through the
    pre-existing checks but never the snapshot/attempts triple."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    _seed_through_roofline_snapshot_1(coord)

    denied = coord._sequence_denial_for_request("kernel", kind)
    # If denied at all, it must NOT be the N13 message
    if denied is not None:
        assert "kernel_opt requires post-cheap-exploration snapshot" not in str(denied)


# ---------------------------------------------------------------------------
# Gate ordering — pre-existing gates fire before N13 layer
# ---------------------------------------------------------------------------
def test_pre_n13_gates_fire_before_n13_layer(session_dir, monkeypatch):
    """If baseline gate fires (no baseline_tput), we must see THAT
    error, not the N13 error — the operator should see the most
    fundamental denial first."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    # No baseline_tput, no trace, nothing — baseline gate must fire
    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)
    assert "kernel_opt requires" not in str(denied)


def test_n13_fires_after_trace_analyze_gate(session_dir, monkeypatch):
    """If last_trace_analyze.trace_input differs from
    last_profile_trace, the pre-existing trace_analyze gate must
    fire BEFORE N13."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.gz"
    # Mismatch — pre-N13 gate should fire
    s.last_trace_analyze = {"trace_input": "/different/trace.gz"}

    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    assert "trace_analyze must run first" in str(denied)
    assert "kernel_opt requires" not in str(denied)


# ---------------------------------------------------------------------------
# Escape hatch helper
# ---------------------------------------------------------------------------
def test_allow_early_kernel_opt_helper_recognises_truthy_values(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "1")
    assert Coordinator._allow_early_kernel_opt() is True
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "FALSE")
    assert Coordinator._allow_early_kernel_opt() is False
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
    assert Coordinator._allow_early_kernel_opt() is False
