"""Roofline-v2 N19c + N21: gain-driven kernel_opt unlock + flags-conditional
roofline re-runs (replaces N13/N14 counter-driven enforcement).

Context — what changed from N13/N14 (counter-driven) to N19c+N21
(gain-driven):

* N13/N14 enforced: kernel_opt unlock iff
    backends_attempts >= 2 AND params_attempts >= 2 AND snapshot_id >= 3
  Rationale was "force multi-round interleaved cheap exploration before
  letting LLM jump to kernel_opt". Empirically wasteful — 82% of 202
  historical sessions never produced a clear single-action winner > 0.5%,
  so forcing them through 2 full b/p rounds before kernel_opt burned
  hours per session for no marginal gain.

* N19c replaces with: kernel_opt unlock iff
    snapshot_id >= 1 AND
    (at least one backends or params attempt has recorded a delta) AND
    last_cheap_delta_gain < EPSILON (default 0.3%, env-overridable)
  Rationale: when cheap exploration stops finding marginal gain (vs
  current_best), continuing to spend wall-clock on cheap rounds is
  pure waste — kernel_opt is where the remaining leverage lives.

* N21 (companion): roofline propose denied if discovered_flags is
  unchanged since the last snapshot. Without this, the LLM might
  re-propose roofline between cheap rounds that didn't promote new
  flags — that would launch sglang with byte-identical args, produce
  a byte-equivalent trace, and waste 20-35min for zero new information.

Together N19c+N21 implement the user's roofline-driven flow:
    baseline -> roofline_1
             -> cheap (b OR p)
             ├─ found gain (flag promoted) -> roofline_2 allowed
             │                              -> next cheap allowed
             └─ no gain (flag unchanged)   -> roofline DENIED (N21)
                                           -> kernel_opt UNLOCKED (N19c)

Escape hatches preserved from N13:
  INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1 — bypass N19c
  INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN=1   — bypass N21
  INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON  — tune EPSILON
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
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _cheap_exhausted_epsilon,
)
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


def _seed_post_baseline_no_snapshot(coord: Coordinator) -> None:
    """Baseline done, no roofline yet."""
    _write_baseline_marker(coord.session_dir)
    coord.shared_state.baseline_tput = 100.0


def _seed_post_snapshot_1(coord: Coordinator) -> None:
    """Baseline + roofline_1 done."""
    _seed_post_baseline_no_snapshot(coord)
    coord.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.trace.json.gz",
        "analysis_md_text": "FAKE",
        "roofline_snapshot_id": 1,
    }
    # Freeze the flags snapshot — empty here because no cheap action ran yet.
    coord.shared_state.discovered_flags_at_last_snapshot = {}


def _seed_cheap_done_exhausted(coord: Coordinator) -> None:
    """Snapshot #1 + one cheap attempt that found 0 gain (delta < EPS)."""
    _seed_post_snapshot_1(coord)
    coord.shared_state.backends_attempts = [{"status": "succeeded"}]
    coord.shared_state.last_cheap_delta_gain = 0.0


def _seed_cheap_done_with_gain(coord: Coordinator) -> None:
    """Snapshot #1 + one cheap attempt that found gain >= EPS."""
    _seed_post_snapshot_1(coord)
    coord.shared_state.params_attempts = [{"status": "succeeded"}]
    coord.shared_state.last_cheap_delta_gain = 1.5  # well above EPS


# ===========================================================================
# N19c — gain-driven kernel_opt unlock
# ===========================================================================
class TestN19cKernelOptUnlock:
    def test_kernel_opt_denied_before_any_snapshot(self, session_dir, monkeypatch):
        """No roofline_1 yet -> kernel_opt denied. The pre-existing
        'profile/roofline must run first' gate fires first (since
        last_profile_trace is empty); N19c is the layer BEHIND it. Either
        way the operator gets a denial with an actionable hint."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_baseline_no_snapshot(coord)

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert isinstance(denied, PolicyDenied)
        # Either pre-gate ("profile must run first") OR N19c
        # ("snapshot_id=0") may fire — both are correct denials. The
        # important contract is: a fresh session with no roofline
        # NEVER unlocks kernel_opt without the escape hatch.
        assert denied.rule == "execution_order"
        assert any(s in str(denied) for s in (
            "profile must run first", "snapshot_id=0",
        ))

    def test_kernel_opt_denied_when_snapshot_but_no_cheap(self, session_dir, monkeypatch):
        """snapshot_id=1 done but no cheap attempts yet -> denied to force
        the LLM to probe the flag space at least once."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert isinstance(denied, PolicyDenied)
        assert "no cheap exploration yet" in str(denied)
        assert "backends OR params" in str(denied)

    def test_kernel_opt_denied_when_cheap_still_finding_gain(
        self, session_dir, monkeypatch,
    ):
        """cheap attempt recorded delta_gain >= EPSILON -> denied,
        cheap exploration should continue."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_cheap_done_with_gain(coord)
        assert coord.shared_state.last_cheap_delta_gain == 1.5

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert isinstance(denied, PolicyDenied)
        assert "last_cheap_delta_gain=1.500%" in str(denied)
        assert "cheap exploration still finding gain" in str(denied)

    def test_kernel_opt_allowed_when_cheap_exhausted(self, session_dir, monkeypatch):
        """cheap attempt recorded delta_gain < EPSILON -> unlocked.
        This is the core N19c win: no need to force more cheap rounds
        when the search space has already converged."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_cheap_done_exhausted(coord)
        assert coord.shared_state.last_cheap_delta_gain == 0.0
        # Note: under N14 this state (only backends=1, params=0, snapshot=1)
        # would have been DENIED. N19c flips it to ALLOWED — that's the
        # whole point of the gain-driven rule.

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert denied is None, (
            f"N19c must unlock kernel_opt when cheap exhausted; got: {denied!r}"
        )

    def test_kernel_opt_denied_when_delta_unset_with_cheap_attempts(
        self, session_dir, monkeypatch,
    ):
        """Defensive: if cheap attempts exist but last_cheap_delta_gain is
        None (corrupted state.json / legacy resume), deny with a hint to
        re-run a cheap action."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        coord.shared_state.backends_attempts = [{"status": "succeeded"}]
        coord.shared_state.last_cheap_delta_gain = None

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert isinstance(denied, PolicyDenied)
        assert "last_cheap_delta_gain not recorded" in str(denied)
        assert "re-run a cheap action" in str(denied)

    @pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "Yes", "on"])
    def test_escape_hatch_overrides_n19c_check(
        self, session_dir, monkeypatch, env_value,
    ):
        """INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1 must bypass the
        N19c-specific check (cheap exploration + delta_gain). Pre-N19c
        gates (baseline/profile/trace_analyze must run first) still
        apply — the escape hatch only targets N19c, not the entire
        request pipeline. Seed past those pre-gates first, then verify
        N19c is bypassed."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", env_value)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        # snapshot=1 + zero cheap attempts -> N19c would normally deny
        # ("no cheap exploration yet"). Escape hatch must bypass it.

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert denied is None, (
            f"escape hatch env={env_value!r} must bypass N19c; got: {denied!r}"
        )

    def test_n19c_does_not_affect_other_request_kinds(self, session_dir, monkeypatch):
        """N19c targets kind='run_optimization' only. Other kinds
        (trace_analyze, integrate, apply_patch, ...) pass through the
        pre-existing gates unchanged."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_baseline_no_snapshot(coord)

        for kind in ("trace_analyze", "integrate", "apply_patch"):
            denied = coord._sequence_denial_for_request("kernel", kind)
            if denied is not None:
                assert "kernel_opt requires either" not in str(denied), (
                    f"kind={kind!r} should not trip N19c rule: {denied!r}"
                )

    def test_hint_mentions_design_doc_and_escape_hatches(
        self, session_dir, monkeypatch,
    ):
        """Operator-visible diagnostics: design doc reference + env var
        names for tuning EPSILON and escaping the gate. Must seed past
        pre-gates so the N19c-specific denial is the one returned."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        # snapshot=1, zero cheap attempts -> N19c "no cheap exploration"

        denied = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert denied is not None and denied.hint is not None
        h = denied.hint
        assert "§6.5.3" in h
        assert "N19c" in h
        assert "INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT" in h
        assert "INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON" in h


# ===========================================================================
# N19c EPSILON env override
# ===========================================================================
class TestN19cEpsilon:
    def test_default_epsilon(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", raising=False)
        assert _cheap_exhausted_epsilon() == 0.3

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", "1.0")
        assert _cheap_exhausted_epsilon() == 1.0

    def test_negative_falls_back_to_default(self, monkeypatch):
        """Defensive: a negative epsilon would make every cheap round
        'exhausted' (any positive gain is >= 0). Fall back to 0.3."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", "-1.0")
        assert _cheap_exhausted_epsilon() == 0.3

    def test_garbage_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", "garbage")
        assert _cheap_exhausted_epsilon() == 0.3

    def test_higher_epsilon_unlocks_sooner(self, session_dir, monkeypatch):
        """EPSILON sets the 'cheap exhausted' threshold. Higher EPSILON
        means even moderate gains are treated as exhausted (faster
        unlock); lower EPSILON keeps the gate closed until cheap really
        flatlines.

        With recorded delta_gain = 1.5:
          * EPSILON = 0.3 (default): 1.5 >= 0.3 -> deny ('still finding gain')
          * EPSILON = 2.0          : 1.5 <  2.0 -> allow (declared exhausted)
        """
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_cheap_done_with_gain(coord)
        # delta_gain = 1.5

        monkeypatch.delenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", raising=False)
        denied_low = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert isinstance(denied_low, PolicyDenied), (
            "EPSILON=0.3, delta=1.5 should deny (still finding gain)"
        )
        assert "EPSILON=0.300%" in str(denied_low)

        monkeypatch.setenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", "2.0")
        denied_high = coord._sequence_denial_for_request("kernel", "run_optimization")
        assert denied_high is None, (
            f"EPSILON=2.0, delta=1.5 should allow (declared exhausted); "
            f"got: {denied_high!r}"
        )


# ===========================================================================
# N21 — flags-conditional roofline re-runs
# ===========================================================================
class TestN21RooflineGate:
    def test_first_roofline_always_allowed(self, session_dir, monkeypatch):
        """No snapshot exists yet -> roofline allowed (this is the
        baseline analysis snapshot)."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_baseline_no_snapshot(coord)
        # snapshot_id absent / 0

        denied = coord._sequence_denial_for_action("roofline")
        assert denied is None

    def test_second_roofline_denied_when_flags_unchanged(
        self, session_dir, monkeypatch,
    ):
        """snapshot=1 already exists, discovered_flags equals frozen
        snapshot -> roofline denied (would produce identical trace)."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        # Both empty {} -> equivalent
        coord.shared_state.discovered_flags = {}
        coord.shared_state.discovered_flags_at_last_snapshot = {}

        denied = coord._sequence_denial_for_action("roofline")
        assert isinstance(denied, PolicyDenied)
        assert "snapshot #1" in str(denied)
        assert "byte-equivalent" in str(denied)
        assert "INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN" in denied.hint

    def test_second_roofline_allowed_when_flags_changed(
        self, session_dir, monkeypatch,
    ):
        """snapshot=1, cheap action promoted a new flag -> new sglang
        launch args -> roofline allowed (new trace will differ)."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        coord.shared_state.discovered_flags_at_last_snapshot = {
            "sglang": {"backend_flags": [], "param_flags": []}
        }
        coord.shared_state.discovered_flags = {
            "sglang": {"backend_flags": ["--attention-backend triton"],
                       "param_flags": []}
        }

        denied = coord._sequence_denial_for_action("roofline")
        assert denied is None

    def test_flags_equivalent_ignores_metadata_fields(self, session_dir, monkeypatch):
        """`source_path` and other infrastructural metadata in
        discovered_flags shouldn't trigger a re-roofline. Only the actual
        backend_flags + param_flags sets count."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        coord.shared_state.discovered_flags_at_last_snapshot = {
            "sglang": {
                "backend_flags": ["--attention-backend triton"],
                "param_flags":   ["--cuda-graph-max-bs 512"],
                "source_path":   "/path/A",
            }
        }
        coord.shared_state.discovered_flags = {
            "sglang": {
                "backend_flags": ["--attention-backend triton"],
                "param_flags":   ["--cuda-graph-max-bs 512"],
                "source_path":   "/path/B",  # different metadata, same flags
            }
        }

        denied = coord._sequence_denial_for_action("roofline")
        assert isinstance(denied, PolicyDenied), (
            "metadata-only diff should still trigger N21 deny"
        )

    @pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "Yes", "on"])
    def test_force_rerun_escape_hatch(self, session_dir, monkeypatch, env_value):
        """INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN=1 bypasses N21 even
        when flags are unchanged (operator debugging / pinning a known
        trace for regression test)."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", env_value)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        coord.shared_state.discovered_flags = {}
        coord.shared_state.discovered_flags_at_last_snapshot = {}

        denied = coord._sequence_denial_for_action("roofline")
        assert denied is None, (
            f"escape hatch env={env_value!r} must bypass N21; got: {denied!r}"
        )

    def test_n21_does_not_affect_other_actions(self, session_dir, monkeypatch):
        """N21 targets action='roofline' only. backends/params/sweep/
        kernel_opt etc. don't go through this gate."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FORCE_ROOFLINE_RERUN", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_snapshot_1(coord)
        coord.shared_state.discovered_flags = {}
        coord.shared_state.discovered_flags_at_last_snapshot = {}

        for action in ("backends", "params", "sweep"):
            denied = coord._sequence_denial_for_action(action)
            if denied is not None:
                assert "byte-equivalent" not in str(denied), (
                    f"action={action!r} should not trip N21: {denied!r}"
                )


# ===========================================================================
# _flags_equivalent helper
# ===========================================================================
class TestFlagsEquivalent:
    def test_empty_dicts_equivalent(self):
        assert Coordinator._flags_equivalent({}, {}) is True

    def test_same_flag_sets_equivalent(self):
        a = {"sglang": {"backend_flags": ["--x", "--y"], "param_flags": []}}
        b = {"sglang": {"backend_flags": ["--y", "--x"], "param_flags": []}}
        # Order-independent (sets) -> equivalent.
        assert Coordinator._flags_equivalent(a, b) is True

    def test_different_flag_sets_not_equivalent(self):
        a = {"sglang": {"backend_flags": ["--x"], "param_flags": []}}
        b = {"sglang": {"backend_flags": ["--y"], "param_flags": []}}
        assert Coordinator._flags_equivalent(a, b) is False

    def test_different_frameworks_not_equivalent(self):
        a = {"sglang": {"backend_flags": [], "param_flags": []}}
        b = {"vllm":   {"backend_flags": [], "param_flags": []}}
        assert Coordinator._flags_equivalent(a, b) is False

    def test_param_flags_independent_of_backend_flags(self):
        a = {"sglang": {"backend_flags": ["--x"], "param_flags": ["--p1"]}}
        b = {"sglang": {"backend_flags": ["--x"], "param_flags": ["--p2"]}}
        assert Coordinator._flags_equivalent(a, b) is False

    def test_handles_missing_keys_gracefully(self):
        a = {"sglang": {"backend_flags": ["--x"]}}  # no param_flags
        b = {"sglang": {"backend_flags": ["--x"], "param_flags": []}}
        # Both treat missing as empty set -> equivalent.
        assert Coordinator._flags_equivalent(a, b) is True

    def test_non_dict_falls_through_to_eq(self):
        assert Coordinator._flags_equivalent({"x": 1}, {"x": 1}) is True
        assert Coordinator._flags_equivalent({"x": 1}, {"x": 2}) is False
