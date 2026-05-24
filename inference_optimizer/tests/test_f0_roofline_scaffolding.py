"""F0 — roofline + framework-agent scaffolding placeholders.

These tests verify that the F0 pre-merge scaffolding lands correctly:

* F0-8: ``TraceAnalyzeSnapshot`` reader + 3 placeholder fields on
  ``SharedState`` (``last_trace_analyze`` / ``roofline_snapshot_id`` /
  ``roofline_saturation_history``).
* F0-9: ``'roofline'`` is in the EXPLORE / KERNEL / CLOSE phase
  allowlists; legacy ``'profile'`` stays in EXPLORE / KERNEL as the
  default-denied (per N9) escape hatch. ``'pmc_roofline'`` has been
  physically removed.
* F0-10: 5 integration toggles default to ``False`` on ``SharedState``
  and the matching ``--use-roofline-composite`` / ... CLI flags
  resolve from env vars.

F1 ports the actual writer (``record_trace_analyze``) and registers
the executor; F2 wires framework-agent into ``serving_specialist``;
F3 adds PolicyGate rules. None of these later steps are exercised here.

Reference: ``plan_roofline_framework/F0_pre_merge.MD`` §9-§11.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# F0-8 — TraceAnalyzeSnapshot + SharedState placeholder fields
# ---------------------------------------------------------------------------


def test_trace_analyze_snapshot_default_empty():
    from inference_optimizer.orchestrator.shared_state import TraceAnalyzeSnapshot

    snap = TraceAnalyzeSnapshot()
    assert snap.trace_input == ""
    assert snap.candidates_path == ""
    assert snap.hot_kernels_top15 == []
    assert snap.task_groups == []
    assert snap.reusable_native_kernel_ids == []
    assert snap.trace_health_warnings == []
    assert snap.analysis_md_path == ""
    assert snap.analysis_md_text == ""
    assert snap.roofline_snapshot_id == 0
    assert snap.roofline_baseline_gain_at_snapshot == 0.0
    assert snap.ts == ""


def test_trace_analyze_snapshot_from_dict_canonical():
    from inference_optimizer.orchestrator.shared_state import TraceAnalyzeSnapshot

    snap = TraceAnalyzeSnapshot.from_dict({
        "trace_input": "/tmp/trace.json",
        "candidates_path": "/tmp/candidates.json",
        "hot_kernels_top15": [{"kernel_id": "k001", "gpu_pct": 28.78}],
        "reusable_native_kernel_ids": ["k001"],
        "trace_health_warnings": [{"code": "high_gpu_idle_pct"}],
        "analysis_md_path": "/tmp/analysis.md",
        "analysis_md_text": "# Hello",
        "roofline_snapshot_id": 3,
        "roofline_baseline_gain_at_snapshot": 12.5,
        "ts": "2026-05-24T14:00:00Z",
    })
    assert snap.trace_input == "/tmp/trace.json"
    assert snap.candidates_path == "/tmp/candidates.json"
    assert snap.hot_kernels_top15 == [{"kernel_id": "k001", "gpu_pct": 28.78}]
    assert snap.reusable_native_kernel_ids == ["k001"]
    assert snap.trace_health_warnings == [{"code": "high_gpu_idle_pct"}]
    assert snap.analysis_md_path == "/tmp/analysis.md"
    assert snap.analysis_md_text == "# Hello"
    assert snap.roofline_snapshot_id == 3
    assert snap.roofline_baseline_gain_at_snapshot == pytest.approx(12.5)
    assert snap.ts == "2026-05-24T14:00:00Z"


def test_trace_analyze_snapshot_from_dict_handles_none():
    from inference_optimizer.orchestrator.shared_state import TraceAnalyzeSnapshot

    snap = TraceAnalyzeSnapshot.from_dict(None)
    assert snap.trace_input == ""
    assert snap.roofline_snapshot_id == 0


def test_shared_state_has_roofline_placeholder_fields():
    from inference_optimizer.orchestrator.shared_state import SharedState

    s = SharedState()
    assert s.last_trace_analyze == {}
    assert isinstance(s.last_trace_analyze, dict)

    assert s.roofline_snapshot_id == 0
    assert s.roofline_saturation_history == []


def test_shared_state_roofline_history_is_independent_per_instance():
    """Catch dataclass field-default-mutable bugs (would silently share lists)."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    a = SharedState()
    b = SharedState()
    a.roofline_saturation_history.append({"snapshot_id": 1})
    assert b.roofline_saturation_history == []


# ---------------------------------------------------------------------------
# F0-9 — phase allowlist gains 'roofline' in EXPLORE / KERNEL / CLOSE
# ---------------------------------------------------------------------------


def test_roofline_in_explore_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_EXPLORE,
        is_action_allowed_in_phase,
    )

    assert is_action_allowed_in_phase("roofline", PHASE_EXPLORE)


def test_roofline_in_kernel_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_KERNEL,
        is_action_allowed_in_phase,
    )

    assert is_action_allowed_in_phase("roofline", PHASE_KERNEL)


def test_roofline_in_close_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_CLOSE,
        is_action_allowed_in_phase,
    )

    assert is_action_allowed_in_phase("roofline", PHASE_CLOSE)


def test_roofline_not_in_prelude_or_sweep():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_PRELUDE,
        PHASE_SWEEP,
        is_action_allowed_in_phase,
    )

    assert not is_action_allowed_in_phase("roofline", PHASE_PRELUDE)
    assert not is_action_allowed_in_phase("roofline", PHASE_SWEEP)


def test_legacy_profile_still_in_explore():
    """The legacy ``profile`` entry point stays in EXPLORE — F3's
    ``--deny-direct-profile`` toggle owns the propose-time denial when
    the operator wants to force the composite path. ``pmc_roofline``
    has been physically removed."""
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_EXPLORE,
        is_action_allowed_in_phase,
    )

    assert is_action_allowed_in_phase("profile", PHASE_EXPLORE)
    assert not is_action_allowed_in_phase("pmc_roofline", PHASE_EXPLORE)


# ---------------------------------------------------------------------------
# Default toggle matrix — the three Roofline-v2 / framework-agent
# toggles default ON; the gain-driven gate + saturation advisory stay
# opt-in.
# ---------------------------------------------------------------------------


def test_shared_state_default_toggles():
    from inference_optimizer.orchestrator.shared_state import SharedState

    s = SharedState()
    # Default-on: roofline composite is the canonical analysis path,
    # framework-agent is wired by default, N9 hard-denies legacy
    # ``profile`` propose so the LLM cannot diverge from the snapshot.
    assert s.use_roofline_composite is True
    assert s.framework_agent_enabled is True
    assert s.deny_direct_profile is True
    # Default-off: opt-in tuning knobs.
    assert s.gain_driven_kernel_opt is False
    assert s.roofline_saturation_advisory is False


def test_cli_parser_exposes_integration_toggles():
    from inference_optimizer import cli

    parser = cli._build_parser()
    namespace = parser.parse_args(["optimize", "--model", "/tmp/x"])
    expected_defaults = {
        "use_roofline_composite": True,
        "framework_agent_enabled": True,
        "deny_direct_profile": True,
        "gain_driven_kernel_opt": False,
        "roofline_saturation_advisory": False,
    }
    for attr, expected in expected_defaults.items():
        assert hasattr(namespace, attr), f"CLI missing attribute {attr}"
        assert getattr(namespace, attr) is expected, (
            f"CLI default for {attr} must be {expected}"
        )


def test_cli_default_on_toggles_can_be_disabled_via_no_flag():
    """``--no-...`` opt-out exists for the three default-on toggles."""
    from inference_optimizer import cli

    parser = cli._build_parser()
    namespace = parser.parse_args([
        "optimize", "--model", "/tmp/x",
        "--no-use-roofline-composite",
        "--no-framework-agent-enabled",
        "--no-deny-direct-profile",
    ])
    assert namespace.use_roofline_composite is False
    assert namespace.framework_agent_enabled is False
    assert namespace.deny_direct_profile is False


def test_cli_default_on_toggles_respect_env_var(monkeypatch):
    """Env=0 turns the default-on toggles off, so a CI box without a
    GPU profile lane can flip them with one shared env var."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE", "0")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED", "0")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE", "0")

    from inference_optimizer import cli

    parser = cli._build_parser()
    namespace = parser.parse_args(["optimize", "--model", "/tmp/x"])
    assert namespace.use_roofline_composite is False
    assert namespace.framework_agent_enabled is False
    assert namespace.deny_direct_profile is False


def test_cli_opt_in_toggles_respect_env_var(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY", "1")

    from inference_optimizer import cli

    parser = cli._build_parser()
    namespace = parser.parse_args(["optimize", "--model", "/tmp/x"])
    assert namespace.gain_driven_kernel_opt is True
    assert namespace.roofline_saturation_advisory is True
