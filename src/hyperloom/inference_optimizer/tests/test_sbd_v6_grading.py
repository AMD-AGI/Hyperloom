# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``metadata.grading``: the axis a session's gains were graded on.

Every throughput field in the breakdown is the output axis by construction, so
without this block an AgentX replay -- ranked on total token throughput, which
runs ~140x the output figure on the canonical corpus -- is indistinguishable
from a synthetic run, and a consumer will sort one against the other.
"""

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT, GRADED_TOTAL
from hyperloom.inference_optimizer.breakdown.collectors.v6 import (
    collect_v6_metadata,
    collect_v6_outcome,
)

# A baseline that carries the graded pair. ``perf_snapshot_from_mapping``
# returns None unless BOTH total (or input+output) and intvty p90 are positive.
_AXES = {"total_throughput": 25978.0, "output_throughput": 183.0, "intvty_p90": 4.2}


def _grading(monkeypatch, *, state=None, framework="sglang", env=None):
    """Project ``metadata.grading`` for one state/env combination."""
    # The helper consults these directly, so a developer machine that happens
    # to export them must not decide the outcome of the test.
    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_PERF_METRIC", "HYPERLOOM_PERF_NOISE_PCT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    metadata = collect_v6_metadata(
        exported_at_utc="2026-09-02T00:00:00+00:00",
        session={"session_id": "s1"},
        workload={"framework_name": framework},
        model_info={},
        langfuse={},
        versions={},
        state=state or {},
        warnings=[],
    )
    return metadata["grading"]


def test_a_synthetic_session_declares_the_output_axis(monkeypatch):
    grading = _grading(monkeypatch)

    assert grading["benchmark_mode"] == "synthetic"
    assert grading["objective"] == GRADED_OUTPUT
    assert grading["intvty_veto"]["enabled"] is False
    assert grading["degrade_reason"] is None


def test_an_agentx_session_with_both_axes_declares_the_total_axis(monkeypatch):
    grading = _grading(
        monkeypatch,
        state={
            "benchmark_mode": "agentx",
            "baseline_perf": dict(_AXES),
            "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 5.0},
        },
    )

    assert grading["benchmark_mode"] == "agentx"
    assert grading["objective"] == GRADED_TOTAL
    assert grading["intvty_veto"]["enabled"] is True
    assert grading["intvty_veto"]["noise_pct"] == pytest.approx(5.0)
    assert grading["degrade_reason"] is None


def test_the_mode_is_read_from_state_not_from_the_exporting_shell(monkeypatch):
    """An export driven from a shell that never had the env var still reports agentx.

    ``benchmark_mode`` is stamped at seed precisely so it outlives the shell;
    CLOSE frequently runs from a subprocess that did not inherit it.
    """
    grading = _grading(
        monkeypatch,
        state={"benchmark_mode": "agentx", "baseline_perf": dict(_AXES)},
    )

    assert grading["objective"] == GRADED_TOTAL


def test_an_agentx_session_missing_the_axes_reports_the_axis_it_actually_used(monkeypatch):
    """Asking for the total axis is not the same as having graded on it.

    ``objective`` names what the gains in this document are on, so a degraded
    session must not be read as a comparable AgentX result.
    """
    grading = _grading(
        monkeypatch,
        state={"benchmark_mode": "agentx", "baseline_perf": {"output_throughput": 183.0}},
    )

    assert grading["objective"] == GRADED_OUTPUT
    assert grading["degrade_reason"] == "baseline_axes_missing"
    assert grading["intvty_veto"]["enabled"] is False


def test_a_scriptable_framework_keeps_the_output_axis(monkeypatch):
    """xDiT reports an image-quality gate, so it has no total-token axis at all."""
    grading = _grading(
        monkeypatch,
        state={"benchmark_mode": "agentx", "baseline_perf": dict(_AXES)},
        framework="xdit",
    )

    assert grading["objective"] == GRADED_OUTPUT
    assert grading["degrade_reason"] is None


def test_an_explicit_perf_metric_override_is_honoured_through_the_recording(monkeypatch):
    """``HYPERLOOM_PERF_METRIC`` decides at seed, and the seed value is what ships.

    The override still works in both directions; it is simply resolved where
    the run can see it rather than where the export happens to run.
    """
    grading = _grading(
        monkeypatch,
        state={
            "benchmark_mode": "agentx",
            "baseline_perf": dict(_AXES),
            "grading": {"objective": GRADED_OUTPUT, "intvty_noise_pct": 5.0},
        },
    )

    assert grading["objective"] == GRADED_OUTPUT
    # Still an AgentX run -- the mode records what ran, the objective records
    # how it was graded, and the two are allowed to disagree.
    assert grading["benchmark_mode"] == "agentx"


def test_the_veto_band_comes_from_the_recording(monkeypatch):
    grading = _grading(
        monkeypatch,
        state={
            "benchmark_mode": "agentx",
            "baseline_perf": dict(_AXES),
            "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 3.5},
        },
    )

    assert grading["intvty_veto"]["noise_pct"] == pytest.approx(3.5)


def test_a_recorded_axis_wins_over_the_exporting_shell(monkeypatch):
    """CLOSE exports from a subprocess whose env can disagree with the run's.

    Re-deriving there would report an axis the session never graded on, so a
    recorded objective must not be second-guessed.
    """
    grading = _grading(
        monkeypatch,
        state={
            "benchmark_mode": "agentx",
            "baseline_perf": dict(_AXES),
            "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 2.5},
        },
        # The exporting shell says output; the recording says total.
        env={"HYPERLOOM_PERF_METRIC": "output_tput", "HYPERLOOM_PERF_NOISE_PCT": "9.9"},
    )

    assert grading["objective"] == GRADED_TOTAL
    assert grading["intvty_veto"]["noise_pct"] == pytest.approx(2.5)


def test_a_session_seeded_before_the_field_still_derives_its_axis(monkeypatch):
    """Sessions predating SharedState.grading carry nothing; only they fall back."""
    grading = _grading(
        monkeypatch,
        state={"benchmark_mode": "agentx", "baseline_perf": dict(_AXES)},
    )

    assert grading["objective"] == GRADED_TOTAL
    # The band was never recorded for such a session, and today's environment
    # is not evidence of what it applied.
    assert grading["intvty_veto"]["noise_pct"] is None


def test_the_fallback_never_lets_the_exporting_shell_rename_the_axis(monkeypatch):
    """No recorded axis is not licence to read the env.

    The export can run days later from any shell. Honouring an override there
    would relabel a finished session's axis, and the block would look
    perfectly well-formed while saying the wrong thing.
    """
    grading = _grading(
        monkeypatch,
        state={"benchmark_mode": "agentx", "baseline_perf": dict(_AXES)},
        env={"HYPERLOOM_PERF_METRIC": "output_tput", "HYPERLOOM_PERF_NOISE_PCT": "9.9"},
    )

    assert grading["objective"] == GRADED_TOTAL
    assert grading["intvty_veto"]["noise_pct"] is None


def test_seed_records_the_axis_the_run_will_grade_on(monkeypatch):
    # Importing the CLI package pulls in the POSIX-only fcntl; the rest of this
    # module stays importable on any platform.
    pytest.importorskip("fcntl", reason="hyperloom.inference_optimizer.cli is POSIX-only")
    from hyperloom.inference_optimizer.cli.bootstrap import seed_grading

    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_PERF_METRIC", "HYPERLOOM_PERF_NOISE_PCT"):
        monkeypatch.delenv(name, raising=False)

    assert seed_grading("sglang", "agentx")["objective"] == GRADED_TOTAL
    assert seed_grading("sglang", "synthetic")["objective"] == GRADED_OUTPUT
    # A scriptable framework reports an image-quality gate, not token throughput.
    assert seed_grading("xdit", "agentx")["objective"] == GRADED_OUTPUT

    # Seed is where the environment gets its say -- both directions.
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_tput")
    assert seed_grading("sglang", "agentx")["objective"] == GRADED_OUTPUT
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    assert seed_grading("sglang", "synthetic")["objective"] == GRADED_TOTAL
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "3.5")
    assert seed_grading("sglang", "agentx")["intvty_noise_pct"] == pytest.approx(3.5)


class _State:
    """A SharedState stand-in: the comparator reads attributes, not keys."""

    def __init__(self, **fields):
        self.framework = "sglang"
        self.benchmark_mode = "agentx"
        self.baseline_tput = 183.0
        self.baseline_perf = dict(_AXES)
        self.current_best = {}
        self.grading = {}
        for name, value in fields.items():
            setattr(self, name, value)


def test_a_resume_that_lost_the_env_keeps_the_axis_the_session_started_with(monkeypatch):
    """A resume is a new process; the KEEP rule must not change under it.

    Without the recorded objective this session would flip back to the total
    axis, and every later KEEP/REVERT would be decided on a different
    quantity than the ones before it.
    """
    from hyperloom.orchestrator.state.shared_state import resolve_graded_comparison

    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_PERF_METRIC", "HYPERLOOM_PERF_NOISE_PCT"):
        monkeypatch.delenv(name, raising=False)
    state = _State(grading={"objective": GRADED_OUTPUT, "intvty_noise_pct": 3.5})

    graded = resolve_graded_comparison(state, dict(_AXES), against_baseline=True)

    assert graded.objective == GRADED_OUTPUT


def test_the_recorded_veto_band_survives_a_resume(monkeypatch):
    """A lost HYPERLOOM_PERF_NOISE_PCT must not widen 3.5% back to the default 5%."""
    from hyperloom.orchestrator.state.shared_state import resolve_graded_comparison

    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_PERF_METRIC", "HYPERLOOM_PERF_NOISE_PCT"):
        monkeypatch.delenv(name, raising=False)
    # 4.2 -> 4.03 is a 4.05% regression: inside the 5% default, outside 3.5%.
    candidate = {"total_throughput": 26000.0, "intvty_p90": 4.03}

    tight = resolve_graded_comparison(
        _State(grading={"objective": GRADED_TOTAL, "intvty_noise_pct": 3.5}),
        candidate,
        against_baseline=True,
    )
    wide = resolve_graded_comparison(
        _State(grading={"objective": GRADED_TOTAL, "intvty_noise_pct": 5.0}),
        candidate,
        against_baseline=True,
    )

    assert tight.vetoed is True
    assert wide.vetoed is False


def _outcome(monkeypatch, state):
    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_PERF_METRIC", "HYPERLOOM_PERF_NOISE_PCT"):
        monkeypatch.delenv(name, raising=False)
    return collect_v6_outcome(
        session={"stop_reason": "global_converged"},
        baseline={"throughput_tok_s_per_gpu": 183.0},
        final={"throughput_tok_s_per_gpu": 210.0, "cumulative_gain_pct_validated": 14.7},
        optimizations={},
        state=state,
        timeline=[],
    )


def test_outcome_carries_the_agentx_axes_at_both_ends(monkeypatch):
    outcome = _outcome(
        monkeypatch,
        {
            "benchmark_mode": "agentx",
            "framework": "sglang",
            "baseline_perf": dict(_AXES),
            "current_best": {"total_throughput": 29000.0, "intvty_p90": 4.1},
        },
    )

    assert outcome["baseline"]["total_throughput_tok_s"] == pytest.approx(25978.0)
    assert outcome["baseline"]["intvty_p90"] == pytest.approx(4.2)
    assert outcome["final"]["total_throughput_tok_s"] == pytest.approx(29000.0)
    assert outcome["final"]["graded_on"] == GRADED_TOTAL
    # The reconciliation must name the same axis it is summing on.
    assert outcome["validation"]["graded_on"] == GRADED_TOTAL


def test_outcome_axes_are_explicit_nulls_on_a_synthetic_session(monkeypatch):
    """Absent would be ambiguous and zero would be a lie; null says "not measured"."""
    outcome = _outcome(monkeypatch, {"framework": "sglang", "baseline_perf": {"tput": 183.0}})

    assert outcome["baseline"]["total_throughput_tok_s"] is None
    assert outcome["baseline"]["intvty_p90"] is None
    assert outcome["final"]["graded_on"] == GRADED_OUTPUT
    # The output axis is untouched -- old consumers keep reading what they read.
    assert outcome["baseline"]["throughput_tok_s_per_gpu"] == pytest.approx(183.0)
    assert outcome["final"]["gain_pct"] == pytest.approx(14.7)


def test_outcome_reports_the_axis_the_run_decided_on_not_the_configured_one(monkeypatch):
    """A promotion can fall back to output on a session configured for total.

    That happens whenever either side of the comparison cannot supply the axis
    pair. Declaring the configured axis on ``outcome.final`` would put a total
    label on an output number.
    """
    state = {
        "benchmark_mode": "agentx",
        "framework": "sglang",
        "baseline_perf": dict(_AXES),
        "current_best": dict(_AXES),
        "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 5.0},
    }
    outcome = collect_v6_outcome(
        session={"stop_reason": "global_converged"},
        baseline={},
        final={"cumulative_gain_pct_validated": 3.0},
        optimizations={
            "validation": {
                "graded_on": GRADED_OUTPUT,
                "graded_degrade_reason": "current_best_axes_missing",
            }
        },
        state=state,
        timeline=[],
    )

    assert outcome["final"]["graded_on"] == GRADED_OUTPUT
    assert outcome["validation"]["graded_on"] == GRADED_OUTPUT
    # The session-level setting is unchanged and still says total.
    assert _grading(monkeypatch, state=state)["objective"] == GRADED_TOTAL


def test_final_axes_come_from_the_promotion_that_produced_the_gain():
    """A revalidation moves the gain without re-promoting current_best.

    Reading the axes off ``current_best`` would pair the newest gain with the
    measurement taken when the recipe was first promoted.
    """
    outcome = collect_v6_outcome(
        session={"stop_reason": "global_converged"},
        baseline={},
        final={"cumulative_gain_pct_validated": 9.0},
        optimizations={
            "validation": {
                "graded_on": GRADED_TOTAL,
                "validated_perf": {"total_throughput": 31000.0, "intvty_p90": 3.9},
            }
        },
        state={
            "benchmark_mode": "agentx",
            "framework": "sglang",
            # Stale: the recipe was promoted on this, the gain came later.
            "current_best": {"total_throughput": 25978.0, "intvty_p90": 4.2},
            "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 5.0},
        },
        timeline=[],
    )

    assert outcome["final"]["total_throughput_tok_s"] == pytest.approx(31000.0)
    assert outcome["final"]["intvty_p90"] == pytest.approx(3.9)


def test_final_axes_fall_back_to_current_best_when_nothing_was_promoted(monkeypatch):
    outcome = _outcome(
        monkeypatch,
        {
            "benchmark_mode": "agentx",
            "framework": "sglang",
            "current_best": {"total_throughput": 25978.0, "intvty_p90": 4.2},
            "grading": {"objective": GRADED_TOTAL, "intvty_noise_pct": 5.0},
        },
    )

    assert outcome["final"]["total_throughput_tok_s"] == pytest.approx(25978.0)
