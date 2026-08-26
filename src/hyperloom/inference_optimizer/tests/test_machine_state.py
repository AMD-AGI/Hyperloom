# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for machine_state pure helpers: escalate hints, budget normalization,
time/budget remaining math, post-prelude target, and history-row builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases import machine_state as ps


def test_is_valid_escalate_hint() -> None:
    assert ps.is_valid_escalate_hint("not-a-real-hint") is False
    # at least one vocab member should validate
    some_vocab = next(iter(ps.ESCALATE_HINT_VOCAB))
    assert ps.is_valid_escalate_hint(some_vocab) is True


def test_normalize_budget_pct_defaults_and_filters() -> None:
    assert ps.normalize_budget_pct(None) == dict(ps.DEFAULT_PHASE_BUDGET_PCT)
    out = ps.normalize_budget_pct(
        {
            ps.PHASE_FRAMEWORK_AGENT: 0.4,
            "BOGUS_PHASE": 0.5,  # unknown phase dropped
            ps.PHASE_KERNEL_AGENT: "bad",  # non-numeric dropped
            ps.PHASE_SWEEP: 2.0,  # out of (0,1] dropped
        }
    )
    assert out[ps.PHASE_FRAMEWORK_AGENT] == 0.4
    assert "BOGUS_PHASE" not in out
    # A dropped entry falls back to its default rather than vanishing: a phase
    # with no share would run to whatever it costs.
    assert set(out) == set(ps.PHASE_NAMES)
    assert out[ps.PHASE_SWEEP] == ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_SWEEP]


def test_apply_escalate_budget_bump() -> None:
    phase = ps.PHASE_FRAMEWORK_AGENT
    base = {phase: 0.3}
    assert ps.apply_escalate_budget_bump(base, phase="nope") == base
    out = ps.apply_escalate_budget_bump({phase: 0.3}, phase=phase, delta=0.1, cap=0.8)
    assert out[phase] == pytest.approx(0.4)
    capped = ps.apply_escalate_budget_bump({phase: 0.75}, phase=phase, delta=0.5, cap=0.8)
    assert capped[phase] == 0.8


def test_now_unix_injected() -> None:
    state = SimpleNamespace(_now_unix=lambda: 1234.0)
    assert ps._now_unix(state) == 1234.0


def test_phase_started_unix_bad_value() -> None:
    assert ps._phase_started_unix(SimpleNamespace(phase_started_unix="bad")) == 0.0
    assert ps._phase_started_unix(SimpleNamespace(phase_started_unix=10.0)) == 10.0


def test_pending_escalate_hint() -> None:
    valid = next(iter(ps.ESCALATE_HINT_VOCAB))
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint=valid)) == valid
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint="garbage")) == ""
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint="")) == ""


def test_max_minutes_coercion() -> None:
    assert ps._max_minutes(SimpleNamespace(max_minutes=30)) == 30.0
    assert ps._max_minutes(SimpleNamespace(max_minutes="bad")) == 0.0
    assert ps._max_minutes(SimpleNamespace(max_minutes=0)) == 0.0


def test_phase_elapsed_seconds() -> None:
    # not started -> 0
    assert ps.phase_elapsed_seconds(SimpleNamespace(phase_started_unix=0.0)) == 0.0
    # started -> now - started, clamped non-negative.
    state = SimpleNamespace(phase_started_unix=100.0)
    assert ps.phase_elapsed_seconds(state, now_unix=160.0) == 60.0
    assert ps.phase_elapsed_seconds(state, now_unix=50.0) == 0.0


def test_phase_budget_remaining_seconds() -> None:
    # unlimited -> None
    assert ps.phase_budget_remaining_seconds(SimpleNamespace(max_minutes=0)) is None
    # phase not in the budget map -> None
    state = SimpleNamespace(
        max_minutes=60,
        phase="UNKNOWN_PHASE",
        phase_started_unix=0.0,
        phase_budget_pct={ps.PHASE_FRAMEWORK_AGENT: 0.5},
    )
    assert ps.phase_budget_remaining_seconds(state) is None
    # 60min * 0.5 = 1800s budget, minus elapsed.
    state2 = SimpleNamespace(
        max_minutes=60,
        phase=ps.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1000.0,
        phase_budget_pct={ps.PHASE_FRAMEWORK_AGENT: 0.5},
    )
    rem = ps.phase_budget_remaining_seconds(state2, now_unix=1300.0)
    assert rem == pytest.approx(1800.0 - 300.0)


def test_session_remaining_seconds() -> None:
    assert ps.session_remaining_seconds(SimpleNamespace(max_minutes=0)) is None
    # no start_ts -> None
    assert (
        ps.session_remaining_seconds(
            SimpleNamespace(max_minutes=60, start_ts=""),
        )
        is None
    )
    # bad ts -> None
    assert (
        ps.session_remaining_seconds(
            SimpleNamespace(max_minutes=60, start_ts="not-a-date"),
        )
        is None
    )
    # valid recent start -> positive remaining
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    rem = ps.session_remaining_seconds(
        SimpleNamespace(max_minutes=60, start_ts=now_iso),
    )
    assert rem is not None and 0.0 < rem <= 3600.0
    assert (
        ps.session_remaining_seconds(
            SimpleNamespace(max_minutes=60, start_ts=now_iso, deadline_unix=1000.0),
            now_unix=400.0,
        )
        == 600.0
    )
    assert (
        ps.session_remaining_seconds(
            SimpleNamespace(max_minutes=0, deadline_unix=2000.0),
            now_unix=1400.0,
        )
        == 600.0
    )


@pytest.mark.parametrize(
    ("optimize", "kernel", "target"),
    [
        (True, True, ps.PHASE_FRAMEWORK_AGENT),
        (True, False, ps.PHASE_FRAMEWORK_AGENT),
        # --no-framework-agent collapses past the optimisation phase.
        (False, True, ps.PHASE_KERNEL_AGENT),
        (False, False, ps.PHASE_SWEEP),
    ],
)
def test_post_prelude_target(optimize: bool, kernel: bool, target: str) -> None:
    assert ps._post_prelude_target(optimize_enabled=optimize, kernel_enabled=kernel) == target


def test_make_history_row() -> None:
    row = ps.make_history_row(
        from_phase="framework_agent",
        to_phase="kernel_agent",
        reason="  plateau  ",
        evidence={"k": 1},
        ts="2026-06-09T00:00:00Z",
        ts_unix=12.0,
    )
    assert row["from_phase"] == "FRAMEWORK_AGENT"
    assert row["to_phase"] == "KERNEL_AGENT"
    assert row["reason"] == "plateau"
    assert row["evidence"] == {"k": 1}
    assert row["ts_unix"] == 12.0


def test_phase_budget_help_quotes_the_real_default() -> None:
    """``--help`` must quote the default the run will actually use.

    These flags default to None and fall through to DEFAULT_PHASE_BUDGET_PCT,
    so the number in the help text is the only place a user can read the real
    value, and nothing recomputes it. Both the KERNEL_AGENT and SWEEP shares
    had been retuned without the help text following.
    """
    import re

    from hyperloom.inference_optimizer.cli.parser import _build_parser

    # The flags live on the ``optimize`` subparser, so walk the tree.
    pending = [_build_parser()]
    quoted: dict[str, float] = {}
    while pending:
        for action in pending.pop()._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                pending.extend(sub for sub in choices.values() if hasattr(sub, "_actions"))
                continue
            match = re.fullmatch(r"phase_budget_(\w+)_pct", action.dest or "")
            if not match:
                continue
            default_text = re.search(r"Default:\s*([0-9.]+)\.", action.help or "")
            assert default_text, f"{action.dest} help does not quote a default"
            quoted[match.group(1).upper()] = float(default_text.group(1))

    assert quoted, "no phase-budget flags found; this guard would pass vacuously"

    real = {phase.upper(): value for phase, value in ps.DEFAULT_PHASE_BUDGET_PCT.items()}
    # The FRAMEWORK_AGENT flag is spelled --phase-budget-framework-pct.
    real["FRAMEWORK"] = real.pop("FRAMEWORK_AGENT")
    real["KERNEL"] = real.pop("KERNEL_AGENT")

    assert quoted == real
