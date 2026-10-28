# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression: KERNEL phase-budget-pct CLI override must reach KERNEL_AGENT."""

from __future__ import annotations

import inspect

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.loop.coordinator import CoordinatorSettings
from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PHASE_BUDGET_PCT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_PRELUDE,
    PHASE_SWEEP,
    normalize_budget_pct,
    phase_budget_remaining_seconds,
    redistribute_budget_pct,
)
from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import make_coordinator


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_kernel_pct_override_reaches_kernel_agent() -> None:
    """The override must survive normalize_budget_pct as KERNEL_AGENT."""
    args = _parse_optimize(["--phase-budget-kernel-pct", "0.78"])
    raw = cli._build_phase_budget_pct(args)
    assert raw.get(PHASE_KERNEL_AGENT) == pytest.approx(0.78)

    normalized = normalize_budget_pct(raw)
    assert normalized[PHASE_KERNEL_AGENT] == pytest.approx(0.78)
    # Must not silently fall back to the default.
    assert normalized[PHASE_KERNEL_AGENT] != pytest.approx(DEFAULT_PHASE_BUDGET_PCT[PHASE_KERNEL_AGENT])


def test_kernel_pct_key_is_canonical_phase_name() -> None:
    """The override key must be the canonical phase name, not the legacy alias."""
    args = _parse_optimize(["--phase-budget-kernel-pct", "0.5"])
    raw = cli._build_phase_budget_pct(args)
    assert "KERNEL" not in raw
    assert PHASE_KERNEL_AGENT in raw


def test_framework_pct_override_reaches_framework_agent() -> None:
    """FRAMEWORK_AGENT is a budgeted phase, so the override must survive normalize_budget_pct as FRAMEWORK_AGENT."""
    args = _parse_optimize(["--phase-budget-framework-pct", "0.42"])
    raw = cli._build_phase_budget_pct(args)
    assert raw.get(PHASE_FRAMEWORK_AGENT) == pytest.approx(0.42)

    normalized = normalize_budget_pct(raw)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.42)
    # Must not silently fall back to the default.
    assert normalized[PHASE_FRAMEWORK_AGENT] != pytest.approx(DEFAULT_PHASE_BUDGET_PCT[PHASE_FRAMEWORK_AGENT])


def test_framework_pct_key_is_canonical_phase_name() -> None:
    """The override key must be the canonical phase name, not the legacy alias."""
    args = _parse_optimize(["--phase-budget-framework-pct", "0.2"])
    raw = cli._build_phase_budget_pct(args)
    assert "FRAMEWORK" not in raw
    assert PHASE_FRAMEWORK_AGENT in raw


def test_all_phase_budget_pct_spellings_parse() -> None:
    """Every phase accepts its phase-budget flag."""
    argv = [
        "--phase-budget-prelude-pct",
        "0.05",
        "--phase-budget-framework-pct",
        "0.20",
        "--phase-budget-kernel-pct",
        "0.40",
        "--phase-budget-sweep-pct",
        "0.15",
        "--phase-budget-close-pct",
        "0.03",
    ]
    args = _parse_optimize(argv)
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    assert normalized["PRELUDE"] == pytest.approx(0.05)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.20)
    assert normalized[PHASE_KERNEL_AGENT] == pytest.approx(0.40)
    assert normalized["SWEEP"] == pytest.approx(0.15)
    assert normalized["CLOSE"] == pytest.approx(0.03)


def test_qwen3_8b_3h_no_kernel_budget_shape() -> None:
    """Pin the 3h demo's framework-on, kernel-off budget shape.

    Values mirror its SKILL.md; phase budgets are independent caps, so their
    total may exceed 1.0.
    """
    args = _parse_optimize(
        [
            "--max-hours",
            "3",
            "--phase-budget-framework-pct",
            "0.50",
            "--phase-budget-sweep-pct",
            "0.01",
            "--no-kernel",
            "--no-enable-conc-sweep",
        ]
    )
    assert not args.no_framework_agent
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    # Every override survived the range check — none fell back to a default.
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.50)
    assert normalized["SWEEP"] == pytest.approx(0.01)

    out = redistribute_budget_pct(
        normalized,
        kernel_enabled=not args.no_kernel,
        optimize_enabled=not args.no_framework_agent,
        sweep_enabled=args.enable_conc_sweep is not False,
    )
    assert out[PHASE_KERNEL_AGENT] == 0.0
    assert out[PHASE_SWEEP] == 0.0
    assert out["PRELUDE"] == pytest.approx(0.03)
    assert out["CLOSE"] == pytest.approx(0.02)
    # FRAMEWORK_AGENT is the only work phase left, so it takes both freed shares.
    assert out[PHASE_FRAMEWORK_AGENT] == pytest.approx(
        min(1.0, 0.50 + 0.01 + DEFAULT_PHASE_BUDGET_PCT[PHASE_KERNEL_AGENT])
    )
    # The redistributed value must survive the downstream re-normalize instead
    # of being dropped back to the default.
    assert normalize_budget_pct(out)[PHASE_FRAMEWORK_AGENT] == pytest.approx(out[PHASE_FRAMEWORK_AGENT])


def test_redistribute_caps_absorber_at_full_wall_clock() -> None:
    """An override that over-absorbs is capped, not silently defaulted.

    ``--phase-budget-framework-pct 0.90`` plus the share freed by ``--no-kernel``
    sums past ``1.0``. Before the cap that value failed
    :func:`normalize_budget_pct`'s range check downstream and reverted to the
    FRAMEWORK_AGENT default — handing the caller *less* budget than the smaller
    override would have. Clamp at a full wall clock and discard the excess.
    """
    args = _parse_optimize(
        [
            "--max-hours",
            "3",
            "--phase-budget-framework-pct",
            "0.90",
            "--phase-budget-sweep-pct",
            "0.01",
            "--no-kernel",
        ]
    )
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    out = redistribute_budget_pct(
        normalized,
        kernel_enabled=not args.no_kernel,
        optimize_enabled=not args.no_framework_agent,
    )
    assert out[PHASE_FRAMEWORK_AGENT] == pytest.approx(1.0)
    assert out[PHASE_KERNEL_AGENT] == 0.0

    # The regression this guards: surviving the re-normalize rather than being
    # dropped back to a default that is *below* the requested override.
    renormalized = normalize_budget_pct(out)
    assert renormalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(1.0)
    assert renormalized[PHASE_FRAMEWORK_AGENT] > DEFAULT_PHASE_BUDGET_PCT[PHASE_FRAMEWORK_AGENT]


def test_optimize_path_is_wired_to_helper() -> None:
    """Guard: the live optimize path must build the budget via the helper."""
    src = inspect.getsource(cli)
    assert "_build_phase_budget_pct(args)" in src
    assert '("phase_budget_kernel_pct", "KERNEL")' not in src


def test_redistribute_disabled_phase_share_goes_to_work_phases() -> None:
    """A disabled phase's pct is zeroed and spread across enabled work phases."""
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    out = redistribute_budget_pct(base, kernel_enabled=False, optimize_enabled=True)
    assert out[PHASE_KERNEL_AGENT] == 0.0
    assert out["PRELUDE"] == base["PRELUDE"]
    assert out["CLOSE"] == base["CLOSE"]
    # Freed KERNEL share landed on the enabled work phases.
    assert out[PHASE_FRAMEWORK_AGENT] > base[PHASE_FRAMEWORK_AGENT]
    assert out["SWEEP"] > base["SWEEP"]
    assert sum(out.values()) == pytest.approx(sum(base.values()))


def test_redistribute_hands_a_disabled_sweep_share_to_the_work_phases() -> None:
    """A run without the concurrency sweep keeps no SWEEP share, and the freed share is not parked back on SWEEP."""
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    out = redistribute_budget_pct(base, kernel_enabled=True, optimize_enabled=True, sweep_enabled=False)
    assert out[PHASE_SWEEP] == 0.0
    assert out[PHASE_FRAMEWORK_AGENT] / out[PHASE_KERNEL_AGENT] == pytest.approx(
        base[PHASE_FRAMEWORK_AGENT] / base[PHASE_KERNEL_AGENT]
    )
    assert sum(out.values()) == pytest.approx(sum(base.values()))

    nothing_left = redistribute_budget_pct(base, kernel_enabled=False, optimize_enabled=False, sweep_enabled=False)
    assert nothing_left[PHASE_SWEEP] == 0.0


def test_a_session_without_conc_sweep_gives_its_sweep_share_to_framework(tmp_path) -> None:
    """AgentX turns the concurrency sweep off, so SWEEP must not keep a share that shrinks FRAMEWORK_AGENT.

    The numbers are a TopCI AgentX session's: kernel off, ``--phase-budget-framework-pct 0.65``,
    ``--phase-budget-sweep-pct 0.30`` and 614 minutes left when FRAMEWORK_AGENT opened. SWEEP's share used to stay in
    the charge-back denominator and cut FRAMEWORK_AGENT to about 414 minutes.
    """
    _persisted_session(tmp_path, kernel_enabled=False, conc_sweep_enabled=False)
    coord = make_coordinator(
        tmp_path,
        settings=CoordinatorSettings(phase_budget_pct={PHASE_FRAMEWORK_AGENT: 0.65, PHASE_SWEEP: 0.30}),
    )
    state = coord.shared_state
    assert state.phase_budget_pct[PHASE_SWEEP] == 0.0
    assert state.phase_budget_pct[PHASE_KERNEL_AGENT] == 0.0

    now = 1_800_000_000.0
    state.max_minutes = 614
    state.elapsed_charged_sec = 0.0
    state.leg_anchor_unix = now
    state.phase = PHASE_FRAMEWORK_AGENT
    state.phase_started_unix = now
    remaining_min = phase_budget_remaining_seconds(state, now_unix=now) / 60.0
    assert remaining_min == pytest.approx(614 * 1.0 / (1.0 + DEFAULT_PHASE_BUDGET_PCT["CLOSE"]))


def test_redistribute_all_enabled_is_noop_and_idempotent() -> None:
    """No disabled phase → unchanged; re-running never drifts."""
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    once = redistribute_budget_pct(base, kernel_enabled=True, optimize_enabled=True)
    assert once == base
    twice = redistribute_budget_pct(once, kernel_enabled=True, optimize_enabled=True)
    assert twice == once


# ---------------------------------------------------------------------------
# Resume budget semantics
# ---------------------------------------------------------------------------


def _persisted_session(session_dir, **fields) -> None:
    SharedState(**fields).save(session_dir)


def test_fresh_session_budget_args_override_state(tmp_path) -> None:
    """Fresh session: explicit budget args reach state after renormalise + redistribute."""
    coord = make_coordinator(tmp_path, settings=CoordinatorSettings(phase_budget_pct={PHASE_KERNEL_AGENT: 0.55}))

    budget = coord.shared_state.phase_budget_pct
    assert budget[PHASE_KERNEL_AGENT] == pytest.approx(0.55)
    # And defaults survive for other phases.
    assert PHASE_FRAMEWORK_AGENT in budget


def test_resume_without_explicit_budget_args_preserves_state(tmp_path) -> None:
    """Resume without explicit args: the persisted state budget is unchanged."""
    # A saved budget that differs from the default, so keeping it is observable.
    persisted = {**DEFAULT_PHASE_BUDGET_PCT, PHASE_KERNEL_AGENT: 0.65}
    _persisted_session(tmp_path, phase_budget_pct=persisted, phase=PHASE_PRELUDE)

    # No explicit budget args (None means "operator said nothing").
    coord = make_coordinator(tmp_path)

    assert coord.shared_state.phase_budget_pct[PHASE_KERNEL_AGENT] == pytest.approx(0.65)


def test_resume_with_explicit_budget_args_overwrites_state(tmp_path) -> None:
    """Resume with explicit args: state budget is overwritten by the CLI values."""
    persisted = {**DEFAULT_PHASE_BUDGET_PCT, PHASE_KERNEL_AGENT: 0.65}
    _persisted_session(tmp_path, phase_budget_pct=persisted, phase=PHASE_PRELUDE)

    # Operator explicitly passes a new kernel share.
    coord = make_coordinator(tmp_path, settings=CoordinatorSettings(phase_budget_pct={PHASE_KERNEL_AGENT: 0.30}))

    # The CLI value wins over the persisted share.
    assert coord.shared_state.phase_budget_pct[PHASE_KERNEL_AGENT] == pytest.approx(0.30)
