# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression: KERNEL phase-budget-pct CLI override must reach KERNEL_AGENT."""

from __future__ import annotations

import inspect

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PHASE_BUDGET_PCT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    normalize_budget_pct,
    redistribute_budget_pct,
)


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


@pytest.mark.parametrize(
    "flag",
    ["--max-minutes-kernel-pct", "--phase-budget-kernel-pct"],
)
def test_kernel_pct_override_reaches_kernel_agent(flag: str) -> None:
    """Both flag spellings must survive normalize_budget_pct as KERNEL_AGENT."""
    args = _parse_optimize([flag, "0.78"])
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


@pytest.mark.parametrize(
    "flag",
    ["--max-minutes-framework-pct", "--phase-budget-framework-pct"],
)
def test_framework_pct_override_reaches_framework_agent(flag: str) -> None:
    """FRAMEWORK_AGENT is a budgeted phase, so both flag spellings must parse and survive normalize_budget_pct as FRAMEWORK_AGENT."""
    args = _parse_optimize([flag, "0.42"])
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
    """Every phase accepts both the legacy and the phase-budget spelling."""
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
            "--max-minutes-framework-pct",
            "0.50",
            "--max-minutes-sweep-pct",
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
    )
    assert out[PHASE_KERNEL_AGENT] == 0.0
    assert out["PRELUDE"] == pytest.approx(0.03)
    assert out["CLOSE"] == pytest.approx(0.02)
    # FRAMEWORK_AGENT and SWEEP split the freed KERNEL share by base weight,
    # so FRAMEWORK_AGENT ends up with essentially the whole wall clock.
    assert out[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.9902, abs=1e-4)
    assert out["SWEEP"] == pytest.approx(0.0198, abs=1e-4)
    # The redistributed value must survive the downstream re-normalize instead
    # of being dropped back to the default.
    assert normalize_budget_pct(out)[PHASE_FRAMEWORK_AGENT] == pytest.approx(out[PHASE_FRAMEWORK_AGENT])


def test_redistribute_caps_absorber_at_full_wall_clock() -> None:
    """An override that over-absorbs is capped, not silently defaulted.

    ``--max-minutes-framework-pct 0.90`` plus the share freed by ``--no-kernel``
    sums past ``1.0``. Before the cap that value failed
    :func:`normalize_budget_pct`'s range check downstream and reverted to the
    FRAMEWORK_AGENT default — handing the caller *less* budget than the smaller
    override would have. Clamp at a full wall clock and discard the excess.
    """
    args = _parse_optimize(
        [
            "--max-hours",
            "3",
            "--max-minutes-framework-pct",
            "0.90",
            "--max-minutes-sweep-pct",
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


def test_redistribute_all_enabled_is_noop_and_idempotent() -> None:
    """No disabled phase → unchanged; re-running never drifts."""
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    once = redistribute_budget_pct(base, kernel_enabled=True, optimize_enabled=True)
    assert once == base
    twice = redistribute_budget_pct(once, kernel_enabled=True, optimize_enabled=True)
    assert twice == once
