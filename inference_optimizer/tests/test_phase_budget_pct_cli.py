# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: KERNEL phase-budget-pct CLI override must reach KERNEL_AGENT.

Two coupled bugs made ``--*-kernel-pct`` a no-op after the KERNEL -> KERNEL_AGENT
phase rename (commit 33ac6ccc):

* Root cause 1: :func:`cli._build_phase_budget_pct` emitted the key ``"KERNEL"``
  while the canonical phase name is ``"KERNEL_AGENT"``, so
  :func:`normalize_budget_pct` dropped it and the phase fell back to its default
  (0.305).
* Root cause 2: only ``--max-minutes-kernel-pct`` was registered; the
  ``--phase-budget-kernel-pct`` spelling used by callers raised a parse error.
"""

from __future__ import annotations

import inspect

import pytest

from inference_optimizer import cli
from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PHASE_BUDGET_PCT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    normalize_budget_pct,
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
    # Regression guard: value must not silently fall back to the default.
    assert normalized[PHASE_KERNEL_AGENT] != pytest.approx(
        DEFAULT_PHASE_BUDGET_PCT[PHASE_KERNEL_AGENT]
    )


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
    """FRAMEWORK_AGENT is a budgeted phase, so both flag spellings must parse
    and survive normalize_budget_pct as FRAMEWORK_AGENT."""
    args = _parse_optimize([flag, "0.42"])
    raw = cli._build_phase_budget_pct(args)
    assert raw.get(PHASE_FRAMEWORK_AGENT) == pytest.approx(0.42)

    normalized = normalize_budget_pct(raw)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.42)
    # Regression guard: value must not silently fall back to the default.
    assert normalized[PHASE_FRAMEWORK_AGENT] != pytest.approx(
        DEFAULT_PHASE_BUDGET_PCT[PHASE_FRAMEWORK_AGENT]
    )


def test_framework_pct_key_is_canonical_phase_name() -> None:
    """The override key must be the canonical phase name, not the legacy alias."""
    args = _parse_optimize(["--phase-budget-framework-pct", "0.2"])
    raw = cli._build_phase_budget_pct(args)
    assert "FRAMEWORK" not in raw
    assert PHASE_FRAMEWORK_AGENT in raw


def test_all_phase_budget_pct_spellings_parse() -> None:
    """Every phase accepts both the legacy and the phase-budget spelling."""
    argv = [
        "--phase-budget-prelude-pct", "0.05",
        "--phase-budget-framework-pct", "0.20",
        "--phase-budget-explore-pct", "0.30",
        "--phase-budget-kernel-pct", "0.40",
        "--phase-budget-sweep-pct", "0.15",
        "--phase-budget-close-pct", "0.03",
    ]
    args = _parse_optimize(argv)
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    assert normalized["PRELUDE"] == pytest.approx(0.05)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.20)
    assert normalized["EXPLORE"] == pytest.approx(0.30)
    assert normalized[PHASE_KERNEL_AGENT] == pytest.approx(0.40)
    assert normalized["SWEEP"] == pytest.approx(0.15)
    assert normalized["CLOSE"] == pytest.approx(0.03)


def test_optimize_path_is_wired_to_helper() -> None:
    """Guard: the live optimize path must build the budget via the helper.

    A prior fix defined ``_build_phase_budget_pct`` but left the real code path
    using an inline ``("phase_budget_kernel_pct", "KERNEL")`` mapping, so the
    unit test passed while production still dropped the override. Assert the
    helper is actually called and the buggy literal is gone from the module.
    """
    src = inspect.getsource(cli)
    assert "_build_phase_budget_pct(args)" in src
    assert '("phase_budget_kernel_pct", "KERNEL")' not in src
