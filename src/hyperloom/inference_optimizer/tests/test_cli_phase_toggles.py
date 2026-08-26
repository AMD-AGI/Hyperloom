# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLI toggles that decide which phases a run enters.

Replaces ``test_cli_no_explore``, most of which exercised a retroactive
resume write-back for a phase toggle that no longer exists on its own:
configuration search and source landing are two arms of one phase, so
``--no-explore`` is an alias for ``--no-framework-agent`` and there is nothing
left to disable separately.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.state.shared_state import SharedState


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


# --------------------------------------------------------------------------- #
# The optimisation phase toggle, and its alias.
# --------------------------------------------------------------------------- #
def test_the_optimisation_phase_is_on_by_default():
    args = _parse_optimize([])
    assert args.no_framework_agent is False
    assert args.no_explore is False
    assert SharedState(session_id="t").framework_agent_phase_enabled is True


@pytest.mark.parametrize("flag", ["--no-framework-agent", "--no-explore"])
def test_either_spelling_disables_the_whole_optimisation_phase(flag):
    """``--no-explore`` names one arm but disables the phase both arms share."""
    args = _parse_optimize([flag])
    disabled = bool(args.no_framework_agent) or bool(args.no_explore)
    assert disabled


@pytest.mark.parametrize(
    "flag",
    [
        "--phase-budget-framework-pct",
        "--phase-budget-explore-pct",
        "--max-minutes-explore-pct",
    ],
)
def test_every_budget_spelling_lands_on_the_one_phase_share(flag):
    """One phase, one budget: a separate explore share would be unread."""
    args = _parse_optimize([flag, "0.6"])
    assert args.phase_budget_framework_pct == 0.6
    assert not hasattr(args, "phase_budget_explore_pct")


def test_no_explore_has_no_env_fallback(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NO_EXPLORE", "1")
    assert _parse_optimize([]).no_explore is False


# --------------------------------------------------------------------------- #
# The eval toggle, which is orthogonal and unchanged.
# --------------------------------------------------------------------------- #
def test_no_eval_default_false():
    assert _parse_optimize([]).no_eval is False


def test_no_eval_flag_sets_true():
    assert _parse_optimize(["--no-eval"]).no_eval is True


def test_shared_state_eval_disabled_defaults_false():
    assert SharedState(session_id="t").eval_disabled is False
