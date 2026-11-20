# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLI toggles that decide which phases a run enters."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.orchestrator.state.shared_state import SharedState


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


# --------------------------------------------------------------------------- # The optimisation phase toggle.
# --------------------------------------------------------------------------- #
def test_the_optimisation_phase_is_on_by_default():
    args = _parse_optimize([])
    assert args.no_framework_agent is False
    assert SharedState(session_id="t").framework_agent_phase_enabled is True


def test_the_optimisation_phase_toggle_disables_both_arms():
    assert _parse_optimize(["--no-framework-agent"]).no_framework_agent is True


def test_no_explore_is_gone_rather_than_aliased():
    """It named one arm, and the phase it would now disable is both."""
    with pytest.raises(SystemExit):
        _parse_optimize(["--no-explore"])


def test_framework_budget_covers_the_whole_optimize_phase():
    """One phase, one budget: a separate explore share would be unread."""
    args = _parse_optimize(["--phase-budget-framework-pct", "0.6"])
    assert args.phase_budget_framework_pct == 0.6
    assert not hasattr(args, "phase_budget_explore_pct")


@pytest.mark.parametrize(
    "flag",
    [
        "--phase-budget-explore-pct",
        "--max-minutes-explore-pct",
        "--max-minutes-prelude-pct",
        "--max-minutes-framework-pct",
        "--max-minutes-kernel-pct",
        "--max-minutes-sweep-pct",
        "--max-minutes-close-pct",
    ],
)
def test_removed_budget_spellings_are_rejected(flag):
    with pytest.raises(SystemExit) as exc:
        _parse_optimize([flag, "0.6"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- # The eval toggle, which is orthogonal
# and unchanged. --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- # --no-source-changes: keep only server
# arguments and environment variables. --------------------------------------------------------------------------- #
def test_source_changes_are_kept_by_default():
    args = _parse_optimize([])
    assert args.no_source_changes is False
    assert SharedState(session_id="t").no_source_changes is False
    assert bootstrap._phase_toggle_fields(args)["no_source_changes"] is False


def test_no_source_changes_turns_on_no_other_flag():
    """The kernel phase, warm replay, local explore and static recon keep their own switches."""
    args = _parse_optimize(["--no-source-changes"])
    assert args.no_source_changes is True
    assert bootstrap._phase_toggle_fields(args)["no_source_changes"] is True
    assert args.no_kernel is False
    assert args.no_warm_replay is False
    assert args.no_framework_local_explore is False
    assert args.static_recon is True
    assert args.no_framework_agent is False
    assert args.enablement == "all"


def test_a_resume_keeps_a_persisted_no_source_changes(tmp_path):
    state = SharedState(session_id="t", no_source_changes=True)
    args = _parse_optimize([])
    cli._restore_phase_toggles(args, state, tmp_path)
    assert state.no_source_changes is True
    assert args.no_source_changes is True


def test_a_resume_can_turn_no_source_changes_on_and_persists_it(tmp_path):
    state = SharedState(session_id="t")
    cli._restore_phase_toggles(_parse_optimize(["--no-source-changes"]), state, tmp_path)
    assert state.no_source_changes is True
    assert SharedState.load_or_init(tmp_path).no_source_changes is True


def test_a_resume_without_the_flag_leaves_source_changes_kept(tmp_path):
    state = SharedState(session_id="t")
    args = _parse_optimize([])
    cli._restore_phase_toggles(args, state, tmp_path)
    assert state.no_source_changes is False
    assert args.no_source_changes is False
