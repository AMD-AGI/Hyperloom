# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover the --no-explore CLI flag + resume write-back semantics."""

from __future__ import annotations


import pytest

from hyperloom.inference_optimizer import cli


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_no_explore_default_false():
    args = _parse_optimize([])
    assert getattr(args, "no_explore") is False


def test_no_explore_flag_sets_true():
    args = _parse_optimize(["--no-explore"])
    assert getattr(args, "no_explore") is True


def test_no_explore_has_no_env_fallback(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NO_EXPLORE", "1")
    args = _parse_optimize([])
    assert getattr(args, "no_explore") is False


def test_shared_state_explore_enabled_defaults_true():
    from hyperloom.orchestrator.state.shared_state import SharedState

    assert SharedState(session_id="t").explore_enabled is True


# Pre-EXPLORE guard: a resume may retroactively honour --no-explore only while the
# persisted phase is still upstream of EXPLORE. The list must include both the legacy
# "FRAMEWORK" name and the current "FRAMEWORK_AGENT".
@pytest.mark.parametrize(
    "phase,expected",
    [
        ("", True),
        ("PRELUDE", True),
        ("prelude", True),
        ("FRAMEWORK", True),
        ("framework", True),
        ("FRAMEWORK_AGENT", True),
        ("EXPLORE", False),
        ("KERNEL_AGENT", False),
        ("SWEEP", False),
        ("CLOSE", False),
    ],
)
def test_resume_can_disable_explore(phase: str, expected: bool) -> None:
    assert cli._resume_can_disable_explore(phase) is expected


class _ResumeStateStub:
    def __init__(self, *, explore_enabled: bool, phase: str) -> None:
        self.explore_enabled = explore_enabled
        self.phase = phase


class _ArgsStub:
    def __init__(self, *, no_explore: bool) -> None:
        self.no_explore = no_explore


def _apply_resume_writeback(state: _ResumeStateStub, args: _ArgsStub) -> str:
    """Mirror the resume-branch control flow from cli.py, delegating the phase
    check to the real ``cli._resume_can_disable_explore`` helper."""
    msg = ""
    if not bool(getattr(state, "explore_enabled", True)):
        args.no_explore = True
        msg = "DISABLED_PERSISTED"
    elif bool(getattr(args, "no_explore", False)):
        cur_phase = (getattr(state, "phase", "") or "").strip().upper()
        if cli._resume_can_disable_explore(cur_phase):
            state.explore_enabled = False
            msg = "DISABLING_RESUME"
        else:
            msg = "WARN_IGNORED"
    return msg


def test_resume_writeback_disables_state_when_prelude_and_flag_passed():
    state = _ResumeStateStub(explore_enabled=True, phase="PRELUDE")
    args = _ArgsStub(no_explore=True)
    assert _apply_resume_writeback(state, args) == "DISABLING_RESUME"
    assert state.explore_enabled is False


def test_resume_writeback_allows_disable_in_framework():
    state = _ResumeStateStub(explore_enabled=True, phase="FRAMEWORK_AGENT")
    args = _ArgsStub(no_explore=True)
    assert _apply_resume_writeback(state, args) == "DISABLING_RESUME"
    assert state.explore_enabled is False


def test_resume_writeback_allows_disable_in_legacy_framework():
    state = _ResumeStateStub(explore_enabled=True, phase="FRAMEWORK")
    args = _ArgsStub(no_explore=True)
    assert _apply_resume_writeback(state, args) == "DISABLING_RESUME"
    assert state.explore_enabled is False


def test_resume_writeback_warns_when_already_in_explore():
    state = _ResumeStateStub(explore_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_explore=True)
    assert _apply_resume_writeback(state, args) == "WARN_IGNORED"
    assert state.explore_enabled is True


def test_resume_writeback_honours_persisted_disable_over_flag_absent():
    state = _ResumeStateStub(explore_enabled=False, phase="KERNEL_AGENT")
    args = _ArgsStub(no_explore=False)
    assert _apply_resume_writeback(state, args) == "DISABLED_PERSISTED"
    assert args.no_explore is True


def test_resume_writeback_no_op_when_state_and_flag_both_enabled():
    state = _ResumeStateStub(explore_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_explore=False)
    assert _apply_resume_writeback(state, args) == ""
    assert state.explore_enabled is True
    assert args.no_explore is False
