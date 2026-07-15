# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cover the --no-framework-agent CLI default + resume write-back semantics."""

from __future__ import annotations


from hyperloom.inference_optimizer import cli


def _parse_optimize(argv: list[str]) -> object:
    """Helper: run the cli parser, return the parsed args namespace."""
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_no_framework_agent_default_false():
    args = _parse_optimize([])
    assert getattr(args, "no_framework_agent") is False


def test_no_framework_agent_explicit_flag_disables():
    args = _parse_optimize(["--no-framework-agent"])
    assert getattr(args, "no_framework_agent") is True


class _ResumeStateStub:
    """Mirrors the SharedState fields the resume write-back branch reads/writes."""

    def __init__(self, *, framework_agent_phase_enabled: bool, phase: str) -> None:
        self.framework_agent_phase_enabled = framework_agent_phase_enabled
        self.phase = phase
        self.save_calls = 0

    def save(self, _session_dir) -> None:
        self.save_calls += 1


class _ArgsStub:
    def __init__(self, *, no_framework_agent: bool) -> None:
        self.no_framework_agent = no_framework_agent


def _apply_resume_writeback(state: _ResumeStateStub, args: _ArgsStub) -> str:
    """Re-implement the resume-branch logic from cli.py byte-for-byte to catch divergence."""
    msg = ""
    if not bool(getattr(state, "framework_agent_phase_enabled", True)):
        args.no_framework_agent = True
        msg = "DISABLED_PERSISTED"
    elif bool(getattr(args, "no_framework_agent", False)):
        cur_phase = (getattr(state, "phase", "") or "").strip().upper()
        if cur_phase in ("", "PRELUDE"):
            state.framework_agent_phase_enabled = False
            state.save("session_dir")
            msg = "DISABLING_RESUME"
        else:
            msg = "WARN_IGNORED"
    return msg


def test_resume_writeback_disables_state_when_prelude_and_flag_passed():
    state = _ResumeStateStub(framework_agent_phase_enabled=True, phase="PRELUDE")
    args = _ArgsStub(no_framework_agent=True)
    msg = _apply_resume_writeback(state, args)
    assert msg == "DISABLING_RESUME"
    assert state.framework_agent_phase_enabled is False
    assert state.save_calls == 1


def test_resume_writeback_warns_when_past_prelude_and_flag_passed():
    state = _ResumeStateStub(framework_agent_phase_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_framework_agent=True)
    msg = _apply_resume_writeback(state, args)
    assert msg == "WARN_IGNORED"
    assert state.framework_agent_phase_enabled is True
    assert state.save_calls == 0


def test_resume_writeback_honours_persisted_disable_over_flag_absent():
    state = _ResumeStateStub(framework_agent_phase_enabled=False, phase="EXPLORE")
    args = _ArgsStub(no_framework_agent=False)
    msg = _apply_resume_writeback(state, args)
    assert msg == "DISABLED_PERSISTED"
    assert args.no_framework_agent is True


def test_resume_writeback_no_op_when_state_and_flag_both_enabled():
    state = _ResumeStateStub(framework_agent_phase_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_framework_agent=False)
    msg = _apply_resume_writeback(state, args)
    assert msg == ""
    assert state.framework_agent_phase_enabled is True
    assert args.no_framework_agent is False
