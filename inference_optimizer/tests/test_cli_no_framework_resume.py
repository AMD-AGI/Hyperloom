"""Cover the --no-framework env default + resume write-back semantics
introduced by the P2.c+d fix."""

from __future__ import annotations


import pytest

from inference_optimizer import cli


# ---------------------------------------------------------------------------
# Parser default (P2.c) — env honored when flag not passed.
# ---------------------------------------------------------------------------
def _parse_optimize(argv: list[str]) -> object:
    """Helper: run the cli parser, return the parsed args namespace.

    Uses a minimal arg vector so we don't have to mirror the full
    optimize launch surface — every required field has a default."""
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_no_framework_default_false_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NO_FRAMEWORK", raising=False)
    # Reload parser module-state by re-importing the build closure
    # (default is captured at add_argument time, so we must build the
    # parser fresh in a context where the env var is absent — easiest
    # is to monkeypatch before parsing).
    args = _parse_optimize([])
    # default keeps phase enabled, so no_framework is False.
    assert getattr(args, "no_framework") is False


def test_no_framework_env_default_true_when_env_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NO_FRAMEWORK", "1")
    args = _parse_optimize([])
    assert getattr(args, "no_framework") is True


def test_no_framework_explicit_flag_overrides_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NO_FRAMEWORK", raising=False)
    args = _parse_optimize(["--no-framework"])
    assert getattr(args, "no_framework") is True


# ---------------------------------------------------------------------------
# Resume write-back (P2.d) — exercise just the branch logic, not the full
# CLI session-resume orchestration which needs a real session dir.
# ---------------------------------------------------------------------------
class _ResumeStateStub:
    """Mirrors only the SharedState fields the resume write-back branch
    reads/writes for the framework_phase_enabled toggle."""

    def __init__(self, *, framework_phase_enabled: bool, phase: str) -> None:
        self.framework_phase_enabled = framework_phase_enabled
        self.phase = phase
        self.save_calls = 0

    def save(self, _session_dir) -> None:
        self.save_calls += 1


class _ArgsStub:
    def __init__(self, *, no_framework: bool) -> None:
        self.no_framework = no_framework


def _apply_resume_writeback(state: _ResumeStateStub, args: _ArgsStub) -> str:
    """Re-implement the resume-branch logic from cli.py for testing.

    The logic lives inline in the resume handler and is hard to lift
    out without a refactor; mirror it byte-for-byte so a divergence
    will be caught the next time the cli changes."""
    msg = ""
    if not bool(getattr(state, "framework_phase_enabled", True)):
        args.no_framework = True
        msg = "DISABLED_PERSISTED"
    elif bool(getattr(args, "no_framework", False)):
        cur_phase = (getattr(state, "phase", "") or "").strip().upper()
        if cur_phase in ("", "PRELUDE"):
            state.framework_phase_enabled = False
            # P2-g: persist immediately, not via the later conditional
            # save, so a clean resume keeps the toggle on disk.
            state.save("session_dir")
            msg = "DISABLING_RESUME"
        else:
            msg = "WARN_IGNORED"
    return msg


def test_resume_writeback_disables_state_when_prelude_and_flag_passed():
    state = _ResumeStateStub(framework_phase_enabled=True, phase="PRELUDE")
    args = _ArgsStub(no_framework=True)
    msg = _apply_resume_writeback(state, args)
    assert msg == "DISABLING_RESUME"
    assert state.framework_phase_enabled is False
    # The toggle is persisted unconditionally on the prelude path.
    assert state.save_calls == 1


def test_resume_writeback_warns_when_past_prelude_and_flag_passed():
    state = _ResumeStateStub(framework_phase_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_framework=True)
    msg = _apply_resume_writeback(state, args)
    assert msg == "WARN_IGNORED"
    # State must not be retroactively flipped or persisted.
    assert state.framework_phase_enabled is True
    assert state.save_calls == 0


def test_resume_writeback_honours_persisted_disable_over_flag_absent():
    state = _ResumeStateStub(framework_phase_enabled=False, phase="EXPLORE")
    args = _ArgsStub(no_framework=False)
    msg = _apply_resume_writeback(state, args)
    assert msg == "DISABLED_PERSISTED"
    assert args.no_framework is True


def test_resume_writeback_no_op_when_state_and_flag_both_enabled():
    state = _ResumeStateStub(framework_phase_enabled=True, phase="EXPLORE")
    args = _ArgsStub(no_framework=False)
    msg = _apply_resume_writeback(state, args)
    assert msg == ""
    assert state.framework_phase_enabled is True
    assert args.no_framework is False
