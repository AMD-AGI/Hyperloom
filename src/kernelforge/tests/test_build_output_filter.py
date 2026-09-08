# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The build is the one command rtk has no filter for, and it is the noisiest.

`rtk err` is the answer: command-agnostic, keeps errors and warnings, preserves
the exit status. These guard the wiring that reaches for it and the tail that
gets handed to the agent when a build fails.
"""

from __future__ import annotations

import inspect

import pytest

from kernelforge import rtk
from kernelforge.loop.runner import _build_failure_tail


def test_err_wrap_reaches_for_the_command_agnostic_filter(monkeypatch) -> None:
    """`rtk err` is what a build goes through -- bare `rtk` passes ninja straight on."""
    monkeypatch.setattr(rtk, "_RTK_PATH", "/opt/bin/rtk")
    assert rtk.err_wrap(["ninja", "-j4"]) == ["/opt/bin/rtk", "err", "ninja", "-j4"]
    # Contrast: the generic wrap produces the no-op this replaces.
    assert rtk.wrap_command(["ninja", "-j4"]) == ["/opt/bin/rtk", "ninja", "-j4"]


def test_err_wrap_is_a_no_op_without_rtk(monkeypatch) -> None:
    """A box without rtk builds exactly as before, not through a missing binary."""
    monkeypatch.setattr(rtk, "_RTK_PATH", None)
    assert rtk.err_wrap(["ninja", "-j4"]) == ["ninja", "-j4"]


def test_err_wrap_leaves_an_empty_command_alone(monkeypatch) -> None:
    """No build configured means no `rtk err` with nothing to run."""
    monkeypatch.setattr(rtk, "_RTK_PATH", "/opt/bin/rtk")
    assert rtk.err_wrap([]) == []


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        # ninja routes the compiler's own output to stdout; reading stderr alone
        # reported a build failure with no reason attached.
        (b"src/x.cpp:1:1: error: boom\n", b"", "src/x.cpp:1:1: error: boom"),
        (b"", b"ld: undefined symbol\n", "ld: undefined symbol"),
        (b"out line\n", b"err line\n", "out line\nerr line"),
        (b"", b"", "no build output"),
        (b"   \n", b"  ", "no build output"),
    ],
)
def test_the_failure_tail_comes_from_whichever_stream_carried_it(stdout, stderr, expected) -> None:
    assert _build_failure_tail(stdout, stderr, 500) == expected


def test_the_failure_tail_keeps_the_end_not_the_beginning() -> None:
    """A long build's last lines are the diagnostic; its first are the progress."""
    tail = _build_failure_tail(b"x" * 900 + b"error: the real one", b"", 30)
    assert tail.endswith("error: the real one")
    assert len(tail) == 30


def test_undecodable_build_output_does_not_crash_the_iteration() -> None:
    """A compiler that emits invalid UTF-8 must not turn a build failure into one of ours."""
    assert "error" in _build_failure_tail(b"\xff\xfe error: bad byte", b"", 500)


def test_the_prompt_asks_for_filters_that_exist() -> None:
    """The paragraph the agent reads must name rtk's real coverage.

    It used to ask for `rtk ninja -j4` and `rtk rocprofv3`. rtk 0.48 has a
    filter for neither and passes an unknown command through, so every one of
    those prefixes was output the prompt paid for and got nothing back. Guarded
    at source level because the paragraph is assembled inside `make_agent_fn`
    from the local runtime, and its wording -- not its plumbing -- is what
    decides whether the agent's shell output is trimmed.
    """
    from kernelforge.orchestrator import agent

    source = inspect.getsource(agent)
    guidance_start = source.index("_rtk_guidance = (")
    guidance = source[guidance_start : source.index("workspace_hygiene_rule")]

    assert "rtk err" in guidance, "builds need the command-agnostic error filter"
    assert "rtk test" in guidance, "test runs need the command-agnostic failure filter"
    for dead in ("`rtk ninja", "`rtk cmake", "`rtk rocprofv3"):
        assert dead not in guidance, f"{dead}` is a passthrough: rtk ships no filter for it"
