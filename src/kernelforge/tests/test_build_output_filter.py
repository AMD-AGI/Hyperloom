# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The build is the one command rtk has no filter for, and it is the noisiest.

`rtk err` is the answer: command-agnostic, keeps errors and warnings, preserves
the exit status. These guard the wiring that reaches for it and the tail that
gets handed to the agent when a build fails.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess

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


# ─── the wiring, executed rather than asserted about ───


_NOISY_BUILD = (
    "sh",
    "-c",
    'for i in $(seq 1 400); do echo "[$i/400] Compiling object_$i.cpp"; done; '
    "echo 'src/kernel.cpp:12:5: error: no matching function for call to bar()' >&2; "
    "exit 7",
)


def _rtk_on_path() -> bool:
    import shutil

    return shutil.which("rtk") is not None


@pytest.mark.skipif(not _rtk_on_path(), reason="rtk is not installed on this machine")
def test_the_wrapped_build_actually_shrinks_and_keeps_the_diagnostic() -> None:
    """Run the real wrapper on a real noisy build, rather than assert about it.

    The forge-loop hands ``err_wrap(build_command)`` to ``subprocess`` and reads
    the tail of what comes back. This runs that exact path: the 400 progress
    lines have to go, the compiler error has to survive, and the inner exit
    status has to reach the caller -- the loop branches on it to decide BUILD
    FAILED.
    """
    importlib.reload(rtk)
    raw = subprocess.run(_NOISY_BUILD, capture_output=True)
    wrapped = subprocess.run(rtk.err_wrap(list(_NOISY_BUILD)), capture_output=True)

    assert wrapped.returncode == 7, "the inner exit status must survive the wrapper"
    raw_size = len(raw.stdout) + len(raw.stderr)
    tail = _build_failure_tail(wrapped.stdout, wrapped.stderr, 500)
    assert "no matching function" in tail, f"the one line worth reading was dropped: {tail!r}"
    assert "Compiling object_200" not in tail
    filtered = len(wrapped.stdout) + len(wrapped.stderr)
    assert filtered < raw_size / 5, f"{raw_size} -> {filtered} bytes is not a useful reduction"


@pytest.mark.skipif(not _rtk_on_path(), reason="rtk is not installed on this machine")
def test_a_succeeding_build_is_not_reported_as_a_failure() -> None:
    """A clean build must still exit 0 through the wrapper."""
    importlib.reload(rtk)
    ok = ("sh", "-c", "echo building; echo done")
    result = subprocess.run(rtk.err_wrap(list(ok)), capture_output=True)
    assert result.returncode == 0


@pytest.mark.skipif(not _rtk_on_path(), reason="rtk is not installed on this machine")
@pytest.mark.parametrize(
    "cmd, expected_code",
    [
        (["sh", "-c", "for i in 1 2; do echo $i; done; exit 5"], 5),
        (["python3", "-c", "print('one two'); import sys; sys.exit(3)"], 3),
        (["false"], 1),
        (["true"], 0),
    ],
)
def test_the_wrapper_runs_the_command_it_was_given(cmd: list[str], expected_code: int) -> None:
    """The wrapped command must be the same command, not a re-split lookalike.

    ``rtk err`` joins its arguments into one line for ``sh``. Handed a raw argv
    it re-splits any argument holding a space or a metacharacter, which turns
    ``sh -c "for …; do …; done"`` into a syntax error and a build flag like
    ``-DCMAKE_CXX_FLAGS=-O3 -g`` into two flags. Matching exit status against a
    direct run is the check that the round trip changed nothing.
    """
    importlib.reload(rtk)
    direct = subprocess.run(cmd, capture_output=True)
    assert direct.returncode == expected_code, "the fixture itself is wrong"
    assert subprocess.run(rtk.err_wrap(cmd), capture_output=True).returncode == expected_code


@pytest.mark.skipif(not _rtk_on_path(), reason="rtk is not installed on this machine")
def test_an_argument_holding_a_space_stays_one_argument(tmp_path) -> None:
    """A path with a space must reach the command whole."""
    importlib.reload(rtk)
    folder = tmp_path / "has space"
    folder.mkdir()
    (folder / "f.txt").write_text("x")
    assert subprocess.run(rtk.err_wrap(["ls", str(folder)]), capture_output=True).returncode == 0
