# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/check_broad_suppress.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from check_broad_suppress import main


def write(tmp_path: Path, name: str, code: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return p


def test_clean_file_exits_zero(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "clean.py",
        """\
        import contextlib
        with contextlib.suppress(OSError):
            open("x")
        with contextlib.suppress(ValueError, TypeError):
            int("abc")
        """,
    )
    assert main([str(p)]) == 0


@pytest.mark.parametrize(
    "caught",
    ["Exception", "BaseException", "OSError, Exception"],
    ids=["exception", "base_exception", "exception_in_a_tuple"],
)
def test_suppress_exception_detected(tmp_path: Path, caught: str) -> None:
    p = write(
        tmp_path,
        "broad.py",
        f"""\
        import contextlib
        with contextlib.suppress({caught}):
            some_call()
        """,
    )
    assert main([str(p)]) == 1


@pytest.mark.parametrize(
    "relative",
    ["tests/helpers.py", "test_broad.py"],
    ids=["in_a_tests_directory", "test_prefix"],
)
def test_test_files_are_skipped(tmp_path: Path, relative: str) -> None:
    p = tmp_path / relative
    p.parent.mkdir(exist_ok=True)
    p.write_text(
        "import contextlib\nwith contextlib.suppress(Exception):\n    pass\n",
        encoding="utf-8",
    )
    assert main([str(tmp_path)]) == 0


def test_waiver_comment_allows_broad_suppress(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "waived.py",
        """\
        import contextlib
        with contextlib.suppress(Exception):  # broad-suppress: caller-supplied callback
            cb()
        """,
    )
    assert main([str(p)]) == 0


def test_waiver_survives_a_formatter_wrapped_header(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "wrapped.py",
        """\
        import contextlib
        with contextlib.suppress(
            Exception
        ):  # broad-suppress: caller-supplied callback
            cb()
        """,
    )
    assert main([str(p)]) == 0


def test_waiver_without_a_reason_is_rejected(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "empty_reason.py",
        """\
        import contextlib
        with contextlib.suppress(Exception):  # broad-suppress:
            cb()
        """,
    )
    assert main([str(p)]) == 1


def test_syntax_error_skipped(tmp_path: Path) -> None:
    p = tmp_path / "broken.py"
    p.write_text("def (\n", encoding="utf-8")
    assert main([str(p)]) == 0


def test_multiple_files_one_broad(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    write(tmp_path, "a.py", "with __import__('contextlib').suppress(OSError): pass\n")
    write(tmp_path, "broad.py", "import contextlib\nwith contextlib.suppress(Exception): pass\n")
    result = main([str(tmp_path)])
    assert result == 1
    captured = capsys.readouterr()
    assert "broad.py" in captured.out
