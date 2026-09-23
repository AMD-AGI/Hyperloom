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


def test_suppress_exception_detected(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "broad.py",
        """\
        import contextlib
        with contextlib.suppress(Exception):
            some_call()
        """,
    )
    assert main([str(p)]) == 1


def test_suppress_base_exception_detected(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "broad_base.py",
        """\
        import contextlib
        with contextlib.suppress(BaseException):
            some_call()
        """,
    )
    assert main([str(p)]) == 1


def test_exception_in_tuple_detected(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "tuple_broad.py",
        """\
        import contextlib
        with contextlib.suppress(OSError, Exception):
            some_call()
        """,
    )
    assert main([str(p)]) == 1


def test_test_files_are_skipped(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    p = tests_dir / "test_something.py"
    p.write_text(
        "import contextlib\nwith contextlib.suppress(Exception):\n    pass\n",
        encoding="utf-8",
    )
    assert main([str(tmp_path)]) == 0


def test_test_prefix_files_skipped(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "test_broad.py",
        """\
        import contextlib
        with contextlib.suppress(Exception):
            pass
        """,
    )
    assert main([str(p)]) == 0


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


def test_multiple_files_all_clean(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "with __import__('contextlib').suppress(OSError): pass\n")
    write(tmp_path, "b.py", "x = 1\n")
    assert main([str(tmp_path)]) == 0


def test_multiple_files_one_broad(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    write(tmp_path, "a.py", "with __import__('contextlib').suppress(OSError): pass\n")
    write(tmp_path, "broad.py", "import contextlib\nwith contextlib.suppress(Exception): pass\n")
    result = main([str(tmp_path)])
    assert result == 1
    captured = capsys.readouterr()
    assert "broad.py" in captured.out
