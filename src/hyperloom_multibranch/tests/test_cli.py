# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``hyperloom-multibranch`` command line."""

from __future__ import annotations

import pytest

from hyperloom_multibranch import cli


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert cli.PROG in capsys.readouterr().out


def test_run_without_runner_fails_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 1
    assert "not implemented" in capsys.readouterr().err
