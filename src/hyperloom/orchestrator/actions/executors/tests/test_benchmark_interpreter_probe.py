# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Magpie import probe only treats a failed launch or a hung probe as "cannot import"."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _benchmark_interpreter as bi

_CANDIDATE = "/fake/bin/python3"


@pytest.fixture
def resolve(monkeypatch):
    """Resolve against an environment where only ``_CANDIDATE`` exists and no ``/opt/venv`` is present."""
    monkeypatch.setattr(bi.shutil, "which", lambda *_a, **_k: _CANDIDATE)
    monkeypatch.setattr(Path, "is_file", lambda _self: False)
    return lambda: bi._resolve_magpie_python({"MAGPIE_PYTHON": "/stale/python", "PATH": "/fake/bin"})


@pytest.mark.parametrize(
    "error",
    [OSError("exec failed"), subprocess.TimeoutExpired(cmd="python", timeout=10)],
)
def test_a_probe_that_cannot_run_answers_false(monkeypatch, resolve, error):
    def _fail(*_a, **_k):
        raise error

    monkeypatch.setattr(bi, "run_with_session_kill", _fail)

    assert resolve() == _CANDIDATE


def test_an_unexpected_probe_failure_propagates(monkeypatch, resolve):
    def _fail(*_a, **_k):
        raise RuntimeError("probe bug")

    monkeypatch.setattr(bi, "run_with_session_kill", _fail)

    with pytest.raises(RuntimeError, match="probe bug"):
        resolve()
