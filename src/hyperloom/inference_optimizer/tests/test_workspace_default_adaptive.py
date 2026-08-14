# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The unset-USER_DATA_PATH default must suit the host it lands on.

``/workspace`` is a container convention: the ROCm images ship it writable, and
a bare-metal host off root has neither the directory nor permission to create
it, so the installers' ``mkdir -p`` aborted under ``set -e``.

Three copies of this resolver exist because the modules holding them are barred
from importing one another -- ``tools/`` scripts run standalone on remote nodes
and the ``fa`` CLI must not depend on inference_optimizer. Nothing but a test
keeps them in step.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.agents.framework import kb
from hyperloom.agents.kernel.tools import _paths as tool_paths
from hyperloom.inference_optimizer.session import paths as session_paths

_RESOLVERS = (
    ("session.paths", lambda: str(session_paths.default_workspace_root())),
    ("tools._paths", tool_paths.default_workspace_root),
    ("framework.kb", kb._default_workspace_root),
)


@pytest.mark.parametrize("name,resolve", _RESOLVERS, ids=[n for n, _ in _RESOLVERS])
def test_falls_back_to_the_caller_directory_without_a_writable_workspace(name, resolve, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)
    monkeypatch.chdir(tmp_path)

    assert Path(resolve()) == tmp_path / "session"


@pytest.mark.parametrize("name,resolve", _RESOLVERS, ids=[n for n, _ in _RESOLVERS])
def test_keeps_the_pod_local_path_when_workspace_is_writable(name, resolve, monkeypatch):
    """Container behaviour must not change: the image provides /workspace."""
    monkeypatch.setattr(os, "access", lambda _path, _mode: True)

    assert Path(resolve()) == Path("/workspace/hyperloom")


@pytest.mark.parametrize("name,resolve", _RESOLVERS, ids=[n for n, _ in _RESOLVERS])
def test_a_creatable_workspace_is_still_used(name, resolve, monkeypatch):
    """``/workspace`` absent but creatable must not divert the run.

    ``os.access`` is False for a path that does not exist, so testing the target
    itself sends root -- who could have created it, and whose earlier runs did --
    to <cwd>/session, and its existing sessions appear to vanish.
    """
    monkeypatch.setattr(os.path, "exists", lambda p: str(p) == "/")
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/")
    monkeypatch.setattr(os, "access", lambda path, _mode: str(path) == "/")

    assert Path(resolve()) == Path("/workspace/hyperloom")


def test_the_three_mirrors_agree(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for writable in (True, False):
        monkeypatch.setattr(os, "access", lambda _path, _mode, _w=writable: _w)
        resolved = {str(Path(resolve())) for _name, resolve in _RESOLVERS}
        assert len(resolved) == 1, f"writable={writable}: mirrors disagree: {resolved}"


def test_an_explicit_user_data_path_still_wins(monkeypatch, tmp_path):
    """The adaptive default must not shadow an operator's choice."""
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("USER_DATA_PATH", str(chosen))
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)

    assert session_paths.workspace_root() == chosen
    assert tool_paths.workspace_root() == str(chosen)
