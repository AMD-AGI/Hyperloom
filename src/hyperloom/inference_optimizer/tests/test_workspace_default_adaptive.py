# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The unset-USER_DATA_PATH default must suit the host it lands on."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from hyperloom.inference_optimizer.session import paths as session_paths


def test_falls_back_to_the_caller_directory_without_a_writable_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)
    monkeypatch.chdir(tmp_path)

    assert session_paths.default_workspace_root() == tmp_path / "session"


def test_keeps_the_pod_local_path_when_workspace_is_writable(monkeypatch):
    """Container behaviour must not change: the image provides /workspace."""
    monkeypatch.setattr(os, "access", lambda _path, _mode: True)

    assert session_paths.default_workspace_root() == Path("/workspace/hyperloom")


def test_a_creatable_workspace_is_still_used(monkeypatch):
    """``/workspace`` absent but creatable must not divert the run."""
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/")
    monkeypatch.setattr(os, "access", lambda path, _mode: str(path) == "/")

    assert session_paths.default_workspace_root() == Path("/workspace/hyperloom")


def test_an_explicit_user_data_path_still_wins(monkeypatch, tmp_path):
    """The adaptive default must not shadow an operator's choice."""
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("USER_DATA_PATH", str(chosen))
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)

    assert session_paths.workspace_root() == chosen


def test_workspace_root_warns_once_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    monkeypatch.setattr(session_paths, "_WARNED_NO_USER_DATA", False)
    expected = session_paths.default_workspace_root()

    with caplog.at_level(logging.WARNING, logger=session_paths.log.name):
        assert session_paths.workspace_root() == expected
        assert session_paths.workspace_root() == expected

    warnings = [r for r in caplog.records if "USER_DATA_PATH" in r.message]
    assert len(warnings) == 1
