"""Tests for standalone kernel-agent path resolution helpers."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _reload_paths():
    import _paths

    return importlib.reload(_paths)


def test_workspace_root_returns_user_data_path_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    paths = _reload_paths()

    assert paths.workspace_root() == str(tmp_path)


def test_workspace_root_warns_once_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    paths = _reload_paths()

    with caplog.at_level(logging.WARNING, logger="_paths"):
        assert paths.workspace_root() == "/workspace/hyperloom"
        assert paths.workspace_root() == "/workspace/hyperloom"

    warnings = [r for r in caplog.records if "USER_DATA_PATH" in r.message]
    assert len(warnings) == 1
