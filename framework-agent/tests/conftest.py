"""Shared pytest fixtures for framework_agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kb_tmp_root(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the framework_agent KB resolver to a clean tmp_path.

    Sets ``FRAMEWORK_AGENT_KB_DIR`` and unsets ``FRAMEWORK_AGENT_ROOT``
    so :func:`framework_agent.kb._resolve_kb_root` returns the tmp
    directory regardless of the surrounding shell environment.
    """
    root = Path(str(tmp_path))
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(root))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    return root
