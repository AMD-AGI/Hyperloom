# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared pytest fixtures for framework_agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kb_tmp_root(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the KB resolver to a clean tmp_path via FRAMEWORK_AGENT_KB_DIR (and unset FRAMEWORK_AGENT_ROOT)."""
    root = Path(str(tmp_path))
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(root))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    return root
