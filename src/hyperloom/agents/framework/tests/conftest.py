# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the framework-agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kb_tmp_root(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the KB resolver to a clean tmp_path through the supported override."""
    root = Path(str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(root))
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    return root
