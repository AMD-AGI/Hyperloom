# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the Critic runtime tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_session_root(tmp_path: Path) -> Path:
    """Per-test session-memory root that's isolated from $HOME / /var."""
    root = tmp_path / "critic-session-memory"
    root.mkdir()
    return root
