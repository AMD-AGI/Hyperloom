# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for robustness-agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.robustness.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "storage").mkdir()
    return Config(
        session_dir=session_dir,
        agent_stall_timeout_s=10.0,
    )
