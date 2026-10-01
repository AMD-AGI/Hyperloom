# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the operator-script tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def rsi_config_dict(tmp_path: Path) -> dict:
    """A minimal valid Meta RSI round configuration rooted in ``tmp_path``."""
    return {
        "round_dir": str(tmp_path / "round"),
        "target": {"repo": str(tmp_path / "repo"), "base_ref": "main", "branch": "rsi/test"},
        "data": {"bundles_dir": str(tmp_path / "bundles"), "recent_since": "2026-09-15"},
        "agent": {"model": "claude-opus-5", "total_budget_usd": 10},
        "checks": {"python": sys.executable, "lint": ["true"]},
        "ab": {
            "python": sys.executable,
            "sessions_dir": str(tmp_path / "arms"),
            "arms": [{"name": "A", "tree": "base"}, {"name": "B", "tree": "candidate"}],
        },
    }
