# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared pytest fixtures for the Critic runtime tests.

Keeping these tiny and orthogonal makes individual test files readable and
prevents the runtime tests from depending on the wider Hyperloom test
harness — the runtime is intended to be pip-installable on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def tmp_session_root(tmp_path: Path) -> Path:
    """Per-test session-memory root that's isolated from $HOME / /var."""
    root = tmp_path / "critic-session-memory"
    root.mkdir()
    return root


@pytest.fixture()
def write_json(tmp_path: Path):
    """Helper to drop a JSON object on disk and return the path."""

    def _write(name: str, body: dict[str, Any]) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(body), encoding="utf-8")
        return p

    return _write
