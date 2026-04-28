"""Shared pytest fixtures for the inference-optimizer test suite."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make the package importable when running pytest from the repo root.
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Avoid talking to the production NFS path during tests.
os.environ.setdefault("INFERENCE_OPTIMIZER_SESSION_ROOT", str(Path(tempfile.gettempdir()) / "io-tests"))


@pytest.fixture
def session_dir(tmp_path):
    """A clean session directory laid out like production NFS."""
    sd = tmp_path / "session_xyz"
    for sub in ("storage", "personas", "checkpoints", "kb", "results", "findings"):
        (sd / sub).mkdir(parents=True, exist_ok=True)
    return sd


@pytest.fixture
def db_path(session_dir):
    return session_dir / "storage" / "conductor.db"


@pytest.fixture
def db(db_path):
    """Fresh ``SqliteConnection`` for one test."""
    from inference_optimizer.storage.connection import SqliteConnection

    conn = SqliteConnection(db_path)
    yield conn
    conn.close()
