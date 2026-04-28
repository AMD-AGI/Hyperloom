"""Filesystem path resolver.

Production NFS layout:
    /hyperloom/inference-optimizer-sessions/<session_id>/
        storage/conductor.db
        state.json
        personas/
        checkpoints/
        kb/
        results/
        findings/

Local dev/test override: set ``INFERENCE_OPTIMIZER_SESSION_ROOT`` to any
directory; e.g. a tempfile.TemporaryDirectory() during pytest.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

DEFAULT_PROD_ROOT = Path("/hyperloom/inference-optimizer-sessions")
ENV_OVERRIDE_ROOT = "INFERENCE_OPTIMIZER_SESSION_ROOT"
ENV_OVERRIDE_DB_PATH = "INFERENCE_OPTIMIZER_DB_PATH"


def session_root() -> Path:
    """Root directory that holds *all* sessions.

    Falls back to ``DEFAULT_PROD_ROOT`` only when the env override is missing
    *and* it actually exists (so we don't pretend to write to /hyperloom on
    a Windows dev box).
    """
    override = os.environ.get(ENV_OVERRIDE_ROOT)
    if override:
        return Path(override)
    return DEFAULT_PROD_ROOT


def make_session_dir(session_id: str | None = None) -> Path:
    """Create and return ``<root>/<session_id>/`` along with subdirs."""
    sid = session_id or uuid.uuid4().hex[:12]
    root = session_root()
    session_dir = root / sid
    for sub in ("storage", "personas", "checkpoints", "kb", "results", "findings"):
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    return session_dir


def db_path_for(session_dir: Path) -> Path:
    """Resolve the SQLite path for a session.

    Production deploys may set ``INFERENCE_OPTIMIZER_DB_PATH`` to keep the DB
    on local sandbox disk (path A in DESIGN §3.5.8) while ``session_dir``
    still points at NFS for backups, results, personas, etc.
    """
    explicit = os.environ.get(ENV_OVERRIDE_DB_PATH)
    if explicit:
        return Path(explicit)
    return session_dir / "storage" / "conductor.db"
