"""Filesystem path resolver (DESIGN v0.6 §23).

Two distinct path concepts:

1. **Session paths** — per-run mutable artifacts (SQLite DB, state.json,
   personas, results, findings). Production NFS layout::

        /hyperloom/inference_optimizer-sessions/<session_id>/
            storage/coordinator.db
            state.json
            personas/  checkpoints/  kb/  results/  findings/

   Override for local dev / tests: ``INFERENCE_OPTIMIZER_SESSION_ROOT``.
   Override DB location only: ``INFERENCE_OPTIMIZER_DB_PATH`` (lets prod
   keep DB on local sandbox disk while session_dir lives on NFS).

2. **Runtime asset paths** — read-only files shipped with the package
   (shell scripts, kernel-opt prompt templates, action metadata, agent
   system prompts). The orchestrator only *reads* these — it never
   modifies them. Override: ``INFERENCE_OPTIMIZER_ASSET_ROOT``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

DEFAULT_PROD_ROOT = Path("/workspace/hyperloom/inference_optimizer-sessions")
ENV_OVERRIDE_ROOT = "INFERENCE_OPTIMIZER_SESSION_ROOT"
ENV_OVERRIDE_DB_PATH = "INFERENCE_OPTIMIZER_DB_PATH"
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"

PACKAGE_ROOT = Path(__file__).resolve().parent

_SESSION_SUBDIRS = (
    "storage",
    "personas",
    "checkpoints",
    "kb",
    "results",
    "findings",
    "agents",
)


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def session_root() -> Path:
    override = os.environ.get(ENV_OVERRIDE_ROOT)
    if override:
        return Path(override)
    return DEFAULT_PROD_ROOT


def make_session_dir(session_id: str | None = None) -> Path:
    """Create ``<session_root>/<session_id>/`` with all standard subdirs."""
    sid = session_id or uuid.uuid4().hex[:12]
    session_dir = session_root() / sid
    for sub in _SESSION_SUBDIRS:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    return session_dir


def db_path_for(session_dir: Path) -> Path:
    explicit = os.environ.get(ENV_OVERRIDE_DB_PATH)
    if explicit:
        return Path(explicit)
    path = session_dir / "storage" / "coordinator.db"
    legacy_path = session_dir / "storage" / "conductor.db"
    if legacy_path.exists() and not path.exists():
        return legacy_path
    return path


def asset_root() -> Path:
    """Return the package runtime-asset root (shipped read-only files)."""
    override = os.environ.get(ENV_OVERRIDE_ASSET_ROOT)
    if override:
        root = Path(override).expanduser()
        if not root.exists():
            raise AssetRootNotFound(
                f"{ENV_OVERRIDE_ASSET_ROOT} points at missing dir: {root}"
            )
        return root
    return PACKAGE_ROOT


def asset_scripts_dir() -> Path:
    return asset_root() / "scripts"


def asset_actions_dir() -> Path:
    return asset_root() / "actions"


def asset_system_prompts_dir() -> Path:
    return asset_root() / "orchestrator" / "system_prompts"


def asset_kernel_opt_dir() -> Path:
    return asset_root() / "kernel_opt"


def agent_session_dir(session_dir: Path, agent_name: str) -> Path:
    """Per-agent inbox/outbox directory under the session.

    Used by Multi-CLI runtime; SINGLE_PROC mode also writes here for
    debugging parity with multi-cli (DESIGN §20).
    """
    d = session_dir / "agents" / agent_name
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "AssetRootNotFound",
    "DEFAULT_PROD_ROOT",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_OVERRIDE_DB_PATH",
    "ENV_OVERRIDE_ROOT",
    "PACKAGE_ROOT",
    "agent_session_dir",
    "asset_actions_dir",
    "asset_kernel_opt_dir",
    "asset_root",
    "asset_scripts_dir",
    "asset_system_prompts_dir",
    "db_path_for",
    "make_session_dir",
    "session_root",
]
