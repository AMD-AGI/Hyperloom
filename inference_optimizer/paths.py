"""Filesystem path resolver.

Two distinct path concepts:

1. **Session paths** — per-run mutable artifacts (SQLite DB, state.json,
   personas, results, kernel-agent workspace, patches, logs). State
   lives in one fixed directory::

        /workspace/hyperloom/
            manifest.json
            state.json
            storage/coordinator.db
            agents/{orchestration,kernel,critic,robustness}/...
            personas/  checkpoints/  findings/  kb/
            runs/{baseline,profile,backends,params,sweep,
                  integrate,kernel_opt}/<task_id>/...
            kernel-agent-workspace/<kernel_id>/
            patches/<kernel_id>/
            reports/  logs/

   Each sandbox is single-use, so there is no session_id subdirectory.

   Resolution order (see :func:`session_dir`):

   1. ``USER_DATA_PATH`` — user-facing env (documented in
      ``.env.template`` and ``SKILL.md``); production launchers and
      the SDK set this.
   2. ``DEFAULT_SESSION_DIR`` (``/workspace/hyperloom``).

2. **Runtime asset paths** — read-only files shipped with the package
   (shell scripts, kernel-opt prompt templates, action metadata, agent
   system prompts). The orchestrator only *reads* these — it never
   modifies them. Override: ``INFERENCE_OPTIMIZER_ASSET_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"

PACKAGE_ROOT = Path(__file__).resolve().parent

# Directory skeleton mkdir-ed on `make_session_dir()`. Order is irrelevant
# (each dir is created with `parents=True, exist_ok=True`), but the listing
# below is the canonical layout — keep it in sync with the docstring above
# and SKILL.md "Session Layout".
_SESSION_SKELETON: tuple[str, ...] = (
    "storage",
    "personas",
    "checkpoints",
    "kb",
    "findings",
    "reports",
    "logs",
    "agents/orchestration",
    "agents/kernel",
    "agents/critic",
    "agents/robustness",
    "runs/baseline",
    "runs/profile",
    "runs/backends",
    "runs/params",
    "runs/sweep",
    "runs/integrate",
    "runs/kernel_opt",
    "kernel-agent-workspace",
    "patches",
)


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def session_dir() -> Path:
    """Return the absolute session directory for the current run.

    Resolution order:

    1. ``$USER_DATA_PATH`` env var (documented in ``.env.template`` and
       ``SKILL.md``; production launchers and the SDK set this).
    2. ``DEFAULT_SESSION_DIR`` (``/workspace/hyperloom``).
    """
    user_data = os.environ.get(ENV_USER_DATA_PATH)
    if user_data:
        return Path(user_data)
    return DEFAULT_SESSION_DIR


def make_session_dir() -> Path:
    """Create the session directory + full subdirectory skeleton.

    Idempotent (``mkdir -p`` semantics). Returns the absolute path.

    The CLI calls this exactly once at startup, before the Coordinator
    is instantiated. Tests call it after pinning ``USER_DATA_PATH`` to
    ``tmp_path``.
    """
    sd = session_dir()
    sd.mkdir(parents=True, exist_ok=True)
    for sub in _SESSION_SKELETON:
        (sd / sub).mkdir(parents=True, exist_ok=True)
    return sd


def db_path_for(session_dir: Path) -> Path:
    """Canonical SQLite location for a session: ``<sd>/storage/coordinator.db``."""
    return Path(session_dir) / "storage" / "coordinator.db"


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

    Created up-front by ``make_session_dir()``; this helper only computes
    the path. Used by Multi-CLI runtime; SINGLE_PROC mode also writes
    here for debugging parity with multi-cli (DESIGN §20).
    """
    return Path(session_dir) / "agents" / agent_name


__all__ = [
    "AssetRootNotFound",
    "DEFAULT_SESSION_DIR",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_USER_DATA_PATH",
    "PACKAGE_ROOT",
    "agent_session_dir",
    "asset_actions_dir",
    "asset_kernel_opt_dir",
    "asset_root",
    "asset_scripts_dir",
    "asset_system_prompts_dir",
    "db_path_for",
    "make_session_dir",
    "session_dir",
]
