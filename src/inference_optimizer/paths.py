"""Filesystem path resolver.

Two distinct path concepts live here:

1. **Session paths** — per-run artifacts (SQLite DB, state.json, personas,
   results, findings, ...). Production NFS layout::

        /hyperloom/inference-optimizer-sessions/<session_id>/
            storage/conductor.db
            state.json
            personas/  checkpoints/  kb/  results/  findings/

   Local dev / test override: ``INFERENCE_OPTIMIZER_SESSION_ROOT``.

2. **Runtime asset paths** — read-only files shipped with the Python package
   itself (shell scripts that touch the GPU, kernel-opt prompt templates,
   action metadata). Layout::

        src/inference_optimizer/
            scripts/         ← run_baseline.sh, eval_accuracy.sh, geak_ray_submit.py, ...
            kernel_opt/      ← geak.md / claude.md / codex.md / llm.md prompts
            actions/         ← <name>.md + _meta/<name>.yaml (active catalogue)
            orchestrator/system_prompts/

   The Python orchestrator never *modifies* these files — it only reads
   them and shells out to the scripts via :class:`ActionExecutor`.

   Override: ``INFERENCE_OPTIMIZER_ASSET_ROOT`` for tests / vendored
   deploys. ``INFERENCE_OPTIMIZER_SKILL_ROOT`` is still accepted as a
   legacy alias while older launch scripts are drained.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# Session root (per-run mutable state)
# --------------------------------------------------------------------------
DEFAULT_PROD_ROOT = Path("/hyperloom/inference-optimizer-sessions")
ENV_OVERRIDE_ROOT = "INFERENCE_OPTIMIZER_SESSION_ROOT"
ENV_OVERRIDE_DB_PATH = "INFERENCE_OPTIMIZER_DB_PATH"


def session_root() -> Path:
    """Root directory that holds *all* sessions.

    Falls back to ``DEFAULT_PROD_ROOT`` only when the env override is
    missing — we still return the path so callers can inspect it, but
    never silently create the production NFS path on a dev box.
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

    Production deploys may set ``INFERENCE_OPTIMIZER_DB_PATH`` to keep
    the DB on local sandbox disk (path A in DESIGN §3.5.8) while
    ``session_dir`` still points at NFS for backups, results, personas.
    """
    explicit = os.environ.get(ENV_OVERRIDE_DB_PATH)
    if explicit:
        return Path(explicit)
    return session_dir / "storage" / "conductor.db"


# --------------------------------------------------------------------------
# Runtime asset root (read-only shipped files)
# --------------------------------------------------------------------------
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"
ENV_OVERRIDE_SKILL_ROOT = "INFERENCE_OPTIMIZER_SKILL_ROOT"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILL_REL_PATH = Path(".cursor") / "skills" / "inference-optimizer"


class SkillRootNotFound(RuntimeError):
    """Raised when an explicit runtime asset override is invalid.

    Kept for backward compatibility with callers that already catch this
    exception around action-registry setup.
    """


def _override_root() -> Path | None:
    """Return the explicit runtime-asset root, if one was configured."""
    for env_name in (ENV_OVERRIDE_ASSET_ROOT, ENV_OVERRIDE_SKILL_ROOT):
        override = os.environ.get(env_name)
        if not override:
            continue
        root = Path(override).expanduser()
        if not root.exists():
            raise SkillRootNotFound(
                f"{env_name} points at a missing runtime asset root: {root}"
            )
        return root
    return None


def asset_root() -> Path:
    """Return the package runtime-asset root.

    The default is the installed ``inference_optimizer`` package directory,
    so the optimizer can run outside Cursor and outside a Git checkout.
    """
    return _override_root() or PACKAGE_ROOT


def skill_root() -> Path:
    """Backward-compatible alias for :func:`asset_root`."""
    return asset_root()


def asset_scripts_dir() -> Path:
    """Shell + Python tools shipped with the package."""
    return asset_root() / "scripts"


def asset_kernel_opt_dir() -> Path:
    """Per-backend prompt templates (geak.md / claude.md / codex.md / llm.md)."""
    return asset_root() / "kernel_opt"


def asset_actions_dir() -> Path:
    """Active action catalogue: ``<name>.md`` + ``_meta/<name>.yaml``."""
    return asset_root() / "actions"


def asset_system_prompts_dir() -> Path:
    """Markdown system prompts loaded by :class:`AgentRole`."""
    return asset_root() / "orchestrator" / "system_prompts"


def asset_script(name: str) -> Path:
    """Resolve a specific shell / Python script under ``asset_scripts_dir``."""
    return asset_scripts_dir() / name


def skill_scripts_dir() -> Path:
    """Backward-compatible alias for :func:`asset_scripts_dir`."""
    return asset_scripts_dir()


def skill_kernel_opt_dir() -> Path:
    """Backward-compatible alias for :func:`asset_kernel_opt_dir`."""
    return asset_kernel_opt_dir()


def skill_actions_dir() -> Path:
    """Backward-compatible alias for :func:`asset_actions_dir`."""
    return asset_actions_dir()


def skill_actions_old_dir() -> Path:
    """Legacy alias kept for callers that still probe the old directory."""
    return asset_root() / "actions_old"


def skill_system_prompts_dir() -> Path:
    """Backward-compatible alias for :func:`asset_system_prompts_dir`."""
    return asset_system_prompts_dir()


def skill_script(name: str) -> Path:
    """Backward-compatible wrapper around :func:`asset_script`.

    Existing executor call sites can keep using ``skill_script("...")``
    while runtime assets live in the package directory.
    """
    return asset_script(name)


__all__ = [
    "DEFAULT_PROD_ROOT",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_OVERRIDE_DB_PATH",
    "ENV_OVERRIDE_ROOT",
    "ENV_OVERRIDE_SKILL_ROOT",
    "PACKAGE_ROOT",
    "SKILL_REL_PATH",
    "SkillRootNotFound",
    "asset_actions_dir",
    "asset_kernel_opt_dir",
    "asset_root",
    "asset_script",
    "asset_scripts_dir",
    "asset_system_prompts_dir",
    "db_path_for",
    "make_session_dir",
    "session_root",
    "skill_actions_dir",
    "skill_actions_old_dir",
    "skill_kernel_opt_dir",
    "skill_root",
    "skill_script",
    "skill_scripts_dir",
    "skill_system_prompts_dir",
]
