"""Filesystem path resolver.

Two distinct path concepts live here:

1. **Session paths** — per-run artifacts (SQLite DB, state.json, personas,
   results, findings, ...). Production NFS layout::

        /hyperloom/inference-optimizer-sessions/<session_id>/
            storage/conductor.db
            state.json
            personas/  checkpoints/  kb/  results/  findings/

   Local dev / test override: ``INFERENCE_OPTIMIZER_SESSION_ROOT``.

2. **Skill asset paths** — read-only files shipped with the skill itself
   (shell scripts that touch the GPU, kernel-opt prompt templates,
   action metadata). Layout::

        <repo>/.cursor/skills/inference-optimizer/
            scripts/         ← run_baseline.sh, eval_accuracy.sh, geak_ray_submit.py, ...
            kernel-opt/      ← geak.md / claude.md / codex.md / llm.md prompts
            actions/         ← <name>.md + _meta/<name>.yaml (active catalogue)
            actions_old/     ← legacy reference (kept until full deprecation)
            system_prompts/  ← per-role markdown bodies

   The Python orchestrator never *modifies* these files — it only reads
   them and shells out to the scripts via :class:`ActionExecutor`.

   Override: ``INFERENCE_OPTIMIZER_SKILL_ROOT`` for tests / vendored
   deploys where the skill lives outside ``.cursor/``.
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
# Skill asset root (read-only shipped files)
# --------------------------------------------------------------------------
ENV_OVERRIDE_SKILL_ROOT = "INFERENCE_OPTIMIZER_SKILL_ROOT"
SKILL_REL_PATH = Path(".cursor") / "skills" / "inference-optimizer"
_SKILL_SENTINEL = "SKILL.md"


class SkillRootNotFound(RuntimeError):
    """Raised when ``skill_root()`` cannot locate the skill directory.

    Either set ``INFERENCE_OPTIMIZER_SKILL_ROOT`` or run from inside a
    repository checkout that contains
    ``.cursor/skills/inference-optimizer/SKILL.md``.
    """


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` looking for a checkout that contains the skill.

    The sentinel is the SKILL.md file itself rather than ``.git`` so this
    works in vendored / sub-tree layouts where there is no top-level
    Git directory.
    """
    cur = Path(start).resolve()
    candidates: list[Path] = []
    if cur.is_file():
        candidates.append(cur.parent)
    else:
        candidates.append(cur)
    candidates.extend(cur.parents)
    for parent in candidates:
        if (parent / SKILL_REL_PATH / _SKILL_SENTINEL).exists():
            return parent
    return None


def skill_root() -> Path:
    """Return absolute path to ``.cursor/skills/inference-optimizer/``.

    Resolution order (cheapest → fallback):

        1. ``INFERENCE_OPTIMIZER_SKILL_ROOT`` env override (must point
           directly at the skill directory).
        2. Walk up from this module's location looking for
           ``.cursor/skills/inference-optimizer/SKILL.md``.
        3. Walk up from the current working directory.
        4. Raise :class:`SkillRootNotFound`.
    """
    override = os.environ.get(ENV_OVERRIDE_SKILL_ROOT)
    if override:
        return Path(override)
    repo = _find_repo_root(Path(__file__))
    if repo is None:
        repo = _find_repo_root(Path.cwd())
    if repo is None:
        raise SkillRootNotFound(
            "could not locate .cursor/skills/inference-optimizer/SKILL.md "
            "from this module or cwd; set "
            f"{ENV_OVERRIDE_SKILL_ROOT}=<path-to-skill-dir> to override."
        )
    return repo / SKILL_REL_PATH


def skill_scripts_dir() -> Path:
    """Shell + Python tools under the skill (run_baseline.sh, eval_accuracy.sh,
    geak_ray_submit.py, ...). All ``ActionExecutor`` shells out here."""
    return skill_root() / "scripts"


def skill_kernel_opt_dir() -> Path:
    """Per-backend prompt templates (geak.md / claude.md / codex.md / llm.md)."""
    return skill_root() / "kernel-opt"


def skill_actions_dir() -> Path:
    """Active action catalogue: ``<name>.md`` + ``_meta/<name>.yaml``."""
    return skill_root() / "actions"


def skill_actions_old_dir() -> Path:
    """Legacy single-skill action descriptions; kept for cross-referencing
    until the full sister-skill deprecation lands."""
    return skill_root() / "actions_old"


def skill_system_prompts_dir() -> Path:
    """Markdown system prompts loaded by :class:`AgentRole`."""
    return skill_root() / "system_prompts"


def skill_script(name: str) -> Path:
    """Resolve a specific shell / Python script under ``skill_scripts_dir``.

    Convenience wrapper used by ``ActionExecutor`` subclasses so the call
    site reads ``skill_script("run_baseline.sh")`` instead of nesting
    path joins.
    """
    return skill_scripts_dir() / name


__all__ = [
    "DEFAULT_PROD_ROOT",
    "ENV_OVERRIDE_DB_PATH",
    "ENV_OVERRIDE_ROOT",
    "ENV_OVERRIDE_SKILL_ROOT",
    "SKILL_REL_PATH",
    "SkillRootNotFound",
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
