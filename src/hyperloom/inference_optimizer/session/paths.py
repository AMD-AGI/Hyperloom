# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filesystem path resolver."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from uuid import uuid4

from hyperloom.common.timeutil import utc_now_compact

log = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
#: Mirrored verbatim in agents/kernel/tools/_paths.py and agents/framework/kb.py,
#: which cannot import this module. Keep the three in step.
POD_LOCAL_WORKSPACE = Path("/workspace")
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"
ENV_CURRENT_SESSION_DIR = "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"
ENV_CACHE_DIR = "HYPERLOOM_CACHE_DIR"
ENV_REPO_ROOT = "REPO_ROOT"

# Shipped read-only asset dirs live directly under ``inference_optimizer/``, one level up from this ``session/``
# module — hence ``.parent.parent``.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# One-shot guard so the "USER_DATA_PATH unset" fallback warning fires at most once per process (workspace_root() is on
# a hot path).
_WARNED_NO_USER_DATA = False

# Per-session directory skeleton mkdir-ed by make_session_dir().
_SESSION_SKELETON: tuple[str, ...] = (
    "storage",
    "personas",
    "checkpoints",
    "kb",
    "findings",
    "reports",
    "agents/orchestration",
    "agents/critic",
    "agents/robustness",
    "runs/baseline",
    "runs/profile",
    "runs/backends",
    "runs/params",
    "runs/integrate",
    "runs/kernel_opt",
    "kernel-agent-workspace",
    "kernel-agent",  # tools/<name>.py output root (runs/<session_id>/...)
    "patches",
    "optimizer_runs",  # launcher stdout / pid / robustness monitor logs
)

# Workspace-shared layout (one copy per $USER_DATA_PATH). mkdir-ed by install.sh + reused for every session_dir
# launched from this workspace.
_WORKSPACE_SKELETON: tuple[str, ...] = (
    "runtime",  # install-generated env files (kernel-agent.env.sh, GEAK litellm config)
    # Workspace-level KB root; it only coincides with the KB path when session_dir == workspace_root.
    "runtime/recipe_kb",
    "logs",  # launcher stdout (workspace-shared)
)

# Filename-safety regex for model_basename (ROCm/Magpie/Claude CLI choke on ``:`` / ``/`` / whitespace).
_MODEL_BASENAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def default_workspace_root() -> Path:
    """The workspace to use when ``$USER_DATA_PATH`` is unset."""
    # The nearest *existing* ancestor decides: os.access is False for a path that does not exist yet, which would
    # divert root off a /workspace it can create.
    probe = POD_LOCAL_WORKSPACE
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if os.access(probe, os.W_OK):
        return DEFAULT_SESSION_DIR
    return Path.cwd() / "session"


def workspace_root() -> Path:
    """Operator-facing workspace root: ``$USER_DATA_PATH`` (else ``DEFAULT_SESSION_DIR``), regardless of layout mode."""
    global _WARNED_NO_USER_DATA
    user_data = (os.environ.get(ENV_USER_DATA_PATH) or "").strip()
    if user_data:
        return Path(user_data).expanduser()
    if not _WARNED_NO_USER_DATA:
        _WARNED_NO_USER_DATA = True
        log.warning(
            "%s is not set; falling back to %s. All session/run artefacts "
            "will be written there, NOT to an operator-chosen location. "
            "Export %s to the intended workspace root before launching to "
            "avoid silently writing to the default.",
            ENV_USER_DATA_PATH,
            default_workspace_root(),
            ENV_USER_DATA_PATH,
        )
    return default_workspace_root()


def _sanitize_model_basename(model_name: str | os.PathLike[str]) -> str:
    """Reduce ``model_name`` (path, HF id, or Path) to a filename-safe basename (trailing path component)."""
    stem = ("" if model_name is None else str(model_name)).strip()
    if not stem:
        return "session"
    stem = stem.rstrip("/")
    if "/" in stem:
        stem = stem.rsplit("/", 1)[1]
    stem = _MODEL_BASENAME_SANITIZE.sub("_", stem).strip("_.-")
    return stem or "session"


def session_dir() -> Path:
    """Absolute session directory for the current run."""
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
    if pinned:
        return Path(pinned)
    return workspace_root()


def make_session_dir(model_name: str | os.PathLike[str] | None = None) -> Path:
    """Create the session directory + per-session + workspace-shared skeletons."""
    ws = workspace_root()
    ws.mkdir(parents=True, exist_ok=True)
    for sub in _WORKSPACE_SKELETON:
        (ws / sub).mkdir(parents=True, exist_ok=True)

    if model_name:
        basename = _sanitize_model_basename(model_name)
        sd = ws / basename / f"{utc_now_compact()}-{uuid4().hex[:8]}"
    else:
        sd = ws

    sd.mkdir(parents=True, exist_ok=True)
    for sub in _SESSION_SKELETON:
        (sd / sub).mkdir(parents=True, exist_ok=True)
    # Pin for downstream callers + subprocesses; the most recent call wins.
    os.environ[ENV_CURRENT_SESSION_DIR] = str(sd)
    return sd


def db_path_for(session_dir: Path) -> Path:
    """Return the canonical SQLite database path for a session."""
    return Path(session_dir) / "storage" / "coordinator.db"


def asset_root() -> Path:
    """Return the package runtime-asset root (shipped read-only files)."""
    override = os.environ.get(ENV_OVERRIDE_ASSET_ROOT)
    if override:
        root = Path(override).expanduser()
        if not root.exists():
            raise AssetRootNotFound(f"{ENV_OVERRIDE_ASSET_ROOT} points at missing dir: {root}")
        return root
    return PACKAGE_ROOT


def asset_system_prompts_dir() -> Path:
    """Return the directory of shipped agent system prompts."""
    if os.environ.get(ENV_OVERRIDE_ASSET_ROOT):
        return asset_root() / "orchestrator" / "prompts"
    import hyperloom.orchestrator.prompts as _prompts_pkg

    return Path(_prompts_pkg.__file__).resolve().parent


def asset_prompt_references_dir() -> Path:
    """Return the directory of shipped on-demand prompt reference documents."""
    return asset_system_prompts_dir() / "references"


# Workspace-/session-scoped artefact helpers.
def runtime_dir() -> Path:
    """``<workspace_root>/runtime/`` — workspace-shared writable runtime (kernel-agent env file, GEAK litellm config)."""
    return workspace_root() / "runtime"


def deps_cache_root() -> Path:
    """Writable cache root for open-source dependency checkouts."""
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override)
    repo_root = os.environ.get(ENV_REPO_ROOT) or os.getcwd()
    return Path(repo_root) / ".cache"


def _dir_mtime(p: Path) -> float:
    """``p``'s mtime, or ``0.0`` if it vanished mid-resolution (race-safe)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def resolve_dep_dir(name: str, env_var: str | None = None) -> Path:
    """Resolve a dependency checkout, bridging install.sh's per-revision ``<name>@<sha>`` layout to runtime callers."""
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return Path(override)
    root = deps_cache_root()
    pinned = [p for p in root.glob(f"{name}@*") if p.is_dir()]
    if pinned:
        return max(pinned, key=_dir_mtime)
    return root / name


def magpie_dir() -> Path:
    """Magpie checkout root, via :func:`resolve_dep_dir` (``$MAGPIE_PATH`` else newest ``Magpie@<sha>`` else bare — Magpie is pip-installed, so bare is the common case)."""
    return resolve_dep_dir("Magpie", "MAGPIE_PATH")


def tracelens_root() -> Path:
    """TraceLens checkout root, via :func:`resolve_dep_dir` (``$TRACELENS_ROOT`` else newest ``TraceLens@<sha>`` else bare)."""
    return resolve_dep_dir("TraceLens", "TRACELENS_ROOT")


def is_path_within(path: Path, root: Path) -> bool:
    """Whether ``path`` provably resolves to ``root`` or a location below it."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def mn_profile_trace_root() -> Path:
    """``<workspace_root>/profile-traces/`` — multi-node torch profile shared root (``<rayjob_id>/torch_trace/`` per provision)."""
    return workspace_root() / "profile-traces"


__all__ = [
    "AssetRootNotFound",
    "DEFAULT_SESSION_DIR",
    "ENV_CURRENT_SESSION_DIR",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_USER_DATA_PATH",
    "PACKAGE_ROOT",
    "asset_prompt_references_dir",
    "asset_root",
    "asset_system_prompts_dir",
    "db_path_for",
    "is_path_within",
    "magpie_dir",
    "make_session_dir",
    "mn_profile_trace_root",
    "deps_cache_root",
    "resolve_dep_dir",
    "runtime_dir",
    "session_dir",
    "workspace_root",
]
