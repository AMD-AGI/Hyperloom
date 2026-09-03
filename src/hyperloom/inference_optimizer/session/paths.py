# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filesystem path resolver.

Two path concepts:

1. **Session paths** — per-run mutable artifacts (SQLite DB, state.json,
   agents, runs, patches, logs). Resolved from ``USER_DATA_PATH`` (else
   ``DEFAULT_SESSION_DIR`` = ``/workspace/hyperloom``).
2. **Runtime asset paths** — read-only files shipped with the package
   (scripts, prompt templates, action metadata, system prompts). Override:
   ``INFERENCE_OPTIMIZER_ASSET_ROOT``.
"""

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

# Shipped read-only asset dirs live directly under ``inference_optimizer/``,
# one level up from this ``session/`` module — hence ``.parent.parent``.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# One-shot guard so the "USER_DATA_PATH unset" fallback warning fires at most
# once per process (workspace_root() is on a hot path).
_WARNED_NO_USER_DATA = False

# Per-session directory skeleton mkdir-ed by make_session_dir(). Splits into
# workspace-shared roots (runtime/, logs/ — one per $USER_DATA_PATH) and
# per-session roots (one per launch).
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

# Workspace-shared layout (one copy per $USER_DATA_PATH). mkdir-ed by
# install.sh + reused for every session_dir launched from this workspace.
_WORKSPACE_SKELETON: tuple[str, ...] = (
    "runtime",  # install-generated env files (kernel-agent.env.sh, GEAK litellm config)
    # Workspace-level KB root; it only coincides with the KB path when
    # session_dir == workspace_root. The per-session bookkeeping
    # (.kb_warm.json / .kb_pitfalls.json / .kb_lessons.json) lives at
    # <session_dir>/runtime/recipe_kb (session_paths.recipe_kb_dir) and is
    # mkdir-ed by its writers in recipe_kb_t0.
    "runtime/recipe_kb",
    "logs",  # launcher stdout (workspace-shared)
)

# Filename-safety regex for model_basename (ROCm/Magpie/Claude CLI choke
# on ``:`` / ``/`` / whitespace).
_MODEL_BASENAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def default_workspace_root() -> Path:
    """The workspace to use when ``$USER_DATA_PATH`` is unset.

    Container images ship a writable ``/workspace``; a bare-metal non-root host
    has neither that directory nor permission to create it, and the mkdir would
    abort installation. Falling back to the caller's directory also matches what
    the setup skill offers.
    """
    # The nearest *existing* ancestor decides: os.access is False for a path that
    # does not exist yet, which would divert root off a /workspace it can create.
    probe = POD_LOCAL_WORKSPACE
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if os.access(probe, os.W_OK):
        return DEFAULT_SESSION_DIR
    return Path.cwd() / "session"


def workspace_root() -> Path:
    """Operator-facing workspace root: ``$USER_DATA_PATH`` (else
    ``DEFAULT_SESSION_DIR``), regardless of layout mode. Workspace-shared
    artefacts (runtime/, logs/) live here. Falling back to the default emits
    one warning so a misconfigured launcher is visible.

    Returns:
        The workspace root path.
    """
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
    """Reduce ``model_name`` (path, HF id, or Path) to a filename-safe
    basename (trailing path component). Empty/all-invalid -> ``"session"``.

    Args:
        model_name: Model path, HF id, or Path to reduce to a basename.

    Returns:
        A filename-safe basename, or ``"session"`` when empty/all-invalid.
    """
    stem = ("" if model_name is None else str(model_name)).strip()
    if not stem:
        return "session"
    stem = stem.rstrip("/")
    if "/" in stem:
        stem = stem.rsplit("/", 1)[1]
    stem = _MODEL_BASENAME_SANITIZE.sub("_", stem).strip("_.-")
    return stem or "session"


def session_dir() -> Path:
    """Absolute session directory for the current run. Resolution order:
    ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` (pin from make_session_dir,
    inherited by subprocesses) -> ``$USER_DATA_PATH`` -> ``DEFAULT_SESSION_DIR``.

    Returns:
        The absolute session directory for the current run.
    """
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
    if pinned:
        return Path(pinned)
    return workspace_root()


def make_session_dir(model_name: str | os.PathLike[str] | None = None) -> Path:
    """Create the session directory + per-session + workspace-shared
    skeletons. With a ``model_name`` the session_dir is
    ``<workspace_root>/<model>/<UTC_ts>-<rand8>/`` and is pinned via
    ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``; otherwise it is
    workspace_root. Idempotent.

    The random suffix keeps two same-second launches of one model apart; the
    fixed-width timestamp stays first so lexical order remains chronological.

    Args:
        model_name: Model name selecting the per-model subtree, or ``None``
            to use workspace_root directly.

    Returns:
        The created (and pinned) session directory.
    """
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
    """Return the canonical SQLite database path for a session.

    Args:
        session_dir (Path): The session directory root.

    Returns:
        Path: ``<session_dir>/storage/coordinator.db``.
    """
    return Path(session_dir) / "storage" / "coordinator.db"


def asset_root() -> Path:
    """Return the package runtime-asset root (shipped read-only files).

    Honours the ``$INFERENCE_OPTIMIZER_ASSET_ROOT`` override (with ``~``
    expansion) when set; otherwise returns the installed package root.

    Returns:
        Path: The asset root directory.

    Raises:
        AssetRootNotFound: If the override env var is set but points at a
            path that does not exist.
    """
    override = os.environ.get(ENV_OVERRIDE_ASSET_ROOT)
    if override:
        root = Path(override).expanduser()
        if not root.exists():
            raise AssetRootNotFound(f"{ENV_OVERRIDE_ASSET_ROOT} points at missing dir: {root}")
        return root
    return PACKAGE_ROOT


def asset_actions_dir() -> Path:
    """Return the directory of shipped action-metadata files.

    Returns:
        Path: ``<asset_root>/actions``.
    """
    return asset_root() / "actions"


def asset_system_prompts_dir() -> Path:
    """Return the directory of shipped agent system prompts.

    When ``$INFERENCE_OPTIMIZER_ASSET_ROOT`` is set (its per-run override
    symlinks ``orchestrator/`` into the override root) the override is honoured.
    Otherwise the path is resolved via the ``hyperloom.orchestrator.prompts``
    package's own ``__file__``, which is correct in both a source checkout and
    an installed distribution.

    Returns:
        Path: ``<asset_root>/orchestrator/prompts`` (override) or
            ``<hyperloom.orchestrator.prompts package dir>`` (default).
    """
    if os.environ.get(ENV_OVERRIDE_ASSET_ROOT):
        return asset_root() / "orchestrator" / "prompts"
    import hyperloom.orchestrator.prompts as _prompts_pkg

    return Path(_prompts_pkg.__file__).resolve().parent


def asset_prompt_references_dir() -> Path:
    """Return the directory of shipped on-demand prompt reference documents."""
    return asset_system_prompts_dir() / "references"


# Workspace-/session-scoped artefact helpers. Single source of truth so
# callers go through e.g. magpie_dir() / runtime_dir() instead of concatenating
# paths by hand.
def runtime_dir() -> Path:
    """``<workspace_root>/runtime/`` — workspace-shared writable runtime
    (kernel-agent env file, GEAK litellm config). Survives across sessions.

    Returns:
        ``<workspace_root>/runtime``.
    """
    return workspace_root() / "runtime"


def deps_cache_root() -> Path:
    """Writable cache root for open-source dependency checkouts.

    Resolution: ``$HYPERLOOM_CACHE_DIR`` else ``$REPO_ROOT/.cache``
    (``REPO_ROOT`` falls back to the current working directory). Deps are cloned
    per revision under this root; see :func:`resolve_dep_dir`.
    """
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
    """Resolve a dependency checkout, bridging install.sh's per-revision
    ``<name>@<sha>`` layout to runtime callers.

    Order: ``$<env_var>`` (installer-written exact path) → newest
    ``<deps_cache_root>/<name>@<sha>`` → bare ``<deps_cache_root>/<name>``. The
    glob step lets a process that did not inherit the env var still find the
    installer's checkout rather than a path it never created (#722).

    Args:
        name: Dependency directory name (e.g. ``TraceLens``).
        env_var: Installer-exported override env var, if any.

    Returns:
        The resolved dependency checkout path.
    """
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
    """Magpie checkout root, via :func:`resolve_dep_dir` (``$MAGPIE_PATH`` else
    newest ``Magpie@<sha>`` else bare — Magpie is pip-installed, so bare is the
    common case).

    Returns:
        The Magpie package/check-out root path.
    """
    return resolve_dep_dir("Magpie", "MAGPIE_PATH")


def tracelens_root() -> Path:
    """TraceLens checkout root, via :func:`resolve_dep_dir` (``$TRACELENS_ROOT``
    else newest ``TraceLens@<sha>`` else bare).

    Returns:
        The TraceLens checkout path.
    """
    return resolve_dep_dir("TraceLens", "TRACELENS_ROOT")


def is_path_within(path: Path, root: Path) -> bool:
    """Whether ``path`` provably resolves to ``root`` or a location below it.

    Both sides are resolved first, so symlinks and ``..`` components cannot
    escape ``root``. Fails closed: a path that cannot be resolved (broken
    symlink chain, symlink loop, permission error) is not provably inside.

    Args:
        path (Path): The candidate path.
        root (Path): The directory that must contain ``path``.

    Returns:
        Whether the resolved ``path`` is ``root`` or below it.
    """
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def mn_profile_trace_root() -> Path:
    """``<workspace_root>/profile-traces/`` — multi-node torch profile shared
    root (``<rayjob_id>/torch_trace/`` per provision). Multi-node operators
    MUST set ``$USER_DATA_PATH`` to a cluster-shared path or the sandbox never
    sees pod-side trace files. Single-node never reads this.

    Returns:
        ``<workspace_root>/profile-traces``.
    """
    return workspace_root() / "profile-traces"


__all__ = [
    "AssetRootNotFound",
    "DEFAULT_SESSION_DIR",
    "ENV_CURRENT_SESSION_DIR",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_USER_DATA_PATH",
    "PACKAGE_ROOT",
    "asset_actions_dir",
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
