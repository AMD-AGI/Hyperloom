# Copyright Advanced Micro Devices, Inc. All rights reserved.

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

import datetime
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"
ENV_SESSION_LAYOUT = "INFERENCE_OPTIMIZER_SESSION_LAYOUT"
ENV_CURRENT_SESSION_DIR = "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"

PACKAGE_ROOT = Path(__file__).resolve().parent

# One-shot guard so the "USER_DATA_PATH unset" fallback warning fires at most
# once per process (workspace_root() is on a hot path).
_WARNED_NO_USER_DATA = False

# Per-session directory skeleton mkdir-ed by make_session_dir(). The layout
# splits into workspace-shared roots (runtime/, logs/ — one per
# $USER_DATA_PATH) and per-session roots (storage/, agents/, runs/, ... —
# one per model+timestamp). Default layout is ``per_model_ts``
# (``$USER_DATA_PATH/<model>/<UTC_ts>/``); set
# $INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat for the legacy single-dir layout.
_SESSION_SKELETON: tuple[str, ...] = (
    "storage",
    "personas",
    "checkpoints",
    "kb",
    "findings",
    "reports",
    "agents/orchestration",
    "agents/orchestration/dynamic_actions",  # per-dyn_id subdirs mkdir-ed at dispatch
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
    "kernel-agent",            # tools/<name>.py output root (runs/<session_id>/...)
    "patches",
    "optimizer_runs",          # launcher stdout / pid / robustness monitor logs
)

# Workspace-shared layout (one copy per $USER_DATA_PATH). mkdir-ed by
# install.sh + reused for every session_dir launched from this workspace.
_WORKSPACE_SKELETON: tuple[str, ...] = (
    "runtime",                 # pod-local env files (kernel-agent.env.sh, etc.)
    "runtime/source-mirrors",  # writable mirrors of GEAK / OOB / TraceLens sources
    "runtime/geak-config",     # generated litellm config consumed by GEAK CLI
    # Cortex KB per-session bookkeeping (.kb_sid / .kb_warm.json / ...);
    # created up-front so the KB client never mkdir's on the hot path.
    "runtime/cortex",
    "logs",                    # launcher stdout (workspace-shared)
)

# Filename-safety regex for model_basename (ROCm/Magpie/Claude CLI choke
# on ``:`` / ``/`` / whitespace).
_MODEL_BASENAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def workspace_root() -> Path:
    """Operator-facing workspace root: ``$USER_DATA_PATH`` (else
    ``DEFAULT_SESSION_DIR``), regardless of layout mode. Workspace-shared
    artefacts (runtime/, logs/) live here. Falling back to the default emits
    one loud warning so a misconfigured launcher is visible.
    """
    global _WARNED_NO_USER_DATA
    user_data = os.environ.get(ENV_USER_DATA_PATH)
    if user_data:
        return Path(user_data)
    if not _WARNED_NO_USER_DATA:
        _WARNED_NO_USER_DATA = True
        log.warning(
            "%s is not set; falling back to %s. All session/run artefacts "
            "will be written there, NOT to an operator-chosen location. "
            "Export %s to the intended workspace root before launching to "
            "avoid silently writing to the pod-local default.",
            ENV_USER_DATA_PATH,
            DEFAULT_SESSION_DIR,
            ENV_USER_DATA_PATH,
        )
    return DEFAULT_SESSION_DIR


def _layout_mode() -> str:
    """Effective layout mode: ``flat`` or ``per_model_ts`` (default), pinnable
    via the env override."""
    raw = (os.environ.get(ENV_SESSION_LAYOUT) or "").strip().lower()
    if raw in ("flat", "per_model_ts"):
        return raw
    return "per_model_ts"


def _sanitize_model_basename(model_name: str | os.PathLike[str]) -> str:
    """Reduce ``model_name`` (path, HF id, or Path) to a filename-safe
    basename (trailing path component). Empty/all-invalid -> ``"session"``."""
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
    inherited by subprocesses) -> ``$USER_DATA_PATH`` (flat layout) ->
    ``DEFAULT_SESSION_DIR``.
    """
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
    if pinned:
        return Path(pinned)
    return workspace_root()


def find_latest_per_session_dir(
    model_name: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Latest per-session subdir under :func:`workspace_root` (used by
    ``--resume`` without ``--resume-from``). Selects by the
    ``%Y%m%dT%H%M%SZ`` timestamp in the directory name (lex sort), not mtime.
    Returns None when no matching subdir exists.
    """
    ws = workspace_root()
    if not ws.is_dir():
        return None
    if model_name:
        basename = _sanitize_model_basename(model_name)
        model_root = ws / basename
        if not model_root.is_dir():
            return None
        candidates = [
            p for p in model_root.iterdir()
            if p.is_dir() and len(p.name) == 16 and p.name.endswith("Z")
        ]
    else:
        # Scan every model_basename subdir; the timestamp-shaped name check
        # skips workspace-shared subdirs (runtime/, logs/).
        candidates: list[Path] = []
        for model_dir in ws.iterdir():
            if not model_dir.is_dir() or model_dir.name in ("runtime", "logs"):
                continue
            for p in model_dir.iterdir():
                if p.is_dir() and len(p.name) == 16 and p.name.endswith("Z"):
                    candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)  # lex == chronological for the ts name
    return candidates[-1]


def make_session_dir(model_name: str | os.PathLike[str] | None = None) -> Path:
    """Create the session directory + per-session + workspace-shared
    skeletons. In ``per_model_ts`` mode with a ``model_name`` the session_dir
    is ``<workspace_root>/<model>/<UTC_ts>/`` and is pinned via
    ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``; otherwise it is
    workspace_root (flat layout). Idempotent.
    """
    ws = workspace_root()
    ws.mkdir(parents=True, exist_ok=True)
    for sub in _WORKSPACE_SKELETON:
        (ws / sub).mkdir(parents=True, exist_ok=True)

    if _layout_mode() == "per_model_ts" and model_name:
        basename = _sanitize_model_basename(model_name)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        sd = ws / basename / ts
    else:
        sd = ws

    sd.mkdir(parents=True, exist_ok=True)
    for sub in _SESSION_SKELETON:
        (sd / sub).mkdir(parents=True, exist_ok=True)
    # Pin for downstream callers + subprocesses; overwrite any prior pin so
    # the most recent make_session_dir() call is authoritative.
    os.environ[ENV_CURRENT_SESSION_DIR] = str(sd)
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
    """Per-agent inbox/outbox dir under the session (created by
    make_session_dir; this only computes the path)."""
    return Path(session_dir) / "agents" / agent_name


# Workspace-/session-scoped artefact helpers. Single source of truth so
# callers go through e.g. magpie_dir(session_dir()) instead of concatenating
# paths by hand.
def runtime_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/`` — workspace-shared writable runtime
    (kernel-agent env file, GEAK litellm config, source mirrors). Survives
    across sessions; the ``session_dir`` param is ignored (back-compat).
    """
    return workspace_root() / "runtime"


def source_mirrors_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/source-mirrors/`` — writable GEAK / OOB /
    TraceLens mirrors so ``pip install -e`` works even when the source mount
    is read-only. ``session_dir`` param ignored (back-compat).
    """
    return runtime_dir() / "source-mirrors"


def magpie_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/Magpie/`` — Magpie clone (workspace-shared;
    ``$MAGPIE_DIR`` overrides). ``session_dir`` param ignored (back-compat).
    """
    return runtime_dir() / "Magpie"


def kernel_agent_runs_root(session_dir: Path) -> Path:
    """``<sd>/kernel-agent/`` — kernel-agent CLI tool output root (one
    ``runs/<session_id>/`` per invocation). Distinct from the kernel_id-keyed
    ``<sd>/kernel-agent-workspace/``.
    """
    return Path(session_dir) / "kernel-agent"


def optimizer_runs_dir(session_dir: Path) -> Path:
    """``<sd>/optimizer_runs/`` — launcher stdout / PID / robustness monitor
    logs."""
    return Path(session_dir) / "optimizer_runs"


def mn_profile_trace_root() -> Path:
    """``<workspace_root>/profile-traces/`` — multi-node torch profile shared
    root (``<rayjob_id>/torch_trace/`` per provision). Multi-node operators
    MUST set ``$USER_DATA_PATH`` to a cluster-shared path or the sandbox never
    sees pod-side trace files. Single-node never reads this.
    """
    return workspace_root() / "profile-traces"


__all__ = [
    "AssetRootNotFound",
    "DEFAULT_SESSION_DIR",
    "ENV_CURRENT_SESSION_DIR",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_SESSION_LAYOUT",
    "ENV_USER_DATA_PATH",
    "PACKAGE_ROOT",
    "agent_session_dir",
    "asset_actions_dir",
    "asset_kernel_opt_dir",
    "asset_root",
    "asset_scripts_dir",
    "asset_system_prompts_dir",
    "db_path_for",
    "kernel_agent_runs_root",
    "magpie_dir",
    "make_session_dir",
    "mn_profile_trace_root",
    "optimizer_runs_dir",
    "runtime_dir",
    "session_dir",
    "source_mirrors_dir",
    "find_latest_per_session_dir",
    "workspace_root",
]
