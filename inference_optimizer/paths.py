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
#
# The ``runtime/`` subtree (Magpie clone, source mirrors, pod-local env files,
# GEAK config) and the ``kernel-agent/runs/`` + ``optimizer_runs/`` trees were
# folded into the session_dir as part of the "all artefacts under
# ``$USER_DATA_PATH``" migration. Older deployments may still find writable
# defaults at ``/workspace/Magpie`` / ``/opt/hyperloom``; new defaults all
# live under the session root so a single ``$USER_DATA_PATH`` move relocates
# everything the operator could possibly want to monitor.
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
    "kernel-agent",            # tools/<name>.py output root (runs/<session_id>/...)
    "patches",
    "optimizer_runs",          # launcher stdout / pid / robustness monitor logs
    "runtime",                 # pod-local env files (kernel-agent.env.sh, etc.)
    "runtime/source-mirrors",  # writable mirrors of GEAK / OOB / TraceLens sources
    "runtime/geak-config",     # generated litellm config consumed by GEAK CLI
    # v0.8 M1 — Cortex KB integration. Holds the per-session ``.kb_sid`` /
    # ``.kb_warm.json`` / ``.kb_pitfalls.json`` / ``.kb_pending.ndjson`` /
    # ``.kb_flushed.ndjson`` / ``.kb_dead_letter.ndjson`` / ``.kb_audit.jsonl``
    # / ``.kb_flusher.pid`` files described in KB_design §3.6 + §3.13 M1.
    # Created up-front so the CortexKBClient never has to ``mkdir -p`` on
    # the hot path; absent files imply ``--no-cortex`` or pre-T0 state.
    "runtime/cortex",
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


# ---------------------------------------------------------------------------
# "All artefacts under $USER_DATA_PATH" helpers
# ---------------------------------------------------------------------------
# Every writable per-pod / per-session product (Magpie clone, source mirrors,
# kernel-agent tool runs, launcher logs) is derived from the session_dir.
# The single source of truth lives here so anywhere that needs e.g. the
# Magpie clone path goes through ``magpie_dir(session_dir())`` instead of
# string-concatenating ``$WORKSPACE_ROOT/Magpie``.
def runtime_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/`` — pod-local writable runtime resources.

    Holds the generated kernel-agent env file (sourced by every CLI
    invocation), the generated GEAK litellm config, and the writable
    source mirrors created by ``kernel-agent/scripts/install.sh``.
    """
    return Path(session_dir) / "runtime"


def source_mirrors_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/source-mirrors/`` — writable mirrors of read-only sources.

    GEAK clone, OOB CLI mirror, and TraceLens mirror all land here so
    the installer's ``pip install -e`` calls land on a writable
    filesystem regardless of whether the original source mount (WekaFS)
    is read-only. Replaces the legacy ``$HYPERLOOM_ROOT`` (default
    ``/opt/hyperloom``) location.
    """
    return runtime_dir(session_dir) / "source-mirrors"


def magpie_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/Magpie/`` — Magpie clone owned by this session.

    The legacy default lived at ``$WORKSPACE_ROOT/Magpie`` (``/workspace/Magpie``);
    moving it under ``runtime/`` keeps every writable artefact under one
    monitorable root. ``$MAGPIE_DIR`` still overrides this for operators
    who want to share a pre-cloned Magpie across sessions.
    """
    return runtime_dir(session_dir) / "Magpie"


def kernel_agent_runs_root(session_dir: Path) -> Path:
    """``<sd>/kernel-agent/`` — output root for kernel-agent CLI tools.

    Distinct from ``<sd>/kernel-agent-workspace/`` (cross-task GEAK/OOB
    work artefacts keyed by ``kernel_id``). This root holds per-tool-call
    logs, status JSON, TraceLens runs, optimization_attempts.jsonl, etc.
    — one ``runs/<session_id>/`` subdirectory per tool invocation. See
    ``kernel-agent/SKILL.md`` "Artifacts" for the on-disk schema.
    """
    return Path(session_dir) / "kernel-agent"


def optimizer_runs_dir(session_dir: Path) -> Path:
    """``<sd>/optimizer_runs/`` — launcher-side stdout / PID / monitor logs.

    Replaces ``$REPO_ROOT/optimizer_runs/``. Lives inside the session so
    a single ``$USER_DATA_PATH`` move relocates the entire run-time tail
    (including ``robustness_monitor*.log`` and ``run_<tag>.{log,pid}``).
    """
    return Path(session_dir) / "optimizer_runs"


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
    "kernel_agent_runs_root",
    "magpie_dir",
    "make_session_dir",
    "optimizer_runs_dir",
    "runtime_dir",
    "session_dir",
    "source_mirrors_dir",
]
