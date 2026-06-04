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

import datetime
import os
import re
from pathlib import Path

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
ENV_OVERRIDE_ASSET_ROOT = "INFERENCE_OPTIMIZER_ASSET_ROOT"
ENV_SESSION_LAYOUT = "INFERENCE_OPTIMIZER_SESSION_LAYOUT"
ENV_CURRENT_SESSION_DIR = "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"

PACKAGE_ROOT = Path(__file__).resolve().parent

# Directory skeleton mkdir-ed on `make_session_dir()`. Order is irrelevant
# (each dir is created with `parents=True, exist_ok=True`), but the listing
# below is the canonical layout — keep it in sync with the docstring above
# and SKILL.md "Session Layout".
#
# The layout used to fold ``runtime/`` (Magpie clone, source mirrors,
# pod-local env files, GEAK config) + ``optimizer_runs/`` +
# ``kernel-agent/`` into the session_dir, on the rationale that
# ``$USER_DATA_PATH`` was the one knob an operator might move. That worked
# when each pod ran a single session end-to-end, but a multi-tenant
# workspace (operator pinning ``$USER_DATA_PATH`` and launching multiple
# optimisation runs against different models back-to-back) silently
# collapsed every session into the same flat dir — state.json / agents /
# runs / manifest all overwritten, no per-session audit possible.
#
# The layout now splits into two roots:
#
# * **Workspace-shared** (one copy per ``$USER_DATA_PATH``, regardless of
#   how many sessions launch from there): ``runtime/`` (Magpie clone,
#   source mirrors, kernel-agent.env.sh, GEAK config) and ``logs/``
#   (launcher stdout). install.sh writes these once and reuses them.
# * **Per-session** (one copy per session, keyed by model + UTC timestamp):
#   ``storage/``, ``agents/``, ``runs/``, ``state.json``, ``manifest.json``,
#   ``personas/``, ``checkpoints/``, ``kb/``, ``findings/``, ``reports/``,
#   ``patches/``, ``optimizer_runs/``, ``kernel-agent/`` runs, ``kernel-
#   agent-workspace/``, ``critic-workdir/``, ``robustness-workdir/``,
#   ``target_analysis/``, ``session_breakdown.json``.
#
# The default layout is now ``per_model_ts`` —
# ``$USER_DATA_PATH/<model_basename>/<UTC_YYYYMMDDTHHMMSSZ>/`` — driven
# by ``make_session_dir(model_name=...)``. Set
# ``$INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat`` to restore the legacy
# behaviour (session_dir == workspace_root). Tests that don't care about
# layout call ``make_session_dir()`` without model_name and get the flat
# layout pinned at the tmp_path that USER_DATA_PATH points to — no
# fixture migration required.
_SESSION_SKELETON: tuple[str, ...] = (
    "storage",
    "personas",
    "checkpoints",
    "kb",
    "findings",
    "reports",
    "agents/orchestration",
    # dispatch-time artefact root for every dyn_id. Per-<dyn_id> subdirs
    # are mkdir-ed on-demand at dispatch.
    "agents/orchestration/dynamic_actions",
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
    # Cortex KB integration. Holds the per-session ``.kb_sid`` /
    # ``.kb_warm.json`` / ``.kb_pitfalls.json`` / ``.kb_audit.jsonl``
    # files. Created up-front so the KB client never has to ``mkdir -p``
    # on the hot path; absent files imply ``--degraded-kb`` or pre-T0
    # state.
    "runtime/cortex",
    "logs",                    # launcher stdout (workspace-shared)
)

# Filename-safety regex for model_basename. WekaFS allows almost any byte
# but ROCm + Magpie + Claude CLI all choke on ``:`` / ``/`` / whitespace,
# so be conservative.
_MODEL_BASENAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class AssetRootNotFound(RuntimeError):
    """Raised when an explicit asset root override points at a missing dir."""


def workspace_root() -> Path:
    """Return the operator-facing root for this Hyperloom workspace.

    This is ``$USER_DATA_PATH`` (falling back to ``DEFAULT_SESSION_DIR``)
    REGARDLESS of layout mode. Everything shared across sessions
    (runtime/, logs/, install.sh's kernel-agent.env.sh) lives here.

    Per-session artefacts live under :func:`session_dir`, which is
    either this same path (legacy ``flat`` layout) or
    ``<workspace_root>/<model>/<ts>/`` (``per_model_ts`` layout).
    """
    user_data = os.environ.get(ENV_USER_DATA_PATH)
    if user_data:
        return Path(user_data)
    return DEFAULT_SESSION_DIR


def _layout_mode() -> str:
    """Effective layout mode for this process. ``flat`` or
    ``per_model_ts``. Defaults to ``per_model_ts`` (the N17 default);
    operators / tests can pin ``flat`` via the env override."""
    raw = (os.environ.get(ENV_SESSION_LAYOUT) or "").strip().lower()
    if raw in ("flat", "per_model_ts"):
        return raw
    return "per_model_ts"


def _sanitize_model_basename(model_name: str | os.PathLike[str]) -> str:
    """Reduce ``model_name`` (may be a full filesystem path, HF id, or
    a Path object) to a filename-safe basename. Empty / all-invalid
    input -> ``"session"``."""
    stem = ("" if model_name is None else str(model_name)).strip()
    if not stem:
        return "session"
    # Treat ``/wekafs/models/DeepSeek-R1-0528`` -> ``DeepSeek-R1-0528``
    # and ``meta-llama/Llama-3.1-70B`` -> ``Llama-3.1-70B`` consistently
    # by always taking the trailing path component.
    stem = stem.rstrip("/")
    if "/" in stem:
        stem = stem.rsplit("/", 1)[1]
    stem = _MODEL_BASENAME_SANITIZE.sub("_", stem).strip("_.-")
    return stem or "session"


def session_dir() -> Path:
    """Return the absolute session directory for the current run.

    Resolution order:

    1. ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` (process-local pin set
       by :func:`make_session_dir`; survives across in-process re-imports
       AND propagates to every subprocess that inherits the env). This
       is the path the CLI and every executor will see at runtime.
    2. ``$USER_DATA_PATH`` (legacy ``flat`` layout — pre-N17 callers and
       test fixtures that monkeypatch USER_DATA_PATH to tmp_path but
       never call make_session_dir() still get a sensible answer).
    3. ``DEFAULT_SESSION_DIR`` (``/workspace/hyperloom``).
    """
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
    if pinned:
        return Path(pinned)
    return workspace_root()


def find_latest_per_session_dir(
    model_name: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Locate the latest per-session subdir under :func:`workspace_root`.

    Used by ``--resume`` (no explicit ``--resume-from``) to pick up where
    the most recent run left off. The per-model/ts layout produced by
    :func:`make_session_dir` writes
    ``<workspace_root>/<model_basename>/<UTC_YYYYMMDDTHHMMSSZ>/``; we
    return the latest such timestamp directory under the matching model
    basename (or, when ``model_name`` is None, the latest across all
    models). Returns None when no matching subdir exists (caller falls
    back to flat layout / errors out).

    The selection criterion is the timestamp **in the directory name**
    (sorted lexicographically — works because the name is fixed
    ``%Y%m%dT%H%M%SZ``), not filesystem mtime, so a touched
    ``state.json`` in an older subdir doesn't shadow a freshly-created
    later run.
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
        # Scan every model_basename subdir under workspace_root for the
        # latest per-launch timestamp. Skip workspace-shared subdirs
        # (runtime/, logs/) by checking the timestamp-shaped name.
        candidates: list[Path] = []
        for model_dir in ws.iterdir():
            if not model_dir.is_dir() or model_dir.name in ("runtime", "logs"):
                continue
            for p in model_dir.iterdir():
                if p.is_dir() and len(p.name) == 16 and p.name.endswith("Z"):
                    candidates.append(p)
    if not candidates:
        return None
    # Lex sort on the YYYYMMDDTHHMMSSZ name works as chronological sort.
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def make_session_dir(model_name: str | os.PathLike[str] | None = None) -> Path:
    """Create the session directory + per-session subdirectory skeleton.

    Layout is governed by :func:`_layout_mode`. When mode is
    ``per_model_ts`` (the N17 default) AND ``model_name`` is non-empty,
    the new session_dir is
    ``<workspace_root>/<sanitized_model_basename>/<UTC_YYYYMMDDTHHMMSSZ>/``
    and is pinned for the rest of this process (and inherited by every
    subprocess) via ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``. In
    every other case (``flat`` layout OR no model_name passed — e.g.
    legacy tests that just call ``make_session_dir()``), the session_dir
    is the workspace_root itself, matching pre-N17 behaviour exactly.

    Also mkdir-s the workspace-shared skeleton (``runtime/``, ``logs/``)
    under workspace_root so install.sh's ``kernel-agent.env.sh`` lands
    in a stable location regardless of how many sessions have been
    created. Idempotent.

    The CLI passes ``model_name=args.model`` from ``_run_optimize`` so
    every production session ends up in its own subdir; tests don't
    have to migrate.
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
    # Pin for downstream callers + subprocesses in this run. We
    # overwrite any prior pin (e.g. from a leaked test) so the most
    # recent make_session_dir() call is always authoritative. The CLI
    # only calls this once per process, so 'overwrite' has no
    # production downside; in tests it guarantees clean isolation
    # without needing every fixture to delenv the var by hand.
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
def runtime_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/`` — workspace-shared writable runtime.

    Holds the generated kernel-agent env file (sourced by every CLI
    invocation), the generated GEAK litellm config, and the writable
    source mirrors created by ``kernel-agent/scripts/install.sh``.

    N17: relocated from ``<session_dir>/runtime/`` to
    ``<workspace_root>/runtime/`` so it survives across multiple
    sessions launched from the same ``$USER_DATA_PATH``. The
    ``session_dir`` parameter is accepted for backward-compat with
    every existing caller (``runtime_dir(session_dir())``) but ignored
    — runtime is workspace-scoped, not session-scoped.
    """
    return workspace_root() / "runtime"


def source_mirrors_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/source-mirrors/`` — writable mirrors.

    GEAK clone, OOB CLI mirror, and TraceLens mirror all land here so
    the installer's ``pip install -e`` calls land on a writable
    filesystem regardless of whether the original source mount (WekaFS)
    is read-only. Replaces the legacy ``$HYPERLOOM_ROOT`` (default
    ``/opt/hyperloom``) location.

    N17: workspace-shared (see :func:`runtime_dir`). ``session_dir``
    param accepted for backward-compat but ignored.
    """
    return runtime_dir() / "source-mirrors"


def magpie_dir(session_dir: Path | None = None) -> Path:
    """``<workspace_root>/runtime/Magpie/`` — Magpie clone (workspace-shared).

    The legacy default lived at ``$WORKSPACE_ROOT/Magpie`` (``/workspace/
    Magpie``); moving it under ``runtime/`` keeps every writable artefact
    under one monitorable root. ``$MAGPIE_DIR`` still overrides this for
    operators who want to share a pre-cloned Magpie across sessions.

    N17: workspace-shared (see :func:`runtime_dir`). ``session_dir``
    param accepted for backward-compat but ignored.
    """
    return runtime_dir() / "Magpie"


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


def mn_profile_trace_root() -> Path:
    """``<workspace_root>/profile-traces/`` — multi-node torch profile shared root.

    Multi-node-only: in ``--nodes >= 2`` mode the sandbox-side optimizer
    and the RayJob GPU pods need a single path both sides can read/write
    so the sandbox can consume the per-rank torch trace files that the
    pod-side sglang/vllm server emits. Anchoring this on
    ``workspace_root()`` (= ``$USER_DATA_PATH``, with the legacy
    ``DEFAULT_SESSION_DIR`` fallback) lets the operator pin a single env
    knob — typically a cluster-shared filesystem path like
    ``/wekafs/<tenant>/sessions`` — and the whole multi-node profile
    pipeline follows. Layout::

        <workspace_root>/profile-traces/<rayjob_id>/torch_trace/

    The ``<rayjob_id>`` subdir partitions traces across RayJob
    provisions (OOM recreate, SaFE reschedule) so stale traces never
    poison newer rounds; ``cli._gc_old_profile_traces`` sweeps sibling
    RayJob dirs older than 7 days.

    Operator caveat: when ``$USER_DATA_PATH`` is left unset the default
    is the pod-local ``DEFAULT_SESSION_DIR`` (``/workspace/hyperloom``),
    which is INVISIBLE to the sandbox in multi-node mode — the sandbox
    will then never see the trace files. Multi-node operators MUST set
    ``$USER_DATA_PATH`` to a cluster-shared path before launching.

    Single-node mode never reads this — server traces stay under
    ``<benchmark_workspace>/torch_trace`` inside the per-round workspace
    dir (see ``ProfileExecutor._resolve_mn_round_trace_root``).
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
