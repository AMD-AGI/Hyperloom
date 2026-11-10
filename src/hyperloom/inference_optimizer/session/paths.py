# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filesystem path resolver."""

from __future__ import annotations

import errno
import json
import logging
import os
import hashlib
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hyperloom_common.shared_storage import SHARED_DIR_MODE, make_shared_dirs, ownership, writable_problems
from hyperloom_common.timeutil import utc_now_compact

log = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
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
    "reports",
    "agents/orchestration",
    "agents/critic",
    "runs/baseline",
    "runs/profile",
    "runs/integrate",
    "kernel-agent-workspace",
    "kernel-agent",  # tools/<name>.py output root (runs/<session_id>/...)
    "patches",
    "optimizer_runs",  # launcher stdout / pid logs
)

#: Files whose presence means a directory holds a session that was started: a fresh session is never created there.
SESSION_FILES: tuple[str, ...] = ("manifest.json", "state.json")

#: The application-private result archive, created by a hosted run beside the skeleton above (not part of it).
ARCHIVE_DIRNAME = "archive"
#: The archive's mode: the run's accounts (one group) share it, no one else may even read it.
ARCHIVE_MODE = 0o2770

#: Top-level session subdirectories that belong to the application alone: never copied into a sandbox, packaged or
#: shown to a specialist.  Whatever copies a session directory elsewhere must leave these out.
PRIVATE_SESSION_SUBDIRS: tuple[str, ...] = (ARCHIVE_DIRNAME,)

#: Where an account keeps the coordinator database of a session that holds a credential, below ``${TMPDIR:-/tmp}``:
#: ``hyperloom-ledger-<uid>/<session id>-<hash of the id>/``, keyed by the session's id, not by the path it is mounted at. Not in the session directory, which the
#: group can write: SQLite reaches a database, and makes its journal files, by path, so any directory on that path that
#: another account can rename entries in lets it put a database of its own there while one is opened.
PRIVATE_DB_DIR_PREFIX = "hyperloom-ledger-"
#: A directory of this account's, ``0700``, that outlives the container, named instead of the one below ``TMPDIR`` (an
#: operator prepares it: below a sticky parent, or one only this account can write in): a container's ``TMPDIR`` is its
#: own disk, so a run resumed in a new container (after a preemption, say) finds no database there.
ENV_LEDGER_DIR = "HYPERLOOM_LEDGER_DIR"

#: The workspace root and its ``runtime/``, where the installer puts what later steps source or run (GEAK litellm config,
#: source mirrors): the group may enter and read them, only their owner may change them, so no other account can swap
#: what a run uses. The session directory below the root, with its own ``runtime/`` (lock, ledger, breakdown), is
#: ``2775`` instead.
_INSTALL_OWNED_MODE = 0o2755

# Workspace-shared layout (one copy per $USER_DATA_PATH), with each entry's mode. mkdir-ed by install.sh + reused for
# every session_dir launched from this workspace.
_WORKSPACE_SKELETON: dict[str, int] = {
    "runtime": _INSTALL_OWNED_MODE,  # install-generated runtime state (GEAK litellm config, source mirrors)
    # Workspace-level KB root; it only coincides with the KB path when session_dir == workspace_root. What the accounts
    # of a run write there is a marker nothing sources.
    "runtime/recipe_kb": SHARED_DIR_MODE,
    "logs": SHARED_DIR_MODE,  # launcher stdout (workspace-shared)
}

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


def hosted_session_dir(model_path: str | os.PathLike[str], run_created_at: datetime, run_id: str) -> Path:
    """The session directory of one hosted run: ``<workspace>/<model>/<UTC creation time>-<sha256(run id)[:8]>``.

    Both parts come from the run, so a restart of the same run lands in the same directory and another run never does.
    """
    stamp = run_created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return workspace_root() / _sanitize_model_basename(model_path) / f"{stamp}-{digest}"


def archive_root(session_dir: Path) -> Path:
    """Where a hosted run archives its results: ``<session_dir>/archive``."""
    return Path(session_dir) / ARCHIVE_DIRNAME


def ensure_archive_root(session_dir: Path) -> Path:
    """Create :func:`archive_root` with mode :data:`ARCHIVE_MODE` (the session directory, when missing, as
    :func:`ensure_session_skeleton` makes it) and return it.

    A run's legs may be different accounts of one group on shared storage, so an existing directory is accepted, whoever
    made it, when this account can create and remove a file in it; it is never chmodded. Who owns it is not compared:
    in a user namespace another account's directory shows as the overflow id.

    Raises:
        PermissionError: The archive is a link or not a directory, any account outside its group can read, write or
            enter it, or this one cannot write it.
    """
    make_shared_dirs(session_dir)
    root = archive_root(session_dir)
    try:
        make_shared_dirs(root, mode=ARCHIVE_MODE)
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ELOOP):
            raise
        # what already stands there, a link included, is judged below
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"the archive {root} is not a directory (a link to one is refused too)")
    if info.st_mode & stat.S_IRWXO:
        raise PermissionError(
            f"the archive {root} is open to accounts outside its group (mode {stat.S_IMODE(info.st_mode):04o}, "
            f"not {ARCHIVE_MODE:04o})"
        )
    problems = writable_problems({"the archive": root})
    if problems:
        raise PermissionError(f"this account cannot write {problems[0]}")
    return root


def _make_dirs(path: Path, mode: int = SHARED_DIR_MODE) -> None:
    """:func:`~hyperloom_common.shared_storage.make_shared_dirs`, with a refusal that names the directory this account
    cannot create in, its owner and mode, and what it needs to be."""
    try:
        make_shared_dirs(path, mode=mode)
    except PermissionError as exc:
        failed = Path(str(exc.filename or path))
        parent = failed.parent
        raise PermissionError(
            exc.errno,
            f"cannot create {failed}: the directory {parent}{ownership(parent)} is not writable by this account; the "
            f"accounts of a run share it, so it needs group write and the setgid bit in the run's group (chmod 2775)",
        ) from exc


def ensure_session_skeleton(session_dir: Path) -> None:
    """Create ``session_dir`` and its per-session directory skeleton, each missing directory ``2775``
    (:func:`~hyperloom_common.shared_storage.make_shared_dirs`): the next leg of a hosted run may be another account of
    the group.

    Raises:
        PermissionError: This account cannot create in a directory that exists, which the message names.
    """
    for sub in _SESSION_SKELETON:
        _make_dirs(session_dir / sub)


def make_session_dir(model_name: str | os.PathLike[str] | None = None, *, target: Path | None = None) -> Path:
    """Create the session directory + per-session + workspace-shared skeletons.

    With ``target`` the session is created at exactly that path (which must lie below the workspace) instead of at a
    generated one.
    """
    ws = workspace_root()
    _make_dirs(ws, _INSTALL_OWNED_MODE)
    for sub, mode in _WORKSPACE_SKELETON.items():
        _make_dirs(ws / sub, mode)

    if target is not None:
        sd = Path(target)
        if not is_path_within(sd, ws):
            raise ValueError(f"session directory {sd} is not below the workspace {ws}")
        if any((sd / name).exists() for name in SESSION_FILES):
            raise ValueError(f"session directory {sd} already holds a session; resume it instead")
    elif model_name:
        basename = _sanitize_model_basename(model_name)
        sd = ws / basename / f"{utc_now_compact()}-{uuid4().hex[:8]}"
    else:
        sd = ws

    ensure_session_skeleton(sd)
    # Pin for downstream callers + subprocesses; the most recent call wins.
    os.environ[ENV_CURRENT_SESSION_DIR] = str(sd)
    return sd


def shared_db_path_for(session_dir: Path) -> Path:
    """Return the coordinator database path in the session directory, where every account of the group can open it."""
    return Path(session_dir) / "storage" / "coordinator.db"


def named_ledger_dir() -> Path | None:
    """The directory ``HYPERLOOM_LEDGER_DIR`` names for this account's ledgers, or ``None`` when it is not set.

    Raises:
        ValueError: When it is not an absolute path: a relative one names a different ledger from each working directory.
    """
    named = os.environ.get(ENV_LEDGER_DIR, "").strip()
    if named and not os.path.isabs(named):
        raise ValueError(f"{ENV_LEDGER_DIR} must be an absolute path, not {named!r}")
    return Path(named) if named else None


def _recorded_session_id(session_dir: Path) -> str:
    """The id ``manifest.json`` records for the session, else the one ``state.json`` does; ``""`` when neither is
    readable (a session not yet started, or one another account's pins keep owner-only)."""
    for name in SESSION_FILES:
        try:
            recorded = json.loads((Path(session_dir) / name).read_text(encoding="utf-8")).get("session_id")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(recorded, str) and recorded.strip():
            return recorded.strip()
    return ""


def session_ledger_key(session_dir: Path) -> str:
    """The name of the session's directory among an account's ledgers: its recorded session id (the id ``manifest.json``
    gives it when it starts, so the same for every leg and for every path the session directory is mounted at), else the
    name of the session directory, which a mount of it at another path keeps too. Neither is the path."""
    ident = _recorded_session_id(session_dir) or Path(os.path.realpath(session_dir)).name or "session"
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', ident)[:40]}-{hashlib.sha256(ident.encode()).hexdigest()[:16]}"


def private_db_path_for(session_dir: Path) -> Path:
    """Return the path of this account's own coordinator database for a session, outside the session directory (see
    :data:`PRIVATE_DB_DIR_PREFIX`, :data:`ENV_LEDGER_DIR` and :func:`session_ledger_key`); nothing is made there.

    Raises:
        ValueError: When ``HYPERLOOM_LEDGER_DIR`` is not an absolute path.
    """
    ledgers = named_ledger_dir() or Path(os.environ.get("TMPDIR") or "/tmp") / f"{PRIVATE_DB_DIR_PREFIX}{os.geteuid()}"
    return ledgers / session_ledger_key(session_dir) / "coordinator.db"


def private_db_exists(session_dir: Path) -> bool:
    """Whether this account has made its own coordinator database for a session: a regular file of this account's, a
    link never followed at the end. An entry another account planted there (a link, a file of its own) is not one."""
    try:
        info = os.lstat(private_db_path_for(session_dir))
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()


def db_path_for(session_dir: Path) -> Path:
    """Return the coordinator database this account uses for a session: its own outside the session directory once it
    has made one, else the one in the session directory."""
    return private_db_path_for(session_dir) if private_db_exists(session_dir) else shared_db_path_for(session_dir)


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
    return PACKAGE_ROOT.parent / "orchestrator" / "prompts"


def asset_prompt_references_dir() -> Path:
    """Return the directory of shipped on-demand prompt reference documents."""
    return asset_system_prompts_dir() / "references"


# Workspace-/session-scoped artefact helpers.
def runtime_dir() -> Path:
    """``<workspace_root>/runtime/`` — workspace-shared writable runtime (GEAK litellm config, source mirrors)."""
    return workspace_root() / "runtime"


def _installed_repo_root() -> Path:
    """The root install.sh's ``resolve_repo_root`` picks when ``$REPO_ROOT`` is unset.

    A source checkout is the directory holding ``pyproject.toml`` above ``src/``; a packaged install (the wheel in
    site-packages) is the directory holding the ``hyperloom`` package. Deriving it from this file, not the working
    directory, makes a process started elsewhere (a hosted run's MB entry) find the checkouts install.sh made.
    """
    package_parent = Path(__file__).resolve().parents[3]
    source_root = package_parent.parent
    return source_root if (source_root / "pyproject.toml").is_file() else package_parent


def deps_cache_root() -> Path:
    """Writable cache root for open-source dependency checkouts: install.sh's ``$REPO_ROOT/.cache``."""
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override)
    repo_root = os.environ.get(ENV_REPO_ROOT)
    return (Path(repo_root) if repo_root else _installed_repo_root()) / ".cache"


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
    """``<workspace_root>/profile-traces/`` — multi-node torch profile shared root.

    A profile run without ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` writes its traces to ``default-<pid>/torch_trace/`` below it.
    """
    return workspace_root() / "profile-traces"


__all__ = [
    "ARCHIVE_DIRNAME",
    "ARCHIVE_MODE",
    "AssetRootNotFound",
    "DEFAULT_SESSION_DIR",
    "ENV_CURRENT_SESSION_DIR",
    "ENV_LEDGER_DIR",
    "ENV_OVERRIDE_ASSET_ROOT",
    "ENV_USER_DATA_PATH",
    "PACKAGE_ROOT",
    "PRIVATE_DB_DIR_PREFIX",
    "PRIVATE_SESSION_SUBDIRS",
    "SESSION_FILES",
    "asset_prompt_references_dir",
    "asset_root",
    "archive_root",
    "asset_system_prompts_dir",
    "db_path_for",
    "ensure_archive_root",
    "ensure_session_skeleton",
    "hosted_session_dir",
    "is_path_within",
    "magpie_dir",
    "make_session_dir",
    "mn_profile_trace_root",
    "deps_cache_root",
    "named_ledger_dir",
    "private_db_exists",
    "private_db_path_for",
    "resolve_dep_dir",
    "runtime_dir",
    "session_dir",
    "session_ledger_key",
    "shared_db_path_for",
    "workspace_root",
]


def stop_request_path(session_dir: Path) -> Path:
    """The file whose presence is an operator's request to stop the run: ``<session>/hosted/control/stop.json``."""
    return Path(session_dir) / "hosted" / "control" / "stop.json"
