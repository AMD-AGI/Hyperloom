"""Resolve and bind session-scoped multi-node state file paths."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from inference_optimizer.paths import ENV_CURRENT_SESSION_DIR

from ._internal.log import warn

_LEGACY_STATE_FILE = Path("/tmp/multi_node_state.json")
_RUNTIME_REL = Path("runtime") / "multi_node_state.json"


def resolve_state_file() -> Path:
    """Return the multi-node CLI state file path for the current process.

    Resolution order:
    1. ``$MULTI_NODE_STATE_FILE`` when set.
    2. ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/runtime/multi_node_state.json``.
    3. Legacy ``/tmp/multi_node_state.json`` (standalone CLI / migration).

    Returns:
        Path: The resolved state file location.
    """
    explicit = os.environ.get("MULTI_NODE_STATE_FILE", "").strip()
    if explicit:
        return Path(explicit)
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR, "").strip()
    if pinned:
        return Path(pinned) / _RUNTIME_REL
    return _LEGACY_STATE_FILE


def legacy_state_file() -> Path:
    """Return the legacy default state file path.

    Returns:
        Path: ``/tmp/multi_node_state.json``.
    """
    return _LEGACY_STATE_FILE


def state_file_safe_to_read(path: Path) -> bool:
    """Return True when ``path`` is owned by the current uid and not group/world-writable.

    Args:
        path: Candidate state file path.

    Returns:
        bool: True when the file passes ownership and permission checks.
    """
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_uid != os.getuid():
        return False
    if st.st_mode & stat.S_IWGRP or st.st_mode & stat.S_IWOTH:
        return False
    return True


def _chmod_state_file(path: Path) -> None:
    """Restrict state file permissions to owner read/write.

    Args:
        path: State file to chmod.
    """
    try:
        path.chmod(0o600)
    except OSError as exc:
        warn(f"could not chmod state file {path} to 0600: {exc}")


def _ensure_runtime_dir(runtime_dir: Path) -> None:
    """Create the runtime directory with owner-only access.

    Args:
        runtime_dir: Parent directory for the state file and SSH keys.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_dir.chmod(0o700)
    except OSError as exc:
        warn(f"could not chmod runtime dir {runtime_dir} to 0700: {exc}")


def bind_state_file_to_session(session_dir: Path) -> Path:
    """Pin ``$MULTI_NODE_STATE_FILE`` under ``session_dir`` and migrate legacy state.

    When the session-scoped file is absent, copies a safe legacy
    ``/tmp/multi_node_state.json`` (or an explicit ``$MULTI_NODE_STATE_FILE``)
    into the session runtime directory.

    Args:
        session_dir: Active optimizer session directory.

    Returns:
        Path: The bound session-scoped state file path.
    """
    target = session_dir / _RUNTIME_REL
    _ensure_runtime_dir(target.parent)

    if not target.is_file():
        candidates: list[Path] = []
        env_path = os.environ.get("MULTI_NODE_STATE_FILE", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        if legacy_state_file() not in candidates:
            candidates.append(legacy_state_file())
        for src in candidates:
            if src == target or not src.is_file():
                continue
            if not state_file_safe_to_read(src):
                warn(f"skipping unsafe multi-node state migration from {src}")
                continue
            try:
                shutil.copy2(src, target)
                _chmod_state_file(target)
                warn(f"migrated multi-node state {src} -> {target}")
                break
            except OSError as exc:
                warn(f"could not migrate multi-node state {src} -> {target}: {exc}")

    os.environ["MULTI_NODE_STATE_FILE"] = str(target)
    return target
