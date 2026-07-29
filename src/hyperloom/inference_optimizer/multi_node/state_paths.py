"""Resolve and bind session-scoped multi-node state file paths."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ..session.paths import ENV_CURRENT_SESSION_DIR

from ._internal.log import warn

_RUNTIME_REL = Path("runtime") / "multi_node_state.json"


def resolve_state_file() -> Path:
    """Return the multi-node CLI state file path for the current process.

    Resolution order:
    1. ``$MULTI_NODE_STATE_FILE`` when set.
    2. ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/runtime/multi_node_state.json``.

    Returns:
        Path: The resolved state file location.

    Raises:
        RuntimeError: When neither ``$MULTI_NODE_STATE_FILE`` nor
            ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` is set.
    """
    explicit = os.environ.get("MULTI_NODE_STATE_FILE", "").strip()
    if explicit:
        return Path(explicit)
    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR, "").strip()
    if pinned:
        return Path(pinned) / _RUNTIME_REL
    raise RuntimeError(
        "cannot resolve multi-node state file: set $MULTI_NODE_STATE_FILE explicitly "
        "or call bind_state_file_to_session() to pin it under the active session dir"
    )


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
    """Pin ``$MULTI_NODE_STATE_FILE`` under ``session_dir``.

    Args:
        session_dir: Active optimizer session directory.

    Returns:
        Path: The bound session-scoped state file path.
    """
    target = session_dir / _RUNTIME_REL
    _ensure_runtime_dir(target.parent)
    os.environ["MULTI_NODE_STATE_FILE"] = str(target)
    return target
