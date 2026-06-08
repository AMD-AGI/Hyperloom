# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mirror ``session_breakdown.json`` into the claw-synced subtree.

Hyperloom's canonical write lands in ``<session_dir>/`` which under the
default ``per_model_ts`` layout is ``<workspace>/<model>/<ts>/``. The
claw sandbox checkpoint sync only watches the ``hyperloom/`` subtree
under ``$USER_DATA_PATH``, so without an explicit mirror the canonical
artifact is lost the moment the sandbox is reaped.

This module exposes a single best-effort helper, kept off ``cli.py`` so
the unit tests can exercise it without dragging in cli's full import
graph (``orchestrator/action_executors`` pulls ``fcntl`` and friends
that aren't available on every dev box).

Contract:

* On success, returns the absolute mirror path.
* On any failure (empty ``session_id``, missing source, unwritable
  destination, etc.) returns ``None`` and logs — the caller MUST treat
  the canonical write as the source of truth and never let a mirror
  failure mask the real ``stop_reason``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..paths import workspace_root

log = logging.getLogger(__name__)


def mirror_breakdown_to_claw_storage(
    breakdown_path: Path,
    *,
    session_id: str,
) -> Path | None:
    """Copy ``breakdown_path`` to ``<workspace_root>/hyperloom/<sid>/``.

    See module docstring for the full contract. Never raises.

    Args:
        breakdown_path (Path): Absolute path to the canonical
            ``session_breakdown.json`` to mirror.
        session_id (str): The session id used as the mirror subdirectory name.

    Returns:
        Path | None: The absolute mirror path on success, or ``None`` if the
            session id is empty or any copy step fails.
    """
    sid = (session_id or "").strip()
    if not sid:
        log.warning("session_breakdown claw-mirror skipped: empty session_id")
        return None
    try:
        mirror_dir = workspace_root() / "hyperloom" / sid
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_path = mirror_dir / breakdown_path.name
        shutil.copy2(breakdown_path, mirror_path)
        return mirror_path
    except Exception:  # noqa: BLE001
        log.exception("session_breakdown claw-mirror failed (non-fatal)")
        return None
