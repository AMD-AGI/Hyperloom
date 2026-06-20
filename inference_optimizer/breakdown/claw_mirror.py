"""Mirror session artifacts into claw-synced storage.

Hyperloom's canonical write lands in ``session_dir`` which, under the
default ``per_model_ts`` layout, is ``<USER_DATA_PATH>/<model>/<ts>/``.
In a Primus-Claw sandbox the brain only uploads ``/workspace`` to object
storage (``syncWorkspaceToS3`` walks ``/workspace`` and maps a file at
``/workspace/<rel>`` to the S3 key
``users/<user_id>/sessions/<claw_session_id>/<rel>``), and the session
collector reads the breakdown straight from the deterministic ROOT key
``users/<uid>/sessions/<csid>/{manifest,session_breakdown}.json``.

Two best-effort mirrors keep that pipeline whole (belt-and-suspenders):

1. :func:`mirror_breakdown_to_claw_storage` — copies the breakdown into
   ``<workspace_root>/hyperloom/<session_id>/`` (the original
   ``$USER_DATA_PATH``-relative checkpoint-synced subtree). Kept for
   backward-compat with consumers that watch that location.
2. :func:`mirror_session_artifacts_to_claw_storage` — ADDITIONALLY copies
   ``manifest.json`` + ``session_breakdown.json`` to the claw S3-sync ROOT
   (``/workspace`` by default), so primus-claw uploads them to the exact
   root key the collector's cheap MinIO-first lookup reads.

Both are best-effort: the canonical weka write under ``session_dir`` is
the source of truth, and a mirror failure MUST never mask the real
``stop_reason`` (never raises).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ..paths import workspace_root

log = logging.getLogger(__name__)

#: Env var primus-claw can set to point at the dir it syncs to the S3
#: session-prefix root. Defaults to ``/workspace`` (the path the brain's
#: ``syncWorkspaceToS3`` hard-codes).
ENV_CLAW_WORKSPACE = "CLAW_WORKSPACE_PATH"
ENV_CLAW_SESSION_ID = "CLAW_SESSION_ID"
DEFAULT_CLAW_WORKSPACE = Path("/workspace")

#: Artifacts mirrored to the sync root. ``manifest.json`` is required by the
#: collector's MinIO-first locate (it keys on the manifest's
#: ``claw_session_id`` and derives the weka dir from ``manifest.session_dir``);
#: ``session_breakdown.json`` is the payload it forwards.
_MIRRORED_NAMES: tuple[str, ...] = ("manifest.json", "session_breakdown.json")


def mirror_breakdown_to_claw_storage(
    breakdown_path: Path,
    *,
    session_id: str,
) -> Path | None:
    """Copy ``breakdown_path`` to ``<workspace_root>/hyperloom/<sid>/``.

    The original ``$USER_DATA_PATH``-relative checkpoint-synced mirror.
    On success returns the absolute mirror path; on any failure (empty
    ``session_id``, missing source, unwritable destination, etc.) returns
    ``None`` and logs — never raises.
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


def claw_sync_root() -> Path | None:
    """Dir primus-claw uploads to the S3 session-prefix ROOT, or None.

    None means we're not in a claw sandbox (no ``CLAW_SESSION_ID``) or the
    configured root doesn't exist on disk -- the caller then skips the
    root mirror.
    """
    if not (os.environ.get(ENV_CLAW_SESSION_ID) or "").strip():
        return None
    raw = (os.environ.get(ENV_CLAW_WORKSPACE) or "").strip()
    root = Path(raw) if raw else DEFAULT_CLAW_WORKSPACE
    return root if root.is_dir() else None


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy ``src`` onto ``dst`` via a same-dir temp + ``os.replace`` so a
    concurrent S3 sync never sees a half-written file at the final name."""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def mirror_session_artifacts_to_claw_storage(
    session_dir: Path | str,
    *,
    sync_root: Path | None = None,
) -> list[Path]:
    """Copy the session's breakdown + manifest to the claw S3-sync ROOT.

    This is the redundant copy (in addition to
    :func:`mirror_breakdown_to_claw_storage`) that lands the artifacts at
    the deterministic root key the collector reads. See the module
    docstring for the full contract. Never raises.

    ``sync_root`` is resolved via :func:`claw_sync_root` when not given
    (the production path); tests pass it explicitly.
    """
    root = sync_root if sync_root is not None else claw_sync_root()
    if root is None:
        return []
    sd = Path(session_dir)
    written: list[Path] = []
    for name in _MIRRORED_NAMES:
        src = sd / name
        if not src.is_file():
            continue
        try:
            dst = root / name
            _copy_atomic(src, dst)
            written.append(dst)
        except Exception:  # noqa: BLE001 - mirror failure must stay non-fatal
            log.exception("claw-mirror of %s failed (non-fatal)", name)
    if written:
        log.info(
            "claw-mirrored %d artifact(s) to %s: %s",
            len(written), root, ", ".join(p.name for p in written),
        )
    return written
