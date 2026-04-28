"""SQLite hot-backup helpers — DESIGN §3.5.8 path A.

DB lives on sandbox-local disk (path A). Periodic ``VACUUM INTO`` drops a
self-consistent backup onto Shared NFS so a fresh sandbox can resume.

STATUS (v0.7):
    Pure-Python implementation. ``vacuum_into`` issues the SQL command
    over the existing :class:`SqliteConnection`. ``periodic_backup`` is a
    cancellable asyncio task; ``restore_from_backup`` is a synchronous
    file copy.

References:
    - DESIGN §3.5.8 deployment path A
    - DESIGN §3.5.4 / §13 atomic semantics
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from .connection import SqliteConnection


__all__ = [
    "DEFAULT_PERIOD_MIN",
    "vacuum_into",
    "periodic_backup",
    "force_backup_after_keep",
    "restore_from_backup",
]


DEFAULT_PERIOD_MIN: int = 30
log = logging.getLogger(__name__)


def _utc_iso_compact() -> str:
    """``20260427T193000Z`` — filesystem-safe UTC stamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
async def vacuum_into(
    db: "SqliteConnection", dest_path: Path
) -> Path:
    """Run ``VACUUM INTO ?`` and return the backup path.

    The backup file is fully self-contained — opening it as a fresh
    SQLite database recovers the exact state at the time of the call.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        # ``VACUUM INTO`` refuses to overwrite, so we move the previous
        # file out of the way first. Failure here is non-fatal — the
        # fresh backup will replace it on the rename anyway.
        try:
            dest_path.unlink()
        except OSError:
            pass
    sql = "VACUUM INTO ?"

    def _do_vacuum() -> None:
        with db._sync_lock:  # type: ignore[attr-defined]
            db._conn.execute(sql, (str(dest_path),))  # type: ignore[attr-defined]
            db._conn.commit()  # type: ignore[attr-defined]

    await asyncio.to_thread(_do_vacuum)
    return dest_path


async def periodic_backup(
    db: "SqliteConnection",
    checkpoint_dir: Path,
    *,
    period_min: int = DEFAULT_PERIOD_MIN,
    stop_event: asyncio.Event | None = None,
    on_complete: Callable[[Path], Awaitable[None]] | None = None,
) -> None:
    """Long-running task launched by Conductor.run().

    Every ``period_min`` minutes (or sooner if ``stop_event`` is set), runs
    ``vacuum_into`` and writes the backup under
    ``<checkpoint_dir>/<ts>/conductor.db.bak``. ``on_complete`` is awaited
    after every successful backup, with the path of the new file.

    The loop tolerates failures — a single failed backup logs a warning
    and continues.
    """
    period_s = max(1.0, float(period_min) * 60.0)
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            ts = _utc_iso_compact()
            dest = Path(checkpoint_dir) / ts / "conductor.db.bak"
            await vacuum_into(db, dest)
            if on_complete is not None:
                try:
                    await on_complete(dest)
                except Exception:  # noqa: BLE001
                    log.exception("periodic_backup: on_complete handler raised")
            log.info("periodic_backup wrote %s", dest)
        except Exception:  # noqa: BLE001
            log.exception("periodic_backup failed")
        try:
            if stop_event is not None:
                await asyncio.wait_for(stop_event.wait(), timeout=period_s)
                return
            await asyncio.sleep(period_s)
        except asyncio.TimeoutError:
            continue


async def force_backup_after_keep(
    db: "SqliteConnection", checkpoint_dir: Path
) -> Path:
    """Synchronous (await-able) backup; called from Conductor on KEEP.

    Returns the path of the new ``conductor.db.bak``.
    """
    ts = _utc_iso_compact()
    dest = Path(checkpoint_dir) / ts / "conductor.db.bak"
    await vacuum_into(db, dest)
    return dest


def restore_from_backup(backup_path: Path, target_db: Path) -> None:
    """Copy ``backup_path`` over ``target_db``.

    Used when a fresh sandbox starts and finds the local DB missing /
    corrupt. Caller MUST not have an open ``SqliteConnection`` to
    ``target_db`` when invoking this.
    """
    backup_path = Path(backup_path)
    target_db = Path(target_db)
    if not backup_path.is_file():
        raise FileNotFoundError(f"backup not found: {backup_path}")
    target_db.parent.mkdir(parents=True, exist_ok=True)
    # also wipe any straggling WAL / SHM files so we don't mix state
    for sidecar in (
        target_db.with_suffix(target_db.suffix + "-wal"),
        target_db.with_suffix(target_db.suffix + "-shm"),
    ):
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("could not remove sidecar %s", sidecar)
    shutil.copy2(backup_path, target_db)
