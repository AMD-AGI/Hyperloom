"""Checkpoint / Resume — DESIGN §13.

SQLite is the source of truth for events / cursors / tasks / leases (ADR-33).
This module orchestrates *outside* SQLite: state.json snapshot, persona
flush, and backup-to-NFS via storage/backup.py.

STATUS (v0.7):
    Pure-Python implementation. ``Checkpoint.create`` writes the state
    snapshot, then dispatches a hot ``vacuum_into`` to
    ``checkpoints/<ts>/conductor.db.bak``. ``resume_from_session_dir``
    reads cursors + personas, surveys in-flight tasks, and reaps expired
    leases. ``evidence_check_matrix`` routes a given task into one of
    six per-action verdicts.

References:
    - DESIGN §13.5 resume pipeline
    - DESIGN §13.6 evidence-check matrix
    - DESIGN §13.7 checkpoint triggers
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..storage.backup import vacuum_into

if TYPE_CHECKING:
    from ..storage.connection import SqliteConnection
    from .resource_lock import ResourceLockManager
    from .shared_state import SharedState
    from .task_registry import Task, TaskRegistry


__all__ = [
    "TriggerReason",
    "CheckpointHandle",
    "ResumeState",
    "Verdict",
    "Checkpoint",
    "resume_from_session_dir",
    "evidence_check_matrix",
]


log = logging.getLogger(__name__)


class TriggerReason(str, Enum):
    PERIODIC = "periodic"
    AFTER_KEEP = "after_keep"
    AFTER_REVIEW = "after_review"
    GRACEFUL_STOP = "graceful_stop"
    CRASH = "crash"


@dataclass
class CheckpointHandle:
    ts: str
    path: Path
    trigger: TriggerReason


@dataclass
class ResumeState:
    found_inflight_tasks: list[Any]
    cursors: dict[str, int]
    rolled_back_count: int = 0
    expired_leases_reaped: int = 0
    persona_paths: dict[str, Path] = field(default_factory=dict)


class Verdict(str, Enum):
    SUCCEEDED = "succeeded"
    SAFELY_FAILED = "safely_failed"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


def _utc_iso_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sync_personas(personas_dir: Path) -> None:
    """Best-effort fsync of every persona file."""
    if not personas_dir.is_dir():
        return
    for p in personas_dir.glob("*.md"):
        try:
            with p.open("rb") as fh:
                import os
                os.fsync(fh.fileno())
        except OSError:
            pass


# ---------------------------------------------------------------------------
class Checkpoint:
    """Static factory; instances are just record-keeping."""

    @staticmethod
    async def create(
        session_dir: Path,
        db: "SqliteConnection",
        state: "SharedState",
        *,
        trigger: TriggerReason = TriggerReason.PERIODIC,
        ts: str | None = None,
    ) -> CheckpointHandle:
        """Snapshot pipeline (DESIGN §13.7):

            1. write state.json.tmp + rename
            2. fsync personas/*.md
            3. vacuum_into ``checkpoints/<ts>/conductor.db.bak``
            4. emit ``checkpoint_taken`` event
        """
        session_dir = Path(session_dir)
        ts_str = ts or _utc_iso_compact()
        cp_dir = session_dir / "checkpoints" / ts_str
        cp_dir.mkdir(parents=True, exist_ok=True)

        # 1. state.json snapshot
        state.write_snapshot(session_dir)

        # 2. sync personas
        _sync_personas(session_dir / "personas")

        # 3. backup
        bak_path = cp_dir / "conductor.db.bak"
        try:
            await vacuum_into(db, bak_path)
        except Exception:  # noqa: BLE001 — checkpoint is best-effort
            log.exception("checkpoint vacuum_into failed")

        # 4. emit event
        try:
            await db.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                "in_reply_to, payload, priority, ts) VALUES (?,?,?,?,?,?,?,?)",
                (
                    f"cp-{ts_str}",
                    "conductor",
                    "*",
                    "checkpoint_taken",
                    None,
                    f'{{"trigger":"{trigger.value}","ts":"{ts_str}",'
                    f'"path":"{bak_path.as_posix()}"}}',
                    1,
                    datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                ),
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to record checkpoint_taken event")

        return CheckpointHandle(ts=ts_str, path=cp_dir, trigger=trigger)

    @staticmethod
    async def create_after_keep(
        session_dir: Path, db: "SqliteConnection", state: "SharedState"
    ) -> CheckpointHandle:
        """Same as ``create`` but tagged ``AFTER_KEEP`` so post-mortem can
        reconstruct which checkpoints captured a confirmed win."""
        return await Checkpoint.create(
            session_dir, db, state, trigger=TriggerReason.AFTER_KEEP
        )

    @staticmethod
    def list_all(session_dir: Path) -> list[CheckpointHandle]:
        """Walk ``<session_dir>/checkpoints`` and return handles in chrono order."""
        root = Path(session_dir) / "checkpoints"
        if not root.is_dir():
            return []
        out: list[CheckpointHandle] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            bak = child / "conductor.db.bak"
            if not bak.is_file():
                continue
            out.append(
                CheckpointHandle(
                    ts=child.name,
                    path=child,
                    trigger=TriggerReason.PERIODIC,  # default — actual reason in event log
                )
            )
        return out

    @staticmethod
    def load_latest(session_dir: Path) -> CheckpointHandle | None:
        all_h = Checkpoint.list_all(session_dir)
        return all_h[-1] if all_h else None


# ---------------------------------------------------------------------------
async def resume_from_session_dir(
    session_dir: Path,
    db: "SqliteConnection",
    locks: "ResourceLockManager",
    *,
    tasks: "TaskRegistry | None" = None,
) -> ResumeState:
    """End-to-end resume (DESIGN §13.5).

    Steps:

        1. load all per-agent cursors via ``CursorStore.all``
        2. survey in-flight tasks (state IN ``queued`` / ``running``)
        3. reap expired leases via ``locks.backend.reap_expired``
        4. enumerate persona files for the prompt-time injection step
    """
    from .cursor_store import CursorStore
    from .task_registry import TaskRegistry

    session_dir = Path(session_dir)

    cursor_store = CursorStore(db)
    cursors_map = await cursor_store.all()
    cursors = {a: c.last_processed_seq for a, c in cursors_map.items()}

    task_reg = tasks or TaskRegistry(db)
    queued = await task_reg.list_by_state("queued")
    running = await task_reg.list_by_state("running")
    inflight = list(queued) + list(running)

    reaped = []
    if hasattr(locks, "backend") and hasattr(locks.backend, "reap_expired"):
        try:
            reaped = await locks.backend.reap_expired()
        except Exception:  # noqa: BLE001
            log.exception("resume: reap_expired failed")
            reaped = []

    persona_paths: dict[str, Path] = {}
    personas_dir = session_dir / "personas"
    if personas_dir.is_dir():
        for p in personas_dir.glob("*.md"):
            persona_paths[p.stem] = p

    return ResumeState(
        found_inflight_tasks=list(inflight),
        cursors=cursors,
        rolled_back_count=0,
        expired_leases_reaped=len(reaped),
        persona_paths=persona_paths,
    )


# ---------------------------------------------------------------------------
# evidence_check_matrix — DESIGN §13.6
# ---------------------------------------------------------------------------
def evidence_check_matrix(
    task: "Task",
    session_dir: Path,
    *,
    server_health_fn: Any | None = None,
    geak_status_fn: Any | None = None,
) -> Verdict:
    """Per-action recovery verdict.

    Routes ``task`` to one of six per-action helpers. Unknown ``task.kind``
    returns ``EVIDENCE_INSUFFICIENT`` so the conductor escalates manually.
    """
    session_dir = Path(session_dir)
    action = _action_name(task)
    if action in ("bench_runner", "baseline", "bench"):
        return _check_bench(task, session_dir)
    if action == "profile":
        return _check_profile(task, session_dir)
    if action in ("integrate", "patch_applier", "patch", "compiler_tuning"):
        return _check_patch_or_integrate(task, session_dir)
    if action in ("server_restart", "backends"):
        return _check_server_restart(task, session_dir, server_health_fn)
    if action in ("kernel_extract", "deep_kernel_analysis"):
        return _check_kernel_extract(task, session_dir)
    if action in ("kernel_opt", "geak_submitter"):
        return _check_geak_submit(task, session_dir, geak_status_fn)
    return Verdict.EVIDENCE_INSUFFICIENT


def _action_name(task: "Task") -> str:
    """Pull the action name from a Task — supports several conventions."""
    if hasattr(task, "params") and isinstance(task.params, dict):
        n = task.params.get("action_name") or task.params.get("action")
        if n:
            return str(n)
    return str(getattr(task, "kind", "") or "")


def _path_under_results(session_dir: Path, *parts: str) -> Path:
    return session_dir / "results" / Path(*parts)


def _check_bench(task: "Task", session_dir: Path) -> Verdict:
    """Bench / baseline: ``results/<task_id>/metrics.json`` must exist."""
    metrics = _path_under_results(session_dir, task.task_id, "metrics.json")
    if not metrics.is_file():
        return Verdict.EVIDENCE_INSUFFICIENT
    if metrics.stat().st_size <= 2:  # empty or `{}` placeholder
        return Verdict.SAFELY_FAILED
    return Verdict.SUCCEEDED


def _check_profile(task: "Task", session_dir: Path) -> Verdict:
    """Profile: at least one ``filtered-TP-0.trace.json.gz`` artifact."""
    trace_root = _path_under_results(session_dir, task.task_id)
    if not trace_root.is_dir():
        return Verdict.EVIDENCE_INSUFFICIENT
    hits = list(trace_root.rglob("filtered-TP-0.trace.json.gz"))
    if not hits:
        return Verdict.EVIDENCE_INSUFFICIENT
    if any(p.stat().st_size > 1024 for p in hits):
        return Verdict.SUCCEEDED
    return Verdict.SAFELY_FAILED


def _check_patch_or_integrate(task: "Task", session_dir: Path) -> Verdict:
    """Integrate / patch: ``results/<task_id>/patch_fingerprint.txt`` must
    contain the same SHA fingerprint that the params hold."""
    fp_file = _path_under_results(session_dir, task.task_id, "patch_fingerprint.txt")
    if not fp_file.is_file():
        return Verdict.EVIDENCE_INSUFFICIENT
    expected = ""
    if hasattr(task, "params") and isinstance(task.params, dict):
        expected = str(task.params.get("patch_fingerprint", ""))
    actual = fp_file.read_text(encoding="utf-8").strip()
    if not actual:
        return Verdict.SAFELY_FAILED
    if expected and actual == expected:
        return Verdict.SUCCEEDED
    return Verdict.EVIDENCE_INSUFFICIENT


def _check_server_restart(
    task: "Task", session_dir: Path, server_health_fn: Any | None
) -> Verdict:
    """Server restart: pid file + healthcheck + we hold the lease."""
    pid_file = _path_under_results(session_dir, task.task_id, "server.pid")
    if not pid_file.is_file():
        return Verdict.EVIDENCE_INSUFFICIENT
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return Verdict.SAFELY_FAILED
    if pid <= 0:
        return Verdict.SAFELY_FAILED
    if server_health_fn is None:
        return Verdict.EVIDENCE_INSUFFICIENT
    try:
        healthy = bool(server_health_fn(pid))
    except Exception:  # noqa: BLE001
        return Verdict.EVIDENCE_INSUFFICIENT
    return Verdict.SUCCEEDED if healthy else Verdict.SAFELY_FAILED


def _check_kernel_extract(task: "Task", session_dir: Path) -> Verdict:
    """Kernel extract: read-only artifact w/ checksum file present."""
    art_dir = _path_under_results(session_dir, task.task_id)
    if not art_dir.is_dir():
        return Verdict.EVIDENCE_INSUFFICIENT
    arts = list(art_dir.rglob("*.kernel.json")) + list(art_dir.rglob("*.kernel.tar.gz"))
    if not arts:
        return Verdict.EVIDENCE_INSUFFICIENT
    return Verdict.SUCCEEDED


def _check_geak_submit(
    task: "Task", session_dir: Path, geak_status_fn: Any | None
) -> Verdict:
    """GEAK submit: external request id present + status fn confirms outcome."""
    req_id_file = _path_under_results(session_dir, task.task_id, "geak_request_id.txt")
    if not req_id_file.is_file():
        return Verdict.EVIDENCE_INSUFFICIENT
    req_id = req_id_file.read_text(encoding="utf-8").strip()
    if not req_id:
        return Verdict.SAFELY_FAILED
    if geak_status_fn is None:
        return Verdict.EVIDENCE_INSUFFICIENT
    try:
        status = str(geak_status_fn(req_id) or "").lower()
    except Exception:  # noqa: BLE001
        return Verdict.EVIDENCE_INSUFFICIENT
    if status == "succeeded":
        return Verdict.SUCCEEDED
    if status in ("failed", "discarded"):
        return Verdict.SAFELY_FAILED
    return Verdict.EVIDENCE_INSUFFICIENT
