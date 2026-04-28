"""IPC — all 6 artifacts (work_queue, results, merge_ready, event_log, findings, rca_reports).

Every function handles file-not-exists and empty-file gracefully.
All writes are atomic (tmp + rename) to avoid partial reads.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_append(path: Path, entry: dict[str, Any]) -> None:
    """Append a single JSON line atomically with file-level locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=str) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _log.warning("Failed to read %s: %s", path, exc)
        return []
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _read_after(path: Path, after_id: str) -> list[dict[str, Any]]:
    """Read entries from a JSONL file after the entry with `after_id`.

    If after_id is empty, returns all entries.  If after_id is not found
    (e.g. file was rotated or IDs were reset), returns all entries so the
    caller re-processes rather than silently stalling forever.
    """
    entries = _read_jsonl(path)
    if not after_id:
        return entries
    for i, e in enumerate(entries):
        if e.get("id") == after_id:
            return entries[i + 1:]
    _log.warning(
        "IPC cursor desync: after_id=%s not found in %s (%d entries), returning all",
        after_id, path.name, len(entries))
    return entries


def _km_dir(session_dir: str | Path) -> Path:
    return Path(session_dir) / "kernel_manager"


def _safe_path_id(event_id: str) -> str:
    """Replace characters illegal on Windows (: and others) so event IDs are safe as directory names."""
    return event_id.replace(":", "-")


# -----------------------------------------------------------------------
# work_queue.jsonl  (W: orchestrator  |  R: kernel-manager, watchdog)
# -----------------------------------------------------------------------

def write_work_queue_entry(session_dir: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Append a target to work_queue.jsonl with write-side dedup.

    If an entry with the same non-empty ID already exists (any status),
    the write is skipped and the existing entry is returned instead.
    """
    entry.setdefault("id", f"wq_{uuid.uuid4().hex[:12]}")
    entry.setdefault("timestamp", _now_iso())
    entry.setdefault("status", "pending")
    entry.setdefault("attempts", 0)

    entry_id = entry.get("id", "")
    if entry_id:
        wq_path = _km_dir(session_dir) / "work_queue.jsonl"
        existing = None
        for e in _read_jsonl(wq_path):
            if e.get("id") == entry_id:
                existing = e
        if existing is not None:
            _log.debug("WQ dedup: skipping duplicate id=%s", entry_id)
            return existing

    _atomic_append(_km_dir(session_dir) / "work_queue.jsonl", entry)
    return entry


def read_work_queue(session_dir: str, *, after_id: str = "") -> list[dict[str, Any]]:
    return _read_after(_km_dir(session_dir) / "work_queue.jsonl", after_id)


def read_work_queue_entry(session_dir: str, entry_id: str) -> dict[str, Any] | None:
    for e in _read_jsonl(_km_dir(session_dir) / "work_queue.jsonl"):
        if e.get("id") == entry_id or e.get("task_id") == entry_id:
            return e
    return None


def read_work_queue_all(session_dir: str) -> list[dict[str, Any]]:
    return _read_jsonl(_km_dir(session_dir) / "work_queue.jsonl")


def compact_work_queue(session_dir: str, processed_ids: set[str] | None = None) -> int:
    """Compact work_queue.jsonl: deduplicate by ID, drop stale terminal entries.

    Holds an exclusive lock on the WQ file for the entire read-modify-write
    to prevent concurrent appends from being lost during the rewrite.

    - Keeps only the LAST entry per unique ID (so status updates win).
    - Removes entries whose status is terminal (completed/APPLIED/failed) AND
      whose ID is in *processed_ids* (already consumed by the KM).
    - Returns the number of entries removed.
    """
    wq_path = _km_dir(session_dir) / "work_queue.jsonl"
    if not wq_path.exists():
        return 0

    processed = processed_ids or set()

    # Hold exclusive lock for the entire read-modify-write cycle
    with open(wq_path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            entries: list[dict[str, Any]] = []
            for line in fh.read().splitlines():
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not entries:
                return 0

            by_id: dict[str, dict[str, Any]] = {}
            no_id: list[dict[str, Any]] = []
            for e in entries:
                eid = e.get("id", "")
                if eid:
                    by_id[eid] = e
                else:
                    no_id.append(e)

            TERMINAL = {"completed", "APPLIED", "failed", "exhausted", "cancelled"}
            kept = []
            for eid, e in by_id.items():
                if e.get("status") in TERMINAL and eid in processed:
                    continue
                kept.append(e)
            kept.extend(no_id)

            removed = len(entries) - len(kept)
            if removed > 0:
                fh.seek(0)
                fh.truncate()
                for e in kept:
                    fh.write(json.dumps(e, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                _log.info("WQ compacted: %d → %d entries (removed %d)",
                          len(entries), len(kept), removed)
            return removed
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# -----------------------------------------------------------------------
# results.jsonl  (W: kernel-manager  |  R: orchestrator)
# 12 fields: id, status, strategy_used, backend_used, micro_speedup,
#   patch_dir, patch_type, rebuild_required, rollback_command,
#   verification_command, error_message, timestamp
# -----------------------------------------------------------------------

def write_result(session_dir: str, result: dict[str, Any]) -> dict[str, Any]:
    result.setdefault("timestamp", _now_iso())
    _atomic_append(_km_dir(session_dir) / "results.jsonl", result)
    return result


def read_new_results(session_dir: str, *, after_id: str = "") -> list[dict[str, Any]]:
    """Read results newer than *after_id* (keyed on result['id'])."""
    return _read_after(_km_dir(session_dir) / "results.jsonl", after_id)


# -----------------------------------------------------------------------
# merge_ready/<task_id>/  (W: kernel-manager  |  R: orchestrator)
# -----------------------------------------------------------------------

def write_merge_ready(session_dir: str, task_id: str, metadata: dict[str, Any],
                      artifacts: dict[str, str | bytes] | None = None) -> Path:
    """Create merge_ready/<task_id>/ with metadata.json + artifact files.

    Writes all files first, then drops a `.ready` marker so readers never
    see a half-written merge package.
    """
    staging = _km_dir(session_dir) / "merge_ready" / f".{task_id}.staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    for name, content in (artifacts or {}).items():
        if isinstance(content, bytes):
            (staging / name).write_bytes(content)
        else:
            (staging / name).write_text(content)
    final = _km_dir(session_dir) / "merge_ready" / task_id
    try:
        staging.rename(final)
    except OSError:
        import shutil
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    (final / ".ready").write_text("")
    return final


def read_merge_ready_metadata(session_dir: str, task_id: str) -> dict[str, Any] | None:
    ready = _km_dir(session_dir) / "merge_ready" / task_id / ".ready"
    if not ready.exists():
        return None
    p = _km_dir(session_dir) / "merge_ready" / task_id / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


MERGE_READY_REQUIRED = {"task_id", "kernel_name", "patch_type", "target_file",
                        "rollback_command", "apply_instructions"}


def validate_merge_ready(session_dir: str, task_id: str) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    meta = read_merge_ready_metadata(session_dir, task_id)
    if meta is None:
        return ["metadata.json missing"]
    errors: list[str] = []
    for req in MERGE_READY_REQUIRED:
        if not meta.get(req):
            errors.append(f"missing required field: {req}")
    d = _km_dir(session_dir) / "merge_ready" / task_id
    has_artifact = any(
        f.name.startswith("optimized_") or f.name == "patch.diff"
        for f in d.iterdir() if f.is_file()
    )
    if not has_artifact:
        errors.append("no optimized_* or patch.diff artifact")
    return errors


# -----------------------------------------------------------------------
# event_log.jsonl  (W: orchestrator, kernel-manager  |  R: watchdog)
# -----------------------------------------------------------------------

EVENT_TYPES = frozenset({
    "segfault", "crash", "regression", "compilation-fail",
    "merge-fail", "merge-keep", "merge-revert", "exhausted",
    "rebuild-fail", "rebuild-crash", "tuning-crash", "tuning-fail",
    "comm-hang", "comm-fail", "codegen-fail", "cache-corrupt",
    "server-crash", "server-hang", "dispatch-fix-fail", "accuracy-fail",
    # Anomaly events (soft failures for discovery)
    "micro-e2e-gap", "regime-divergence", "stale-change", "interaction-regression",
})

EVENT_SNIPPET_CHARS = 2000


def write_event(session_dir: str, event: dict[str, Any]) -> dict[str, Any]:
    event.setdefault("id", f"evt_{uuid.uuid4().hex[:12]}")
    event.setdefault("timestamp", _now_iso())
    event.setdefault("severity", "error")
    details = event.get("details")
    if isinstance(details, dict):
        snippet = details.get("crash_log_snippet", "")
        if len(snippet) > EVENT_SNIPPET_CHARS:
            details["crash_log_snippet"] = snippet[:EVENT_SNIPPET_CHARS]
    _atomic_append(_km_dir(session_dir) / "event_log.jsonl", event)
    return event


def read_new_events(session_dir: str, *, after_id: str = "") -> list[dict[str, Any]]:
    return _read_after(_km_dir(session_dir) / "event_log.jsonl", after_id)


# -----------------------------------------------------------------------
# findings.jsonl  (W: watchdog  |  R: orchestrator, kernel-manager)
# Finding cursor uses finding["event_id"] (not a separate finding id).
# -----------------------------------------------------------------------

def write_finding(session_dir: str, finding: dict[str, Any]) -> dict[str, Any]:
    finding.setdefault("timestamp", _now_iso())
    _atomic_append(_km_dir(session_dir) / "findings.jsonl", finding)
    return finding


def read_new_findings(session_dir: str, *, after_event_id: str = "") -> list[dict[str, Any]]:
    entries = _read_jsonl(_km_dir(session_dir) / "findings.jsonl")
    if not after_event_id:
        return entries
    for i, e in enumerate(entries):
        if e.get("event_id") == after_event_id:
            return entries[i + 1:]
    import logging
    logging.getLogger(__name__).warning(
        "IPC cursor desync: after_event_id=%s not found in findings.jsonl (%d entries), returning all",
        after_event_id, len(entries))
    return entries


def get_findings_for_kernel(session_dir: str, kernel_name: str) -> list[dict[str, Any]]:
    return [f for f in _read_jsonl(_km_dir(session_dir) / "findings.jsonl")
            if f.get("kernel_name") == kernel_name]


# -----------------------------------------------------------------------
# insights.jsonl  (W: orchestrator, kernel-manager, watchdog  |  R: all)
# Insight types: pattern-discovery, design-space-found, transfer-opportunity,
#   anomaly-detected, interaction-effect
# -----------------------------------------------------------------------

INSIGHT_TYPES = frozenset({
    "pattern-discovery", "design-space-found", "transfer-opportunity",
    "anomaly-detected", "interaction-effect",
})


def write_insight(session_dir: str, insight: dict[str, Any]) -> dict[str, Any]:
    """Append an insight to insights.jsonl."""
    insight.setdefault("id", f"ins_{uuid.uuid4().hex[:12]}")
    insight.setdefault("timestamp", _now_iso())
    insight.setdefault("confidence", "medium")
    _atomic_append(_km_dir(session_dir) / "insights.jsonl", insight)
    return insight


def read_new_insights(session_dir: str, *, after_id: str = "") -> list[dict[str, Any]]:
    """Read insights newer than *after_id*."""
    return _read_after(_km_dir(session_dir) / "insights.jsonl", after_id)


def read_all_insights(session_dir: str) -> list[dict[str, Any]]:
    """Read all insights (for dream phase analysis)."""
    return _read_jsonl(_km_dir(session_dir) / "insights.jsonl")


# -----------------------------------------------------------------------
# rca_reports/<event_id>/  (W: watchdog  |  R: human review)
# -----------------------------------------------------------------------

def create_rca_report_dir(session_dir: str, event_id: str) -> Path:
    d = _km_dir(session_dir) / "rca_reports" / event_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "evidence").mkdir(exist_ok=True)
    return d


def read_rca_summary(session_dir: str, event_id: str) -> dict[str, Any] | None:
    p = _km_dir(session_dir) / "rca_reports" / event_id / "rca_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# -----------------------------------------------------------------------
# Initialisation — touch all IPC files (non-destructive)
# -----------------------------------------------------------------------

def init_ipc_files(session_dir: str) -> None:
    km = _km_dir(session_dir)
    km.mkdir(parents=True, exist_ok=True)
    for name in ("work_queue.jsonl", "results.jsonl", "event_log.jsonl",
                 "findings.jsonl", "insights.jsonl"):
        (km / name).touch(exist_ok=True)
    (km / "merge_ready").mkdir(exist_ok=True)
    (km / "rca_reports").mkdir(exist_ok=True)
