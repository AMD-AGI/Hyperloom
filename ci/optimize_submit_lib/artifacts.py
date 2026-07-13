from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import config as _config
from . import records as _records

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_records).items() if not k.startswith("__")})

from . import delivery as _delivery
from . import backfill as _backfill
from . import nfs_collect as _nfs_collect
from . import artifact_paths as _artifact_paths

globals().update({k: v for k, v in vars(_artifact_paths).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_delivery).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_backfill).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_nfs_collect).items() if not k.startswith("__")})

def _is_wanted_artifact(path: str, all_artifacts: bool) -> bool:
    """Decide whether an artifact path should be downloaded.

    Args:
        path (str): The remote artifact path.
        all_artifacts (bool): When True, accept every artifact.

    Returns:
        bool: True when ``all_artifacts`` is set or the path matches a default
            artifact pattern.
    """
    if all_artifacts:
        return True
    p = path.lower()
    return any(pat in p for pat in DEFAULT_ARTIFACT_PATTERNS)


def _safe_local_path(artifacts_dir: Path, task_id: str, remote_path: str) -> Path:
    """Map a session-relative remote path to a local file path, stripping leading
    slashes / '..' so we never escape the artifacts dir.

    Args:
        artifacts_dir (Path): Local artifacts root.
        task_id (str): SaFE task id used as the per-task subdirectory.
        remote_path (str): Session-relative remote artifact path.

    Returns:
        Path: A safe local destination path under ``artifacts_dir/task_id``.
    """
    rel = remote_path.lstrip("/").replace("\\", "/")
    parts = [seg for seg in rel.split("/") if seg and seg != ".." and seg != "."]
    return artifacts_dir / task_id / Path(*parts) if parts else artifacts_dir / task_id / "artifact.bin"


def _record_artifact_source(
    rec: SubmissionRecord,
    local_path: Path,
    source_type: str,
    *,
    remote_path: str | None = None,
    source_path: str | None = None,
    session_dir: str | None = None,
) -> None:
    """Append a provenance entry describing where an artifact came from.

    Args:
        rec (SubmissionRecord): Record whose ``artifact_sources`` is appended.
        local_path (Path): Local path the artifact was written to.
        source_type (str): Origin label (e.g. ``safe_artifact_api``, ``nfs_*``).
        remote_path (str | None): Remote artifact path, when applicable.
        source_path (str | None): Source filesystem path, when applicable.
        session_dir (str | None): Originating session directory, when known.
    """
    entry = {
        "source_type": source_type,
        "local_path": str(local_path).replace("\\", "/"),
        "file_name": local_path.name,
    }
    if remote_path:
        entry["remote_path"] = remote_path
    if source_path:
        entry["source_path"] = source_path
    if session_dir:
        entry["session_dir"] = session_dir
    rec.artifact_sources.append(entry)


def _write_artifact_sources(task_dir: Path, rec: SubmissionRecord) -> None:
    """Write the per-task ``artifact_sources.json`` provenance file.

    No-op when the record has no recorded artifact sources.

    Args:
        task_dir (Path): Per-task directory the file is written into.
        rec (SubmissionRecord): Record carrying the artifact source entries.
    """
    if not rec.artifact_sources:
        return
    payload = {
        "task_id": rec.task_id,
        "model": rec.model,
        "claw_session_id": rec.claw_session_id,
        "artifact_dir": str(task_dir).replace("\\", "/"),
        "files": rec.artifact_sources,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "artifact_sources.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# Files required for a run to count as "delivered". The NFS legacy fallback
# scans these suffixes; a miss triggers the wekafs rescue path.
_KEY_RESULT_SUFFIXES: tuple[str, ...] = (
    "optimization_report.md",
    "ci_metrics.json",
    "session_breakdown.json",
)

def wait_and_collect_one(
    safe: SafeOptimizeClient,
    rec: SubmissionRecord,
    artifacts_dir: Path,
    task_timeout_min: int,
    poll_s: int,
    collect: bool,
    all_artifacts: bool,
) -> SubmissionRecord:
    """Wait for one task to finish, then optionally download its artifacts:
    (1) SaFE artifacts API, then (2) NFS fallback when session_breakdown.json
    (the CI delivery contract) is still missing.

    Args:
        safe (SafeOptimizeClient): Client used to poll and download artifacts.
        rec (SubmissionRecord): Record to update with status and artifacts.
        artifacts_dir (Path): Local artifacts root for this task.
        task_timeout_min (int): Max minutes to wait for the task.
        poll_s (int): Polling interval in seconds.
        collect (bool): When False, skip artifact collection.
        all_artifacts (bool): When True, download every artifact / full session.

    Returns:
        SubmissionRecord: The same ``rec``, mutated in place with results.
    """
    if not rec.task_id:
        return rec  # nothing to wait for (skipped/failed during submit)

    final_status, last_task = safe.wait_task_done(rec.task_id, timeout_min=task_timeout_min, poll_s=poll_s)
    rec.final_status = final_status
    rec.final_phase = last_task.get("currentPhase")
    rec.final_message = (last_task.get("message") or "")[:500] or None
    rec.claw_session_id = _resolve_record_claw_session_id(safe, rec, last_task)
    rec.model_path = (last_task.get("modelPath") or "").strip() or rec.model_path
    rec.safe_user_id = (last_task.get("userId") or "").strip() or rec.safe_user_id
    rec.safe_started_at = (last_task.get("startedAt") or "").strip() or rec.safe_started_at
    rec.safe_finished_at = (last_task.get("finishedAt") or "").strip() or rec.safe_finished_at
    rec.sandbox_duration_seconds = _sandbox_duration_seconds(last_task)
    if rec.claw_session_id:
        log.info(
            "[task %s] clawSessionId=%s duration=%ss",
            rec.task_id,
            rec.claw_session_id,
            rec.sandbox_duration_seconds if rec.sandbox_duration_seconds is not None else "?",
        )

    if not collect:
        rec.ci_status = "Not collected"
        rec.delivery_reason = "artifact collection disabled"
        return rec

    # Stage 1: SaFE artifacts API (most reliable when the agent copied files to
    # /workspace/hyperloom). Retry — terminal status can beat Claw's file index.
    items = []
    wanted = []
    for attempt in range(3):
        try:
            items = safe.list_artifacts(rec.task_id)
        except Exception as e:
            log.warning("[task %s] list_artifacts attempt %d failed: %s", rec.task_id, attempt + 1, e)
            items = []
        wanted = [it for it in items if _is_wanted_artifact(it.get("path", ""), all_artifacts)]
        wanted_paths = [it.get("path", "").lower() for it in wanted]
        # session_breakdown.json is the CI delivery contract (cli.finally always
        # writes it, even on abort), so retry only until it shows up.
        has_safe_breakdown = any(p.endswith("session_breakdown.json") for p in wanted_paths)
        if has_safe_breakdown:
            break
        if attempt < 2:
            log.info(
                "[task %s] safe artifacts missing session_breakdown.json on attempt %d; retrying",
                rec.task_id,
                attempt + 1,
            )
            time.sleep(15)
    log.info("[task %s] safe artifacts: %d total, %d to download", rec.task_id, len(items), len(wanted))
    current_session_hints = _session_hints_from_artifact_items(items)
    if current_session_hints:
        log.info(
            "[task %s] current session timestamp hints from artifacts: %s",
            rec.task_id,
            ", ".join(sorted(current_session_hints)),
        )

    if not has_safe_breakdown:
        waited_session = _wait_for_nfs_session_delivery(
            rec,
            current_session_hints=current_session_hints,
            poll_s=poll_s,
        )
        if waited_session:
            # The SaFE artifact index may lag behind the agent's final writes.
            # Re-list once after the grace wait before falling back to NFS.
            try:
                items = safe.list_artifacts(rec.task_id)
                wanted = [it for it in items if _is_wanted_artifact(it.get("path", ""), all_artifacts)]
                wanted_paths = [it.get("path", "").lower() for it in wanted]
                current_session_hints.update(_session_hints_from_artifact_items(items))
                log.info(
                    "[task %s] safe artifacts after NFS grace wait: %d total, %d to download",
                    rec.task_id,
                    len(items),
                    len(wanted),
                )
            except Exception as e:
                log.warning("[task %s] post-grace list_artifacts failed: %s", rec.task_id, e)

    task_dir = artifacts_dir / rec.task_id
    rec.artifacts_dir = str(task_dir)
    for it in wanted:
        path = it.get("path", "")
        if not path:
            continue
        local = _safe_local_path(artifacts_dir, rec.task_id, path)
        try:
            n = safe.download_artifact_to(rec.task_id, it, str(local))
            _backfill_ci_metrics_file(local, rec)
            _record_artifact_source(
                rec,
                local,
                "safe_artifact_api",
                remote_path=path,
            )
            rec.artifact_files.append(str(local))
            rec.artifact_count += 1
            log.info("[task %s] saved %s (%d bytes)", rec.task_id, path, n)
        except Exception as e:
            log.warning("[task %s] failed to download %s: %s", rec.task_id, path, e)

    # Stage 2: NFS fallback when Stage 1 didn't deliver session_breakdown.json
    # (the CI delivery contract). ci_metrics.json / optimization_report.md no
    # longer gate the fallback.
    has_breakdown = any(p.endswith("session_breakdown.json") for p in rec.artifact_files)
    if not has_breakdown:
        log.info("[task %s] missing session_breakdown.json — trying NFS fallback", rec.task_id)
        copy_full_session = all_artifacts or _env_truthy("SAFE_OPTIMIZE_COPY_FULL_SESSION")
        n_added = _nfs_fallback_collect(
            rec,
            artifacts_dir,
            copy_full_session=copy_full_session,
            current_session_hints=current_session_hints,
        )
        if n_added:
            log.info("[task %s] NFS fallback added %d files", rec.task_id, n_added)
        else:
            log.info("[task %s] NFS fallback found nothing", rec.task_id)

    _mark_record_delivery(rec)
    if rec.ci_status:
        log.info("[task %s] CI delivery status: %s (%s)", rec.task_id, rec.ci_status, rec.delivery_reason or "-")

    # Stage 3: reverse-backfill audit fields into the wekafs SOURCE files so
    # operators see them without the GHA artifact zip. No-op when wekafs unmounted.
    if rec.artifact_count:
        try:
            n_wkfs = _backfill_wekafs_in_place(rec)
            if n_wkfs:
                log.info("[task %s] wekafs in-place backfill updated %d file(s)", rec.task_id, n_wkfs)
        except Exception as e:
            log.warning("[task %s] wekafs in-place backfill skipped due to %s: %s", rec.task_id, type(e).__name__, e)
    else:
        log.info("[task %s] wekafs in-place backfill skipped: no artifacts collected", rec.task_id)
    try:
        _write_artifact_sources(task_dir, rec)
    except Exception as e:
        log.warning("[task %s] artifact source metadata write skipped: %s", rec.task_id, e)
    return rec


def process_completion(
    safe: SafeOptimizeClient,
    records: list[SubmissionRecord],
    artifacts_dir: Path,
    task_timeout_min: int,
    poll_s: int,
    collect: bool,
    all_artifacts: bool,
    parallel: int,
) -> None:
    """Wait + collect for all submitted records, in parallel up to ``parallel``.

    Args:
        safe (SafeOptimizeClient): Client used to poll and download artifacts.
        records (list[SubmissionRecord]): All submission records; only
            ``submitted`` ones with a task id are awaited.
        artifacts_dir (Path): Destination root for downloaded artifacts.
        task_timeout_min (int): Max minutes to wait per task.
        poll_s (int): Polling interval in seconds.
        collect (bool): When True, collect artifacts after each task finishes.
        all_artifacts (bool): When True, also copy full session trees.
        parallel (int): Max concurrent wait/collect workers (<=1 runs serially).
    """
    pending = [r for r in records if r.status == "submitted" and r.task_id]
    if not pending:
        log.info("no submitted tasks to wait for")
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "waiting for %d task(s) to finish (parallel=%d, timeout=%dm each)", len(pending), parallel, task_timeout_min
    )

    if parallel <= 1:
        for rec in pending:
            try:
                wait_and_collect_one(safe, rec, artifacts_dir, task_timeout_min, poll_s, collect, all_artifacts)
            except Exception as e:
                log.exception("[task %s] unexpected wait/collect error", rec.task_id)
                rec.final_status = rec.final_status or "Error"
                rec.final_message = (rec.final_message or "") + f" | wait error: {e}"
            finally:
                if not rec.ci_status:
                    _mark_record_delivery(rec)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(
                wait_and_collect_one, safe, rec, artifacts_dir, task_timeout_min, poll_s, collect, all_artifacts
            ): rec
            for rec in pending
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                fut.result()  # mutates rec in place
            except Exception as e:
                log.exception("[task %s] unexpected wait/collect error", rec.task_id)
                rec.final_status = rec.final_status or "Error"
                rec.final_message = (rec.final_message or "") + f" | wait error: {e}"
            finally:
                if not rec.ci_status:
                    _mark_record_delivery(rec)
