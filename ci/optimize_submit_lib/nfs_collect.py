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

globals().update({k: v for k, v in vars(_delivery).items() if not k.startswith("__")})

from . import backfill as _backfill
from . import artifact_paths as _artifact_paths

globals().update({k: v for k, v in vars(_backfill).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_artifact_paths).items() if not k.startswith("__")})

def _nfs_fallback_collect(
    rec: SubmissionRecord,
    artifacts_dir: Path,
    copy_full_session: bool = False,
    current_session_hints: set[str] | None = None,
) -> int:
    """Scan NFS result directories for files matching this model, used when the
    SaFE artifact API returns nothing.

    Two stages: A. legacy canonical CI dirs, matched by dir name; B. per-user
    session dirs, matched by each ci_metrics.json's `model` field.

    Args:
        rec (SubmissionRecord): Record used to match sessions and record
            collected artifacts.
        artifacts_dir (Path): Local artifacts root the files are copied into.
        copy_full_session (bool): Also copy the full matched session tree.
        current_session_hints (set[str] | None): Session-timestamp hints used to
            scope the scan.

    Returns:
        int: Number of files copied.
    """
    current_session_hints = set(current_session_hints or set())
    nfs_root = os.environ.get("NFS_ROOT", "/mnt/shared")
    model_basename = (rec.model or "").split("/")[-1]
    model_short = model_basename.lower().replace("-", "").replace("_", "").replace(".", "")
    if not model_short or not rec.task_id:
        return 0

    task_dir = artifacts_dir / rec.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    has_task_scoped_model_dir = bool(rec.safe_user_id and _record_has_task_window(rec))
    if not current_session_hints and not has_task_scoped_model_dir:
        log.info(
            "[task %s] NFS fallback skipped: no current-session "
            "timestamp hints from SaFE artifacts or task time window",
            rec.task_id,
        )
        return copied

    # Stage A: legacy canonical dirs, matched by current timestamp.
    legacy_scan_dirs = [
        f"{nfs_root}/hyperloom-results",
        f"{nfs_root}/results/ci",
        f"{nfs_root}/inference-optimization/results",
    ]
    for scan_dir in legacy_scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        try:
            entries = sorted(os.listdir(scan_dir), reverse=True)
        except Exception:
            continue
        for entry in entries:
            if not _path_has_session_hint(entry, current_session_hints):
                continue
            entry_clean = entry.lower().replace("-", "").replace("_", "").replace(".", "")
            if model_short not in entry_clean:
                continue
            candidate = os.path.join(scan_dir, entry)
            if not os.path.isdir(candidate):
                continue
            for suffix in _KEY_RESULT_SUFFIXES:
                for root, _, fnames in os.walk(candidate):
                    for fn in fnames:
                        if not fn.endswith(suffix):
                            continue
                        src = os.path.join(root, fn)
                        dst = task_dir / fn
                        if dst.exists():
                            continue
                        try:
                            shutil.copy2(src, dst)
                            _backfill_ci_metrics_file(dst, rec)
                            _record_artifact_source(
                                rec,
                                dst,
                                "nfs_legacy",
                                source_path=src,
                                session_dir=candidate,
                            )
                        except Exception as e:
                            log.warning("[task %s] NFS legacy copy %s -> %s failed: %s", rec.task_id, src, dst, e)
                            continue
                        rec.artifact_files.append(str(dst))
                        rec.artifact_count += 1
                        copied += 1
                        log.info("[task %s] NFS legacy: copied %s -> %s", rec.task_id, src, dst)
            if copied:
                break
        if copied:
            break
    if copied:
        return copied

    # Stage B: <shared-root>/users/<uid>/<session>/... matched by timestamp.
    users_root = f"{nfs_root}/users"
    if not os.path.isdir(users_root):
        return copied

    # Primary match: JSON model fields when present; secondary: session-dir match.
    target = _norm_token(model_basename)
    if not target:
        return copied

    def _model_field_matches(model_field: str) -> bool:
        """Check a JSON ``model`` field against the record's expected names.

        Args:
            model_field (str): The model id read from a candidate result file.

        Returns:
            bool: True if it normalizes to one of the record's allowed names.
        """
        observed = _norm_token(model_field)
        allowed = {
            target,
            _norm_token((rec.model or "").replace("/", "-")),
            _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        }
        allowed.discard("")
        return observed in allowed or _norm_token(model_field.split("/")[-1]) in allowed

    def _consider_result_file(path: str, session_dir: str, score_base: int) -> None:
        """Score a candidate result file and append it to ``candidates``.

        Validates the file's model/session/claw fields against the record and,
        when it matches, records a ``(score, mtime, path, session_dir)`` tuple.

        Args:
            path (str): Path to a candidate result JSON file.
            session_dir (str): The session directory containing the file.
            score_base (int): Base score reflecting match-source confidence.
        """
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return

        model_field = _json_model_field(data)
        if model_field:
            if not _model_field_matches(model_field):
                return
            score = score_base + 100
        else:
            session_name = os.path.basename(session_dir.rstrip("/"))
            parent_name = os.path.basename(os.path.dirname(session_dir.rstrip("/")))
            if not (_record_matches_session_dir(rec, session_name) or _record_matches_session_dir(rec, parent_name)):
                return
            score = score_base + 60

        claw = _json_claw_session_id(data)
        if rec.claw_session_id and claw:
            if rec.claw_session_id != claw:
                return
            score += 30
        elif claw and not rec.claw_session_id:
            rec.claw_session_id = claw
            score += 20
        if _json_positive_perf(data):
            score += 10
        else:
            log.info(
                "[task %s] candidate %s matched but has no positive throughput (will use only if no better match)",
                rec.task_id,
                path,
            )
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        candidates.append((score, mtime, path, session_dir))

    candidates: list[tuple[int, float, str, str]] = []
    uid_dirs = [rec.safe_user_id] if rec.safe_user_id else os.listdir(users_root)
    for uid_dir in uid_dirs:
        if not uid_dir:
            continue
        uid_path = os.path.join(users_root, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        for sess in os.listdir(uid_path):
            sess_path = os.path.join(uid_path, sess)
            if not os.path.isdir(sess_path):
                continue
            if not _path_has_session_hint(sess_path, current_session_hints):
                continue
            for sub in ("", "phase10_report", "results"):
                ci_path = (
                    os.path.join(sess_path, sub, "ci_metrics.json")
                    if sub
                    else os.path.join(sess_path, "ci_metrics.json")
                )
                _consider_result_file(ci_path, sess_path, 0)

        # <shared-root>/users/<uid>/<model-basename>/<ts>/: with no artifact hint,
        # accept timestamp dirs under this user+model dir within the task window.
        if rec.safe_user_id and uid_dir == rec.safe_user_id:
            for model_dir_name in _candidate_model_dir_names(rec):
                model_dir = os.path.join(uid_path, model_dir_name)
                if not os.path.isdir(model_dir):
                    continue
                try:
                    ts_entries = sorted(os.listdir(model_dir), reverse=True)
                except Exception:
                    continue
                for ts_entry in ts_entries:
                    session_dir = os.path.join(model_dir, ts_entry)
                    if not os.path.isdir(session_dir):
                        continue
                    ts = _session_timestamp_from_path(ts_entry)
                    if current_session_hints:
                        if not _path_has_session_hint(session_dir, current_session_hints):
                            continue
                        score_base = 40
                    else:
                        if not _timestamp_in_task_window(ts, rec):
                            continue
                        score_base = 30
                    for sub, filename in (
                        ("", "ci_metrics.json"),
                        ("phase10_report", "ci_metrics.json"),
                        ("results", "ci_metrics.json"),
                        ("", "session_breakdown.json"),
                        ("phase10_report", "session_breakdown.json"),
                        ("v2_session", "session_breakdown.json"),
                    ):
                        result_path = (
                            os.path.join(session_dir, sub, filename) if sub else os.path.join(session_dir, filename)
                        )
                        _consider_result_file(result_path, session_dir, score_base)

    if not candidates:
        log.info(
            "[task %s] no shared users candidate matched model=%s (user_id=%s, hints=%s, task_window=%s)",
            rec.task_id,
            model_basename,
            rec.safe_user_id or "?",
            ",".join(sorted(current_session_hints)) or "-",
            "yes" if _record_has_task_window(rec) else "no",
        )
        return copied

    # Highest-confidence, then freshest.
    candidates.sort(reverse=True)
    _score, _mtime, best_ci, best_sess = candidates[0]
    log.info(
        "[task %s] NFS user-session match: %s (from %d candidate(s), session=%s)",
        rec.task_id,
        best_ci,
        len(candidates),
        best_sess,
    )

    # Copy ci_metrics + any optimization_report.md flat under task_dir/.
    targets = [best_ci]
    for cand in [
        os.path.join(best_sess, "optimization_report.md"),
        os.path.join(best_sess, "phase10_report", "optimization_report.md"),
        os.path.join(best_sess, "reports", "final.md"),
    ]:
        if os.path.isfile(cand):
            targets.append(cand)
            break
    # Optional audit artifact.
    for cand in [
        os.path.join(best_sess, "session_breakdown.json"),
        os.path.join(best_sess, "phase10_report", "session_breakdown.json"),
    ]:
        if os.path.isfile(cand):
            targets.append(cand)
            break

    for src in targets:
        dst_name = "optimization_report.md" if src.endswith("final.md") else os.path.basename(src)
        dst = task_dir / dst_name
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
            _backfill_ci_metrics_file(dst, rec)
            _record_artifact_source(
                rec,
                dst,
                "nfs_user_session",
                source_path=src,
                session_dir=best_sess,
            )
        except Exception as e:
            log.warning("[task %s] NFS user-session copy %s -> %s failed: %s", rec.task_id, src, dst, e)
            continue
        rec.artifact_files.append(str(dst))
        rec.artifact_count += 1
        copied += 1
        log.info("[task %s] NFS user-session: copied %s -> %s", rec.task_id, src, dst)

    if copy_full_session:
        session_dst = task_dir / "session"
        n_full = _copy_session_tree(best_sess, session_dst)
        if n_full:
            copied += n_full
            rec.artifact_count += n_full
            rec.artifact_files.append(str(session_dst))
            log.info(
                "[task %s] NFS full-session: copied %d file(s) %s -> %s", rec.task_id, n_full, best_sess, session_dst
            )
        else:
            log.info("[task %s] NFS full-session: no new files copied from %s", rec.task_id, best_sess)

    return copied
