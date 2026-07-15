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

def _norm_token(s: str) -> str:
    """Aggressively normalise a string for fuzzy equality comparison.

    Lowercases and strips dashes, underscores, dots, slashes, and spaces.

    Args:
        s (str): String to normalise.

    Returns:
        str: The normalised token.
    """
    return (s or "").lower().replace("-", "").replace("_", "").replace(".", "").replace("/", "").replace(" ", "")


def _slug_token(s: str) -> str:
    """Convert a string to a lowercase dash-separated slug.

    Collapses any run of non-alphanumeric characters to a single dash and
    trims leading/trailing dashes.

    Args:
        s (str): String to slugify.

    Returns:
        str: The slugified token.
    """
    out = []
    prev_dash = False
    for ch in (s or "").lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def _metrics_have_positive_throughput(path: str) -> bool:
    """Report whether a ci_metrics.json file has real, non-zero throughput.

    Args:
        path (str): Path to a ``ci_metrics.json`` file.

    Returns:
        bool: True when both baseline and optimized throughput parse to values
            greater than zero; False on any read/parse error.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    baseline = data.get("baseline_throughput") or data.get("tok_per_gpu_baseline")
    optimized = data.get("optimized_throughput") or data.get("tok_per_gpu_optimized")
    try:
        return float(baseline) > 0 and float(optimized) > 0
    except (TypeError, ValueError):
        return False


def _json_has_any_number(value) -> bool:
    """Recursively test whether a JSON value contains any real number.

    Booleans are not counted as numbers; lists are scanned up to the first 100
    elements.

    Args:
        value: Any JSON-decoded value.

    Returns:
        bool: True when an int/float (non-bool) is found anywhere within.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_json_has_any_number(v) for v in value.values())
    if isinstance(value, list):
        return any(_json_has_any_number(v) for v in value[:100])
    return False


def _breakdown_has_basic_data(path: Path) -> bool:
    """True when a session_breakdown JSON carries usable audit/perf payload.

    The delivery contract is "has structured data", not "has positive gain".

    Args:
        path (Path): Path to a ``session_breakdown`` JSON file.

    Returns:
        bool: True when the JSON is an object carrying a known audit/perf key or
        any numeric value; False on read/parse error.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    keys = (
        "baseline",
        "best",
        "final",
        "optimized",
        "steps",
        "actions",
        "phases",
        "session",
        "session_meta",
        "workload",
        "baseline_throughput",
        "optimized_throughput",
        "tok_per_gpu_baseline",
        "tok_per_gpu_optimized",
        "gain_pct",
        "cumulative_gain_pct",
    )
    if any(data.get(k) not in (None, {}, [], "") for k in keys):
        return True
    return _json_has_any_number(data)


def _mark_record_delivery(rec: SubmissionRecord) -> None:
    """Set CI-level delivery status from collected artifacts.

    Scans the record's artifact files/dir for a publishable
    ``session_breakdown`` JSON and updates ``ci_success``, ``ci_status``, and
    ``delivery_reason`` accordingly.

    Args:
        rec (SubmissionRecord): The submission record to mutate in place.
    """
    candidates: list[Path] = []
    for raw in rec.artifact_files:
        p = Path(raw)
        if p.is_file() and p.name.startswith("session_breakdown") and p.suffix == ".json":
            candidates.append(p)
    if rec.artifacts_dir:
        root = Path(rec.artifacts_dir)
        if root.is_dir():
            candidates.extend(p for p in root.glob("**/session_breakdown*.json") if p.is_file())
    seen: set[str] = set()
    unique = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for p in unique:
        if _breakdown_has_basic_data(p):
            rec.ci_success = True
            rec.ci_status = "Delivered"
            rec.delivery_reason = f"publishable session_breakdown: {p.name}"
            return

    if rec.artifact_count:
        rec.ci_status = "Artifacts incomplete"
        rec.delivery_reason = "artifacts collected but no usable session_breakdown"
    else:
        rec.ci_status = "Missing artifacts"
        rec.delivery_reason = "no artifacts collected"


def _timestamp_hint_variants(value: str) -> set[str]:
    """Return path-matchable variants for skill session timestamps.

    Args:
        value (str): A session timestamp string (e.g. ``"20260512T010203Z"``).

    Returns:
        set[str]: Case- and separator-normalized variants for substring
            matching against paths; empty if ``value`` is blank.
    """
    raw = value.strip()
    if not raw:
        return set()
    compact = raw.replace("T", "").replace("t", "").replace("Z", "").replace("z", "")
    variants = {raw, raw.lower()}
    if compact and compact != raw:
        variants.update({compact, compact.lower()})
    return variants


def _session_hints_from_artifact_items(items: list[dict]) -> set[str]:
    """Collect session-timestamp hints from artifact item metadata.

    Args:
        items (list[dict]): Artifact item dicts with ``path``/``downloadPath``/
            ``name`` fields.

    Returns:
        set[str]: Timestamp hint variants discovered across the items.
    """
    hints: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(key) or "") for key in ("path", "downloadPath", "name"))
        for match in re.findall(r"\b\d{8}T\d{6}Z\b", text, flags=re.IGNORECASE):
            hints.update(_timestamp_hint_variants(match))
    return hints


def _path_has_session_hint(path: str, hints: set[str]) -> bool:
    """Report whether a path contains any of the session timestamp hints.

    Args:
        path (str): The path to test.
        hints (set[str]): Timestamp hint variants to look for.

    Returns:
        bool: True if any hint appears in the normalized path.
    """
    if not hints:
        return False
    norm_path = _norm_token(path)
    return any(_norm_token(hint) in norm_path for hint in hints)


def _parse_safe_timestamp(value: str | None) -> datetime | None:
    """Parse a SaFE ISO-8601 timestamp into a UTC datetime.

    Args:
        value (str | None): An ISO-8601 string (``Z`` suffix accepted).

    Returns:
        datetime | None: A timezone-aware UTC datetime, or ``None`` if blank
            or unparseable.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_session_timestamp(value: str) -> datetime | None:
    """Parse a compact ``YYYYMMDDTHHMMSSZ`` session timestamp.

    Args:
        value (str): The compact session timestamp string.

    Returns:
        datetime | None: A UTC datetime, or ``None`` if blank or unparseable.
    """
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw.upper(), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _session_timestamp_from_path(path: str) -> str:
    """Extract the last ``YYYYMMDDTHHMMSSZ`` timestamp found in a path.

    Args:
        path (str): The path to scan.

    Returns:
        str: The matched timestamp (upper-cased), or ``""`` if none found.
    """
    matches = re.findall(r"\b\d{8}T\d{6}Z\b", path, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def _timestamp_in_task_window(timestamp: str, rec: SubmissionRecord, margin_hours: int = 2) -> bool:
    """Check whether a session timestamp falls within the task's run window.

    Args:
        timestamp (str): A compact session timestamp string.
        rec (SubmissionRecord): The record providing SaFE start/finish times.
        margin_hours (int): Slack added on each side of the window.

    Returns:
        bool: True if the timestamp lies within the (padded) task window.
    """
    ts = _parse_session_timestamp(timestamp)
    start = _parse_safe_timestamp(rec.safe_started_at)
    end = _parse_safe_timestamp(rec.safe_finished_at)
    if ts is None or start is None:
        return False
    if end is None:
        end = start + timedelta(hours=24)
    return (start - timedelta(hours=margin_hours)) <= ts <= (end + timedelta(hours=margin_hours))


def _record_has_task_window(rec: SubmissionRecord) -> bool:
    """Report whether a record has a usable SaFE start timestamp.

    Args:
        rec (SubmissionRecord): The submission record to inspect.

    Returns:
        bool: True if ``safe_started_at`` parses into a timestamp.
    """
    return _parse_safe_timestamp(rec.safe_started_at) is not None


def _record_model_field_matches(rec: SubmissionRecord, model_field: str) -> bool:
    """Compare a JSON model field with a SubmissionRecord conservatively."""
    observed = _norm_token(str(model_field or ""))
    if not observed:
        return False
    allowed = {
        _norm_token((rec.model or "").split("/")[-1]),
        _norm_token((rec.model or "").replace("/", "-")),
        _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        _norm_token(rec.display_name or ""),
    }
    allowed.discard("")
    return observed in allowed or _norm_token(str(model_field).split("/")[-1]) in allowed


def _candidate_model_dir_names(rec: SubmissionRecord) -> list[str]:
    """Derive plausible per-model directory basenames for a record."""
    names: list[str] = []
    for value in (
        rec.model_path or "",
        (rec.model or "").replace("/", "-"),
        (rec.model or "").split("/")[-1],
        rec.display_name or "",
    ):
        name = str(value or "").strip().rstrip("/\\").split("/")[-1]
        if name and name not in names:
            names.append(name)
    return names


def _read_session_state(session_dir: str | Path) -> dict:
    """Best-effort read of a session's state.json.

    Args:
        session_dir (str | Path): Session directory containing ``state.json``.

    Returns:
        dict: The parsed state dict, or an empty dict when missing/unreadable
        or not a JSON object.
    """
    path = Path(session_dir) / "state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _session_has_terminal_marker(session_dir: str | Path) -> bool:
    """True when a session has reached a state where CI should collect now.

    Args:
        session_dir (str | Path): Session directory to inspect.

    Returns:
        bool: True when a ``session_breakdown.json`` / ``complete`` marker
        exists or ``state.json`` reports ``close_sequence_done``.
    """
    root = Path(session_dir)
    if (root / "session_breakdown.json").is_file():
        return True
    if (root / "complete").is_file():
        return True
    state = _read_session_state(root)
    if state.get("close_sequence_done") is True:
        return True
    return False


def _session_activity_mtime(session_dir: str | Path) -> float:
    """Return a bounded best-effort activity timestamp for a session.

    Scans state files plus the runtime subtrees CI relies on, with a file cap to
    avoid expensive walks for very large sessions.

    Args:
        session_dir (str | Path): Session directory to scan.

    Returns:
        float: The newest relevant file mtime, or ``0.0`` when nothing is found.
    """
    root = Path(session_dir)
    mtimes: list[float] = []
    for rel in (
        "state.json",
        "session_breakdown.json",
        "complete",
        "reports/final.md",
        "reports/final.json",
    ):
        p = root / rel
        try:
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        except OSError:
            continue

    seen = 0
    for sub in ("optimizer_runs", "runs", "reports"):
        base = root / sub
        if not base.exists():
            continue
        for walk_root, _dirs, files in os.walk(base):
            for name in files:
                seen += 1
                if seen > 5000:
                    return max(mtimes) if mtimes else 0.0
                if not name.endswith((".log", ".json", ".txt", ".md", ".csv", ".gz")):
                    continue
                try:
                    mtimes.append((Path(walk_root) / name).stat().st_mtime)
                except OSError:
                    continue
    return max(mtimes) if mtimes else 0.0


def _find_nfs_state_session_dir(
    rec: SubmissionRecord,
    current_session_hints: set[str] | None = None,
) -> str | None:
    """Locate the current NFS session using state.json, not breakdown files.

    Args:
        rec (SubmissionRecord): Record providing user id, model names, and task
            window for matching.
        current_session_hints (set[str] | None): Optional session-timestamp
            hints that, when present, must appear in the matched path.

    Returns:
        str | None: The best-matching session directory path, or ``None`` when
        none matches.
    """
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    users_root = Path(nfs_root) / "users"
    if not rec.safe_user_id or not users_root.is_dir():
        return None
    uid_path = users_root / rec.safe_user_id
    if not uid_path.is_dir():
        return None

    hints = set(current_session_hints or set())
    candidates: list[tuple[int, float, str]] = []
    for model_dir_name in _candidate_model_dir_names(rec):
        model_dir = uid_path / model_dir_name
        if not model_dir.is_dir():
            continue
        try:
            ts_entries = sorted(os.listdir(model_dir), reverse=True)
        except OSError:
            continue
        for ts_entry in ts_entries:
            session_dir = model_dir / ts_entry
            if not session_dir.is_dir():
                continue
            state_path = session_dir / "state.json"
            if not state_path.is_file():
                continue
            if hints:
                if not _path_has_session_hint(str(session_dir), hints):
                    continue
                score = 40
            else:
                ts = _session_timestamp_from_path(ts_entry)
                if not _timestamp_in_task_window(ts, rec):
                    continue
                score = 30
            state = _read_session_state(session_dir)
            workload = state.get("workload") if isinstance(state.get("workload"), dict) else {}
            model_field = str(state.get("model") or state.get("model_name") or workload.get("model_name") or "")
            if model_field and not _record_model_field_matches(rec, model_field):
                continue
            if model_field:
                score += 100
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((score, mtime, str(session_dir)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _wait_for_nfs_session_delivery(
    rec: SubmissionRecord,
    current_session_hints: set[str] | None = None,
    poll_s: int = 60,
    grace_min: float | None = None,
    idle_min: float | None = None,
) -> str | None:
    """After SaFE early terminal, wait while the NFS session is still active.

    Args:
        rec (SubmissionRecord): Record used to locate the NFS session.
        current_session_hints (set[str] | None): Optional session-timestamp
            hints to constrain the matched session.
        poll_s (int): Seconds between activity polls.
        grace_min (float | None): Max minutes to wait; ``None`` resolves from
            ``$SAFE_OPTIMIZE_NFS_LIVE_GRACE_MIN``.
        idle_min (float | None): Idle minutes that end the wait; ``None``
            resolves from ``$SAFE_OPTIMIZE_NFS_IDLE_GRACE_MIN``.

    Returns:
        str | None: The session directory once delivery/idle settles, or
        ``None`` when none is found or grace waiting is disabled.
    """
    grace_min = _env_float("SAFE_OPTIMIZE_NFS_LIVE_GRACE_MIN", 180.0) if grace_min is None else grace_min
    idle_min = _env_float("SAFE_OPTIMIZE_NFS_IDLE_GRACE_MIN", 20.0) if idle_min is None else idle_min
    if grace_min <= 0 or idle_min <= 0:
        return None

    session_dir = _find_nfs_state_session_dir(rec, current_session_hints)
    if not session_dir:
        return None

    now = time.time()
    activity = _session_activity_mtime(session_dir)
    if not activity:
        return session_dir
    if now - activity > idle_min * 60 and not _session_has_terminal_marker(session_dir):
        log.info(
            "[task %s] NFS session %s found but inactive for %.1fmin; collecting without grace wait",
            rec.task_id,
            session_dir,
            (now - activity) / 60,
        )
        return session_dir

    deadline = now + grace_min * 60
    idle_deadline = activity + idle_min * 60
    log.warning(
        "[task %s] SaFE status=%s but NFS session still appears active: %s; "
        "waiting up to %.1fmin (idle %.1fmin) for delivery contract files",
        rec.task_id,
        rec.final_status,
        session_dir,
        grace_min,
        idle_min,
    )
    while time.time() < deadline:
        if _session_has_terminal_marker(session_dir):
            log.info(
                "[task %s] NFS session reached terminal/delivery marker: %s",
                rec.task_id,
                session_dir,
            )
            return session_dir
        latest = _session_activity_mtime(session_dir)
        if latest > activity:
            activity = latest
            idle_deadline = latest + idle_min * 60
            log.info(
                "[task %s] NFS session still active (last activity %s)",
                rec.task_id,
                datetime.fromtimestamp(latest, tz=timezone.utc).isoformat(),
            )
        if time.time() > idle_deadline:
            log.warning(
                "[task %s] NFS session idle for %.1fmin without delivery marker; proceeding to collect",
                rec.task_id,
                idle_min,
            )
            return session_dir
        time.sleep(max(1, min(poll_s, 60)))

    log.warning(
        "[task %s] NFS live-session grace wait expired after %.1fmin; proceeding to collect",
        rec.task_id,
        grace_min,
    )
    return session_dir


def _category_from_arch(arch: str | None) -> str:
    """Coarse model-shape classification: "moe" if arch contains "moe", else
    "dense"; "" when unknown.

    Args:
        arch (str | None): HF architecture class name.

    Returns:
        str: ``"moe"``, ``"dense"``, or ``""`` for empty/unknown input.
    """
    if not arch:
        return ""
    return "moe" if "moe" in arch.lower() else "dense"
