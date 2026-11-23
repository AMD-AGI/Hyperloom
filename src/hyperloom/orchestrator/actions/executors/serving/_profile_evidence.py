# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a profiled server run left behind: orphans, liveness, trace files and capture-failure evidence."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hyperloom_common.jsonio import read_json_strict
from inference_suite.contracts import FailedStep, Report
from inference_suite.contracts.layout import REPORT_FILE

log = logging.getLogger(__name__)

#: Suffixes trace discovery will consider. One of these already present in a task's own run directory is
#: a leftover discovery can lead with instead of what this run produces.
_TRACE_SUFFIXES = (".pt.trace.json.gz", ".pt.trace.json", ".trace.json.gz", ".trace.json")
#: Enough leftovers to establish the pattern; the preflight record is context, not an inventory.
_PREFLIGHT_STALE_TRACE_LIMIT = 20


async def reap_session_orphans(session_dir: Path | str) -> list[int]:
    """Reap this session's own orphaned serving processes."""
    from ._server_lifecycle import reap_orphaned_servers

    return await asyncio.to_thread(reap_orphaned_servers, Path(session_dir))


def marker_hit_context(sources: dict[str, str], marker: str, radius: int = 6) -> list[dict[str, Any]]:
    """Return the lines around each source's first hit on ``marker``.

    The classifier matches lowercased substrings, so it can be fooled; the excerpt is the original text, because
    that is what a reader needs to overrule a classification they think is wrong.
    """
    needle = marker.split(" + ")[0].lower()
    hits: list[dict[str, Any]] = []
    for name, text in sources.items():
        if not text:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if needle in line.lower():
                hits.append(
                    {
                        "source": name,
                        "line_no": idx + 1,
                        "excerpt": lines[max(0, idx - radius) : idx + radius + 1],
                    }
                )
                break
    return hits


def trace_dir_inventory(profile_result: Any, limit: int = 200) -> dict[str, Any]:
    """List what a failed profile left on disk, so a truncated export reads differently from no export at all."""
    if not isinstance(profile_result, dict):
        return {}
    base = profile_result.get("trace_dir") or profile_result.get("workspace")
    if not base:
        return {}
    root = Path(str(base))
    try:
        entries = sorted(p for p in root.rglob("*") if p.is_file())
    except OSError as exc:
        # The inventory is evidence about a failure, so it reports its own trouble rather than raising over it.
        return {"root": str(root), "error": repr(exc)}
    files: list[dict[str, Any]] = []
    for p in entries[:limit]:
        try:
            files.append({"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size})
        except OSError:
            continue
    out: dict[str, Any] = {"root": str(root), "file_count": len(entries), "files": files}
    if len(entries) > limit:
        out["truncated"] = True
    return out


def _iso_from_epoch(ts: float) -> str:
    """Render a filesystem timestamp in the same UTC ISO form the rest of the event uses."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def preflight_probe(session_dir: Path, task_id: str, *, reaped: Sequence[int]) -> dict[str, Any]:
    """Describe the conditions the action is about to profile under. Never raises.

    Reported, not acted on: a leftover server, a nearly-full disk or a trace file already sitting in this task's
    own output directory all change how the result should be read, but deciding what to do about them is not a
    measurement stage's call.
    """
    out: dict[str, Any] = {
        "orphans_reaped": list(reaped),
        "orphans_reaped_count": len(reaped),
    }
    try:
        usage = shutil.disk_usage(session_dir)
        out["disk"] = {
            "path": str(session_dir),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "free_pct": round(100.0 * usage.free / usage.total, 2) if usage.total else None,
        }
    except OSError as exc:
        out["disk"] = {"path": str(session_dir), "error": repr(exc)}

    # A trace already under this task's own run directory can lead trace discovery instead of what this run
    # produces, and by then there is no way to tell which run the numbers came from.
    stale: list[dict[str, Any]] = []
    try:
        for run_dir in sorted(session_dir.glob(f"runs/*/{task_id}")) if task_id else []:
            for p in sorted(run_dir.rglob("*")):
                if len(stale) >= _PREFLIGHT_STALE_TRACE_LIMIT:
                    break
                if p.is_file() and str(p).endswith(_TRACE_SUFFIXES):
                    info = p.stat()
                    stale.append(
                        {
                            "path": str(p.relative_to(session_dir)),
                            "size_bytes": info.st_size,
                            "mtime_iso": _iso_from_epoch(info.st_mtime),
                        }
                    )
    except OSError as exc:
        out["stale_traces_error"] = repr(exc)
    out["stale_traces"] = stale
    out["stale_trace_count"] = len(stale)
    return out


def drain_instrumentation(executor: Any) -> dict[str, Any] | None:
    """Take the profile executor's per-attempt instrumentation report, if it keeps one."""
    drain = getattr(executor, "drain_instrumentation_report", None)
    if not callable(drain):
        return None
    report = drain()
    return report if isinstance(report, dict) else None


_SERVER_DIED_STEPS = frozenset({FailedStep.SERVER_DIED, FailedStep.SERVER_EXITED_BEFORE_READY})


def server_liveness_probe(session_dir: Path, task_id: str) -> dict[str, Any]:
    """Report, from this task's run_dir ``report.json`` files, whose server died and whose outlived its teardown.

    A run whose engine died can still have exported a complete trace, which is exactly why the result dict alone
    cannot show it.
    """
    runs_dir = session_dir / "runs"
    if not runs_dir.is_dir():
        return {}
    reports = [path for path in sorted(runs_dir.rglob(REPORT_FILE)) if task_id in str(path)]
    steps = {path.parent: Report.from_document(read_json_strict(path)).failed_step for path in reports}
    entries = [
        {"run_dir": str(run_dir.relative_to(session_dir)), "failed_step": step.value}
        for run_dir, step in steps.items()
        if step in _SERVER_DIED_STEPS or step is FailedStep.TEARDOWN_INCOMPLETE
    ]
    if not entries:
        return {"reports": len(reports)}
    return {
        "reports": len(reports),
        "server_died": sum(1 for step in steps.values() if step in _SERVER_DIED_STEPS),
        "server_outlived_teardown": sum(1 for step in steps.values() if step is FailedStep.TEARDOWN_INCOMPLETE),
        "entries": entries,
    }


def write_capture_failure_diagnosis(path: Path, payload: dict[str, Any]) -> str:
    """Write the capture-failure evidence file; return its path, or ``""`` when it could not be written.

    A failure to write the evidence must not displace the capture failure it describes, so this degrades to an
    empty path and lets the caller's message stand on its own.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("roofline: could not write capture-failure diagnosis to %s: %r", path, exc)
        return ""
    return str(path)
