"""Top-level builder for ``session_breakdown.json``.

Entry points:
  - CLI finally block (end-of-session safety net)
  - Orchestrator action (agent-driven export)
  - Offline script (post-mortem / historical)

All call :func:`build` (pure, read-only) or :func:`write_breakdown_json`
(atomic write to disk).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import collectors
from .schema import SCHEMA_VERSION

log = logging.getLogger(__name__)

EXPORTER_VERSION = "hyperloom-breakdown-1.0.0"
BREAKDOWN_FILENAME = "session_breakdown.json"


def _load_state(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read state.json as a plain dict. Falls back to empty with warning."""
    state_path = session_dir / "state.json"
    if not state_path.exists():
        warnings.append(f"state.json missing at {state_path}")
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to parse state.json: {exc!r}")
        return {}


def _load_manifest(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read session manifest (top-level manifest.json)."""
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        warnings.append(f"manifest.json missing at {manifest_path}")
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to parse manifest.json: {exc!r}")
        return {}


def build(
    session_dir: Path | str,
    *,
    include_agent_logs: bool = False,
) -> dict[str, Any]:
    """Build a complete SessionBreakdown for ``session_dir``.

    Pure function — reads from disk, never mutates state.

    Args:
        session_dir: absolute path to a hyperloom session directory.
        include_agent_logs: when True, inlines process.log content for
            each agent (large; default False).

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    warnings: list[str] = []

    state = _load_state(sd, warnings)
    manifest = _load_manifest(sd, warnings)

    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    session_meta = _safe("session",
                         lambda: collectors.collect_session(sd, state, manifest, warnings),
                         warnings)
    workload = _safe("workload",
                     lambda: collectors.collect_workload(state, manifest, warnings),
                     warnings)
    baseline = _safe("baseline",
                     lambda: collectors.collect_baseline(sd, state, warnings),
                     warnings)
    final = _safe("final",
                  lambda: collectors.collect_final(sd, state, warnings),
                  warnings)
    agent_timeline = _safe("agent_timeline",
                           lambda: collectors.collect_agent_timeline(sd, state, warnings),
                           warnings, default=[])
    capability_summary = _safe("capability_summary",
                               lambda: collectors.collect_capability_summary(
                                   sd, state, agent_timeline, warnings),
                               warnings)
    geak_invocations, oob_invocations = _safe(
        "kernel_invocations",
        lambda: collectors.collect_kernel_invocations(sd, warnings),
        warnings, default=([], []),
    )
    kernel_lifecycle = _safe("kernel_lifecycle",
                             lambda: collectors.collect_kernel_lifecycle(
                                 sd, state, geak_invocations, oob_invocations, warnings),
                             warnings)
    profiling = _safe("profiling",
                      lambda: collectors.collect_profiling(sd, state, warnings),
                      warnings)
    sweep = _safe("sweep",
                  lambda: collectors.collect_sweep(sd, state, warnings),
                  warnings)
    watchdog_events = _safe("watchdog_events",
                            lambda: collectors.collect_watchdog_events(sd, warnings),
                            warnings, default=[])
    attribution = _safe("attribution",
                        lambda: collectors.collect_attribution(
                            state, geak_invocations, oob_invocations, warnings),
                        warnings)
    telemetry = _safe("telemetry",
                      lambda: collectors.collect_telemetry(sd, state, warnings),
                      warnings)
    source_files = _safe("source_files",
                         lambda: collectors.collect_source_files(sd, warnings),
                         warnings)

    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at_utc": exported_at,
        "exporter_version": EXPORTER_VERSION,

        "session": session_meta,
        "workload": workload,
        "baseline": baseline,
        "final": final,
        "agent_timeline": agent_timeline,
        "capability_summary": capability_summary,
        "geak_invocations": geak_invocations,
        "oob_invocations": oob_invocations,
        "kernel_lifecycle": kernel_lifecycle,
        "profiling": profiling,
        "sweep": sweep,
        "watchdog_events": watchdog_events,
        "attribution": attribution,
        "telemetry": telemetry,
        "source_files": source_files,
        "warnings": warnings,
    }


def _safe(
    name: str,
    fn: Any,
    warnings: list[str],
    *,
    default: Any = None,
) -> Any:
    """Run a collector with broad exception catching.

    A bug in one collector must never poison the whole export.
    """
    try:
        return fn()
    except Exception as exc:
        log.exception("collector %s failed", name)
        warnings.append(f"collector:{name} failed: {type(exc).__name__}: {exc}")
        return default if default is not None else {}


def write_breakdown_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    include_agent_logs: bool = False,
) -> Path:
    """Build + atomically write ``session_breakdown.json``.

    Returns absolute path to the written JSON file.
    """
    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else sd / BREAKDOWN_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    breakdown = build(sd, include_agent_logs=include_agent_logs)
    payload = json.dumps(breakdown, indent=2, sort_keys=True, default=_json_default)

    fd, tmp = tempfile.mkstemp(
        prefix=f".{BREAKDOWN_FILENAME}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    log.info("session_breakdown: wrote %s (%d bytes)", target, len(payload))
    return target


def _json_default(obj: Any) -> Any:
    """Stringify objects json.dumps can't handle natively."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "build",
    "write_breakdown_json",
]
