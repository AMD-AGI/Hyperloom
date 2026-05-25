"""Top-level builder for ``session_breakdown.json``.

Three entrypoints share this builder:

* CLI script (``scripts/dump_session_breakdown.py``) — offline / historical
* Coordinator action (``action_executors/session_breakdown.py``) — agent-driven
* ``cli.py`` finally block — end-of-session safety net

All three call :func:`build` (pure: no side effects) or
:func:`write_breakdown_json` (atomic ``state.json``-style write).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from . import collectors
from .schema import SCHEMA_VERSION

log = logging.getLogger(__name__)

EXPORTER_VERSION = "session-breakdown-1.0.0"
BREAKDOWN_FILENAME = "session_breakdown.json"


def _load_state(
    session_dir: Path,
    warnings: list[str],
) -> tuple[dict[str, Any], bool]:
    """Read ``state.json`` as a plain dict.

    Falls back to an empty dict (recorded as warning) so collectors can
    still surface manifest-only metadata if state is missing.

    Returns ``(state_dict, present_flag)``. ``present_flag`` is True iff
    ``state.json`` exists *and* parsed successfully.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        warnings.append(f"state.json missing at {state_path}")
        return {}, False
    try:
        return json.loads(state_path.read_text(encoding="utf-8")), True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse state.json: {exc!r}")
        return {}, False


def _load_manifest(
    session_dir: Path,
    warnings: list[str],
) -> tuple[dict[str, Any], bool]:
    """Read ``manifest.json`` as a plain dict.

    Returns ``(manifest_dict, present_flag)``. ``present_flag`` is True
    iff ``manifest.json`` exists *and* parsed successfully.
    """
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        warnings.append(f"manifest.json missing at {manifest_path}")
        return {}, False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")), True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse manifest.json: {exc!r}")
        return {}, False


def _coverage_label(state_present: bool, manifest_present: bool) -> str:
    """Map the two presence flags to the documented coverage label."""
    if state_present and manifest_present:
        return "full"
    if state_present or manifest_present:
        return "partial"
    return "shell_only"


def build(session_dir: Path | str, *, detail_level: str = "standard") -> dict[str, Any]:
    """Build a complete :class:`SessionBreakdown` for ``session_dir``.

    Pure function — reads from disk, never mutates state.

    Args:
        session_dir: absolute path to a hyperloom session directory.
                     The directory MUST contain at least ``manifest.json``
                     or ``state.json`` for any usable output; a totally
                     empty dir returns mostly-empty sections with warnings.

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    # Load state + manifest into a *private* warnings buffer first so we
    # can decide post-hoc whether to surface the "missing" lines or
    # consolidate them into a single ``coverage: shell_only`` marker.
    # Doing it this way keeps the partial-coverage warnings explicit
    # (e.g. only manifest missing → one informative line) while
    # collapsing the dual-missing case (post-orchestrator output dirs
    # with no session state) into a single low-noise summary.
    load_warnings: list[str] = []
    state, state_present = _load_state(sd, load_warnings)
    manifest, manifest_present = _load_manifest(sd, load_warnings)
    coverage = _coverage_label(state_present, manifest_present)

    warnings: list[str] = []
    if coverage == "shell_only":
        warnings.append(
            "coverage: shell_only — neither state.json nor manifest.json "
            "found; emitted payload is best-effort filesystem walk only"
        )
    else:
        warnings.extend(load_warnings)

    from datetime import datetime, timezone
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── Section collectors (each catches its own errors via warnings) ──
    session_meta      = _safe_collect("session",
                                       lambda: collectors.collect_session(sd, state, manifest, warnings),
                                       warnings)
    workload          = _safe_collect("workload",
                                       lambda: collectors.collect_workload(state, manifest, warnings),
                                       warnings)
    baseline          = _safe_collect("baseline",
                                       lambda: collectors.collect_baseline(sd, state, warnings),
                                       warnings)
    final             = _safe_collect("final",
                                       lambda: collectors.collect_final(sd, state, warnings),
                                       warnings)
    phase_timeline    = _safe_collect("phase_timeline",
                                       lambda: collectors.collect_phase_timeline(state, warnings, sd),
                                       warnings)
    # Back-fill session timing + closing-event duration once both
    # session_meta and phase_timeline are available. ``collect_session``
    # runs before phase_timeline and therefore can't derive the
    # "session_started/ended from phase_timeline" fallback or fill the
    # closing event's duration_seconds against the session end. We do
    # that here so the two sections stay consistent.
    try:
        collectors.enrich_session_and_timeline(session_meta, phase_timeline, state)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"enrich_session_and_timeline failed: {type(exc).__name__}: {exc}"
        )
    geak_invocations, oob_invocations = _safe_collect(
        "invocations",
        lambda: collectors.collect_kernel_invocations(sd, warnings),
        warnings,
        default=([], []),
    )
    capability_summary = _safe_collect("capability_summary",
                                        lambda: collectors.collect_capability_summary(
                                            state, geak_invocations, oob_invocations, warnings,
                                        ),
                                        warnings)
    kernel_lifecycle   = _safe_collect("kernel_lifecycle",
                                        lambda: collectors.collect_kernel_lifecycle(
                                            sd, state, geak_invocations, oob_invocations, warnings,
                                        ),
                                        warnings)
    param_search       = _safe_collect("param_search",
                                        lambda: collectors.collect_param_search(state, warnings),
                                        warnings)
    sweep              = _safe_collect("sweep",
                                        lambda: collectors.collect_sweep(sd, state, warnings),
                                        warnings)
    critic_robustness  = _safe_collect("critic_robustness",
                                        lambda: collectors.collect_critic_robustness(sd, warnings),
                                        warnings)
    telemetry          = _safe_collect("telemetry",
                                        lambda: collectors.collect_telemetry(sd, state, warnings),
                                        warnings)
    attribution        = _safe_collect("attribution",
                                        lambda: collectors.collect_attribution(
                                            state, geak_invocations, oob_invocations,
                                            kernel_lifecycle.get("adopted") or [],
                                            warnings,
                                        ),
                                        warnings)
    decision_journal     = _safe_collect(
        "decision_journal",
        lambda: collectors.collect_decision_journal(
            sd, state, warnings, detail_level=detail_level,
        ),
        warnings,
        default=[],
    )
    kernel_profiling     = _safe_collect(
        "kernel_profiling",
        lambda: collectors.collect_kernel_profiling(sd, state, warnings),
        warnings,
        default=[],
    )
    kernel_decision_path = _safe_collect(
        "kernel_decision_path",
        lambda: collectors.collect_kernel_decision_path(state, warnings, sd),
        warnings,
        default=[],
    )
    roofline             = _safe_collect(
        "roofline",
        lambda: collectors.collect_roofline(sd, warnings),
        warnings,
        default=[],
    )

    # Pre-assemble the breakdown payload so ``collect_data_provenance``
    # can decide each section's ``populated`` flag without re-deriving
    # the section dicts from disk. The provenance collector only reads
    # the dict values + on-disk artifact existence; it never mutates the
    # payload it receives.
    breakdown_so_far: dict[str, Any] = {
        "session":              session_meta,
        "workload":             workload,
        "baseline":             baseline,
        "final":                final,
        "phase_timeline":       phase_timeline,
        "capability_summary":   capability_summary,
        "geak_invocations":     geak_invocations,
        "oob_invocations":      oob_invocations,
        "kernel_lifecycle":     kernel_lifecycle,
        "param_search":         param_search,
        "sweep":                sweep,
        "critic_robustness":    critic_robustness,
        "telemetry":            telemetry,
        "attribution":          attribution,
        "decision_journal":     decision_journal,
        "kernel_profiling":     kernel_profiling,
        "kernel_decision_path": kernel_decision_path,
        "roofline":             roofline,
    }
    data_provenance      = _safe_collect(
        "data_provenance",
        lambda: collectors.collect_data_provenance(sd, breakdown_so_far, warnings),
        warnings,
        default=[],
    )

    source_files = collectors.collect_source_files(
        sd,
        baseline.get("benchmark_report_path"),
        telemetry.get("profile_report_paths") or [],
        [p.get("benchmark_report_path") for p in (sweep.get("all_variants") or [])
         if p.get("benchmark_report_path")],
    )

    # For ``shell_only`` coverage there is no actionable signal beyond
    # the single coverage marker — every other warning emitted by the
    # downstream collectors (image not configured, server.log missing,
    # framework_args extraction failure, …) is a direct consequence of
    # the absent state/manifest. Suppress them to keep the breakdown
    # readable; consumers still know full diagnostic context from
    # ``coverage`` itself.
    if coverage == "shell_only":
        warnings = [w for w in warnings if w.startswith("coverage:")]

    return {
        "schema_version":      SCHEMA_VERSION,
        "exported_at_utc":     exported_at,
        "exporter_version":    EXPORTER_VERSION,
        "detail_level":        detail_level,
        "coverage":            coverage,

        "session":             session_meta,
        "workload":            workload,
        "baseline":            baseline,
        "final":               final,
        "phase_timeline":      phase_timeline,
        "capability_summary":  capability_summary,
        "geak_invocations":    geak_invocations,
        "oob_invocations":     oob_invocations,
        "kernel_lifecycle":    kernel_lifecycle,
        "param_search":        param_search,
        "sweep":               sweep,
        "critic_robustness":   critic_robustness,
        "telemetry":           telemetry,
        "attribution":         attribution,
        "decision_journal":    decision_journal,
        "kernel_profiling":    kernel_profiling,
        "kernel_decision_path": kernel_decision_path,
        "roofline":            roofline,
        "data_provenance":     data_provenance,

        "warnings":            warnings,
        "source_files":        source_files,
    }


def _safe_collect(
    name: str,
    fn: callable,
    warnings: list[str],
    *,
    default: Any = None,
):
    """Run a collector with broad exception catching.

    A bug in one collector must never poison the whole export. Each
    failure becomes a warning entry; the section becomes ``default``
    (an empty dict / list).
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.exception("collector %s failed", name)
        warnings.append(f"collector:{name} failed: {type(exc).__name__}: {exc}")
        if default is not None:
            return default
        return {}


def write_breakdown_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    detail_level: str = "standard",
) -> Path:
    """Build + atomically write ``session_breakdown.json``.

    Args:
        session_dir: hyperloom session directory.
        output_path: override target path (defaults to
                     ``<session_dir>/session_breakdown.json``).

    Returns:
        Absolute path to the written JSON file.
    """
    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else sd / BREAKDOWN_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    breakdown = build(sd, detail_level=detail_level)
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
    """Stringify objects json.dumps can't handle natively (Path, set, ...)."""
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
