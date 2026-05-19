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


def _load_state(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read ``state.json`` as a plain dict.

    Falls back to an empty dict (recorded as warning) so collectors can
    still surface manifest-only metadata if state is missing.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        warnings.append(f"state.json missing at {state_path}")
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse state.json: {exc!r}")
        return {}


def _load_manifest(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        warnings.append(f"manifest.json missing at {manifest_path}")
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse manifest.json: {exc!r}")
        return {}


def build(
    session_dir: Path | str,
    *,
    include_transcripts: bool | None = None,
) -> dict[str, Any]:
    """Build a complete :class:`SessionBreakdown` for ``session_dir``.

    Pure function — reads from disk, never mutates state.

    Args:
        session_dir: absolute path to a hyperloom session directory.
                     The directory MUST contain at least ``manifest.json``
                     or ``state.json`` for any usable output; a totally
                     empty dir returns mostly-empty sections with warnings.
        include_transcripts: when True, specialist transcripts are
            inlined under ``specialist_runs[i].transcripts[j].body``.
            When None (default) we consult the env var
            ``INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS=1`` so
            CLI / SDK / agent-action call sites converge through one
            switch (KB_design §3.12 §7 step 5). Defaults to False —
            transcripts are large and most dashboards prefer a path
            reference.

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    warnings: list[str] = []
    if include_transcripts is None:
        include_transcripts = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "",
            ).strip().lower() in ("1", "true", "yes")
        )

    state = _load_state(sd, warnings)
    manifest = _load_manifest(sd, warnings)

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
                                       lambda: collectors.collect_phase_timeline(state, warnings),
                                       warnings)
    phase_segments    = _safe_collect("phase_segments",
                                       lambda: collectors.collect_phase_segments(
                                           state, phase_timeline, warnings,
                                       ),
                                       warnings,
                                       default=[])
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
    kb_provenance      = _safe_collect("kb_provenance",
                                        lambda: collectors.collect_kb_provenance(
                                            session_dir, state, manifest, warnings,
                                        ),
                                        warnings)
    # v0.8 §3.12 §4.3 — specialist sub-agent dispatch records. Built
    # from ``state.specialist_rounds`` + the on-disk transcripts so
    # capability_summary.specialist and specialist_runs always agree
    # (Inv-12.2 single source).
    specialist_runs    = _safe_collect("specialist_runs",
                                        lambda: collectors.collect_specialist_runs(
                                            sd, state, warnings,
                                            include_transcripts=include_transcripts,
                                        ),
                                        warnings,
                                        default=[])

    source_files = collectors.collect_source_files(
        sd,
        baseline.get("benchmark_report_path"),
        telemetry.get("profile_report_paths") or [],
        [p.get("benchmark_report_path") for p in (sweep.get("all_variants") or [])
         if p.get("benchmark_report_path")],
    )

    return {
        "schema_version":      SCHEMA_VERSION,
        "exported_at_utc":     exported_at,
        "exporter_version":    EXPORTER_VERSION,

        "session":             session_meta,
        "workload":            workload,
        "baseline":            baseline,
        "final":               final,
        "phase_timeline":      phase_timeline,
        # v0.8 M2 — phase boundary segments with embedded action events
        # (KB_design §3.12 §4 "phase_timeline upgrade"). Additive: v1
        # readers keep using ``phase_timeline`` (flat); v2 readers
        # prefer ``phase_segments``.
        "phase_segments":      phase_segments,
        # v0.8 §3.12 §4.2 — top-level v1-reader alias: the flat
        # per-action timeline used to live under ``phase_timeline``
        # in v1. Mirrors the same list so an old reader picks it up
        # without code change.
        "action_timeline":     phase_timeline,
        "capability_summary":  capability_summary,
        "geak_invocations":    geak_invocations,
        "oob_invocations":     oob_invocations,
        "kernel_lifecycle":    kernel_lifecycle,
        "param_search":        param_search,
        # v0.8 §3.12 §5 — ``explore_search`` is the v2-native name for
        # the merged ledger (KB_design §3.4). Mirror of
        # ``param_search`` so v2 readers can switch with a one-line
        # rename + v1 readers don't break.
        "explore_search":      param_search,
        "sweep":               sweep,
        "critic_robustness":   critic_robustness,
        "telemetry":           telemetry,
        "attribution":         attribution,
        # v0.8 M1 — Cortex KB integration audit (KB_design §3.13 M1 §4
        # "kb_provenance"). Added as a new top-level section rather than
        # bumping ``schema_version`` because every field is optional; the
        # v1 reader simply ignores it.
        "kb_provenance":       kb_provenance,
        # v0.8 §3.12 §4.3 — specialist sub-agent dispatch records.
        "specialist_runs":     specialist_runs,

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
    include_transcripts: bool | None = None,
) -> Path:
    """Build + atomically write ``session_breakdown.json``.

    Args:
        session_dir: hyperloom session directory.
        output_path: override target path (defaults to
                     ``<session_dir>/session_breakdown.json``).
        include_transcripts: see :func:`build` — when None the
            ``INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS=1``
            env var (set by CLI ``--breakdown-include-transcripts``)
            decides.

    Returns:
        Absolute path to the written JSON file.
    """
    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else sd / BREAKDOWN_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    breakdown = build(sd, include_transcripts=include_transcripts)
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
