# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level builder for ``session_breakdown.json``."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_bool
from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom.common.jsonio import read_json

from . import collectors
from .recorder.event_finalize import finalize_events
from .recorder.recorder_warnings import RECORDING_ERRORS
from .schema import SCHEMA_VERSION_V6
from ..session.session_paths import BREAKDOWN_FILENAME, manifest_path, state_path

log = logging.getLogger(__name__)

EXPORTER_VERSION = "session-breakdown-1.0.0"


def _recorded_session_value(value: Any) -> bool:
    """Whether a recorder ``session`` field carries evidence.

    Only an absent field counts as no evidence. A recorded zero is a fact the
    recorder observed -- a session really can have run zero ticks -- so it must
    win over the collector rebuild rather than read as a missing value.
    """
    return value is not None and value != ""


_COLLECTOR_OWNED_SESSION = frozenset({"stop_reason", "ended_at_utc", "elapsed_minutes"})


def _merge_session(fragment: Any, collector_value: Any) -> Any:
    """Overlay the recorder's live ``session`` fields on the collected section.

    Lifecycle fields the collector derives from CLOSE and timestamps stay
    collector-owned when the collector actually produced a value. A degraded
    collect (``None`` / empty) keeps the recorder fragment.
    """
    if not isinstance(fragment, dict) or not fragment:
        return collector_value
    merged = dict(collector_value) if isinstance(collector_value, dict) else {}
    for key, value in fragment.items():
        if key in _COLLECTOR_OWNED_SESSION and _recorded_session_value(merged.get(key)):
            continue
        if _recorded_session_value(value) or key not in merged:
            merged[key] = value
    return merged


def _load_session_json(path: Path, label: str, warnings: list[str]) -> dict[str, Any]:
    """Read a session JSON file as a dict; ``{}`` + warning on failure."""
    if not path.exists():
        warnings.append(f"{label} missing at {path}")
        return {}
    try:
        return read_json(path, require_dict=True, strict=True)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse {label}: {exc!r}")
        return {}


def build(session_dir: Path | str) -> dict[str, Any]:
    """Build a complete :class:`SessionBreakdown` for ``session_dir``.

    Not pure: before the timeline is read this closes any event whose phase was
    killed before it could close itself, which writes those fragments back to
    the spool. The close is idempotent -- the first build settles the orphans
    and every later build over the same session sees the same closed events --
    so repeated builds still return the same breakdown.

    Args:
        session_dir: hyperloom session directory (needs ``manifest.json``
            or ``state.json`` for usable output).

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    warnings: list[str] = []

    state = _load_session_json(state_path(sd), "state.json", warnings)
    manifest = _load_session_json(manifest_path(sd), "manifest.json", warnings)

    # Author-time recorder fragments (write-side spool).
    assembled = _load_assembled(sd, warnings)
    # V6 is the hard-cutover wire shape regardless of whether recorder fragments or collector fallbacks supplied the
    # underlying evidence.
    schema_version = SCHEMA_VERSION_V6

    from datetime import datetime, timezone

    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Section collectors (each catches its own errors via warnings).
    session_section = _merge_session(
        assembled.get("session"),
        _safe_collect("session", lambda: collectors.collect_session(sd, state, manifest, warnings), warnings),
    )
    # Task configuration, read from the launch inputs rather than recorded:
    # both are authored before the recorder exists.
    workload = _safe_collect("workload", lambda: collectors.collect_workload(state, manifest, warnings), warnings)
    # Model basics -- verbatim mirror of ``state.model_info``. Empty {} on
    # non-transformers models.
    model_info = _safe_collect(
        "model_info", lambda: collectors.collect_model_info(state, warnings), warnings, default={}
    )
    # Live-Langfuse push receipt (opt-in second sink). Prefers the post-flush
    # ``langfuse_receipt.json``; falls back to a live emitter read. Authored by
    # the emitter, so it is read here rather than derived.
    langfuse = _safe_collect(
        "langfuse",
        lambda: collectors.collect_langfuse(
            sd,
            manifest,
            warnings,
        ),
        warnings,
        default={},
    )
    # Events whose phase was killed before it could close them are closed here,
    # before the timeline is read: their fragments are on disk, and an event
    # left open would otherwise be read back as still running. Programming
    # errors in finalize are left to raise; spool failures are a dedicated
    # warning rather than a generic collector miss.
    try:
        closed = finalize_events(sd)
    except RECORDING_ERRORS as exc:
        warnings.append(f"timeline_finalize: {type(exc).__name__}: {exc}")
        closed = []
    if closed:
        log.info("timeline_finalize: closed %s orphan event(s)", len(closed))
    timeline = _safe_collect(
        "timeline",
        lambda: collectors.collect_v6_timeline(sd, warnings),
        warnings,
        default=[],
    )
    # Before ``outcome``, which reads the recipe the close-out settled: the
    # session's terminal configuration is only final once the stack has stopped
    # changing, and the close-out is what states it.
    v6_close = _safe_collect(
        "close",
        lambda: collectors.collect_v6_close(
            warnings,
            recorded=assembled.get("close"),
        ),
        warnings,
        default={},
    )
    outcome = _safe_collect(
        "outcome",
        lambda: collectors.collect_v6_outcome(
            session=session_section,
            close=v6_close,
            state=state,
            timeline=timeline,
            warnings=warnings,
        ),
        warnings,
        default={},
    )
    metadata = _safe_collect(
        "metadata",
        lambda: collectors.collect_v6_metadata(
            exported_at_utc=exported_at,
            session=session_section,
            workload=workload,
            model_info=model_info,
            langfuse=langfuse,
            state=state,
            warnings=warnings,
            recorded=assembled.get("metadata"),
        ),
        warnings,
        default={},
    )
    v6_critic = collectors.collect_v6_critic(assembled.get("critic"))
    v6_robustness = collectors.collect_v6_robustness(assembled.get("robustness"))
    # Snapshot last: every collector above feeds this one list, and this is the
    # single place a collection failure surfaces. An export used to also carry
    # a top-level copy taken partway through, which was a strict subset and so
    # disagreed with this one about how the export had gone.
    if isinstance(metadata, dict):
        metadata["warnings"] = list(warnings)

    breakdown = {
        "schema_version": schema_version,
        "exported_at_utc": exported_at,
        "exporter_version": EXPORTER_VERSION,
        "metadata": metadata,
        "outcome": outcome,
        "timeline": timeline,
        "close": v6_close,
        "critic": v6_critic,
        "robustness": v6_robustness,
    }
    return breakdown


def _load_assembled(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble recorder fragments into ``{section: value}`` (empty on opt-out or when no fragments exist)."""
    if env_bool("INFERENCE_OPTIMIZER_BREAKDOWN_DISABLE_RECORDER"):
        warnings.append("recorder: disabled by INFERENCE_OPTIMIZER_BREAKDOWN_DISABLE_RECORDER")
        return {}
    try:
        from .recorder import assemble_parts, has_parts

        if not has_parts(session_dir):
            return {}
        out = assemble_parts(session_dir, warnings=warnings)
        return out if isinstance(out, dict) else {}
    except RECORDING_ERRORS as exc:
        log.exception("recorder: assemble_parts failed")
        warnings.append(f"recorder: assemble_parts failed: {type(exc).__name__}: {exc}")
        return {}


def _safe_collect(
    name: str,
    fn: callable,
    warnings: list[str],
    *,
    default: Any = None,
):
    """Run a collector; any failure becomes a warning plus ``default``.

    Collectors are allowed to see drifted session artifacts, so an unexpected
    shape must not abort the rest of the export. The assembler and event
    finalize paths are narrower: they catch :data:`RECORDING_ERRORS` only.
    """
    try:
        return fn()
    except Exception as exc:
        log.exception("collector %s failed", name)
        warnings.append(f"collector:{name} failed: {type(exc).__name__}: {exc}")
        if default is not None:
            return default
        return {}


def write_breakdown_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Build + atomically write ``session_breakdown.json``; returns the absolute path.

    Also triggers ``reports/trace/decision_trace.jsonl``, which is not part of
    the breakdown (see
    :func:`hyperloom.inference_optimizer.trace.decision_trace.write_session_decision_trace`)
    but is produced from here because every caller that flushes Langfuse goes
    through this function.

    ``output_path`` defaults to ``<session_dir>/session_breakdown.json``.

    Args:
        session_dir: The hyperloom session directory to build from.
        output_path: Destination file; defaults to
            ``<session_dir>/session_breakdown.json``.

    Returns:
        The absolute path of the written breakdown file.
    """
    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else sd / BREAKDOWN_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    # The decision trace is not part of the breakdown -- nothing here reads it,
    # and it lives with the Langfuse emitter and the backfill tool that score a
    # session from it. Only the trigger is here, because every path that writes
    # a session's artifacts goes through this function and the file has to land
    # before any flush reads it.
    try:
        from ..trace.decision_trace import write_session_decision_trace

        for warning in write_session_decision_trace(sd):
            log.debug("decision_trace: %s", warning)
    except Exception:
        # The trace is a side artifact; losing it must not fail the breakdown
        # write, but it is scored downstream so a silent loss has to be visible.
        log.warning("decision_trace write failed for %s; trace artifacts will be missing", sd, exc_info=True)

    breakdown = build(sd)
    payload = json.dumps(breakdown, indent=2, sort_keys=True, default=_json_default)
    atomic_write_text(target, payload)
    log.info("session_breakdown: wrote %s (%d bytes)", target, len(payload))
    return target


def _patch_breakdown(
    session_dir: Path | str,
    section: str,
    revise: Callable[[Path, dict[str, Any]], bool],
) -> bool:
    """Rewrite one section of an already-written breakdown, atomically."""
    sd = Path(session_dir).resolve()
    target = sd / BREAKDOWN_FILENAME
    try:
        if not target.exists():
            return False
        breakdown = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(breakdown, dict):
            return False
        if not revise(sd, breakdown):
            return False
        atomic_write_text(target, json.dumps(breakdown, indent=2, sort_keys=True, default=_json_default))
        log.info("session_breakdown: refreshed %s section in %s", section, target)
        return True
    except Exception:
        log.debug("session_breakdown: %s patch failed (non-fatal)", section, exc_info=True)
        return False


def patch_breakdown_langfuse(session_dir: Path | str) -> bool:
    """Refresh only ``metadata.langfuse`` in an already-written breakdown.

    ``session_breakdown.json`` is written *before* the session-end
    ``flush_session`` (the flush depends on ``decision_trace.jsonl``, which the
    breakdown produces). So the breakdown's first Langfuse block carries the
    pre-flush, in-process counts. Call this right after ``flush_session`` to
    splice in the post-flush ``langfuse_receipt.json`` (final counts) without
    rebuilding the whole file.

    Best-effort and self-skipping: returns False (no-op) when no breakdown or
    no receipt exists yet, when live push was disabled, or on any error. Never
    raises -- it must not mask the session's stop_reason at shutdown.

    Args:
        session_dir: The hyperloom session directory holding the breakdown.

    Returns:
        ``True`` when the langfuse block was refreshed, ``False`` otherwise.
    """
    from ..trace.langfuse_emitter import read_receipt

    def _revise(sd: Path, breakdown: dict[str, Any]) -> bool:
        receipt = read_receipt(sd)
        if receipt is None:
            return False
        metadata = breakdown.get("metadata")
        if not isinstance(metadata, dict):
            return False
        block = collectors.langfuse_block(receipt)
        if metadata.get("langfuse") == block:
            return False  # already current
        metadata["langfuse"] = block
        return True

    return _patch_breakdown(session_dir, "langfuse", _revise)


def patch_breakdown_close(session_dir: Path | str) -> bool:
    """Refresh only the ``close`` section of an already-written breakdown.

    ``session_breakdown`` is a step in the middle of the CLOSE sequencer, so
    the breakdown it writes can only ever describe the close-out up to its own
    step: the steps after it have not run, and no verdict has been recorded.
    Left alone, every healthy session reports ``close.status: "running"``,
    which reads as a session that died during its own wind-down.

    Call this as the last act of the sequencer, after
    ``record_close_settled``. Every close step persists as it settles, so by
    then the full sequence, the artifact paths and the verdict are all on disk
    and the re-read key is the real one.

    Best-effort and self-skipping, exactly like
    :func:`patch_breakdown_langfuse`: returns False on a missing breakdown, an
    unchanged section, or any error. Never raises — it runs after
    ``stop_reason`` and ``close_sequence_done`` are settled and must not mask
    them at shutdown.

    Args:
        session_dir: The hyperloom session directory holding the breakdown.

    Returns:
        ``True`` when the close section was refreshed, ``False`` otherwise.
    """

    def _revise(sd: Path, breakdown: dict[str, Any]) -> bool:
        # A V5-only breakdown has no ``close`` key to refresh, and adding one would change the surface of a payload
        # that never carried it.
        if "close" not in breakdown:
            return False

        fresh_warnings: list[str] = []
        # Re-assembled rather than reused from the export: this pass runs after
        # the sequencer's last act, so the fragments now carry the verdict and
        # the artifact paths that did not exist when the breakdown was written.
        fresh = collectors.collect_v6_close(
            fresh_warnings,
            recorded=_load_assembled(sd, fresh_warnings).get("close"),
        )
        changed = breakdown.get("close") != fresh
        breakdown["close"] = fresh

        # This pass is the only one that ever sees the steps recorded *after* the breakdown was written —
        # ``artifact_package``, ``ndjson_drain``, ``done`` — so drift among them is reported here or nowhere.
        metadata = breakdown.get("metadata")
        if isinstance(metadata, dict) and fresh_warnings:
            existing = [str(row) for row in metadata.get("warnings") or []]
            merged = existing + [row for row in fresh_warnings if row not in existing]
            if merged != existing:
                metadata["warnings"] = merged
                changed = True
        return changed

    return _patch_breakdown(session_dir, "close", _revise)


def _json_default(obj: Any) -> Any:
    """Stringify objects json.dumps can't handle natively (Path, set, ...)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_minimal_final_report(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """cli.finally safety-net for ``reports/final.md`` when the CLOSE sequencer never reached step 1."""
    from ..session.session_paths import reports_dir

    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else reports_dir(sd) / "final.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    state = read_json(state_path(sd), default={}, require_dict=True)
    breakdown_link = sd / BREAKDOWN_FILENAME

    def _fmt_attempt(d: dict[str, Any] | None, label: str) -> str:
        """Format one ``last_*`` attempt record as a markdown bullet."""
        if not isinstance(d, dict) or not d:
            return f"- **{label}**: (none)"
        ts = d.get("ts") or "-"
        body = json.dumps(
            {k: v for k, v in d.items() if k != "ts"},
            sort_keys=True,
            default=str,
        )[:600]
        return f"- **{label}** (`{ts}`): `{body}`"

    from .. import framework_registry

    framework = str(state.get("framework") or "")
    baseline_tput = state.get("baseline_tput") or 0.0
    current_best = state.get("current_best") or {}
    cb_action = current_best.get("action") or "-"
    cb_tput = current_best.get("tput")
    # Framework-aware primary metric: serving shows tok/s/GPU, scriptable xDiT shows per-image latency e2el_mean_ms
    # (ms).
    baseline_metric_s = framework_registry.format_primary_metric(framework, baseline_tput, precision=2)
    cb_metric_s = (
        framework_registry.format_primary_metric(framework, cb_tput, precision=2)
        if isinstance(cb_tput, (int, float))
        else "-"
    )
    lines = [
        "# Inference Optimizer — emergency final report",
        "",
        "> **Auto-generated safety-net.** The CLOSE phase 7-step "
        + "sequencer did not run to completion (process exited before "
        + "phase transition, or ``report`` executor failed). For the "
        + "full audit trail open `session_breakdown.json` next to this "
        + "file.",
        "",
        f"- session_id     : `{state.get('session_id') or '-'}`",
        f"- model_path     : `{state.get('model_path') or '-'}`",
        f"- framework      : `{framework or '-'}`",
        f"- gpu_type       : `{state.get('gpu_type') or '-'}`",
        f"- phase (last)   : `{state.get('phase') or '-'}`",
        f"- stop_reason    : `{state.get('stop_reason') or '-'}`",
        f"- baseline       : `{baseline_metric_s}`",
        f"- current_best   : `{cb_action}` @ `{cb_metric_s}`",
        f"- cumul_gain     : `{state.get('cumulative_gain_validated') or 0.0:.2f}%` (validated)",
        f"- stack_entries  : `{len(state.get('optimization_stack') or [])}`",
        "",
        "## Last action attempts",
        "",
        _fmt_attempt(state.get("last_baseline"), "last_baseline"),
        _fmt_attempt(state.get("last_profile"), "last_profile"),
        _fmt_attempt(state.get("last_explore"), "last_explore"),
        "",
        "## Structured detail",
        "",
        f"See `{breakdown_link.name}` (sibling of session root) for the "
        f"complete `timeline` / `critic` / `robustness` blocks.",
        "",
    ]

    atomic_write_text(target, "\n".join(lines))
    log.info("emergency final report: wrote %s", target)
    return target


def _crash_safe_platform(gpu_type: str | None) -> dict[str, Any]:
    """Platform record for the crash-safe path."""
    from hyperloom.common.platform_probe import platform_fingerprint

    return platform_fingerprint(gpu_type)


#: Who wrote a crash-safe ``final.json``. A producer may replace a fallback
#: written by itself or by a lower-ranked producer, never the full
#: ``ReportExecutor`` output. The supervisor outranks the coordinator: it
#: writes only after observing the coordinator's process end.
FINAL_PRODUCER_COORDINATOR = "coordinator"
FINAL_PRODUCER_SUPERVISOR = "supervisor"
_FINAL_PRODUCER_RANK: dict[str, int] = {
    FINAL_PRODUCER_COORDINATOR: 1,
    FINAL_PRODUCER_SUPERVISOR: 2,
}


def write_minimal_final_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    producer: str = FINAL_PRODUCER_COORDINATOR,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Crash-safe ``reports/final.json`` fallback for any non-graceful exit."""
    from datetime import datetime, timezone

    from ..session.session_paths import reports_dir

    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else reports_dir(sd) / "final.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep a full report (``safety_net`` absent/false) and a fallback from a
    # higher-ranked producer; refresh a fallback from an equal or lower-ranked
    # one; preserve a corrupt file as ``final.json.corrupt`` and overwrite it.
    rank = _FINAL_PRODUCER_RANK.get(producer, 1)
    if target.exists() and target.stat().st_size > 0:
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            overwrite = isinstance(existing, dict) and existing.get("safety_net") is True
            if overwrite:
                held = _FINAL_PRODUCER_RANK.get(existing.get("producer"), 1)
                overwrite = rank >= held
        except (OSError, json.JSONDecodeError):
            try:
                target.replace(target.with_name(target.name + ".corrupt"))
            except OSError:
                log.warning("crash-safe final.json: could not back up corrupt %s", target)
            overwrite = True
        if not overwrite:
            return target

    state = read_json(state_path(sd), default={}, require_dict=True)
    gpu_type = state.get("gpu_type") or ""
    summary: dict[str, Any] = {
        # Crash-safe markers: a consumer can distinguish this from the full ReportExecutor output and know the run did
        # not finish gracefully.
        "safety_net": True,
        "report_complete": False,
        "producer": producer,
        "session_id": state.get("session_id") or "",
        "model_name": state.get("model_name") or "",
        "model_path": state.get("model_path") or "",
        "model_class": state.get("model_class") or "",
        "framework": state.get("framework") or "",
        "gpu_type": gpu_type,
        "phase": state.get("phase") or "",
        "stop_reason": state.get("stop_reason") or "",
        "baseline_tput": state.get("baseline_tput") or 0.0,
        "baseline_accuracy": state.get("baseline_accuracy") or 0.0,
        "current_best": state.get("current_best") or {},
        "cumulative_gain_validated": state.get("cumulative_gain_validated") or 0.0,
        "optimization_stack_len": len(state.get("optimization_stack") or []),
        "crash_count": state.get("crash_count") or 0,
        "max_minutes": state.get("max_minutes") or 0,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        # A run that died unattended is exactly when the host record is most useful, since nobody was watching.
        "platform": _crash_safe_platform(gpu_type),
    }
    if extra:
        summary.update(extra)

    atomic_write_json(target, summary)
    log.info("crash-safe final.json: wrote %s", target)
    return target


__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "FINAL_PRODUCER_COORDINATOR",
    "FINAL_PRODUCER_SUPERVISOR",
    "build",
    "write_breakdown_json",
    "write_minimal_final_json",
    "write_minimal_final_report",
]
