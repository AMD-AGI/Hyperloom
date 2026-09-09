# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level builder for ``session_breakdown.json``."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json

from . import collectors
from .recorder.event_finalize import finalize_events
from .schema import SCHEMA_VERSION_V6
from ..session.session_paths import manifest_path, state_path

log = logging.getLogger(__name__)

EXPORTER_VERSION = "session-breakdown-1.0.0"
BREAKDOWN_FILENAME = "session_breakdown.json"


def _recorded_session_value(value: Any) -> bool:
    """Whether a recorder ``session`` field carries evidence."""
    if value is None or value == "":
        return False
    return not (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


def _merge_session(fragment: Any, collector_value: Any) -> Any:
    """Overlay the recorder's live ``session`` fields on the collected section."""
    if not isinstance(fragment, dict) or not fragment:
        return collector_value
    merged = dict(collector_value) if isinstance(collector_value, dict) else {}
    for key, value in fragment.items():
        if _recorded_session_value(value) or key not in merged:
            merged[key] = value
    merged["elapsed_minutes"] = collectors.session_elapsed_minutes(merged)
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
    """Build a complete :class:`SessionBreakdown` for ``session_dir`` (pure; reads disk).

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
    v6_warnings = list(warnings)
    # Events whose phase was killed before it could close them are closed here, before the timeline is read: their
    # fragments are on disk, and an event left open would otherwise be read back as still running.
    _safe_collect("timeline_finalize", lambda: finalize_events(sd), v6_warnings, default=[])
    timeline = _safe_collect(
        "timeline",
        lambda: collectors.collect_v6_timeline(sd, v6_warnings),
        v6_warnings,
        default=[],
    )
    # Before ``outcome``, which reads the recipe the close-out settled: the
    # session's terminal configuration is only final once the stack has stopped
    # changing, and the close-out is what states it.
    v6_close = _safe_collect(
        "close",
        lambda: collectors.collect_v6_close(
            v6_warnings,
            recorded=assembled.get("close"),
            robustness=assembled.get("robustness"),
        ),
        v6_warnings,
        default={},
    )
    outcome = _safe_collect(
        "outcome",
        lambda: collectors.collect_v6_outcome(
            session=session_section,
            close=v6_close,
            state=state,
            timeline=timeline,
        ),
        v6_warnings,
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
            warnings=v6_warnings,
            recorded=assembled.get("metadata"),
        ),
        v6_warnings,
        default={},
    )
    v6_critic = _safe_collect(
        "critic",
        lambda: collectors.collect_v6_critic(assembled.get("critic")),
        v6_warnings,
        default={},
    )
    v6_robustness = _safe_collect(
        "robustness",
        lambda: collectors.collect_v6_robustness(assembled.get("robustness")),
        v6_warnings,
        default={},
    )
    # Snapshot last: every V6 collector above feeds this list, and it is the only place a V6 failure is allowed to
    # surface.
    if isinstance(metadata, dict):
        metadata["warnings"] = list(v6_warnings)

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
        "warnings": warnings,
    }
    return breakdown


def _load_assembled(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble recorder fragments into ``{section: value}`` (empty on opt-out or when no fragments exist)."""
    disabled = os.environ.get(
        "INFERENCE_OPTIMIZER_BREAKDOWN_DISABLE_RECORDER",
        "",
    ).strip().lower() in ("1", "true", "yes")
    if disabled:
        return {}
    try:
        from .recorder import assemble_parts, has_parts

        if not has_parts(session_dir):
            return {}
        out = assemble_parts(session_dir, warnings=warnings)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
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
    """Run a collector with broad exception catching; failure → warning + ``default``."""
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
) -> Path:
    """Build + atomically write ``session_breakdown.json``; returns the absolute path.

    Also triggers ``reports/trace/decision_trace.jsonl``, which is not part of
    the breakdown (see
    :func:`hyperloom.orchestrator.trace.decision_trace.write_session_decision_trace`)
    but is produced from here so that building a breakdown stays a read-only
    act, and because every caller that flushes Langfuse goes through this
    function.

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
    with suppress(Exception):
        from hyperloom.orchestrator.trace.decision_trace import write_session_decision_trace

        for warning in write_session_decision_trace(sd):
            log.debug("decision_trace: %s", warning)

    breakdown = build(sd)
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
        with suppress(OSError):
            tmp_path.unlink()
        raise
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
            with suppress(OSError):
                tmp_path.unlink()
            raise
        log.info("session_breakdown: refreshed %s section in %s", section, target)
        return True
    except Exception:  # noqa: BLE001
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
    from hyperloom.orchestrator.trace.langfuse_emitter import read_receipt

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
    from hyperloom.orchestrator.state.shared_state import SharedState
    from ..session.session_paths import reports_dir

    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else reports_dir(sd) / "final.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    state = SharedState.load_or_init(sd)
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

    current_best = state.current_best or {}
    cb_action = current_best.get("action") or "-"
    cb_tput = current_best.get("tput")
    # Framework-aware primary metric: serving shows tok/s/GPU, scriptable xDiT shows per-image latency e2el_mean_ms
    # (ms).
    baseline_metric_s = framework_registry.format_primary_metric(state.framework, state.baseline_tput, precision=2)
    cb_metric_s = (
        framework_registry.format_primary_metric(state.framework, cb_tput, precision=2)
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
        f"- session_id     : `{state.session_id or '-'}`",
        f"- model_path     : `{state.model_path or '-'}`",
        f"- framework      : `{state.framework or '-'}`",
        f"- gpu_type       : `{state.gpu_type or '-'}`",
        f"- phase (last)   : `{state.phase or '-'}`",
        f"- stop_reason    : `{state.stop_reason or '-'}`",
        f"- baseline       : `{baseline_metric_s}`",
        f"- current_best   : `{cb_action}` @ `{cb_metric_s}`",
        f"- cumul_gain     : `{state.cumulative_gain_validated:.2f}%` (validated)",
        f"- stack_entries  : `{len(state.optimization_stack or [])}`",
        "",
        "## Last action attempts",
        "",
        _fmt_attempt(getattr(state, "last_baseline", None), "last_baseline"),
        _fmt_attempt(getattr(state, "last_profile", None), "last_profile"),
        _fmt_attempt(getattr(state, "last_explore", None), "last_explore"),
        "",
        "## Structured detail",
        "",
        f"See `{breakdown_link.name}` (sibling of session root) for the "
        f"complete `timeline` / `critic` / `robustness` blocks.",
        "",
    ]

    fd, tmp = tempfile.mkstemp(
        prefix=".final.md.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise
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

    from hyperloom.orchestrator.state.shared_state import SharedState
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

    state = SharedState.load_or_init(sd)
    summary: dict[str, Any] = {
        # Crash-safe markers: a consumer can distinguish this from the full ReportExecutor output and know the run did
        # not finish gracefully.
        "safety_net": True,
        "report_complete": False,
        "producer": producer,
        "session_id": state.session_id,
        "model_name": state.model_name,
        "model_path": state.model_path,
        "model_class": state.model_class,
        "framework": state.framework,
        "gpu_type": state.gpu_type,
        "phase": state.phase,
        "stop_reason": state.stop_reason,
        "baseline_tput": state.baseline_tput,
        "baseline_accuracy": state.baseline_accuracy,
        "current_best": state.current_best,
        "cumulative_gain_validated": state.cumulative_gain_validated,
        "optimization_stack_len": len(state.optimization_stack or []),
        "crash_count": state.crash_count,
        "max_minutes": state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        # A run that died unattended is exactly when the host record is most useful, since nobody was watching.
        "platform": _crash_safe_platform(state.gpu_type),
    }
    if extra:
        summary.update(extra)

    fd, tmp = tempfile.mkstemp(
        prefix=".final.json.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, target)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise
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
