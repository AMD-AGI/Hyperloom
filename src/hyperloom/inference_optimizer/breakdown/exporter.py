# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level builder for ``session_breakdown.json``.

Shared by the dump CLI, the coordinator action, and ``cli.py``'s finally
safety net, all calling :func:`build` (pure) or :func:`write_breakdown_json`
(atomic write).
"""

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


def _merge_phase_timeline(
    fragment: Any,
    collector_value: Any,
) -> list[dict[str, Any]]:
    """Union the recorder ``phase_timeline`` fragment with the collector result.

    The collector merges three sources; the recorder fragment only carries
    audit-action attempts. Keep the collector result as the base and only append
    fragment rows whose dedupe key is missing, staying sorted by ``ts``.

    Args:
        fragment: The recorder ``phase_timeline`` fragment (may be any type).
        collector_value: The collector-computed phase timeline used as the
            base.

    Returns:
        The merged timeline (collector base plus missing fragment rows),
        sorted by ``ts``.
    """
    base: list[dict[str, Any]] = list(collector_value) if isinstance(collector_value, list) else []
    if not isinstance(fragment, list) or not fragment:
        return base
    # Share the collector's identity rule rather than restating it: the two
    # disagreeing is invisible downstream, and costs whole events.
    dedup = collectors.TimelineDedup()
    for ev in base:
        if isinstance(ev, dict):
            dedup.is_new(ev)
    for ev in fragment:
        if not isinstance(ev, dict):
            continue
        # Normalise to the collector's audit-row shape so keys line up.
        norm = dict(ev)
        norm.setdefault("kernel_id", None)
        norm.setdefault("phase", "")
        norm.setdefault("change", str(ev.get("action") or ""))
        if dedup.is_new(norm):
            base.append(norm)
    base.sort(key=lambda e: e.get("ts") or "")
    return base


def _recorded_session_value(value: Any) -> bool:
    """Whether a recorder ``session`` field carries evidence.

    The snapshot writes every key on every save, so an unset field arrives as
    the type's empty value rather than as a missing key -- ``0`` for the
    budget and the tick count exactly as ``""`` for the ids. Only a value that
    says something may overwrite what the collector resolved.

    Args:
        value (Any): A fragment field value.

    Returns:
        bool: ``True`` when the field was actually recorded.
    """
    if value is None or value == "":
        return False
    return not (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


def _merge_session(fragment: Any, collector_value: Any) -> Any:
    """Overlay the recorder's live ``session`` fields on the collected section.

    The recorder snapshots what the running state knows -- ids, phase, tick,
    the start and end timestamps -- while everything derived from
    ``manifest.json`` (container image, host, pid) and everything derived from
    the timestamps (``elapsed_minutes``) only exists on the collector side.
    Replacing the section wholesale dropped those, so a live-recorded run
    reported no wall-clock elapsed time and no image. An empty fragment value
    is absence of evidence and never overwrites a collected one, but it still
    lands on a key the collector does not fill (``phase``), which the section
    carried before this merge existed.

    Args:
        fragment: The recorder ``session`` fragment (may be any type).
        collector_value: The collector-computed session section.

    Returns:
        The merged section, or ``collector_value`` when no fragment was
        recorded.
    """
    if not isinstance(fragment, dict) or not fragment:
        return collector_value
    merged = dict(collector_value) if isinstance(collector_value, dict) else {}
    for key, value in fragment.items():
        if _recorded_session_value(value) or key not in merged:
            merged[key] = value
    merged["elapsed_minutes"] = collectors.session_elapsed_minutes(merged)
    return merged


def _load_session_json(path: Path, label: str, warnings: list[str]) -> dict[str, Any]:
    """Read a session JSON file as a dict; ``{}`` + warning on failure.

    Args:
        path: File to read.
        label: Human-readable file label for warning messages.
        warnings: Accumulator appended to when the file is missing or unparseable.

    Returns:
        The parsed JSON contents, or an empty dict on any failure.
    """
    if not path.exists():
        warnings.append(f"{label} missing at {path}")
        return {}
    try:
        return read_json(path, require_dict=True, strict=True)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse {label}: {exc!r}")
        return {}


_ROOFLINE_NUMERIC_FIELDS = (
    "arithmetic_intensity",
    "flops_per_byte",
    "efficiency_percent",
)


def _attach_kernel_roofline(
    kernel_journey: dict[str, Any],
    kernel_roofline: dict[str, Any],
) -> None:
    """Merge per-kernel roofline metrics into the ``kernel_journey`` view.

    For every journey entry with a matching ``kernel_roofline`` kernel (by
    ``kernel_id``, falling back to ``name``), attach the full roofline entry
    under ``roofline`` and backfill the discovery numeric fields discovery left
    empty. Best-effort: a missing/empty roofline table leaves the journey
    untouched.

    Args:
        kernel_journey: The kernel-journey view mutated in place.
        kernel_roofline: The per-kernel roofline table to merge from.
    """
    if not isinstance(kernel_journey, dict) or not isinstance(
        kernel_roofline,
        dict,
    ):
        return
    kernels = kernel_journey.get("kernels")
    roofline_kernels = kernel_roofline.get("kernels")
    if not isinstance(kernels, list) or not isinstance(roofline_kernels, list):
        return

    by_kid: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for rk in roofline_kernels:
        if not isinstance(rk, dict):
            continue
        kid = str(rk.get("kernel_id") or "")
        name = str(rk.get("name") or "")
        if kid:
            by_kid.setdefault(kid, rk)
        if name:
            by_name.setdefault(name, rk)

    for entry in kernels:
        if not isinstance(entry, dict):
            continue
        rk = by_kid.get(str(entry.get("kernel_id") or "")) or by_name.get(
            str(entry.get("name") or ""),
        )
        if not rk:
            continue
        entry["roofline"] = dict(rk)
        # Promote bound_type onto the entry header when discovery left it blank.
        if not str(entry.get("bound_type") or "") and rk.get("bound_type"):
            entry["bound_type"] = rk.get("bound_type")
        disc = entry.get("discovery")
        if not isinstance(disc, dict):
            continue
        if not str(disc.get("bound_type") or "") and rk.get("bound_type"):
            disc["bound_type"] = rk.get("bound_type")
        for field in _ROOFLINE_NUMERIC_FIELDS:
            if disc.get(field) in (None, 0, 0.0) and rk.get(field) not in (
                None,
                0,
                0.0,
            ):
                disc[field] = rk.get(field)


_DEFAULT_INCLUDE_TRANSCRIPTS = False


def set_default_include_transcripts(value: bool) -> None:
    """Set the process-local transcript inlining default for CLI runs."""
    global _DEFAULT_INCLUDE_TRANSCRIPTS
    _DEFAULT_INCLUDE_TRANSCRIPTS = bool(value)


def build(
    session_dir: Path | str,
    *,
    include_transcripts: bool | None = None,
) -> dict[str, Any]:
    """Build a complete :class:`SessionBreakdown` for ``session_dir`` (pure; reads disk).

    Args:
        session_dir: hyperloom session directory (needs ``manifest.json``
            or ``state.json`` for usable output).
        include_transcripts: inline specialist transcripts. ``None`` defaults
            to False since transcripts are large.

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    warnings: list[str] = []
    if include_transcripts is None:
        include_transcripts = _DEFAULT_INCLUDE_TRANSCRIPTS

    state = _load_session_json(state_path(sd), "state.json", warnings)
    manifest = _load_session_json(manifest_path(sd), "manifest.json", warnings)

    # Author-time recorder fragments (write-side spool). When present they are
    # the source of truth for their section; when absent the collectors are used
    # as fallback.
    assembled = _load_assembled(sd, warnings)
    # V6 is the hard-cutover wire shape regardless of whether recorder
    # fragments or collector fallbacks supplied the underlying evidence.
    schema_version = SCHEMA_VERSION_V6

    def _pick(section: str, collector_value: Any) -> Any:
        """Fragment value if recorded and non-empty, else the collector value.

        Args:
            section: The breakdown section name to look up in the recorder
                fragments.
            collector_value: The fallback collector-computed value.

        Returns:
            The recorder fragment value when present and non-empty, else
            ``collector_value``.
        """
        frag = assembled.get(section)
        if isinstance(frag, list) and frag:
            return frag
        if isinstance(frag, dict) and frag:
            return frag
        return collector_value

    from datetime import datetime, timezone

    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Section collectors (each catches its own errors via warnings).
    session_section = _merge_session(
        assembled.get("session"),
        _safe_collect("session", lambda: collectors.collect_session(sd, state, manifest, warnings), warnings),
    )
    # Author-side ``session_meta`` enrichment, emitted from the manifest +
    # resolved ``session`` block.
    session_meta = _pick(
        "session_meta",
        _safe_collect(
            "session_meta",
            lambda: collectors.collect_session_meta(manifest, session_section, warnings),
            warnings,
            default={},
        ),
    )
    workload = _pick(
        "workload", _safe_collect("workload", lambda: collectors.collect_workload(state, manifest, warnings), warnings)
    )
    # Model basics — verbatim mirror of ``state.model_info``. Empty {} on
    # non-transformers models.
    model_info = _pick(
        "model_info",
        _safe_collect("model_info", lambda: collectors.collect_model_info(state, warnings), warnings, default={}),
    )
    baseline = _pick(
        "baseline", _safe_collect("baseline", lambda: collectors.collect_baseline(sd, state, warnings), warnings)
    )
    final = _safe_collect("final", lambda: collectors.collect_final(sd, state, warnings), warnings)
    # Enablement attempt-runtime observability; {} → dashboard hides the block.
    enablement = _pick(
        "enablement",
        _safe_collect("enablement", lambda: collectors.collect_enablement(sd, state, warnings), warnings, default={}),
    )
    # Merge (not _pick replace): the recorder fragment only carries audit-action
    # attempts, while the collector also folds in optimization_journal KEEP/REVERT
    # and the kernel lanes; union + dedupe instead of fragment-wins.
    phase_timeline = _merge_phase_timeline(
        assembled.get("phase_timeline"),
        _safe_collect("phase_timeline", lambda: collectors.collect_phase_timeline(sd, state, warnings), warnings),
    )
    # Derived from the resolved (fragment-or-collector) phase_timeline.
    phase_segments = _safe_collect(
        "phase_segments",
        lambda: collectors.collect_phase_segments(
            state,
            phase_timeline,
            warnings,
        ),
        warnings,
        default=[],
    )
    geak_c, forge_c = _safe_collect(
        "invocations",
        lambda: collectors.collect_kernel_invocations(sd, warnings),
        warnings,
        default=([], []),
    )
    geak_invocations = _pick("geak_invocations", geak_c)
    forge_invocations = _pick("forge_invocations", forge_c)
    geak = _pick(
        "geak",
        _safe_collect(
            "geak",
            lambda: collectors.collect_geak(sd, state, warnings),
            warnings,
            default={},
        ),
    )
    capability_summary = _safe_collect(
        "capability_summary",
        lambda: collectors.collect_capability_summary(
            state,
            geak_invocations,
            warnings,
            forge_invocations,
            geak,
        ),
        warnings,
    )
    kernel_lifecycle = _safe_collect(
        "kernel_lifecycle",
        lambda: collectors.collect_kernel_lifecycle(
            sd,
            state,
            geak_invocations,
            warnings,
            forge_invocations,
        ),
        warnings,
    )
    explore_search = _pick(
        "explore_search",
        _safe_collect("explore_search", lambda: collectors.collect_explore_search(state, warnings), warnings),
    )
    critic_robustness = _pick(
        "critic_robustness",
        _safe_collect("critic_robustness", lambda: collectors.collect_critic_robustness(sd, warnings), warnings),
    )
    telemetry = _pick(
        "telemetry", _safe_collect("telemetry", lambda: collectors.collect_telemetry(sd, state, warnings), warnings)
    )
    collective = _safe_collect(
        "collective",
        lambda: collectors.collect_collective(state),
        warnings,
        default={},
    )
    # Canonical optimization read model. This is the single downstream entry
    # point for adopted warm-replay, Explore, Framework Agent, and Kernel Agent
    # changes.
    #
    # Only author-time records build it. They carry the owning agent, the
    # verdict, and the threshold behind it as recorded facts; rebuilding the
    # same model from ``state.json`` could only re-infer ownership from phase
    # timestamps, and produced a plausible-looking answer for a session whose
    # records never arrived. Absent records are now reported as absent.
    recorded_operations = [row for row in assembled.get("operations") or [] if isinstance(row, dict)]
    optimizations = (
        _safe_collect(
            "optimizations",
            lambda: collectors.collect_recorded_optimizations(
                str(state.get("session_id") or "session"),
                recorded_operations,
                [row for row in assembled.get("measurements") or [] if isinstance(row, dict)],
                [row for row in assembled.get("adoptions") or [] if isinstance(row, dict)],
                [row for row in assembled.get("artifacts") or [] if isinstance(row, dict)],
                geak_invocations,
                forge_invocations,
                warnings,
            ),
            warnings,
            default=None,
        )
        if recorded_operations
        else None
    )
    if not optimizations:
        optimizations = _unavailable_optimizations(
            "the recorder projection failed" if recorded_operations else "no operations were recorded for this session",
            state=state,
            warnings=warnings,
        )
    geak_capability = capability_summary.get("geak") if isinstance(capability_summary, dict) else {}
    # Same predicate the capability-summary fallback ran on. A private copy here
    # would go quiet exactly when the two computations had drifted apart, which
    # is the disagreement these warnings exist to catch.
    geak_promoted, geak_has_route_evidence = collectors.geak_route_evidence(state, geak)
    if geak_has_route_evidence and isinstance(geak_capability, dict):
        if geak_capability.get("status") == "not_attempted":
            warnings.append(
                "geak consistency: GEAK produced route evidence but capability_summary.geak is not_attempted"
            )
        kernel_summary = (
            ((optimizations.get("summary_by_source") or {}).get("kernel_agent") or {})
            if isinstance(optimizations, dict)
            else {}
        )
        geak_backend = (kernel_summary.get("by_backend") or {}).get("geak") or {}
        geak_gain = geak_backend.get("total_gain_pct")
        # A keep the ledger deliberately declined to sum already explains the
        # zero, and says so in its own warning. Firing here too would report a
        # gap in the accounting where the accounting is working as designed.
        geak_withheld = int(geak_backend.get("non_attributable_keeps") or 0) > 0
        if geak_promoted and not geak_withheld and not (isinstance(geak_gain, (int, float)) and geak_gain > 0):
            warnings.append(
                "geak consistency: a promoted geak_e2e stack entry has no positive gain in "
                "optimizations.summary_by_source.kernel_agent.by_backend.geak"
            )
    # Specialist sub-agent dispatch records (state + on-disk transcripts).
    specialist_runs = _pick(
        "specialist_runs",
        _safe_collect(
            "specialist_runs",
            lambda: collectors.collect_specialist_runs(
                sd,
                state,
                warnings,
                include_transcripts=include_transcripts,
            ),
            warnings,
            default=[],
        ),
    )
    # Hot-kernel roofline table from ``<sd>/reports/kernel_roofline.json``.
    kernel_roofline = _pick(
        "kernel_roofline",
        _safe_collect(
            "kernel_roofline",
            lambda: collectors.collect_kernel_roofline(
                sd,
                warnings,
            ),
            warnings,
            default={},
        ),
    )
    # Kernel-agent attempt outcome summary; mirrors
    # ``reports/kernel_optimization_summary.json``, empty → hides Block 1.
    kernel_optimization_summary = _pick(
        "kernel_optimization_summary",
        _safe_collect(
            "kernel_optimization_summary",
            lambda: collectors.collect_kernel_optimization_summary(sd, warnings),
            warnings,
            default={},
        ),
    )
    # Post-optimization concurrency sweep; mirrors
    # ``reports/conc_sweep_summary.json``, empty → hides Block 2.
    conc_sweep_summary = _pick(
        "conc_sweep_summary",
        _safe_collect(
            "conc_sweep_summary", lambda: collectors.collect_conc_sweep_summary(sd, warnings), warnings, default={}
        ),
    )
    # Per-snapshot roofline comparison list (markdown ``## Roofline`` source) from ``state.roofline_snapshots``.
    roofline = _pick(
        "roofline",
        _safe_collect(
            "roofline",
            lambda: collectors.collect_roofline(
                state,
                warnings,
            ),
            warnings,
            default=[],
        ),
    )
    # Optimization-progress curve: stack ledger + ceiling/target lines from state.json.
    roofline_progress = _pick(
        "roofline_progress",
        _safe_collect(
            "roofline_progress",
            lambda: collectors.collect_roofline_progress(
                sd,
                state,
                manifest,
                warnings,
            ),
            warnings,
            default={},
        ),
    )
    # Full-trace: unified token + decision timeline. Joins the per-call token
    # ledger with the KEEP/REVERT journal + dynamic_action dispatch history.
    # Also writes reports/trace/decision_trace.jsonl as a side effect.
    decision_trace = _safe_collect(
        "decision_trace",
        lambda: collectors.collect_decision_trace(
            sd,
            state,
            warnings,
        ),
        warnings,
        default={},
    )
    # Promoted token-spend rollup, derived from decision_trace's token_rollup
    # plus an action_timeline correlation on task_id.
    token_usage = _safe_collect(
        "token_usage",
        lambda: collectors.collect_token_usage(
            decision_trace,
            phase_timeline,
            warnings,
        ),
        warnings,
        default={},
    )
    # Live-Langfuse push receipt (opt-in second sink). Prefers the post-flush
    # ``langfuse_receipt.json``; falls back to a live emitter read.
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
    # Authoritative external-tool versions, folded into a {tool: meta} map by
    # the recorder assembler. Pure recorder section (no collector fallback).
    versions = _pick("versions", {})
    kernel_journey = _pick("kernel_journey", {})
    # Attach per-kernel roofline metrics onto each journey entry and backfill
    # discovery numeric fields discovery couldn't surface. Best-effort.
    _attach_kernel_roofline(kernel_journey, kernel_roofline)

    source_files = _safe_collect(
        "source_files",
        lambda: collectors.collect_source_files(
            sd,
            baseline.get("benchmark_report_path"),
            telemetry.get("profile_report_paths") or [],
            [],
        ),
        warnings,
        default={},
    )
    v6_warnings = list(warnings)
    # Events whose phase was killed before it could close them are closed here,
    # before the timeline is read: their fragments are on disk, and an event
    # left open would otherwise be read back as still running.
    _safe_collect("timeline_finalize", lambda: finalize_events(sd), v6_warnings, default=[])
    timeline = _safe_collect(
        "timeline",
        lambda: collectors.collect_v6_timeline(
            sd,
            v6_warnings,
            state=state,
            recorded_operations=recorded_operations,
            critic_iterations=(
                critic_robustness.get("critic_iterations", []) if isinstance(critic_robustness, dict) else []
            ),
            conc_sweep_summary=conc_sweep_summary,
            phase_timeline=phase_timeline,
        ),
        v6_warnings,
        default=[],
    )
    outcome = _safe_collect(
        "outcome",
        lambda: collectors.collect_v6_outcome(
            session=session_section,
            baseline=baseline,
            final=final,
            optimizations=optimizations,
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
            versions=versions,
            state=state,
            warnings=v6_warnings,
        ),
        v6_warnings,
        default={},
    )
    v6_close = _safe_collect(
        "close",
        lambda: collectors.collect_v6_close(sd, state, critic_robustness, v6_warnings),
        v6_warnings,
        default={},
    )
    # Snapshot last: every V6 collector above feeds this list, and it is the
    # only place a V6 failure is allowed to surface.
    if isinstance(metadata, dict):
        metadata["warnings"] = list(v6_warnings)

    breakdown = {
        "schema_version": schema_version,
        "exported_at_utc": exported_at,
        "exporter_version": EXPORTER_VERSION,
        "session": session_section,
        # Session metadata enrichment; always present from the exporter.
        "session_meta": session_meta,
        "workload": workload,
        # Structural model summary (state.model_info mirror); empty {} on
        # non-transformers models.
        "model_info": model_info,
        "baseline": baseline,
        "final": final,
        "phase_timeline": phase_timeline,
        # v1 readers use flat ``phase_timeline``, v2 prefer ``phase_segments``.
        "phase_segments": phase_segments,
        "capability_summary": capability_summary,
        # GEAK route diagnostics and accepted artifacts. This is independent
        # from the canonical optimization ledger and remains useful on failed
        # or incomplete runs that produced no adoption.
        "geak": geak,
        "kernel_lifecycle": kernel_lifecycle,
        # Collective lane audit trail; survives a campaign the E2E gate rejected,
        # which never reaches ``optimizations``.
        "collective": collective,
        "param_search": explore_search,
        "critic_robustness": critic_robustness,
        "telemetry": telemetry,
        # Canonical downstream optimization API.
        "optimizations": optimizations,
        "specialist_runs": specialist_runs,
        # Hot-kernel roofline table; empty → hidden.
        "kernel_roofline": kernel_roofline,
        # Kernel-agent attempt outcome summary; empty → hides Block 1.
        "kernel_optimization_summary": kernel_optimization_summary,
        # Post-optimization concurrency sweep; empty → hides Block 2.
        "conc_sweep_summary": conc_sweep_summary,
        # Per-snapshot roofline comparison list (markdown source).
        "roofline": roofline,
        # Optimization-progress curve; ``ceiling_available`` False when the
        # watermark roofline pipeline never ran.
        "roofline_progress": roofline_progress,
        # Full-trace token + decision timeline. ``decision_trace`` is the
        # per-decision join; ``token_rollup`` is the by_phase / by_component /
        # session_total summary.
        "decision_trace": decision_trace,
        # Promoted token-spend summary, derived from decision_trace.token_rollup.
        "token_usage": token_usage,
        # Live-Langfuse push receipt; ``enabled`` False on the default path.
        "langfuse": langfuse,
        # Kernel-major lifecycle view (discovery -> dispatch -> backend
        # attempts -> e2e), composed from the recorder substreams. Empty {} on
        # sessions that predate the substreams.
        "kernel_journey": kernel_journey,
        # Authoritative external-tool versions, one object per tool keyed by
        # tool name. Each carries ``{tool, root_dir, commit, version}``.
        "versions": versions,
        # Enablement attempt-runtime observability; {} → hidden.
        "enablement": enablement,
        "metadata": metadata,
        "outcome": outcome,
        "timeline": timeline,
        "close": v6_close,
        "warnings": warnings,
        "source_files": source_files,
    }
    return breakdown


def _load_assembled(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble recorder fragments into ``{section: value}`` (empty on opt-out
    or when no fragments exist). Never raises.

    Args:
        session_dir: The hyperloom session directory.
        warnings: Accumulator appended to when assembly fails.

    Returns:
        The assembled ``{section: value}`` mapping, or an empty dict when the
        recorder is disabled, no fragments exist, or assembly fails.
    """
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


def _unavailable_optimizations(
    reason: str,
    *,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Declare the optimization read model unavailable rather than rebuild it.

    Rebuilding this section from ``state.json`` was how a session whose
    records never landed became indistinguishable from a session that adopted
    nothing. ``state.json`` is still consulted, but only to tell those two
    apart: a stack it knows about and the recorder does not is a write that
    went missing, which is worth saying loudly.

    Args:
        reason: Why the recorder projection is not available.
        state: The parsed ``state.json``, read only as a tripwire.
        warnings: Accumulator appended to.

    Returns:
        An optimization section in the current wire shape, empty and
        explicitly flagged unavailable.
    """
    stack = state.get("optimization_stack")
    stack_len = len(stack) if isinstance(stack, list) else 0
    if stack_len:
        warnings.append(
            f"optimizations: unavailable ({reason}), yet state.json carries "
            f"{stack_len} adopted optimization(s) -- the recorder did not "
            "capture a session that optimized, so this breakdown is incomplete"
        )
    else:
        warnings.append(f"optimizations: unavailable ({reason})")
    return {
        "schema_version": collectors.OPTIMIZATIONS_SCHEMA_VERSION,
        "source_of_truth": "recorder",
        "available": False,
        "unavailable_reason": reason,
        "attempts": [],
        "entries": [],
        "backend_attempts": [],
        "summary_by_agent": {},
        "summary_by_source": {},
        "summary_by_kind": {},
        "validation": {"method": "unavailable"},
        "gemm_tuning_runs": [],
    }


def _safe_collect(
    name: str,
    fn: callable,
    warnings: list[str],
    *,
    default: Any = None,
):
    """Run a collector with broad exception catching; failure → warning + ``default``.

    Args:
        name: Collector name used in the warning message.
        fn: Zero-argument callable that runs the collector.
        warnings: Accumulator appended to when the collector raises.
        default: Value returned on failure; an empty dict when ``None``.

    Returns:
        The collector result, or ``default`` (or an empty dict) on failure.
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
    """Build + atomically write ``session_breakdown.json``; returns the absolute path.

    ``output_path`` defaults to ``<session_dir>/session_breakdown.json``;
    ``include_transcripts`` is as in :func:`build`.

    Args:
        session_dir: The hyperloom session directory to build from.
        output_path: Destination file; defaults to
            ``<session_dir>/session_breakdown.json``.
        include_transcripts: Whether to embed transcripts, as in
            :func:`build`.

    Returns:
        The absolute path of the written breakdown file.
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
    """Rewrite one section of an already-written breakdown, atomically.

    ``revise`` receives the resolved session directory and the parsed payload,
    mutates it in place, and returns whether anything actually changed; a
    ``False`` skips the write, so a repeated call costs a read.

    Best-effort throughout: a missing or unparsable breakdown, an unchanged
    payload, or any error returns ``False``. Never raises — every caller runs
    at shutdown, after ``stop_reason`` is settled, and must not mask it.

    Args:
        session_dir: The hyperloom session directory holding the breakdown.
        section (str): Section name, for the log line.
        revise (Callable[[Path, dict[str, Any]], bool]): The in-place edit.

    Returns:
        ``True`` when the file was rewritten, ``False`` otherwise.
    """
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
    """Refresh only the ``langfuse`` section of an already-written breakdown.

    ``session_breakdown.json`` is written *before* the session-end
    ``flush_session`` (the flush depends on ``decision_trace.jsonl``, which the
    breakdown produces). So the breakdown's first ``langfuse`` section carries
    the pre-flush, in-process counts (``counts_final=False``). Call this right
    after ``flush_session`` to splice in the post-flush
    ``langfuse_receipt.json`` (final counts) without rebuilding the whole file.

    Best-effort and self-skipping: returns False (no-op) when no breakdown or
    no receipt exists yet, when live push was disabled, or on any error. Never
    raises -- it must not mask the session's stop_reason at shutdown.

    Args:
        session_dir: The hyperloom session directory holding the breakdown.

    Returns:
        ``True`` when the langfuse section was refreshed, ``False`` otherwise.
    """
    from hyperloom.orchestrator.trace.langfuse_emitter import read_receipt

    def _revise(sd: Path, breakdown: dict[str, Any]) -> bool:
        receipt = read_receipt(sd)
        if receipt is None:
            return False
        receipt["receipt_source"] = "receipt_file"
        if breakdown.get("langfuse") == receipt:
            return False  # already current
        breakdown["langfuse"] = receipt
        return True

    return _patch_breakdown(session_dir, "langfuse", _revise)


def patch_breakdown_close(session_dir: Path | str) -> bool:
    """Refresh only the ``close`` section of an already-written breakdown.

    ``session_breakdown`` is step 2 of the CLOSE sequencer, so the breakdown it
    writes can only ever describe the close-out up to its own step: the four
    steps after it are not recorded yet and ``close_sequence_done`` is still
    false. Left alone, every healthy session reports ``close.status:
    "degraded"`` — an accurate statement about the *record*, but one that reads
    as the close-out having gone wrong.

    Call this as the last act of the sequencer. ``_record_close_step``
    persists ``state.json`` on every step, so by then the full sequence and
    ``close_sequence_done`` are on disk and the recomputed key is the real one.

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
        # A V5-only breakdown has no ``close`` key to refresh, and adding one
        # would change the surface of a payload that never carried it.
        if "close" not in breakdown:
            return False

        fresh_warnings: list[str] = []
        state = _load_session_json(state_path(sd), "state.json", fresh_warnings)
        critic_robustness = _safe_collect(
            "critic_robustness",
            lambda: collectors.collect_critic_robustness(sd, fresh_warnings),
            fresh_warnings,
            default={},
        )
        fresh = collectors.collect_v6_close(sd, state, critic_robustness, fresh_warnings)
        changed = breakdown.get("close") != fresh
        breakdown["close"] = fresh

        # This pass is the only one that ever sees the steps recorded *after*
        # the breakdown was written — ``artifact_package``, ``ndjson_drain``,
        # ``done`` — so drift among them is reported here or nowhere.
        # ``metadata.warnings`` is V6's single outlet, so merge into it rather
        # than overwrite: the first pass's findings are still true.
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
    """Stringify objects json.dumps can't handle natively (Path, set, ...).

    Args:
        obj (Any): The object ``json.dumps`` could not serialize.

    Returns:
        Any: ``str(obj)`` for :class:`~pathlib.Path`, a sorted list for
            ``set``.

    Raises:
        TypeError: If ``obj`` is of an unsupported type.
    """
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
    """cli.finally safety-net for ``reports/final.md`` when the CLOSE sequencer never reached step 1.

    Stays minimal (one SharedState read) so it does not block shutdown.
    Idempotent: never overwrites an existing ``reports/final.md``.

    Args:
        session_dir: The hyperloom session directory.
        output_path: Destination file; defaults to
            ``<session_dir>/reports/final.md``.

    Returns:
        The path of the (existing or newly written) ``final.md`` file.

    Raises:
        OSError: If the destination cannot be read or written; returning a
            path to a file that was never written would be worse. The
            teardown caller logs it rather than masking the stop_reason.
    """
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
        """Format one ``last_*`` attempt record as a markdown bullet.

        Args:
            d (dict[str, Any] | None): The attempt record (or ``None``).
            label (str): The bullet label (e.g. ``"last_baseline"``).

        Returns:
            str: A markdown bullet line; ``"(none)"`` when the record is
                empty.
        """
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
    # Framework-aware primary metric: serving shows tok/s/GPU, scriptable xDiT
    # shows per-image latency e2el_mean_ms (ms).
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
        f"complete `phase_history` / `critic_robustness` blocks.",
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
    """Platform record for the crash-safe path.

    Imported lazily to keep this module's import cost off the normal path, but
    from ``hyperloom.common`` rather than from the orchestrator's report
    renderer: this runs when a run has already died, which is the worst moment
    to pull in the message bus and a SQLite connection layer, and the worst
    moment to depend on a private symbol in another layer.

    ``platform_fingerprint`` returns a ``status`` dict on every path and does
    not raise, so there is no second net here. ``multi_node`` is left unset --
    nothing on this path establishes it, and an unearned ``False`` would read as
    a fact about the session.
    """
    from hyperloom.common.platform_probe import platform_fingerprint

    return platform_fingerprint(gpu_type)


def write_minimal_final_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Crash-safe ``reports/final.json`` fallback for any non-graceful exit.

    The full ``ReportExecutor`` writes ``final.json`` only on the graceful
    CLOSE step-1 path. When the run is time-exhausted, killed, or the report
    task fails, this mirror of :func:`write_minimal_final_report` emits a
    compact JSON summary from ``state.json`` so a consumable result always
    exists.

    Stays minimal (one SharedState read) so it does not block shutdown.
    Idempotent: never overwrites an existing non-empty
    ``reports/final.json`` (so it can never clobber a full ReportExecutor
    summary).

    Args:
        session_dir: The hyperloom session directory.
        output_path: Destination file; defaults to
            ``<session_dir>/reports/final.json``.

    Returns:
        The path of the (existing or newly written) ``final.json`` file.

    Raises:
        OSError: If the destination cannot be read or written; returning a
            path to a file that was never written would be worse. The
            teardown caller logs it rather than masking the stop_reason.
    """
    from datetime import datetime, timezone

    from hyperloom.orchestrator.state.shared_state import SharedState
    from ..session.session_paths import reports_dir

    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else reports_dir(sd) / "final.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Decide whether to keep the existing final.json or (re)write the fallback:
    #   * full report (``safety_net`` absent/false) -> keep, never clobber it.
    #   * prior crash-safe fallback (``safety_net: true``) -> refresh.
    #   * corrupt / unreadable -> preserve as ``final.json.corrupt``, then
    #     overwrite so downstream still gets consumable JSON.
    if target.exists() and target.stat().st_size > 0:
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            overwrite = isinstance(existing, dict) and existing.get("safety_net") is True
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
        # Crash-safe markers: a consumer can distinguish this from the full
        # ReportExecutor output and know the run did not finish gracefully.
        "safety_net": True,
        "report_complete": False,
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
        # A run that died unattended is exactly when the host record is most
        # useful, since nobody was watching. One-shot, and non-raising on
        # every path it probes.
        "platform": _crash_safe_platform(state.gpu_type),
    }

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
    "build",
    "write_breakdown_json",
    "write_minimal_final_json",
    "write_minimal_final_report",
]
