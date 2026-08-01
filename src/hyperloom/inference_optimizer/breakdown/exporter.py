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
from contextlib import suppress
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.common.timeutil import iso_z

from . import collectors
from .schema import SCHEMA_VERSION_V5
from ..session.session_paths import manifest_path, state_path

log = logging.getLogger(__name__)

EXPORTER_VERSION = "session-breakdown-1.0.0"
BREAKDOWN_FILENAME = "session_breakdown.json"


def _phase_event_key(ev: dict[str, Any]) -> tuple[str, str, str]:
    """Dedupe key matching :func:`collectors.collect_phase_timeline`.

    ``(action, ts-to-second, change|task_id)`` with the timestamp canonicalised
    to ``...Z`` so a recorder fragment row and the collector's audit-list row
    for the same attempt collapse to one event.

    Args:
        ev: A phase-timeline event dict.

    Returns:
        The ``(action, ts-to-second, change|task_id)`` dedupe key.
    """
    return (
        str(ev.get("action") or ""),
        iso_z(ev.get("ts"))[:19],
        str(ev.get("change") or ev.get("task_id") or ""),
    )


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
    seen = {_phase_event_key(ev) for ev in base if isinstance(ev, dict)}
    for ev in fragment:
        if not isinstance(ev, dict):
            continue
        # Normalise to the collector's audit-row shape so keys line up.
        norm = dict(ev)
        norm.setdefault("kernel_id", None)
        norm.setdefault("phase", "")
        norm.setdefault("change", str(ev.get("action") or ""))
        key = _phase_event_key(norm)
        if key in seen:
            continue
        seen.add(key)
        base.append(norm)
    base.sort(key=lambda e: e.get("ts") or "")
    return base


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
    # V5 is the hard-cutover wire shape regardless of whether recorder
    # fragments or collector fallbacks supplied the underlying evidence.
    schema_version = SCHEMA_VERSION_V5

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
    session_section = _pick(
        "session", _safe_collect("session", lambda: collectors.collect_session(sd, state, manifest, warnings), warnings)
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
    final_collector = _safe_collect("final", lambda: collectors.collect_final(sd, state, warnings), warnings)
    final_frag = assembled.get("final")
    # Merge: fragment live-scalars win, but collector structural fields (invocation,
    # action_path, source_layers) are preserved when absent from the fragment.
    if isinstance(final_frag, dict) and final_frag:
        merged_final = dict(final_collector or {})
        merged_final.update(final_frag)
        for _structural in ("invocation", "action_path", "source_layers"):
            if _structural in (final_collector or {}):
                merged_final[_structural] = (final_collector or {})[_structural]
        final = merged_final
    else:
        final = final_collector
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
    capability_summary = _safe_collect(
        "capability_summary",
        lambda: collectors.collect_capability_summary(
            state,
            geak_invocations,
            warnings,
            forge_invocations,
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
    sweep = _pick("sweep", _safe_collect("sweep", lambda: collectors.collect_sweep(sd, state, warnings), warnings))
    critic_robustness = _pick(
        "critic_robustness",
        _safe_collect("critic_robustness", lambda: collectors.collect_critic_robustness(sd, warnings), warnings),
    )
    telemetry = _pick(
        "telemetry", _safe_collect("telemetry", lambda: collectors.collect_telemetry(sd, state, warnings), warnings)
    )
    attribution = _safe_collect(
        "attribution",
        lambda: collectors.collect_attribution(
            state,
            geak_invocations,
            kernel_lifecycle.get("adopted") or [],
            warnings,
            forge_invocations,
        ),
        warnings,
    )
    gemm_tuning = _safe_collect(
        "gemm_tuning",
        lambda: collectors.collect_gemm_tuning(state),
        warnings,
        default={},
    )
    # Canonical optimization read model.  This is the single downstream entry
    # point for adopted warm-replay, Explore, Framework Agent, and Kernel Agent
    # changes; the historical sections below remain compatibility/audit data.
    optimizations = _safe_collect(
        "optimizations",
        lambda: collectors.collect_optimizations(
            state,
            attribution,
            geak_invocations,
            forge_invocations,
            warnings,
            gemm_tuning=gemm_tuning,
        ),
        warnings,
        default={
            "schema_version": 2,
            "entries": [],
            "backend_attempts": [],
            "summary_by_source": {},
            "summary_by_kind": {},
            "validation": {},
            "gemm_tuning_runs": [],
        },
    )
    kb_provenance = _pick(
        "kb_provenance",
        _safe_collect(
            "kb_provenance",
            lambda: collectors.collect_kb_provenance(
                session_dir,
                state,
                manifest,
                warnings,
            ),
            warnings,
        ),
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

    source_files = collectors.collect_source_files(
        sd,
        baseline.get("benchmark_report_path"),
        telemetry.get("profile_report_paths") or [],
        [p.get("benchmark_report_path") for p in (sweep.get("all_variants") or []) if p.get("benchmark_report_path")],
    )

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
        # v1-reader alias mirroring the flat per-action timeline.
        "action_timeline": phase_timeline,
        "capability_summary": capability_summary,
        "kernel_lifecycle": kernel_lifecycle,
        "param_search": explore_search,
        # v2-native name for the merged ledger; mirrors ``param_search``.
        "explore_search": explore_search,
        "sweep": sweep,
        "critic_robustness": critic_robustness,
        "telemetry": telemetry,
        # Canonical downstream optimization API.
        "optimizations": optimizations,
        # Recipe KB integration audit.
        "kb_provenance": kb_provenance,
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
        "warnings": warnings,
        "source_files": source_files,
    }
    return breakdown


_V4_CANONICAL_STREAMS: dict[str, tuple[str, str | None]] = {
    "run": ("run_snapshot", "run"),
    "workload": ("run_snapshot", "workload"),
    "model": ("run_snapshot", "model"),
    "versions": ("run_snapshot", "versions"),
    "phases": ("phase_transitions", None),
    "subjects": ("subjects", None),
    "operations": ("operations", None),
    "measurements": ("measurements", None),
    "adoptions": ("adoptions", None),
    "outcome": ("run_snapshot", "outcome"),
    "artifacts": ("artifacts", None),
    "trace": ("trace_events", None),
}


def _v4_value_available(value: Any) -> bool:
    """Return whether an assembled canonical value contains authored facts."""
    return bool(value) if isinstance(value, (dict, list)) else value is not None


def _v4_integrity(
    assembled: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Validate canonical streams and references without fallback reads."""
    fields: dict[str, Any] = {}
    available_count = 0
    conflicts: list[dict[str, Any]] = []
    conflict_fields: set[str] = set()

    def add_conflict(field: str, code: str, **evidence: Any) -> None:
        conflict_fields.add(field)
        conflicts.append({"field": field, "code": code, **evidence})

    entity_specs = {
        "phases": ("phase_transitions", ("transition_id", "event_id")),
        "subjects": ("subjects", ("subject_id",)),
        "operations": ("operations", ("operation_id",)),
        "measurements": ("measurements", ("measurement_id",)),
        "adoptions": ("adoptions", ("adoption_id",)),
        "artifacts": ("artifacts", ("artifact_id",)),
        "trace": ("trace_events", ("trace_event_id", "event_id", "span_id")),
    }
    rows_by_field: dict[str, list[dict[str, Any]]] = {}
    for field, (stream, id_fields) in entity_specs.items():
        rows = [
            row
            for row in assembled.get(stream, [])
            if isinstance(row, dict)
        ] if isinstance(assembled.get(stream), list) else []
        rows_by_field[field] = rows
        for index, row in enumerate(rows):
            if not any(str(row.get(id_field) or "").strip() for id_field in id_fields):
                add_conflict(field, "missing_stable_id", record_index=index)

    operations = rows_by_field["operations"]
    subjects = rows_by_field["subjects"]
    measurements = rows_by_field["measurements"]
    adoptions = rows_by_field["adoptions"]
    artifacts = rows_by_field["artifacts"]
    operation_ids = {str(row.get("operation_id")) for row in operations if row.get("operation_id")}
    measurement_ids = {
        str(row.get("measurement_id")) for row in measurements if row.get("measurement_id")
    }
    adoption_ids = {str(row.get("adoption_id")) for row in adoptions if row.get("adoption_id")}
    artifact_ids = {str(row.get("artifact_id")) for row in artifacts if row.get("artifact_id")}
    subject_ids = {str(row.get("subject_id")) for row in subjects if row.get("subject_id")}

    def validate_subject_refs(field: str, entity_id: str, row: dict[str, Any]) -> None:
        subject_refs = []
        if isinstance(row.get("subject"), dict):
            subject_refs.append(row["subject"])
        subject_refs.extend(
            reference
            for reference in row.get("subjects") or []
            if isinstance(reference, dict)
        )
        for reference in subject_refs:
            subject_id = str(reference.get("subject_id") or "")
            if subject_id and subject_id not in subject_ids:
                add_conflict(
                    field,
                    "dangling_subject_reference",
                    entity_id=entity_id,
                    subject_id=subject_id,
                )

    allowed_executors = {"llm_agent", "llm_tool", "deterministic"}

    def nested_reference_ids(raw: Any, id_field: str) -> list[str]:
        values = raw if isinstance(raw, list) else [raw] if raw else []
        references: list[str] = []
        for value in values:
            if isinstance(value, dict):
                reference = str(value.get(id_field) or "")
            else:
                reference = str(value or "")
            if reference:
                references.append(reference)
        return references

    def validate_nested_operation_item(
        operation_id: str,
        container: str,
        item_index: int,
        item: dict[str, Any],
    ) -> None:
        reference_specs = (
            ("measurements", "measurement_id", measurement_ids),
            ("measurement_refs", "measurement_id", measurement_ids),
            ("artifacts", "artifact_id", artifact_ids),
            ("artifact_refs", "artifact_id", artifact_ids),
            ("adoptions", "adoption_id", adoption_ids),
            ("adoption_refs", "adoption_id", adoption_ids),
            ("subjects", "subject_id", subject_ids),
            ("subject_refs", "subject_id", subject_ids),
            ("operations", "operation_id", operation_ids),
            ("operation_refs", "operation_id", operation_ids),
        )
        for ref_field, id_field, known_ids in reference_specs:
            for reference in nested_reference_ids(item.get(ref_field), id_field):
                if reference not in known_ids:
                    add_conflict(
                        "operations",
                        "dangling_nested_reference",
                        operation_id=operation_id,
                        container=container,
                        item_index=item_index,
                        reference_field=ref_field,
                        reference_id=reference,
                    )
        for ref_field, id_field, known_ids in (
            ("measurement_id", "measurement_id", measurement_ids),
            ("artifact_id", "artifact_id", artifact_ids),
            ("adoption_id", "adoption_id", adoption_ids),
            ("subject_id", "subject_id", subject_ids),
        ):
            reference = str(item.get(ref_field) or "")
            if reference and reference not in known_ids:
                add_conflict(
                    "operations",
                    "dangling_nested_reference",
                    operation_id=operation_id,
                    container=container,
                    item_index=item_index,
                    reference_field=ref_field,
                    reference_id=reference,
                )
        if isinstance(item.get("subject"), dict):
            subject_id = str(item["subject"].get("subject_id") or "")
            if subject_id and subject_id not in subject_ids:
                add_conflict(
                    "operations",
                    "dangling_nested_reference",
                    operation_id=operation_id,
                    container=container,
                    item_index=item_index,
                    reference_field="subject",
                    reference_id=subject_id,
                )
        for ref_field, id_field, known_ids in (
            ("measurement", "measurement_id", measurement_ids),
            ("artifact", "artifact_id", artifact_ids),
            ("adoption", "adoption_id", adoption_ids),
            ("operation", "operation_id", operation_ids),
        ):
            reference_value = item.get(ref_field)
            reference = (
                str(reference_value.get(id_field) or "")
                if isinstance(reference_value, dict)
                else str(reference_value or "")
            )
            if reference and reference not in known_ids:
                add_conflict(
                    "operations",
                    "dangling_nested_reference",
                    operation_id=operation_id,
                    container=container,
                    item_index=item_index,
                    reference_field=ref_field,
                    reference_id=reference,
                )
        for ref_field in (
            "operation_id",
            "target_operation_id",
            "parent_operation_id",
            "root_operation_id",
        ):
            reference = str(item.get(ref_field) or "")
            if reference and reference not in operation_ids:
                add_conflict(
                    "operations",
                    "dangling_nested_reference",
                    operation_id=operation_id,
                    container=container,
                    item_index=item_index,
                    reference_field=ref_field,
                    reference_id=reference,
                )

    for transition in rows_by_field["phases"]:
        operation_id = str(transition.get("operation_id") or "")
        if operation_id and operation_id not in operation_ids:
            add_conflict(
                "phases",
                "dangling_operation_reference",
                transition_id=transition.get("transition_id") or transition.get("event_id"),
                reference_field="operation_id",
                reference_id=operation_id,
            )

    for event in rows_by_field["trace"]:
        event_id = event.get("trace_event_id") or event.get("event_id") or event.get("span_id")
        for ref_field in ("operation_id", "parent_operation_id"):
            operation_id = str(event.get(ref_field) or "")
            if operation_id and operation_id not in operation_ids:
                add_conflict(
                    "trace",
                    "dangling_operation_reference",
                    event_id=event_id,
                    reference_field=ref_field,
                    reference_id=operation_id,
                )

    for operation in operations:
        operation_id = str(operation.get("operation_id") or "")
        validate_subject_refs("operations", operation_id, operation)
        executor = operation.get("executor_class")
        if executor and executor not in allowed_executors:
            add_conflict(
                "operations",
                "invalid_executor_class",
                operation_id=operation_id,
                executor_class=executor,
            )
        for ref_field, known_ids, target_field in (
            ("measurement_refs", measurement_ids, "measurements"),
            ("artifact_refs", artifact_ids, "artifacts"),
            ("adoption_refs", adoption_ids, "adoptions"),
        ):
            for reference in operation.get(ref_field) or []:
                if str(reference) not in known_ids:
                    add_conflict(
                        "operations",
                        "dangling_reference",
                        operation_id=operation_id,
                        reference_field=ref_field,
                        reference_id=str(reference),
                        target_field=target_field,
                    )
        for ref_field in ("parent_operation_id", "root_operation_id"):
            reference = str(operation.get(ref_field) or "")
            if reference and reference not in operation_ids:
                add_conflict(
                    "operations",
                    "dangling_operation_reference",
                    operation_id=operation_id,
                    reference_field=ref_field,
                    reference_id=reference,
                )
        for container in ("relations", "attempts", "substeps"):
            for item_index, item in enumerate(operation.get(container) or []):
                if isinstance(item, dict):
                    validate_nested_operation_item(
                        operation_id,
                        container,
                        item_index,
                        item,
                    )

    for field, rows, ref_specs in (
        (
            "measurements",
            measurements,
            (("operation_id", operation_ids, "operations"),),
        ),
        (
            "artifacts",
            artifacts,
            (
                ("operation_id", operation_ids, "operations"),
                ("producer_operation_id", operation_ids, "operations"),
            ),
        ),
        (
            "adoptions",
            adoptions,
            (
                ("operation_id", operation_ids, "operations"),
                ("measurement_ids", measurement_ids, "measurements"),
                ("artifact_ids", artifact_ids, "artifacts"),
            ),
        ),
    ):
        for row in rows:
            row_id = str(
                row.get(
                    {
                        "measurements": "measurement_id",
                        "artifacts": "artifact_id",
                        "adoptions": "adoption_id",
                    }[field]
                )
                or ""
            )
            validate_subject_refs(field, row_id, row)
            has_business_association = (
                field == "adoptions"
                or (
                    field == "measurements"
                    and any(
                        row.get(key) not in (None, "", {}, [])
                        for key in ("kind", "name", "source", "subject")
                    )
                )
                or (
                    field == "artifacts"
                    and any(
                        row.get(key) not in (None, "", {}, [])
                        for key in ("kind", "name", "path", "uri", "digest", "subject")
                    )
                )
            )
            if has_business_association and not row.get("operation_id"):
                add_conflict(
                    field,
                    "missing_operation_reference",
                    entity_id=row_id,
                )
            for ref_field, known_ids, target_field in ref_specs:
                raw = row.get(ref_field)
                references = raw if isinstance(raw, list) else [raw] if raw else []
                for reference in references:
                    if str(reference) not in known_ids:
                        add_conflict(
                            field,
                            "dangling_reference",
                            entity_id=row_id,
                            reference_field=ref_field,
                            reference_id=str(reference),
                            target_field=target_field,
                        )

    for adoption in adoptions:
        status = str(adoption.get("status") or "").lower()
        decision = str(adoption.get("decision") or "").upper()
        validated = adoption.get("validated")
        if (
            (status == "adopted" and (decision != "KEEP" or validated is not True))
            or (status in {"revoked", "reverted"} and (decision != "REVERT" or validated is not False))
        ):
            add_conflict(
                "adoptions",
                "invalid_adoption_state",
                adoption_id=adoption.get("adoption_id"),
                status=status,
                decision=decision,
                validated=validated,
            )

    selections = [
        operation
        for operation in operations
        if operation.get("kind") == "strategy_selection"
        and operation.get("strategy_group") == "kernel_optimizer"
    ]
    active_selections = [
        selection
        for selection in selections
        if str(selection.get("status") or "").lower()
        not in {"revoked", "reverted", "superseded", "skipped"}
    ]
    routes = [
        operation
        for operation in operations
        if operation.get("kind") == "kernel_optimizer_run"
        and operation.get("status") != "skipped"
    ]
    selection_ids = {
        str(selection.get("operation_id") or "") for selection in selections
    }
    selections_by_cycle: dict[str, set[str]] = {}
    for selection in active_selections:
        if selection.get("macro_cycle") is None:
            continue
        cycle_key = str(selection.get("macro_cycle"))
        selections_by_cycle.setdefault(cycle_key, set()).add(
            str(selection.get("operation_id") or "")
        )
    for cycle_key, cycle_selection_ids in selections_by_cycle.items():
        if len(cycle_selection_ids) > 1:
            add_conflict(
                "operations",
                "multiple_active_selections_in_cycle",
                macro_cycle=cycle_key,
                selection_operation_ids=sorted(cycle_selection_ids),
            )
    for selection in active_selections:
        selection_id = str(selection.get("operation_id") or "")
        selection_outputs = selection.get("outputs") or {}
        selected = str(selection_outputs.get("selected_strategy") or "")
        candidates = {
            str(candidate)
            for candidate in selection_outputs.get("candidates") or []
        }
        if not {"geak", "kernel_agent_forge"} <= candidates:
            add_conflict(
                "operations",
                "kernel_route_candidates_incomplete",
                selection_id=selection_id,
                candidates=sorted(candidates),
            )
        active_executed = [
            route
            for route in routes
            if str(route.get("parent_operation_id") or "") == selection_id
            and str(route.get("strategy") or "") == selected
            and str(route.get("status") or "").lower()
            not in {"revoked", "reverted", "superseded", "skipped"}
            and (
                ((route.get("extensions") or {}).get("route_competition") or {}).get("active")
                is not False
            )
        ]
        if len(active_executed) != 1:
            add_conflict(
                "operations",
                "kernel_route_xor_violation",
                selection_id=selection_id,
                macro_cycle=selection.get("macro_cycle"),
                executed_route_count=len(active_executed),
            )
    for route in routes:
        parent = str(route.get("parent_operation_id") or "")
        if parent not in selection_ids:
            add_conflict(
                "operations",
                "kernel_route_without_selection",
                route_operation_id=route.get("operation_id"),
                parent_operation_id=parent,
            )

    for field, (stream, nested_key) in _V4_CANONICAL_STREAMS.items():
        stream_value = assembled.get(stream)
        value = stream_value.get(nested_key) if nested_key and isinstance(stream_value, dict) else stream_value
        available = _v4_value_available(value)
        if available:
            available_count += 1
        record_count = len(value) if isinstance(value, list) else (1 if available else 0)
        fields[field] = {
            "status": (
                "partial"
                if available and field in conflict_fields
                else "exact"
                if available
                else "unavailable"
            ),
            "source": f"recorder:{stream}",
            "reason": "" if available else f"author-time stream {stream!r} did not provide {field!r}",
            "record_count": record_count,
            "warnings": [],
        }

    if available_count == len(fields) and not warnings and not conflicts:
        status = "exact"
    elif available_count == 0:
        status = "unavailable"
    else:
        status = "partial"
    return {
        "status": status,
        "canonical_source": "author_time_recorder_fragments",
        "fields": fields,
        "warnings": list(warnings),
        "conflicts": conflicts,
    }


def _v4_compat_projection(
    *,
    run: dict[str, Any],
    workload: dict[str, Any],
    model: dict[str, Any],
    versions: dict[str, Any],
    transitions: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    trace_events: list[dict[str, Any]],
    outcome: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    """Create a conservative legacy shadow from canonical author-time facts."""
    operation_by_id = {
        str(operation.get("operation_id")): operation
        for operation in operations
        if operation.get("operation_id")
    }
    measurement_by_id = {
        str(measurement.get("measurement_id")): measurement
        for measurement in measurements
        if measurement.get("measurement_id")
    }
    artifact_by_id = {
        str(artifact.get("artifact_id")): artifact
        for artifact in artifacts
        if artifact.get("artifact_id")
    }
    validated_adoptions = [
        adoption
        for adoption in adoptions
        if adoption.get("validated") is True
        and str(adoption.get("status") or "").lower() == "adopted"
        and str(adoption.get("decision") or "").upper() in {"KEEP", "ADOPTED"}
    ]
    selections = [
        operation
        for operation in operations
        if operation.get("kind") == "strategy_selection"
        and operation.get("strategy_group") == "kernel_optimizer"
    ]
    active_selections = [
        candidate_selection
        for candidate_selection in selections
        if str(candidate_selection.get("status") or "").lower()
        not in {"revoked", "reverted", "superseded", "skipped"}
    ]
    selection = active_selections[-1] if active_selections else {}
    selected_strategy = str((selection.get("outputs") or {}).get("selected_strategy") or "")
    route_operations = [
        operation
        for operation in operations
        if operation.get("kind") == "kernel_optimizer_run"
        and operation.get("status") != "skipped"
    ]
    selection_by_id = {
        str(candidate_selection.get("operation_id") or ""): candidate_selection
        for candidate_selection in selections
    }
    routes = []
    for operation in route_operations:
        parent_id = str(operation.get("parent_operation_id") or "")
        parent_selection = selection_by_id.get(parent_id, {})
        parent_selected_strategy = str(
            (parent_selection.get("outputs") or {}).get("selected_strategy") or ""
        )
        competition = (operation.get("extensions") or {}).get("route_competition") or {}
        active = (
            parent_selection in active_selections
            and str(operation.get("strategy") or "") == parent_selected_strategy
            and str(operation.get("status") or "").lower()
            not in {"revoked", "reverted", "superseded", "skipped"}
            and competition.get("active") is not False
        )
        routes.append(
            {
                "operation_id": operation.get("operation_id"),
                "strategy": operation.get("strategy"),
                "status": operation.get("status"),
                "selected": active,
                "active": active,
                "historical": not active,
                "superseded_by": competition.get("superseded_by"),
                "selection_version": competition.get("selection_version"),
                "parent_operation_id": operation.get("parent_operation_id"),
                "macro_cycle": operation.get("macro_cycle"),
                "subject": operation.get("subject"),
            }
        )
    route_cycles = []
    for candidate_selection in active_selections:
        selection_id = str(candidate_selection.get("operation_id") or "")
        selected_for_cycle = str(
            (candidate_selection.get("outputs") or {}).get("selected_strategy") or ""
        )
        cycle_routes = [
            route
            for route in routes
            if str(route.get("parent_operation_id") or "") == selection_id
        ]
        executed = [
            route
            for route in cycle_routes
            if route.get("active") is True
            and str(route.get("strategy") or "") == selected_for_cycle
        ]
        route_cycles.append(
            {
                "selection_operation_id": selection_id,
                "macro_cycle": candidate_selection.get("macro_cycle"),
                "candidates": (candidate_selection.get("outputs") or {}).get("candidates") or [],
                "selected_strategy": (candidate_selection.get("outputs") or {}).get("selected_strategy"),
                "routes": cycle_routes,
                "executed_route": executed[0] if len(executed) == 1 else None,
                "xor": len(executed) == 1,
            }
        )
    latest_cycle = route_cycles[-1] if route_cycles else {}
    active_selection_ids_by_cycle: dict[str, set[str]] = {}
    for candidate_selection in active_selections:
        if candidate_selection.get("macro_cycle") is None:
            continue
        active_selection_ids_by_cycle.setdefault(
            str(candidate_selection.get("macro_cycle")),
            set(),
        ).add(str(candidate_selection.get("operation_id") or ""))
    selection_cycles_unique = all(
        len(selection_ids) == 1
        for selection_ids in active_selection_ids_by_cycle.values()
    )
    kernel_operations = [
        operation
        for operation in operations
        if operation.get("kind") == "kernel_optimization"
        and operation.get("scope") == "kernel"
    ]
    kernel_journey = {
        "route_operation_id": next(
            (
                operation.get("operation_id")
                for operation in reversed(route_operations)
                if operation.get("strategy") == "kernel_agent_forge"
            ),
            None,
        ),
        "kernels": [
            {
                "kernel_id": ((operation.get("subject") or {}).get("name") or operation.get("name")),
                "subject": operation.get("subject"),
                "operation_id": operation.get("operation_id"),
                "status": operation.get("status"),
                "backend_attempts": operation.get("attempts") or [],
                "gates": operation.get("gates") or [],
                "decision": (operation.get("outputs") or {}).get("decision"),
                "measurement_refs": operation.get("measurement_refs") or [],
                "artifact_refs": operation.get("artifact_refs") or [],
                "adoption_refs": operation.get("adoption_refs") or [],
                "outcome": (
                    "adopted"
                    if any(
                        adoption.get("operation_id") == operation.get("operation_id")
                        for adoption in validated_adoptions
                    )
                    else operation.get("status")
                ),
            }
            for operation in kernel_operations
        ],
        "final_validation_precedence": "validated_adoption_then_operation_status",
    }
    geak_operation = next(
        (operation for operation in reversed(route_operations) if operation.get("strategy") == "geak"),
        {},
    )
    forge_invocations = [
        {
            **attempt,
            "operation_id": operation.get("operation_id"),
            "kernel_id": ((operation.get("subject") or {}).get("name") or operation.get("name")),
            "gates": operation.get("gates") or [],
        }
        for operation in kernel_operations
        for attempt in operation.get("attempts") or []
        if isinstance(attempt, dict) and str(attempt.get("backend") or "").lower() == "forge"
    ]
    optimization_stack = [
        {
            "adoption_id": adoption.get("adoption_id"),
            "operation_id": adoption.get("operation_id"),
            "kind": adoption.get("kind"),
            "decision": adoption.get("decision"),
            "validated": True,
            "gain_pct": adoption.get("gain_pct"),
            "configuration": adoption.get("configuration") or {},
            "strategy": (operation_by_id.get(str(adoption.get("operation_id"))) or {}).get("strategy"),
            "measurement_ids": adoption.get("measurement_ids") or [],
            "artifact_ids": adoption.get("artifact_ids") or [],
        }
        for adoption in validated_adoptions
    ]
    attribution_by_strategy: dict[str, dict[str, Any]] = {}
    for entry in optimization_stack:
        strategy = str(entry.get("strategy") or entry.get("kind") or "unknown")
        bucket = attribution_by_strategy.setdefault(
            strategy,
            {"adoption_count": 0, "validated_gain_pct": 0.0, "partial_gain_count": 0},
        )
        bucket["adoption_count"] += 1
        gain = entry.get("gain_pct")
        if isinstance(gain, (int, float)):
            bucket["validated_gain_pct"] += float(gain)
        else:
            bucket["partial_gain_count"] += 1
    geak_measurements = [
        measurement_by_id[measurement_id]
        for measurement_id in geak_operation.get("measurement_refs") or []
        if measurement_id in measurement_by_id
    ]
    geak_artifacts = [
        artifact_by_id[artifact_id]
        for artifact_id in geak_operation.get("artifact_refs") or []
        if artifact_id in artifact_by_id
    ]
    gemm_operations = [
        operation for operation in operations if operation.get("kind") == "gemm_tuning"
    ]

    def matching_operations(*names: str) -> list[dict[str, Any]]:
        wanted = {name.lower() for name in names}
        return [
            operation
            for operation in operations
            if str(operation.get("kind") or "").lower() in wanted
            or str(operation.get("name") or "").lower() in wanted
            or str(operation.get("strategy_group") or "").lower() in wanted
        ]

    def latest_outputs(*names: str) -> dict[str, Any]:
        matches = matching_operations(*names)
        return dict(matches[-1].get("outputs") or {}) if matches else {}

    baseline = latest_outputs("baseline")
    explore_operations = matching_operations("explore")
    explore_search = {
        "operations": [
            {
                "operation_id": operation.get("operation_id"),
                "status": operation.get("status"),
                "decision": (operation.get("outputs") or {}).get("decision"),
                "outputs": operation.get("outputs") or {},
                "measurement_refs": operation.get("measurement_refs") or [],
            }
            for operation in explore_operations
        ]
    } if explore_operations else {}
    sweep = latest_outputs("sweep", "conc_sweep")
    capability_summary = latest_outputs("capability", "capability_summary")
    critic_operations = matching_operations("critic", "robustness")
    critic_robustness = {
        "operations": critic_operations,
    } if critic_operations else {}
    specialist_operations = matching_operations("specialist")
    specialist_runs = [
        {
            "operation_id": operation.get("operation_id"),
            "status": operation.get("status"),
            "outputs": operation.get("outputs") or {},
            "extensions": operation.get("extensions") or {},
        }
        for operation in specialist_operations
    ]
    roofline_operations = matching_operations("roofline")
    roofline = [
        {
            "operation_id": operation.get("operation_id"),
            "status": operation.get("status"),
            "outputs": operation.get("outputs") or {},
            "measurement_refs": operation.get("measurement_refs") or [],
            "artifact_refs": operation.get("artifact_refs") or [],
        }
        for operation in roofline_operations
    ]
    roofline_progress = roofline[-1] if roofline else {}
    phase_segments = [
        {
            "transition_id": transition.get("transition_id") or transition.get("event_id"),
            "phase": transition.get("phase") or transition.get("to_phase"),
            "from_phase": transition.get("from_phase"),
            "status": transition.get("status"),
            "started_at": transition.get("started_at") or transition.get("ts"),
            "ended_at": transition.get("ended_at") or transition.get("ts"),
            "macro_cycle": transition.get("macro_cycle"),
        }
        for transition in transitions
    ]
    decisions = [
        {
            **decision,
            "operation_id": operation.get("operation_id"),
            "phase": operation.get("phase"),
        }
        for operation in operations
        for decision in operation.get("decisions") or []
        if isinstance(decision, dict)
    ]
    decision_trace = {
        "decisions": decisions,
        "events": [
            event
            for event in trace_events
            if str(event.get("kind") or "").lower()
            in {"decision", "operation_finalized", "operation_adopted"}
        ],
    } if decisions or trace_events else {}
    token_usage_rows = [
        {
            "operation_id": operation.get("operation_id"),
            **token_usage,
        }
        for operation in operations
        for token_usage in [
            (operation.get("outputs") or {}).get("token_usage")
            or (operation.get("extensions") or {}).get("token_usage")
        ]
        if isinstance(token_usage, dict)
    ]
    token_usage = {"operations": token_usage_rows} if token_usage_rows else {}
    telemetry = {"events": trace_events} if trace_events else {}
    source_files = {
        str(artifact.get("artifact_id")): {
            "path": artifact.get("path"),
            "kind": artifact.get("kind"),
            "operation_id": artifact.get("operation_id"),
        }
        for artifact in artifacts
        if artifact.get("path")
        and str(artifact.get("kind") or "").lower()
        in {"source", "source_file", "target_file", "patch", "tuned_file"}
    }
    kb_operations = matching_operations("kb_write", "knowledge")
    kb_provenance = {
        "operations": [
            {
                "operation_id": operation.get("operation_id"),
                "status": operation.get("status"),
                "outputs": operation.get("outputs") or {},
                "artifact_refs": operation.get("artifact_refs") or [],
            }
            for operation in kb_operations
        ]
    } if kb_operations else {}
    final = dict(outcome)
    if optimization_stack:
        final.setdefault("latest_adoption", optimization_stack[-1])

    adopted_operation_ids = {
        str(adoption.get("operation_id") or "")
        for adoption in validated_adoptions
    }
    detected_kernels: list[dict[str, Any]] = []
    optimized_kernels: list[dict[str, Any]] = []
    adopted_kernels: list[dict[str, Any]] = []
    rejected_kernels: list[dict[str, Any]] = []
    for operation in kernel_operations:
        kernel_id = str(
            (operation.get("subject") or {}).get("name")
            or operation.get("name")
            or operation.get("operation_id")
            or ""
        )
        base_entry = {
            "kernel_id": kernel_id,
            "name": kernel_id,
            "operation_id": operation.get("operation_id"),
            "status": operation.get("status"),
            "strategy": operation.get("strategy"),
        }
        detected_kernels.append(base_entry)
        if operation.get("attempts") or operation.get("outputs"):
            optimized_kernels.append(
                {
                    **base_entry,
                    "backend_attempts": operation.get("attempts") or [],
                    "outputs": operation.get("outputs") or {},
                }
            )
        if str(operation.get("operation_id") or "") in adopted_operation_ids:
            adopted_kernels.append(base_entry)
        elif str(operation.get("status") or "").lower() in {
            "failed",
            "rejected",
            "reverted",
            "revoked",
            "needs_review",
        }:
            rejected_kernels.append(base_entry)
    kernel_lifecycle = {
        "detected": detected_kernels,
        "recommended": detected_kernels,
        "optimized": optimized_kernels,
        "adopted": adopted_kernels,
        "rejected": rejected_kernels,
        "reverted": [
            entry
            for entry in rejected_kernels
            if str(entry.get("status") or "").lower() in {"reverted", "revoked"}
        ],
    } if kernel_operations else {}

    kernel_operation_ids = {
        str(operation.get("operation_id") or "") for operation in kernel_operations
    }
    kernel_metric_rows: dict[str, dict[str, Any]] = {}
    for measurement in measurements:
        operation_id = str(measurement.get("operation_id") or "")
        dimensions = measurement.get("dimensions") or {}
        kernel_id = str(
            dimensions.get("kernel_id")
            or (
                (operation_by_id.get(operation_id, {}).get("subject") or {}).get("name")
                if operation_id in kernel_operation_ids
                else ""
            )
            or ""
        )
        if not kernel_id:
            continue
        entry = kernel_metric_rows.setdefault(
            kernel_id,
            {"kernel_id": kernel_id, "operation_id": operation_id},
        )
        name = str(measurement.get("name") or measurement.get("kind") or "measurement")
        entry[name] = measurement.get("value")
        entry.setdefault("measurements", []).append(measurement)
    kernel_roofline_operations = matching_operations("kernel_roofline")
    for operation in kernel_roofline_operations:
        outputs = operation.get("outputs") or {}
        rows = outputs.get("kernels") if isinstance(outputs.get("kernels"), list) else []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            kernel_id = str(row.get("kernel_id") or row.get("name") or index)
            kernel_metric_rows.setdefault(kernel_id, {"kernel_id": kernel_id}).update(row)
    kernel_roofline = {
        "source": "canonical_operations_measurements",
        "kernels": list(kernel_metric_rows.values()),
        "operation_ids": [
            operation.get("operation_id") for operation in kernel_roofline_operations
        ],
        "artifact_ids": [
            artifact.get("artifact_id")
            for artifact in artifacts
            if str(artifact.get("kind") or "").lower()
            in {"kernel_roofline", "kernel_roofline_path", "analysis_md"}
        ],
    } if kernel_metric_rows or kernel_roofline_operations else {}

    attempted_count = sum(bool(operation.get("attempts")) for operation in kernel_operations)
    kernel_optimization_summary = {
        "schema_version": 1,
        "session_id": run.get("session_id"),
        "model_name": model.get("name") or workload.get("model_name"),
        "cumulative_gain_validated_pct": outcome.get("cumulative_gain_validated_pct"),
        "totals": {
            "top_candidates": len(kernel_operations),
            "attempted": attempted_count,
            "integrated": len(adopted_kernels),
            "keep_pending": sum(
                str(operation.get("status") or "").lower() == "needs_review"
                for operation in kernel_operations
            ),
            "rejected": len(rejected_kernels),
            "in_flight": sum(
                str(operation.get("status") or "").lower() == "running"
                for operation in kernel_operations
            ),
            "unattempted": max(len(kernel_operations) - attempted_count, 0),
        },
        "by_kernel": optimized_kernels or detected_kernels,
    } if kernel_operations else {}

    conc_sweep_operations = matching_operations("conc_sweep", "concurrency_sweep")
    conc_sweep_summary: dict[str, Any] = {}
    if conc_sweep_operations:
        conc_operation = conc_sweep_operations[-1]
        conc_sweep_summary = dict(conc_operation.get("outputs") or {})
        conc_sweep_summary.setdefault("status", conc_operation.get("status"))
        conc_sweep_summary.setdefault("operation_id", conc_operation.get("operation_id"))
        conc_sweep_summary.setdefault(
            "measurements",
            [
                measurement
                for measurement in measurements
                if measurement.get("operation_id") == conc_operation.get("operation_id")
            ],
        )
        conc_sweep_summary.setdefault(
            "artifacts",
            [
                artifact
                for artifact in artifacts
                if artifact.get("operation_id") == conc_operation.get("operation_id")
            ],
        )

    langfuse_operations = matching_operations("langfuse")
    langfuse_events = [
        event
        for event in trace_events
        if "langfuse" in str(event.get("kind") or event.get("component") or "").lower()
    ]
    langfuse: dict[str, Any] = {}
    if langfuse_operations:
        langfuse.update(langfuse_operations[-1].get("outputs") or {})
        langfuse.setdefault("operation_id", langfuse_operations[-1].get("operation_id"))
    if langfuse_events:
        latest_langfuse_event = langfuse_events[-1]
        for key in (
            "enabled",
            "disabled_reason",
            "config",
            "trace_id",
            "session_id",
            "correlated_on",
            "counts",
            "counts_final",
            "receipt_source",
        ):
            if latest_langfuse_event.get(key) is not None:
                langfuse[key] = latest_langfuse_event.get(key)
        langfuse["events"] = langfuse_events

    projection: dict[str, Any] = {
        "session": run,
        "session_meta": {},
        "workload": workload,
        "model_info": model,
        "baseline": baseline,
        "final": final,
        "phase_timeline": transitions,
        "phase_segments": phase_segments,
        "action_timeline": transitions,
        "capability_summary": capability_summary,
        "kernel_route": {
            "strategy_group": "kernel_optimizer",
            "candidates": (selection.get("outputs") or {}).get("candidates") or [],
            "selected_strategy": selected_strategy or None,
            "actual_path": (selection.get("outputs") or {}).get("actual_path"),
            "xor": (
                bool(route_cycles)
                and selection_cycles_unique
                and all(cycle["xor"] for cycle in route_cycles)
            ),
            "routes": routes,
            "executed_routes": [
                cycle["executed_route"]
                for cycle in route_cycles
                if cycle.get("executed_route")
            ],
            "executed_route": latest_cycle.get("executed_route"),
            "cycles": route_cycles,
            "final_validation_precedence": [
                "orchestrator_final_validation",
                "same_harness_validated_adoption",
                "provisional_internal_result",
            ],
        },
        "geak_invocations": geak_operation.get("substeps") or [],
        "forge_invocations": forge_invocations,
        "kernel_lifecycle": kernel_lifecycle,
        "param_search": explore_search,
        "explore_search": explore_search,
        "sweep": sweep,
        "geak": {
            "operation_id": geak_operation.get("operation_id"),
            "status": geak_operation.get("status"),
            "route": geak_operation.get("strategy"),
            "substeps": geak_operation.get("substeps") or [],
            "gates": geak_operation.get("gates") or [],
            "measurements": geak_measurements,
            "artifacts": geak_artifacts,
            "extensions": geak_operation.get("extensions") or {},
            "adoption_refs": geak_operation.get("adoption_refs") or [],
            "final_validation_precedence": "orchestrator_final_validation",
        },
        "critic_robustness": critic_robustness,
        "telemetry": telemetry,
        "attribution": {
            "basis": "validated_canonical_adoptions",
            "by_strategy": attribution_by_strategy,
        },
        "kb_provenance": kb_provenance,
        "specialist_runs": specialist_runs,
        "optimization_stack": optimization_stack,
        "gemm_tuning": {
            "operations": gemm_operations,
            "keep_only_adoptions": [
                adoption
                for adoption in validated_adoptions
                if adoption.get("kind") == "gemm_tuning"
            ],
        },
        "kernel_roofline": kernel_roofline,
        "kernel_optimization_summary": kernel_optimization_summary,
        "conc_sweep_summary": conc_sweep_summary,
        "roofline": roofline,
        "roofline_progress": roofline_progress,
        "decision_trace": decision_trace,
        "token_usage": token_usage,
        "langfuse": langfuse,
        "kernel_journey": kernel_journey,
        "versions": versions,
        "warnings": list(integrity.get("warnings") or []),
        "source_files": source_files,
    }
    return projection


def build_v4_live(session_dir: Path | str) -> dict[str, Any]:
    """Build v4 exclusively from live recorder fragments.

    The only filesystem reads performed by this path are reads of recorder
    fragment envelopes under the breakdown parts directory. Missing streams
    remain empty and are declared unavailable in ``integrity``.
    """
    from datetime import datetime, timezone

    sd = Path(session_dir).resolve()
    warnings: list[str] = []
    assembled = _load_assembled_v4(sd, warnings)
    snapshot = assembled.get("run_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}

    run = dict(snapshot.get("run") or {}) if isinstance(snapshot.get("run"), dict) else {}
    workload = dict(snapshot.get("workload") or {}) if isinstance(snapshot.get("workload"), dict) else {}
    model = dict(snapshot.get("model") or {}) if isinstance(snapshot.get("model"), dict) else {}
    versions = dict(snapshot.get("versions") or {}) if isinstance(snapshot.get("versions"), dict) else {}
    outcome = dict(snapshot.get("outcome") or {}) if isinstance(snapshot.get("outcome"), dict) else {}

    def _rows(name: str) -> list[dict[str, Any]]:
        """Return only mapping payloads from one assembled item stream."""
        value = assembled.get(name)
        return [dict(row) for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    transitions = _rows("phase_transitions")
    subjects = _rows("subjects")
    operations = _rows("operations")
    measurements = _rows("measurements")
    adoptions = _rows("adoptions")
    artifacts = _rows("artifacts")
    trace_events = _rows("trace_events")
    integrity = _v4_integrity(assembled, warnings)
    optimizations = _safe_collect(
        "optimizations",
        lambda: collectors.collect_v4_optimizations(
            run,
            operations,
            measurements,
            adoptions,
            artifacts,
            warnings,
        ),
        warnings,
        default={
            "schema_version": 2,
            "entries": [],
            "backend_attempts": [],
            "summary_by_source": {},
            "summary_by_kind": {},
            "validation": {},
            "gemm_tuning_runs": [],
        },
    )
    compat = _v4_compat_projection(
        run=run,
        workload=workload,
        model=model,
        versions=versions,
        transitions=transitions,
        operations=operations,
        measurements=measurements,
        adoptions=adoptions,
        artifacts=artifacts,
        trace_events=trace_events,
        outcome=outcome,
        integrity=integrity,
    )
    compat["optimizations"] = optimizations
    # Hard cutover: these legacy optimization projections are used only while
    # constructing ``optimizations`` and are no longer part of the public SBD
    # wire shape. Canonical operations/adoptions remain available for audit.
    for legacy_field in (
        "optimization_stack",
        "attribution",
        "geak",
        "geak_invocations",
        "forge_invocations",
        "gemm_tuning",
    ):
        compat.pop(legacy_field, None)
    breakdown: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V5,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exporter_version": EXPORTER_VERSION,
        "run": run,
        "workload": workload,
        "model": model,
        "versions": versions,
        "phases": {"transitions": transitions},
        "subjects": subjects,
        "operations": operations,
        "measurements": measurements,
        "adoptions": adoptions,
        "optimizations": optimizations,
        "outcome": outcome,
        "artifacts": artifacts,
        "trace": {"events": trace_events},
        "integrity": integrity,
        "projections": compat,
        "compat": compat,
    }
    breakdown.update(compat)
    return breakdown


def _load_assembled_v4(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Load only v4 recorder streams; never invoke collector fallback."""
    try:
        from .recorder.assembler import assemble_v4_parts

        out = assemble_v4_parts(session_dir, warnings=warnings)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.exception("recorder: assemble_v4_parts failed")
        warnings.append(f"recorder: assemble_v4_parts failed: {type(exc).__name__}: {exc}")
        return {}


def _load_assembled(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble recorder fragments into ``{section: value}`` (empty on opt-out
    or when no fragments exist). Never raises; falls back to collectors.

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

    v4_enabled = os.environ.get(
        "INFERENCE_OPTIMIZER_BREAKDOWN_V4",
        "",
    ).strip().lower() in ("1", "true", "yes", "on")
    breakdown = (
        build_v4_live(sd)
        if v4_enabled
        else build(sd, include_transcripts=include_transcripts)
    )
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

    sd = Path(session_dir).resolve()
    target = sd / BREAKDOWN_FILENAME
    try:
        receipt = read_receipt(sd)
        if receipt is None or not target.exists():
            return False
        receipt["receipt_source"] = "receipt_file"
        breakdown = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(breakdown, dict):
            return False
        if breakdown.get("langfuse") == receipt:
            return False  # already current
        breakdown["langfuse"] = receipt
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
        log.info("session_breakdown: refreshed langfuse section in %s", target)
        return True
    except Exception:  # noqa: BLE001
        log.debug("session_breakdown: langfuse patch failed (non-fatal)", exc_info=True)
        return False


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

    Stays minimal (one SharedState read) so it never raises or blocks
    shutdown. Idempotent: never overwrites an existing ``reports/final.md``.

    Args:
        session_dir: The hyperloom session directory.
        output_path: Destination file; defaults to
            ``<session_dir>/reports/final.md``.

    Returns:
        The path of the (existing or newly written) ``final.md`` file.
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
            label (str): The bullet label (e.g. ``"last_sweep"``).

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
    last_sweep = state.last_sweep or {}
    if last_sweep:
        sw_grid = last_sweep.get("grid_size", 0)
        sw_best = last_sweep.get("best_overall") or {}
        sw_tput = sw_best.get("output_throughput")
        sw_line = f"grid_size={sw_grid} best_tput={(f'{sw_tput:.2f}' if isinstance(sw_tput, (int, float)) else '-')}"
    else:
        sw_line = "(none)"

    lines = [
        "# Inference Optimizer — emergency final report",
        "",
        "> **Auto-generated safety-net.** The CLOSE phase 5-step "
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
        f"- cumul_gain     : `{state.cumulative_gain:.2f}%` (validated `{state.cumulative_gain_validated:.2f}%`)",
        f"- stack_entries  : `{len(state.optimization_stack or [])}`",
        f"- sweep summary  : {sw_line}",
        "",
        "## Last action attempts",
        "",
        _fmt_attempt(getattr(state, "last_baseline", None), "last_baseline"),
        _fmt_attempt(getattr(state, "last_profile", None), "last_profile"),
        _fmt_attempt(getattr(state, "last_explore", None), "last_explore"),
        _fmt_attempt(state.last_sweep, "last_sweep"),
        "",
        "## Structured detail",
        "",
        f"See `{breakdown_link.name}` (sibling of session root) for the "
        f"complete `phase_history` / `critic_robustness` / "
        f"`kb_provenance` blocks.",
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

    Stays minimal (one SharedState read) so it never raises or blocks
    shutdown. Idempotent: never overwrites an existing non-empty
    ``reports/final.json`` (so it can never clobber a full ReportExecutor
    summary).

    Args:
        session_dir: The hyperloom session directory.
        output_path: Destination file; defaults to
            ``<session_dir>/reports/final.json``.

    Returns:
        The path of the (existing or newly written) ``final.json`` file.
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
        "cumulative_gain": state.cumulative_gain,
        "cumulative_gain_validated": state.cumulative_gain_validated,
        "optimization_stack_len": len(state.optimization_stack or []),
        "crash_count": state.crash_count,
        "max_minutes": state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
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
    "build_v4_live",
    "write_breakdown_json",
    "write_minimal_final_json",
    "write_minimal_final_report",
]
