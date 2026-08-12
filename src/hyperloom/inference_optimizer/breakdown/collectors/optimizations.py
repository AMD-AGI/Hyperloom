# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical optimization projection for ``session_breakdown.json``.

The historical breakdown contract exposes successful optimizations through
several parallel sections (``optimization_stack``, attribution, Explore,
GEAK, and Forge).  This module projects those records into one stable,
phase-aware read model while leaving the historical sections available as
audit and compatibility data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


OPTIMIZATIONS_SCHEMA_VERSION = 3

_SOURCES = (
    "warm_replay",
    "explore",
    "framework_agent",
    "kernel_agent",
    "unattributed",
)

# Operation kinds that represent one attempt at making the workload faster.
# Everything else the recorder captures (discovery, review, routing, baseline
# measurement) is context, not an attempt.
_ATTEMPT_KINDS = frozenset(
    {
        "kernel_optimization",
        "gemm_tuning",
        "integrate_patch",
        "integrate",
        "framework_agent",
        "explore",
        "replay_warm_recipe",
        "warm_replay",
        "warm_recipe",
    }
)

_KEEP_DECISIONS = frozenset({"KEEP", "KEPT", "KEPT_INERT", "ADOPT", "ADOPTED", "PROMOTED"})
_REVERT_DECISIONS = frozenset(
    {"REVERT", "REVERTED", "REJECTED", "FAILED", "ACCURACY_UNAVAILABLE_REJECT"}
)

_THROUGHPUT_AFTER_NAMES = frozenset(
    {"throughput_after", "final_throughput", "output_throughput", "throughput"}
)
_THROUGHPUT_BEFORE_NAMES = frozenset(
    {"throughput_before", "baseline_throughput", "base_tput"}
)
_GAIN_NAMES = frozenset({"e2e_gain_pct", "gain_pct", "gain", "delta_pct"})

_NON_OPTIMIZATION_ACTIONS = {
    "baseline",
    "profile",
    "roofline",
    "sweep",
    "conc_sweep",
    "validate",
    "validate_stack",
    "target_analysis",
    "trace_analyze",
}


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _normalized_phase(value: Any) -> str:
    phase = str(value or "").strip().upper()
    return {
        "FRAMEWORK": "framework_agent",
        "FRAMEWORK_AGENT": "framework_agent",
        "EXPLORE": "explore",
        "KERNEL": "kernel_agent",
        "KERNEL_AGENT": "kernel_agent",
    }.get(phase, "")


def _phase_timeline(state: dict[str, Any]) -> list[tuple[float, str]]:
    timeline: list[tuple[float, str]] = []
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return timeline
    for row in history:
        if not isinstance(row, dict):
            continue
        phase = _normalized_phase(row.get("to_phase"))
        ts = _parse_ts(row.get("ts_unix"))
        if ts is None:
            ts = _parse_ts(row.get("ts"))
        if phase and ts is not None:
            timeline.append((ts, phase))
    timeline.sort(key=lambda item: item[0])
    return timeline


def _phase_at(ts: float | None, timeline: list[tuple[float, str]]) -> str:
    if ts is None:
        return ""
    current = ""
    for boundary, phase in timeline:
        if boundary > ts:
            break
        current = phase
    return current


def _kernel_backend(
    raw: dict[str, Any],
    *,
    kernel_id: str,
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
) -> str | None:
    values = (
        raw.get("backend"),
        raw.get("engine"),
        raw.get("source"),
        raw.get("action"),
    )
    joined = " ".join(str(value or "").lower() for value in values)
    if "forge" in joined:
        return "forge"
    if "geak" in joined:
        return "geak"

    if kernel_id:
        for backend, invocations in (
            ("forge", forge_invocations),
            ("geak", geak_invocations),
        ):
            if any(
                str(inv.get("kernel_id") or "") == kernel_id
                and str(inv.get("decision") or "").upper() == "KEEP"
                for inv in invocations
                if isinstance(inv, dict)
            ):
                return backend
    return None


def _resolve_source(
    raw: dict[str, Any],
    gain: dict[str, Any],
    *,
    timeline: list[tuple[float, str]],
) -> tuple[str, str]:
    action = str(raw.get("action") or gain.get("action") or "").strip().lower()
    if action in {"replay_warm_recipe", "warm_replay"} or "warm_replay" in action:
        return "warm_replay", "action_family"

    kernel_markers = (
        raw.get("kernel_id"),
        raw.get("backend"),
        raw.get("engine"),
        raw.get("tuned_file"),
        raw.get("final_overlay"),
    )
    if not action.startswith("integrate_patch") and (
        any(value not in (None, "", [], {}) for value in kernel_markers)
        or action in {"geak_e2e", "gemm_tuning", "fusion", "integrate"}
        or action.startswith("kernel_opt")
    ):
        return "kernel_agent", "action_family"
    if action in {"framework", "framework_agent"}:
        return "framework_agent", "action_family"

    explicit = _normalized_phase(
        raw.get("source_phase")
        or gain.get("source_phase")
        or raw.get("phase")
        or gain.get("phase")
    )
    if action.startswith("integrate_patch"):
        if raw.get("framework_agent_authoring") or gain.get(
            "framework_agent_authoring"
        ):
            return "framework_agent", "recorded"
        provenance = str(
            raw.get("provenance") or gain.get("provenance") or ""
        ).strip().lower()
        specialist_owned = bool(
            raw.get("domain")
            or gain.get("domain")
            or provenance.startswith("specialist:")
        )
        if explicit in {"framework_agent", "explore"}:
            return explicit, "recorded"
        if specialist_owned:
            return "explore", "recorded"
        # integrate_patch ownership must be explicit. Never infer it from the
        # phase active when the delayed application happened.
        return "unattributed", "unknown"
    if explicit:
        return explicit, "recorded"

    ts = _parse_ts(gain.get("ts_unix"))
    if ts is None:
        ts = _parse_ts(gain.get("ts") or raw.get("ts"))
    inferred = _phase_at(ts, timeline)
    if inferred:
        return inferred, "phase_history_ts"

    if action in {"explore", "backends", "params"}:
        return "explore", "action_family"
    return "unattributed", "unknown"


def _optimization_kind(
    raw: dict[str, Any],
    *,
    source: str,
) -> str:
    action = str(raw.get("action") or "").strip().lower()
    operation_kind = str(raw.get("operation_kind") or "").strip().lower()
    if source == "warm_replay":
        return "serving_config"
    if source == "kernel_agent":
        if action == "gemm_tuning":
            return "gemm_tuning"
        if action == "fusion":
            return "kernel_fusion"
        return "kernel_patch"
    if source == "framework_agent":
        patch_evidence = (
            raw.get("patch_path"),
            raw.get("target_file"),
            raw.get("source_snapshot"),
            raw.get("framework_root"),
            raw.get("base_sha"),
        )
        return (
            "framework_patch"
            if any(value not in (None, "") for value in patch_evidence)
            else "serving_config"
        )
    if source == "explore":
        return operation_kind if operation_kind in {"backend", "param", "env"} else "serving_config"
    return operation_kind or "unknown"


def _artifacts(raw: dict[str, Any]) -> list[dict[str, str]]:
    fields = (
        ("patch", "patch_path"),
        ("target_file", "target_file"),
        ("tuned_file", "tuned_file"),
        ("report", "final_report_path"),
        ("report", "report_path"),
        ("overlay", "final_overlay"),
        ("source_manifest", "source_manifest"),
    )
    out = [
        {"kind": str(item.get("kind") or ""), "path": str(item.get("path") or "")}
        for item in raw.get("artifacts") or []
        if isinstance(item, dict) and item.get("path")
    ]
    seen: set[tuple[str, str]] = set()
    seen.update((item["kind"], item["path"]) for item in out)
    for kind, field in fields:
        path = str(raw.get(field) or "").strip()
        key = (kind, path)
        if not path or key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "path": path})
    target_files = raw.get("target_files") or []
    if isinstance(target_files, list):
        for value in target_files:
            path = str(value or "").strip()
            key = ("target_file", path)
            if not path or key in seen:
                continue
            seen.add(key)
            out.append({"kind": "target_file", "path": path})
    return out


def _execution_mode(raw: dict[str, Any], backend: str | None) -> str | None:
    explicit = str(raw.get("execution_mode") or "").strip()
    if explicit in {"whole_pipeline", "per_kernel"}:
        return explicit
    action = str(raw.get("action") or "").strip().lower()
    if action == "geak_e2e":
        return "whole_pipeline"
    if action in {"gemm_tuning", "fusion", "integrate"} or action.startswith("kernel_opt"):
        return "per_kernel"
    if backend == "forge":
        return "per_kernel"
    if backend == "geak":
        return "whole_pipeline"
    return None


def _empty_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        source: {"keeps": 0, "total_gain_pct": 0.0}
        for source in _SOURCES
    }
    summary["kernel_agent"]["by_backend"] = {
        "geak": {"keeps": 0, "total_gain_pct": 0.0},
        "forge": {"keeps": 0, "total_gain_pct": 0.0},
        "unattributed": {"keeps": 0, "total_gain_pct": 0.0},
    }
    return summary


def _collect_backend_attempts(
    session_id: str,
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize every backend invocation into one chronological attempt list."""

    attempts: list[dict[str, Any]] = []
    for default_backend, invocations in (
        ("geak", geak_invocations),
        ("forge", forge_invocations),
    ):
        for raw in invocations:
            if not isinstance(raw, dict):
                continue
            duration = next(
                (
                    _to_float(raw.get(field))
                    for field in (
                        "duration_sec",
                        "elapsed_sec",
                        "elapsed_time_sec",
                        "elapsed",
                    )
                    if _to_float(raw.get(field)) is not None
                ),
                None,
            )
            attempts.append(
                {
                    "attempt_id": str(raw.get("attempt_id") or ""),
                    "run_id": str(raw.get("run_id") or ""),
                    "kernel_id": str(raw.get("kernel_id") or ""),
                    "backend": str(raw.get("backend") or default_backend),
                    "decision": str(raw.get("decision") or "").upper(),
                    "ts": str(raw.get("ts") or ""),
                    "duration_sec": duration,
                    "micro_speedup": _to_float(raw.get("micro_speedup")),
                    "compile_passed": raw.get("compile_passed"),
                    "correctness_passed": raw.get("correctness_passed"),
                    "error_class": (
                        str(raw.get("error_class")) if raw.get("error_class") else None
                    ),
                    "error": str(raw.get("error")) if raw.get("error") else None,
                    "result_path": (
                        str(raw.get("result_path")) if raw.get("result_path") else None
                    ),
                    "verification_path": (
                        str(raw.get("verification_path"))
                        if raw.get("verification_path")
                        else None
                    ),
                }
            )

    attempts.sort(
        key=lambda row: (
            str(row.get("kernel_id") or ""),
            _parse_ts(row.get("ts")) if _parse_ts(row.get("ts")) is not None else float("inf"),
            str(row.get("backend") or ""),
            str(row.get("attempt_id") or ""),
        )
    )
    sequence_by_kernel: dict[str, int] = {}
    for attempt in attempts:
        kernel_id = str(attempt.get("kernel_id") or "")
        sequence_by_kernel[kernel_id] = sequence_by_kernel.get(kernel_id, 0) + 1
        attempt["sequence"] = sequence_by_kernel[kernel_id]
        if not attempt.get("attempt_id"):
            attempt["attempt_id"] = (
                f"{session_id}:kernel-attempt:"
                f"{kernel_id or 'unknown'}:{attempt.get('backend') or 'unknown'}:"
                f"{attempt['sequence']}"
            )
    return attempts


def _adopted_attempt_id(
    raw: dict[str, Any],
    *,
    kernel_id: str,
    backend: str | None,
    backend_attempts: list[dict[str, Any]],
    warnings: list[str],
) -> str | None:
    explicit = str(
        raw.get("adopted_attempt_id")
        or raw.get("best_attempt_id")
        or raw.get("attempt_id")
        or ""
    ).strip()
    if explicit:
        return explicit
    matches = [
        str(attempt.get("attempt_id") or "")
        for attempt in backend_attempts
        if str(attempt.get("kernel_id") or "") == kernel_id
        and str(attempt.get("backend") or "") == str(backend or "")
        and str(attempt.get("decision") or "").upper() == "KEEP"
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        warnings.append(
            "optimizations: multiple KEEP attempts match "
            f"kernel={kernel_id!r} backend={backend!r}; "
            "adopted_attempt_id requires explicit producer evidence"
        )
    return None


def _empty_kind_summary() -> dict[str, Any]:
    return {"keeps": 0, "total_gain_pct": 0.0}


def _validation_summary(
    state: dict[str, Any],
    attribution: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_breakdown = attribution.get("source_breakdown")
    if not isinstance(source_breakdown, dict):
        source_breakdown = {}
    validated_total = _to_float(source_breakdown.get("validated_total_pct"))
    if validated_total is None:
        validated_total = _to_float(state.get("cumulative_gain_validated"))
    attributed_total = sum(
        _to_float(entry.get("gain_pct")) or 0.0
        for entry in entries
        if entry.get("validated") is True
        and entry.get("source") != "unattributed"
    )
    phase_breakdown = attribution.get("phase_breakdown")
    if not isinstance(phase_breakdown, dict):
        phase_breakdown = {}
    explore = phase_breakdown.get("explore")
    domain_attribution = (
        dict(explore.get("by_domain") or {})
        if isinstance(explore, dict) and isinstance(explore.get("by_domain"), dict)
        else {}
    )
    notes = attribution.get("notes")
    canonical_source_breakdown = {
        "sweep_gain_pct": round(
            _to_float(source_breakdown.get("sweep_pct_of_total")) or 0.0,
            6,
        ),
        "params_gain_pct": round(
            _to_float(source_breakdown.get("params_pct_of_total")) or 0.0,
            6,
        ),
        "backends_gain_pct": round(
            _to_float(source_breakdown.get("backends_pct_of_total")) or 0.0,
            6,
        ),
        "gemm_tuning_gain_pct": round(
            _to_float(source_breakdown.get("gemm_tuning_pct_of_total")) or 0.0,
            6,
        ),
        "unattributed_gain_pct": round(
            _to_float(source_breakdown.get("unattributed_pct_of_total")) or 0.0,
            6,
        ),
    }
    try:
        validated_at_stack_len = int(
            state.get("cumulative_gain_validated_stack_len") or 0
        )
    except (TypeError, ValueError):
        validated_at_stack_len = 0
    return {
        "method": str(attribution.get("method") or "missing"),
        "validated_at_stack_len": validated_at_stack_len,
        "validated_total_gain_pct": (
            round(validated_total, 6) if validated_total is not None else None
        ),
        "attributed_total_gain_pct": round(attributed_total, 6),
        "attribution_gap_pct": (
            round(validated_total - attributed_total, 6)
            if validated_total is not None
            else None
        ),
        "notes": [str(note) for note in notes] if isinstance(notes, list) else [],
        "source_breakdown": canonical_source_breakdown,
        "phase_breakdown": phase_breakdown,
        "domain_attribution": domain_attribution,
    }


def collect_optimizations(
    state: dict[str, Any],
    attribution: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
    warnings: list[str],
    gemm_tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single canonical optimization read model.

    Only records in ``state.optimization_stack`` are projected: attempts,
    REVERTs, and ``no_promote`` events remain in the action timeline.  The
    legacy sections are intentionally not mutated.
    """

    raw_stack = state.get("optimization_stack") or []
    if not isinstance(raw_stack, list):
        warnings.append("optimizations: state.optimization_stack is not a list")
        raw_stack = []
    gain_rows = attribution.get("gain_per_stack_entry") or []
    if not isinstance(gain_rows, list):
        gain_rows = []
    state_gain_rows = state.get("gain_per_stack_entry")
    if not isinstance(state_gain_rows, list):
        state_gain_rows = []
    if gain_rows and len(gain_rows) != len(raw_stack):
        warnings.append(
            "optimizations: gain_per_stack_entry length does not match "
            "optimization_stack; missing rows use throughput fallback"
        )
    try:
        validated_len = int(state.get("cumulative_gain_validated_stack_len") or 0)
    except (TypeError, ValueError):
        validated_len = 0

    baseline_tput = _to_float(state.get("baseline_tput"))
    session_id = str(state.get("session_id") or "session")
    timeline = _phase_timeline(state)
    backend_attempts = _collect_backend_attempts(
        session_id,
        geak_invocations,
        forge_invocations,
    )
    entries: list[dict[str, Any]] = []
    previous_tput = baseline_tput

    for stack_index, raw_value in enumerate(raw_stack):
        if not isinstance(raw_value, dict):
            warnings.append(f"optimizations: stack entry {stack_index} is not an object")
            continue
        raw = dict(raw_value)
        # A pre-baseline enablement patch is part of the reproducible launch
        # configuration, not a measured optimization. Keep it in SharedState's
        # stack, but omit it from the canonical optimization projection and do
        # not advance the throughput chain used by the next attributable entry.
        if raw.get("baseline_enablement") or raw.get("attribution_eligible") is False:
            continue
        if str(raw.get("action") or "").strip().lower() in _NON_OPTIMIZATION_ACTIONS:
            anchor_tput = _to_float(raw.get("tput"))
            if anchor_tput is not None:
                previous_tput = anchor_tput
            continue
        gain = (
            dict(gain_rows[stack_index])
            if stack_index < len(gain_rows) and isinstance(gain_rows[stack_index], dict)
            else {}
        )
        source, source_method = _resolve_source(raw, gain, timeline=timeline)
        kernel_id = str(raw.get("kernel_id") or raw.get("action_kernel_id") or "").strip()
        backend = (
            _kernel_backend(
                raw,
                kernel_id=kernel_id,
                geak_invocations=geak_invocations,
                forge_invocations=forge_invocations,
            )
            if source == "kernel_agent"
            else None
        )
        throughput_after = _to_float(raw.get("tput"))
        throughput_before = previous_tput
        delta = _to_float(gain.get("delta_pct"))
        cumulative = _to_float(gain.get("cum_gain_after"))
        original_gain = (
            state_gain_rows[stack_index]
            if stack_index < len(state_gain_rows)
            else None
        )
        if isinstance(original_gain, dict) and _to_float(
            original_gain.get("delta_pct")
        ) is not None:
            gain_method = "ledger"
        elif isinstance(original_gain, (int, float)):
            gain_method = "legacy_ledger_derived"
        elif delta is not None:
            gain_method = "reconstructed"
        else:
            gain_method = "missing"
        if delta is None and throughput_before and throughput_after is not None:
            delta = (throughput_after / throughput_before - 1.0) * 100.0
            gain_method = "throughput_derived"
        if cumulative is None and baseline_tput and throughput_after is not None:
            cumulative = (throughput_after / baseline_tput - 1.0) * 100.0

        name = str(
            raw.get("name")
            or raw.get("variant_name")
            or kernel_id
            or _optimization_kind(raw, source=source)
        )
        entry = {
            "id": f"{session_id}:optimization:{stack_index}",
            "stack_index": stack_index,
            "source": source,
            "source_method": source_method,
            "optimization_kind": _optimization_kind(raw, source=source),
            "name": name,
            "backend": backend,
            "execution_mode": _execution_mode(raw, backend),
            "kernel_id": kernel_id or None,
            "adopted_attempt_id": _adopted_attempt_id(
                raw,
                kernel_id=kernel_id,
                backend=backend,
                backend_attempts=backend_attempts,
                warnings=warnings,
            ),
            "action": str(raw.get("action") or ""),
            "variant_name": str(raw.get("variant_name") or ""),
            "fingerprint": str(raw.get("fingerprint") or ""),
            "scope": str(raw.get("scope") or ""),
            "source_phase": str(
                raw.get("source_phase")
                or gain.get("source_phase")
                or raw.get("phase")
                or gain.get("phase")
                or ""
            ),
            "gain_method": gain_method,
            "accepted_heads": (
                list(raw.get("accepted_heads") or [])
                if isinstance(raw.get("accepted_heads"), list)
                else []
            ),
            "extra_server_args_is_invariant": (
                raw.get("extra_server_args_is_invariant")
                if isinstance(raw.get("extra_server_args_is_invariant"), bool)
                else None
            ),
            "candidate_flags": (
                raw.get("candidate_flags")
                if raw.get("candidate_flags") is not None
                else raw.get("flags")
                if raw.get("flags") is not None
                else str(raw.get("candidate_extra_server_args") or "")
            ),
            "gain_pct": round(delta, 6) if delta is not None else None,
            "cumulative_gain_pct": (
                round(cumulative, 6) if cumulative is not None else None
            ),
            "throughput_before": throughput_before,
            "throughput_after": throughput_after,
            "validated": stack_index < validated_len,
            "task_id": str(raw.get("task_id") or gain.get("task_id") or ""),
            "ts": str(gain.get("ts") or raw.get("ts") or ""),
            "provenance": str(raw.get("provenance") or ""),
            "configuration": {
                "extra_server_args": str(
                    raw.get("extra_server_args")
                    or raw.get("candidate_extra_server_args")
                    or gain.get("extra_server_args")
                    or ""
                ),
                "extra_envs": dict(raw.get("extra_envs") or {}),
            },
            "artifacts": _artifacts(raw),
        }
        entries.append(entry)
        if throughput_after is not None:
            previous_tput = throughput_after

    summary = _empty_summary()
    summary_by_kind: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not entry["validated"]:
            continue
        source = str(entry["source"])
        bucket = summary[source]
        bucket["keeps"] += 1
        gain = _to_float(entry.get("gain_pct")) or 0.0
        bucket["total_gain_pct"] += gain
        if source == "kernel_agent":
            backend = str(entry.get("backend") or "unattributed")
            if backend not in bucket["by_backend"]:
                backend = "unattributed"
            bucket["by_backend"][backend]["keeps"] += 1
            bucket["by_backend"][backend]["total_gain_pct"] += gain
        kind = str(entry.get("optimization_kind") or "unknown")
        kind_bucket = summary_by_kind.setdefault(kind, _empty_kind_summary())
        kind_bucket["keeps"] += 1
        kind_bucket["total_gain_pct"] += gain
        if source == "kernel_agent":
            by_backend = kind_bucket.setdefault(
                "by_backend",
                {
                    "geak": _empty_kind_summary(),
                    "forge": _empty_kind_summary(),
                    "unattributed": _empty_kind_summary(),
                },
            )
            backend = str(entry.get("backend") or "unattributed")
            if backend not in by_backend:
                backend = "unattributed"
            by_backend[backend]["keeps"] += 1
            by_backend[backend]["total_gain_pct"] += gain

    for bucket in summary.values():
        bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]), 6)
        for backend_bucket in bucket.get("by_backend", {}).values():
            backend_bucket["total_gain_pct"] = round(
                float(backend_bucket["total_gain_pct"]),
                6,
            )
    for bucket in summary_by_kind.values():
        bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]), 6)
        for backend_bucket in bucket.get("by_backend", {}).values():
            backend_bucket["total_gain_pct"] = round(
                float(backend_bucket["total_gain_pct"]),
                6,
            )

    gemm_tuning_runs = []
    if isinstance(gemm_tuning, dict) and isinstance(gemm_tuning.get("runs"), list):
        gemm_tuning_runs = [
            dict(run) for run in gemm_tuning["runs"] if isinstance(run, dict)
        ]

    return {
        "schema_version": OPTIMIZATIONS_SCHEMA_VERSION,
        "source_of_truth": "state",
        # Business state only retains what was adopted, so this fallback cannot
        # enumerate attempts. An empty list means "not recorded", not "none".
        "attempts": [],
        "entries": entries,
        "backend_attempts": backend_attempts,
        "summary_by_agent": {},
        "summary_by_source": summary,
        "summary_by_kind": summary_by_kind,
        "validation": _validation_summary(state, attribution, entries),
        "gemm_tuning_runs": gemm_tuning_runs,
    }


def _work_kind(operation: dict[str, Any]) -> str:
    """Return the operation's real work kind.

    ``composite`` is a container label the recorder uses for multi-step actions;
    the action name underneath it is what identifies the work.
    """
    kind = str(operation.get("kind") or "").strip().lower()
    if kind in {"", "composite"}:
        return str(operation.get("name") or "").strip().lower()
    return kind


_AGENT_BY_RECORDED_KIND = {
    "kernel_optimization": "kernel_agent",
    "gemm_tuning": "kernel_agent",
    "framework_agent": "framework_agent",
    "explore": "explore",
    "replay_warm_recipe": "warm_replay",
    "warm_replay": "warm_replay",
    "warm_recipe": "warm_replay",
}


def _attempt_agent(operation: dict[str, Any], adoption: dict[str, Any]) -> str:
    """Return the owning agent, preferring what the producer actually recorded.

    Sessions recorded before producers stamped ``agent`` still need a bucket, so
    fall back to the work kind and, for patch application, the authoring markers
    the executor left in its own result.
    """
    recorded = str(operation.get("agent") or adoption.get("agent") or "").strip()
    if recorded:
        return recorded

    kind = _work_kind(operation)
    by_kind = _AGENT_BY_RECORDED_KIND.get(kind)
    if by_kind:
        return by_kind

    if kind in {"integrate_patch", "integrate"}:
        outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
        if outputs.get("framework_agent_authoring") or outputs.get(
            "framework_agent_candidate_id"
        ):
            return "framework_agent"
        provenance = str(outputs.get("provenance") or "").strip().lower()
        if outputs.get("domain") or provenance.startswith("specialist:"):
            return "explore"
        return _normalized_phase(outputs.get("source_phase")) or "unattributed"
    return "unattributed"


def _last_decision(operation: dict[str, Any]) -> dict[str, Any]:
    decisions = [row for row in operation.get("decisions") or [] if isinstance(row, dict)]
    return decisions[-1] if decisions else {}


def _threshold_from_operation(operation: dict[str, Any]) -> float | None:
    """Pull the keep threshold out of whichever gate or decision recorded it."""
    for gate in operation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        for holder in (gate.get("inputs"), gate.get("evidence")):
            if isinstance(holder, dict):
                value = _to_float(holder.get("keep_threshold_pct"))
                if value is not None:
                    return value
    evidence = _last_decision(operation).get("evidence")
    if isinstance(evidence, dict):
        value = _to_float(evidence.get("keep_threshold_pct"))
        if value is not None:
            return value
    outputs = operation.get("outputs")
    if isinstance(outputs, dict):
        return _to_float(outputs.get("keep_threshold_pct"))
    return None


def _gate_rows(operation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in operation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        rows.append(
            {
                "kind": str(gate.get("kind") or ""),
                "name": str(gate.get("name") or ""),
                "status": str(gate.get("status") or ""),
                "decision": str(gate.get("decision") or ""),
                "reason": str(gate.get("reason") or ""),
            }
        )
    return rows


def _sub_attempt_rows(operation: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the per-backend tries nested inside one optimization attempt."""
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(operation.get("attempts") or []):
        if not isinstance(value, dict):
            continue
        outputs = value.get("outputs") if isinstance(value.get("outputs"), dict) else {}
        error = value.get("error")
        rows.append(
            {
                "attempt_id": str(value.get("attempt_id") or ""),
                "sequence": value.get("sequence") or index + 1,
                "backend": str(value.get("backend") or ""),
                "status": str(value.get("status") or ""),
                "decision": str(outputs.get("decision") or "").upper(),
                "started_at": str(value.get("started_at") or ""),
                "duration_sec": _duration_between(
                    value.get("started_at"),
                    value.get("ended_at"),
                ),
                "micro_speedup": _to_float(outputs.get("micro_speedup")),
                "compile_passed": outputs.get("compile_passed"),
                "correctness_passed": outputs.get("correctness_passed"),
                "error": str(error) if error and not isinstance(error, dict) else None,
            }
        )
    return rows


def _recorded_attempt_row(
    operation: dict[str, Any],
    *,
    adoption: dict[str, Any],
    measurement_by_id: dict[str, dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Project one recorded operation into a single canonical attempt row."""
    operation_id = str(operation.get("operation_id") or "")
    outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
    decision_row = _last_decision(operation)
    subject = operation.get("subject") if isinstance(operation.get("subject"), dict) else {}

    recorded_ids = [str(mid) for mid in operation.get("measurement_refs") or []]
    pinned_ids = [str(mid) for mid in adoption.get("measurement_ids") or []]
    if pinned_ids:
        # The adoption named the measurements it was decided on. Re-measuring
        # the same subject now records new ones beside these instead of over
        # them, so these still say what they said when the call was made.
        measurement_ids = pinned_ids
        measurement_source = "adoption_pinned"
    else:
        measurement_ids = _latest_measurement_per_name(recorded_ids, measurement_by_id)
        measurement_source = "latest_occurrence"
    occurrence_index = _occurrence_index(recorded_ids, measurement_by_id)
    measured_before = None
    measured_after = None
    measured_gain = None
    measurements: list[dict[str, Any]] = []
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        value = _to_float(measurement.get("value"))
        measurements.append(
            {
                "name": name,
                "value": measurement.get("value"),
                "unit": str(measurement.get("unit") or ""),
                # Which reading of this name it is, oldest first, so a reader
                # can tell a re-measure from the one that was decided on
                # without having to compare ids.
                "occurrence": occurrence_index.get(str(measurement_id), 0),
                "occurrences_of_name": sum(
                    1
                    for mid, _ in occurrence_index.items()
                    if str((measurement_by_id.get(mid) or {}).get("name") or "")
                    .strip()
                    .lower()
                    == name
                ),
            }
        )
        if value is None:
            continue
        if name in _THROUGHPUT_AFTER_NAMES and measured_after is None:
            measured_after = value
        elif name in _THROUGHPUT_BEFORE_NAMES and measured_before is None:
            measured_before = value
        elif name in _GAIN_NAMES and measured_gain is None:
            measured_gain = value

    # Values frozen on the adoption outrank the referenced measurements, which
    # archives recorded before per-occurrence ids may have since overwritten.
    frozen_before = _to_float(adoption.get("throughput_before"))
    frozen_after = _to_float(adoption.get("throughput_after"))
    throughput_before = frozen_before if frozen_before is not None else measured_before
    throughput_after = frozen_after if frozen_after is not None else measured_after
    if measurement_source == "adoption_pinned" and _disagrees(
        (frozen_before, measured_before),
        (frozen_after, measured_after),
    ):
        # The adoption pinned these readings, yet they no longer say what it
        # was decided on: they were written over before ids carried an
        # occurrence. The frozen values still stand; the citation does not.
        measurement_source = "adoption_pinned_stale"

    artifact_ids = list(operation.get("artifact_refs") or [])
    artifact_ids += [
        aid for aid in adoption.get("artifact_ids") or [] if aid not in artifact_ids
    ]
    artifacts: list[dict[str, str]] = []
    for artifact_id in artifact_ids:
        artifact = artifact_by_id.get(str(artifact_id))
        if not artifact:
            continue
        path = str(artifact.get("path") or artifact.get("uri") or "")
        if not path:
            continue
        artifacts.append({"kind": str(artifact.get("kind") or ""), "path": path})

    decision = str(
        adoption.get("decision")
        or decision_row.get("verdict")
        or outputs.get("decision")
        or outputs.get("status")
        or operation.get("status")
        or ""
    ).strip().upper()
    adopted = (
        bool(adoption)
        and adoption.get("validated") is True
        and str(adoption.get("decision") or "").upper() in _KEEP_DECISIONS
    )
    evidence = decision_row.get("evidence") if isinstance(decision_row.get("evidence"), dict) else {}
    # Deliberately never named ``gain_pct``: this is what the executor measured
    # against its own starting point, which is not the session baseline once
    # anything has been adopted. Summing these across attempts is wrong, and a
    # shared field name is all it takes for someone to try.
    local_gain_pct = _to_float(adoption.get("gain_pct"))
    if local_gain_pct is None:
        local_gain_pct = _to_float(evidence.get("gain_pct"))
    if local_gain_pct is None:
        local_gain_pct = _to_float(outputs.get("delta_pct"))
    if local_gain_pct is None:
        local_gain_pct = measured_gain

    return {
        "attempt_id": operation_id,
        "agent": _attempt_agent(operation, adoption),
        "agent_method": (
            "recorded"
            if str(operation.get("agent") or adoption.get("agent") or "").strip()
            else "derived"
        ),
        "producer": str(operation.get("producer") or ""),
        "kind": _work_kind(operation),
        "name": str(operation.get("name") or subject.get("name") or ""),
        "subject": {
            "type": str(subject.get("subject_type") or ""),
            "name": str(subject.get("name") or ""),
        },
        "kernel_id": (
            str(subject.get("name") or "")
            if "kernel" in str(subject.get("subject_type") or "").lower()
            else None
        ),
        "backend": str(operation.get("strategy") or ""),
        "phase": str(operation.get("phase") or ""),
        "macro_cycle": operation.get("macro_cycle"),
        "started_at": str(operation.get("started_at") or ""),
        "ended_at": str(operation.get("ended_at") or ""),
        "duration_sec": _duration_between(
            operation.get("started_at"),
            operation.get("ended_at"),
        ),
        "status": str(operation.get("status") or ""),
        "decision": decision,
        "decision_reason": str(
            adoption.get("reason")
            or decision_row.get("reason")
            or outputs.get("reason")
            or ""
        ),
        "keep_threshold_pct": _threshold_from_operation(operation),
        "adopted": adopted,
        "attribution_eligible": (
            bool(adoption.get("attribution_eligible", True)) if adoption else None
        ),
        "local_gain_pct": round(local_gain_pct, 6) if local_gain_pct is not None else None,
        "throughput_before": throughput_before,
        "throughput_after": throughput_after,
        "adoption_id": str(adoption.get("adoption_id") or "") or None,
        "gates": _gate_rows(operation),
        "backend_attempts": _sub_attempt_rows(operation),
        "measurements": measurements,
        # Which reading of a repeatedly measured subject these numbers came
        # from, and how many readings the operation has in total. Without this
        # a re-measured kernel looks the same as one measured once.
        "measurement_source": measurement_source,
        "measurement_occurrences": sum(
            1 for mid in recorded_ids if mid in measurement_by_id
        ),
        "artifacts": artifacts,
    }


def _disagrees(*pairs: tuple[float | None, float | None]) -> bool:
    """Report whether a frozen value and its cited measurement have parted ways.

    Compared in relative terms so a re-serialized float never counts, while a
    genuine re-measurement always does.
    """
    for frozen, measured in pairs:
        if frozen is None or measured is None or not frozen:
            continue
        if abs(measured - frozen) / abs(frozen) > 1e-6:
            return True
    return False


def _occurrence_index(
    measurement_ids: list[str],
    measurement_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Number an operation's readings of each metric, oldest first.

    Recorded ids have to be derived from what is being recorded rather than
    from a counter, since several producers replay their records after a
    resume. That makes them stable but unreadable, so the plain ordinal a
    reader wants is assigned here, where the whole set is in hand at once.
    """
    ordered: dict[str, list[tuple[float, str]]] = {}
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        taken_at = _parse_ts(measurement.get("measured_at"))
        ordered.setdefault(name, []).append(
            (taken_at if taken_at is not None else float("inf"), str(measurement_id))
        )
    index: dict[str, int] = {}
    for rows in ordered.values():
        # Ties fall back to the id so the numbering is at least deterministic
        # for readings whose timestamps are identical or missing.
        for position, (_, measurement_id) in enumerate(sorted(rows)):
            index[measurement_id] = position
    return index


def _latest_measurement_per_name(
    measurement_ids: list[str],
    measurement_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Keep one measurement per name: the most recent time it was taken.

    An operation accumulates a reference for every occurrence of every metric
    it measured, because retrying a subject no longer overwrites the earlier
    numbers. With no adoption naming which occurrence a decision used, the
    newest one is the operation's current state.
    """
    newest: dict[str, tuple[float, str]] = {}
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        taken_at = _parse_ts(measurement.get("measured_at")) or float("-inf")
        current = newest.get(name)
        if current is None or taken_at >= current[0]:
            newest[name] = (taken_at, str(measurement_id))
    chosen = {measurement_id for _, measurement_id in newest.values()}
    return [
        measurement_id for measurement_id in measurement_ids if measurement_id in chosen
    ]


def _recorded_baseline_throughput(
    operations: list[dict[str, Any]],
    measurement_by_id: dict[str, dict[str, Any]],
) -> float | None:
    """Return the session baseline throughput the recorder measured.

    The baseline is where the session started, so when it has been measured
    more than once the earliest reading is the one every gain in the report is
    stated against. Taking the newest would quietly move the denominator under
    figures that were already published.
    """
    earliest: tuple[float, float] | None = None
    for operation in operations:
        if not isinstance(operation, dict) or _work_kind(operation) != "baseline":
            continue
        for measurement_id in operation.get("measurement_refs") or []:
            measurement = measurement_by_id.get(str(measurement_id))
            if not measurement:
                continue
            if str(measurement.get("name") or "").strip().lower() != "throughput":
                continue
            value = _to_float(measurement.get("value"))
            if not value:
                continue
            taken_at = _parse_ts(measurement.get("measured_at")) or float("inf")
            if earliest is None or taken_at < earliest[0]:
                earliest = (taken_at, value)
    return earliest[1] if earliest else None


def _summarize_by_agent(
    attempts: list[dict[str, Any]],
    gain_by_attempt: dict[str, float],
) -> dict[str, Any]:
    """Aggregate attempts into the per-agent view (first layer of the report).

    ``gain_by_attempt`` holds each adopted attempt's baseline-relative
    contribution, so per-agent totals add up to
    ``validation.attributed_total_gain_pct``. They fall short of the session's
    end-to-end gain by whatever no attempt accounts for, which is reported
    separately as ``validation.unattributed_gain_pct``.
    """
    summary: dict[str, Any] = {}
    for attempt in attempts:
        agent = str(attempt.get("agent") or "unattributed")
        bucket = summary.setdefault(
            agent,
            {
                "attempts": 0,
                "keeps": 0,
                "reverts": 0,
                "attributable_gain_pct": 0.0,
                "non_attributable_keeps": 0,
                "by_kind": {},
            },
        )
        bucket["attempts"] += 1
        kind_bucket = bucket["by_kind"].setdefault(
            str(attempt.get("kind") or "unknown"),
            {"attempts": 0, "keeps": 0, "attributable_gain_pct": 0.0},
        )
        kind_bucket["attempts"] += 1
        decision = str(attempt.get("decision") or "").upper()
        if attempt.get("adopted"):
            bucket["keeps"] += 1
            kind_bucket["keeps"] += 1
            if attempt.get("attribution_eligible") is False:
                bucket["non_attributable_keeps"] += 1
            else:
                gain = gain_by_attempt.get(str(attempt.get("attempt_id") or ""), 0.0)
                bucket["attributable_gain_pct"] += gain
                kind_bucket["attributable_gain_pct"] += gain
        elif decision in _REVERT_DECISIONS:
            bucket["reverts"] += 1
    for bucket in summary.values():
        bucket["attributable_gain_pct"] = round(float(bucket["attributable_gain_pct"]), 6)
        for kind_bucket in bucket["by_kind"].values():
            kind_bucket["attributable_gain_pct"] = round(
                float(kind_bucket["attributable_gain_pct"]),
                6,
            )
    return summary


def _duration_between(started_at: Any, ended_at: Any) -> float | None:
    started = _parse_ts(started_at)
    ended = _parse_ts(ended_at)
    if started is None or ended is None or ended < started:
        return None
    return round(ended - started, 6)


def _collect_v4_backend_invocations(
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validated_operation_ids = {
        str(adoption.get("operation_id") or "")
        for adoption in adoptions
        if isinstance(adoption, dict)
        and adoption.get("validated") is True
        and str(adoption.get("decision") or "").upper()
        in {"KEEP", "ADOPT", "ADOPTED"}
    }
    micro_speedup_by_attempt = {
        str((measurement.get("dimensions") or {}).get("attempt_id") or ""): _to_float(
            measurement.get("value")
        )
        for measurement in measurements
        if isinstance(measurement, dict)
        and str(measurement.get("name") or "") == "micro_speedup"
        and isinstance(measurement.get("dimensions"), dict)
        and (measurement.get("dimensions") or {}).get("attempt_id")
    }
    geak: list[dict[str, Any]] = []
    forge: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        operation_id = str(operation.get("operation_id") or "")
        subject = (
            operation.get("subject")
            if isinstance(operation.get("subject"), dict)
            else {}
        )
        kernel_id = str(subject.get("name") or operation.get("name") or "")
        outputs = (
            operation.get("outputs")
            if isinstance(operation.get("outputs"), dict)
            else {}
        )
        verification = (
            outputs.get("verification")
            if isinstance(outputs.get("verification"), dict)
            else {}
        )
        proposal = (
            outputs.get("proposal")
            if isinstance(outputs.get("proposal"), dict)
            else {}
        )
        best_attempt_id = str(verification.get("best_attempt_id") or "")
        operation_decision = str(
            outputs.get("decision")
            or proposal.get("decision")
            or ""
        ).upper()
        for index, attempt_value in enumerate(operation.get("attempts") or []):
            if not isinstance(attempt_value, dict):
                continue
            attempt = dict(attempt_value)
            attempt_outputs = (
                attempt.get("outputs")
                if isinstance(attempt.get("outputs"), dict)
                else {}
            )
            attempt_id = str(attempt.get("attempt_id") or "")
            backend = str(
                attempt.get("backend")
                or attempt_outputs.get("backend")
                or operation.get("strategy")
                or ""
            ).lower()
            if "geak" in backend:
                backend = "geak"
                lane = geak
            elif "forge" in backend:
                backend = "forge"
                lane = forge
            else:
                continue
            status = str(attempt.get("status") or "").upper()
            decision = str(attempt_outputs.get("decision") or "").upper()
            if not decision and operation_id in validated_operation_ids:
                if not best_attempt_id or attempt_id == best_attempt_id:
                    decision = "KEEP"
            if not decision and status in {"FAILED", "ERROR", "CANCELLED"}:
                decision = "FAILED"
            error = attempt.get("error")
            error_class = None
            error_message = None
            if isinstance(error, dict):
                error_class = str(
                    error.get("class")
                    or error.get("error_class")
                    or error.get("type")
                    or ""
                ) or None
                error_message = str(
                    error.get("message") or error.get("error") or ""
                ) or None
            elif error:
                error_message = str(error)
            lane.append(
                {
                    "attempt_id": attempt_id,
                    "run_id": operation_id,
                    "kernel_id": kernel_id,
                    "backend": backend,
                    "decision": decision or operation_decision,
                    "ts": str(
                        attempt.get("started_at")
                        or operation.get("started_at")
                        or ""
                    ),
                    "sequence": attempt.get("sequence") or index + 1,
                    "duration_sec": _duration_between(
                        attempt.get("started_at"),
                        attempt.get("ended_at"),
                    ),
                    "micro_speedup": _to_float(
                        attempt_outputs.get("micro_speedup")
                        or attempt_outputs.get("speedup")
                    )
                    or micro_speedup_by_attempt.get(attempt_id),
                    "compile_passed": attempt_outputs.get("compile_passed"),
                    "correctness_passed": attempt_outputs.get(
                        "correctness_passed"
                    ),
                    "error_class": error_class,
                    "error": error_message,
                }
            )
    return geak, forge


def _collect_v4_gemm_tuning_runs(
    operations: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adoption_by_operation = {
        str(adoption.get("operation_id") or ""): adoption
        for adoption in adoptions
        if isinstance(adoption, dict) and adoption.get("operation_id")
    }
    runs: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("kind") != "gemm_tuning":
            continue
        extensions = (
            operation.get("extensions")
            if isinstance(operation.get("extensions"), dict)
            else {}
        )
        gemm_extension = (
            extensions.get("gemm")
            if isinstance(extensions.get("gemm"), dict)
            else {}
        )
        result = (
            dict(gemm_extension.get("result"))
            if isinstance(gemm_extension.get("result"), dict)
            else {}
        )
        outputs = (
            operation.get("outputs")
            if isinstance(operation.get("outputs"), dict)
            else {}
        )
        adoption = adoption_by_operation.get(
            str(operation.get("operation_id") or ""),
            {},
        )
        result.setdefault(
            "engine",
            outputs.get("engine") or operation.get("strategy") or "",
        )
        result.setdefault("status", operation.get("status") or "")
        result.setdefault("decision", outputs.get("decision") or "")
        result.setdefault("source", operation.get("producer") or "")
        result.setdefault(
            "ts",
            operation.get("ended_at") or operation.get("started_at") or "",
        )
        result.setdefault(
            "duration_sec",
            _duration_between(
                operation.get("started_at"),
                operation.get("ended_at"),
            ),
        )
        result.setdefault(
            "adopted",
            adoption.get("validated") is True
            and str(adoption.get("decision") or "").upper()
            in {"KEEP", "ADOPT", "ADOPTED"},
        )
        if result.get("gain_pct") is None:
            result["gain_pct"] = _to_float(adoption.get("gain_pct"))
        if "candidates" not in result:
            result["candidates"] = [
                dict(attempt.get("outputs") or {})
                for attempt in operation.get("attempts") or []
                if isinstance(attempt, dict)
            ]
        runs.append(result)
    return runs


def collect_v4_optimizations(
    run: dict[str, Any],
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Project canonical v4 adoption streams through the same read model.

    V4 never reads business-state files, so a small synthetic stack is built
    exclusively from recorder operations, validated adoptions, measurements,
    and artifact references before delegating to :func:`collect_optimizations`.
    """

    operation_by_id = {
        str(operation.get("operation_id") or ""): operation
        for operation in operations
        if isinstance(operation, dict) and operation.get("operation_id")
    }
    measurement_by_id = {
        str(measurement.get("measurement_id") or ""): measurement
        for measurement in measurements
        if isinstance(measurement, dict) and measurement.get("measurement_id")
    }
    artifact_by_id = {
        str(artifact.get("artifact_id") or ""): artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("artifact_id")
    }

    stack: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    cumulative = 0.0
    for adoption in adoptions:
        if not isinstance(adoption, dict):
            continue
        decision = str(adoption.get("decision") or "").upper()
        if decision not in {"KEEP", "ADOPT", "ADOPTED"}:
            continue
        if adoption.get("validated") is not True:
            continue
        operation_id = str(adoption.get("operation_id") or "")
        operation = operation_by_id.get(operation_id, {})
        strategy = str(operation.get("strategy") or "")
        subject = adoption.get("subject") if isinstance(adoption.get("subject"), dict) else {}
        if not subject:
            subject = operation.get("subject") if isinstance(operation.get("subject"), dict) else {}
        kernel_id = str(subject.get("name") or "") if "kernel" in str(subject.get("kind") or "").lower() else ""
        artifact_rows = [
            artifact_by_id[artifact_id]
            for artifact_id in adoption.get("artifact_ids") or []
            if artifact_id in artifact_by_id
        ]
        raw: dict[str, Any] = {
            "action": str(operation.get("kind") or adoption.get("kind") or ""),
            "variant_name": str(
                operation.get("name")
                or subject.get("name")
                or adoption.get("adoption_id")
                or ""
            ),
            "name": str(
                operation.get("name")
                or subject.get("name")
                or adoption.get("kind")
                or ""
            ),
            "task_id": operation_id,
            "source_phase": str(operation.get("phase") or ""),
            "backend": strategy,
            "execution_mode": (
                "whole_pipeline"
                if strategy == "geak"
                else "per_kernel"
                if "forge" in strategy
                else None
            ),
            "kernel_id": kernel_id,
            "operation_kind": str(operation.get("kind") or adoption.get("kind") or ""),
            "extra_server_args": str((adoption.get("configuration") or {}).get("extra_server_args") or ""),
            "extra_envs": dict((adoption.get("configuration") or {}).get("extra_envs") or {}),
            "ts": str(adoption.get("adopted_at") or operation.get("ended_at") or ""),
            "provenance": str(adoption.get("producer") or operation.get("producer") or ""),
        }
        for artifact in artifact_rows:
            kind = str(artifact.get("kind") or "").lower()
            path = str(artifact.get("path") or artifact.get("uri") or "")
            if not path:
                continue
            if "patch" in kind:
                raw.setdefault("patch_path", path)
            elif "tuned" in kind:
                raw.setdefault("tuned_file", path)
            elif "target" in kind or "source" in kind:
                raw.setdefault("target_file", path)
            elif "overlay" in kind:
                raw.setdefault("final_overlay", path)
            else:
                raw.setdefault("report_path", path)
        raw["artifacts"] = [
            {
                "kind": str(artifact.get("kind") or ""),
                "path": str(artifact.get("path") or artifact.get("uri") or ""),
            }
            for artifact in artifact_rows
            if artifact.get("path") or artifact.get("uri")
        ]

        throughput_after = None
        throughput_before = None
        for measurement_id in adoption.get("measurement_ids") or []:
            measurement = measurement_by_id.get(str(measurement_id), {})
            name = str(measurement.get("name") or "").lower()
            value = _to_float(measurement.get("value"))
            if value is None:
                continue
            if name in {"throughput_after", "final_throughput", "output_throughput"}:
                throughput_after = value
            elif name in {"throughput_before", "baseline_throughput"}:
                throughput_before = value
        if throughput_after is not None:
            raw["tput"] = throughput_after

        delta = _to_float(adoption.get("gain_pct"))
        cumulative += delta or 0.0
        gain_row: dict[str, Any] = {
            "action": raw["action"],
            "variant_name": raw["variant_name"],
            "delta_pct": delta,
            "cum_gain_after": cumulative,
            "ts": raw["ts"],
            "task_id": operation_id,
            "source_phase": raw["source_phase"],
        }
        if throughput_before is not None:
            gain_row["throughput_before"] = throughput_before
        stack.append(raw)
        gains.append(gain_row)

    baseline_tput = next(
        (
            _to_float(gain.get("throughput_before"))
            for gain in gains
            if _to_float(gain.get("throughput_before")) is not None
        ),
        None,
    )
    synthetic_state = {
        "session_id": str(run.get("session_id") or run.get("claw_session_id") or "session"),
        "baseline_tput": baseline_tput,
        "optimization_stack": stack,
        "gain_per_stack_entry": gains,
        "cumulative_gain_validated": cumulative,
        "cumulative_gain_validated_stack_len": len(stack),
        "phase_history": [],
    }
    geak_invocations, forge_invocations = _collect_v4_backend_invocations(
        operations,
        measurements,
        adoptions,
    )
    synthetic_attribution = {
        "gain_per_stack_entry": gains,
        "method": "validated" if gains else "missing",
        "source_breakdown": {
            "validated_total_pct": cumulative,
        },
        "phase_breakdown": {},
        "notes": [
            "Attribution synthesized from validated V4 adoption streams."
        ],
    }
    return collect_optimizations(
        synthetic_state,
        synthetic_attribution,
        geak_invocations,
        forge_invocations,
        warnings,
        gemm_tuning={
            "runs": _collect_v4_gemm_tuning_runs(operations, adoptions),
        },
    )


def collect_recorded_optimizations(
    session_id: str,
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the optimization read model straight from author-time records.

    Every field here traces to something a producer recorded when it happened,
    so ownership, verdicts, and the thresholds behind them survive export
    instead of being re-inferred from business state after the fact.
    """

    measurement_by_id = {
        str(row.get("measurement_id") or ""): row
        for row in measurements
        if isinstance(row, dict) and row.get("measurement_id")
    }
    artifact_by_id = {
        str(row.get("artifact_id") or ""): row
        for row in artifacts
        if isinstance(row, dict) and row.get("artifact_id")
    }
    adoption_by_operation: dict[str, dict[str, Any]] = {}
    for row in adoptions:
        if not isinstance(row, dict):
            continue
        operation_id = str(row.get("operation_id") or "")
        if not operation_id:
            continue
        current = adoption_by_operation.get(operation_id)
        # An operation can transition (adopted then revoked); the last word wins.
        if current is None or str(row.get("adopted_at") or row.get("revoked_at") or "") >= str(
            current.get("adopted_at") or current.get("revoked_at") or ""
        ):
            adoption_by_operation[operation_id] = row

    attempts: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if _work_kind(operation) not in _ATTEMPT_KINDS:
            continue
        attempts.append(
            _recorded_attempt_row(
                operation,
                adoption=adoption_by_operation.get(
                    str(operation.get("operation_id") or ""),
                    {},
                ),
                measurement_by_id=measurement_by_id,
                artifact_by_id=artifact_by_id,
            )
        )
    attempts.sort(
        key=lambda row: (
            _parse_ts(row.get("ended_at") or row.get("started_at")) or float("inf"),
            str(row.get("attempt_id") or ""),
        )
    )

    orphan_adoptions = [
        str(row.get("adoption_id") or "")
        for row in adoptions
        if isinstance(row, dict)
        and str(row.get("operation_id") or "") not in adoption_by_operation
    ]
    if orphan_adoptions:
        warnings.append(
            "optimizations: adoptions reference operations that were never "
            f"recorded: {sorted(set(orphan_adoptions))[:5]}"
        )

    # Reported gain is measured against the session baseline, so each adopted
    # step contributes the percentage points it added to the cumulative figure.
    # An executor's own local gain is relative to whatever it started from,
    # which is not the baseline once anything has already been adopted.
    #
    # ``entries`` is the gain ledger over ``attempts``: one row per adopted and
    # attributable attempt, carrying only what the chain arithmetic needs.
    # Everything descriptive stays on the attempt and is reachable through
    # ``adopted_attempt_id``.
    baseline_tput = _recorded_baseline_throughput(operations, measurement_by_id)
    entries: list[dict[str, Any]] = []
    cumulative = 0.0
    # Throughput the next adopted step is expected to start from: the baseline
    # for the first one, then wherever the previous one left off. A step that
    # starts somewhere else means something moved the workload without being
    # adopted, and that movement belongs to nobody. Crediting it to the next
    # step is how a kernel's reported gain silently absorbs an earlier patch.
    expected_before = baseline_tput
    unattributed = 0.0
    attributed = 0.0
    for attempt in attempts:
        if not attempt.get("adopted") or attempt.get("attribution_eligible") is False:
            continue
        local_gain = _to_float(attempt.get("local_gain_pct"))
        throughput_before = _to_float(attempt.get("throughput_before"))
        throughput_after = _to_float(attempt.get("throughput_after"))
        if baseline_tput and throughput_after:
            started_from = throughput_before or expected_before or baseline_tput
            drift = (
                (started_from - expected_before) / baseline_tput * 100.0
                if expected_before
                else 0.0
            )
            # Percentage points of the baseline this step added. Stated this
            # way the rows sum exactly, with no chaining subtleties to get
            # wrong, and any drift stays outside the sum.
            gain = (throughput_after - started_from) / baseline_tput * 100.0
            gain_method = "baseline_chain"
            unattributed += drift
            cumulative += drift + gain
            expected_before = throughput_after
        else:
            gain = local_gain
            cumulative += local_gain or 0.0
            gain_method = "recorded_adoption" if local_gain is not None else "missing"
        # Summed unrounded, so the reported gap equals the drift exactly rather
        # than trailing it by a rounding step.
        attributed += gain or 0.0
        stack_index = len(entries)
        entries.append(
            {
                "id": f"{session_id}:optimization:{stack_index}",
                "stack_index": stack_index,
                "adopted_attempt_id": attempt["attempt_id"],
                "adoption_id": attempt.get("adoption_id"),
                "source": attempt["agent"],
                "source_method": attempt["agent_method"],
                "optimization_kind": attempt["kind"],
                "name": attempt["name"],
                "backend": attempt.get("backend") or None,
                # Gain against the session baseline: the only figure that may
                # be summed, and the one ``cumulative_gain_pct`` is built from.
                "gain_pct": round(gain, 6) if gain is not None else None,
                "gain_method": gain_method,
                # The executor's own figure, carried so the two are visibly
                # different numbers rather than one ambiguous field.
                "local_gain_pct": (
                    round(local_gain, 6) if local_gain is not None else None
                ),
                "cumulative_gain_pct": round(cumulative, 6),
                "throughput_after": throughput_after,
                "validated": True,
                "ts": attempt.get("ended_at") or "",
            }
        )

    # Below this the drift is float noise from re-serialized throughputs, well
    # under any measurement's own repeatability.
    if abs(unattributed) > 0.01:
        warnings.append(
            "optimizations: "
            f"{unattributed:+.6f}pp of the session's gain sits between adopted "
            "steps and belongs to no attempt; it is reported as "
            "validation.unattributed_gain_pct rather than credited to the step "
            "that follows it"
        )

    summary_by_agent = _summarize_by_agent(
        attempts,
        {
            str(entry["adopted_attempt_id"]): _to_float(entry.get("gain_pct")) or 0.0
            for entry in entries
        },
    )
    backend_attempts = _collect_backend_attempts(
        session_id,
        geak_invocations,
        forge_invocations,
    )

    summary_by_source = _empty_summary()
    summary_by_kind: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source = str(entry["source"])
        bucket = summary_by_source.setdefault(source, {"keeps": 0, "total_gain_pct": 0.0})
        gain = _to_float(entry.get("gain_pct")) or 0.0
        bucket["keeps"] += 1
        bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]) + gain, 6)
        if source == "kernel_agent":
            backend = str(entry.get("backend") or "unattributed")
            by_backend = bucket.setdefault("by_backend", {})
            if backend not in by_backend:
                backend = "unattributed"
            backend_bucket = by_backend.setdefault(
                backend,
                {"keeps": 0, "total_gain_pct": 0.0},
            )
            backend_bucket["keeps"] += 1
            backend_bucket["total_gain_pct"] = round(
                float(backend_bucket["total_gain_pct"]) + gain,
                6,
            )
        kind_bucket = summary_by_kind.setdefault(
            str(entry.get("optimization_kind") or "unknown"),
            _empty_kind_summary(),
        )
        kind_bucket["keeps"] += 1
        kind_bucket["total_gain_pct"] = round(
            float(kind_bucket["total_gain_pct"]) + gain,
            6,
        )

    non_attributable = [
        attempt
        for attempt in attempts
        if attempt.get("adopted") and attempt.get("attribution_eligible") is False
    ]
    return {
        "schema_version": OPTIMIZATIONS_SCHEMA_VERSION,
        "source_of_truth": "recorder",
        "attempts": attempts,
        "entries": entries,
        "backend_attempts": backend_attempts,
        "summary_by_agent": summary_by_agent,
        "summary_by_source": summary_by_source,
        "summary_by_kind": summary_by_kind,
        "validation": {
            "method": "recorded_adoptions",
            "validated_at_stack_len": len(entries),
            # What the session moved end to end, and how much of that any
            # attempt is willing to claim. The difference is the part no
            # adopted step accounts for, and it is stated rather than absorbed.
            "validated_total_gain_pct": round(cumulative, 6),
            "attributed_total_gain_pct": round(attributed, 6),
            "unattributed_gain_pct": round(unattributed, 6),
            "attribution_gap_pct": round(cumulative - attributed, 6),
            "attempt_count": len(attempts),
            "keep_count": len(entries),
            "non_attributable_keep_count": len(non_attributable),
            "notes": [
                "Projected from author-time recorder streams "
                "(operations/adoptions/measurements/artifacts)."
            ],
        },
        "gemm_tuning_runs": _collect_v4_gemm_tuning_runs(operations, adoptions),
    }

