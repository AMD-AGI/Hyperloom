# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a session's recipe attributes say about where its optimizations came from."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Collection, Mapping, NamedTuple, Sequence

from hyperloom.inference_optimizer.session.optimization_journal import OUTCOME_KEEP, OUTCOME_REVERT

from .workload_tags import workload_shape_tags

#: The probe's own word for a version it could not read.
_UNIDENTIFIED_VERSION: str = "unknown"


class AttemptProvenance(NamedTuple):
    """Where a session's proven optimizations came from, and what it reverted.

    Keepers are indexed twice because a stack layer may identify itself either
    by name or by the research gap it closed.
    """

    kept_by_name: dict[str, str]
    kept_by_gap: dict[str, str]
    reverted: list[dict[str, Any]]


def _known_version(raw: Any) -> str:
    version = str(raw or "").strip()
    return "" if version == _UNIDENTIFIED_VERSION else version


def _runtime_version_tags(fingerprint_meta: Mapping[str, Any], framework: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    framework_key = framework.lower()
    if framework_key in {"sglang", "vllm"}:
        active = _known_version(fingerprint_meta.get(framework_key))
        if active:
            tags["framework_version"] = active
    for source_key, tag in (("rocm", "rocm_version"), ("aiter", "aiter_version"), ("image_digest", "image_digest")):
        version = _known_version(fingerprint_meta.get(source_key))
        if version:
            tags[tag] = version
    return tags


def _baseline_workload_tags(workload_extra: Mapping[str, Any]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for key in ("max_running_requests", "max_num_seqs"):
        value = workload_extra.get(key)
        if isinstance(value, int) and value > 0:
            tags[key] = value
    for key in ("chunked_prefill_enabled", "enable_torch_compile"):
        value = workload_extra.get(key)
        if isinstance(value, bool):
            tags[key] = value
    for key in ("quant_scheme", "workload_mode"):
        value = workload_extra.get(key)
        if isinstance(value, str) and value.strip():
            tags[key] = value.strip()
    return tags


def workload_kb_tags(state: Any) -> dict[str, Any]:
    """The workload-shape tags describing the session this recipe came from."""
    tags: dict[str, Any] = {}
    # ``model_family`` is deliberately absent, being no part of the
    # seven-dimension Recipe identity.
    for attr in ("framework", "model_class", "model_name"):
        text = str(getattr(state, attr, "") or "").strip()
        if text:
            tags[attr] = text
    precision = getattr(state, "precision", None)
    if precision not in (None, "", 0):
        tags["precision"] = precision
    tags.update(workload_shape_tags(state))
    fingerprint_meta = state.stack_fingerprint_meta or {}
    if isinstance(fingerprint_meta, dict):
        tags.update(_runtime_version_tags(fingerprint_meta, str(tags.get("framework") or "")))
    workload_extra = state.baseline_workload_extra or {}
    if isinstance(workload_extra, dict):
        tags.update(_baseline_workload_tags(workload_extra))
    return tags


def _reverted_attempt_row(name: str, attempt: Mapping[str, Any], source: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "reason": "reverted",
        "gain_pct": attempt.get("gain_pct"),
    }
    if source:
        row["source"] = source
    return row


def _gap_attempt_provenance(gap: Mapping[str, Any]) -> AttemptProvenance:
    source = str(gap.get("provenance") or "").strip()
    canonical_id = str(gap.get("canonical_id") or "").strip()
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for attempt in gap.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        variant = str(attempt.get("variant_name") or "").strip()
        kernel = str(attempt.get("kernel_id") or "").strip()
        outcome = str(attempt.get("outcome") or "").strip().upper()
        if outcome == OUTCOME_KEEP and source:
            for name in (variant, kernel):
                if name:
                    kept_by_name.setdefault(name, source)
            if canonical_id:
                kept_by_gap.setdefault(canonical_id, source)
        elif outcome == OUTCOME_REVERT and (variant or kernel):
            reverted.append(_reverted_attempt_row(variant or kernel, attempt, source))
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def attempt_provenance(state: Any) -> AttemptProvenance:
    """Trace each proven optimization back to the research hint it came from."""
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for gap in state.gaps or []:
        if not isinstance(gap, dict):
            continue
        from_gap = _gap_attempt_provenance(gap)
        for name, source in from_gap.kept_by_name.items():
            kept_by_name.setdefault(name, source)
        for canonical_id, source in from_gap.kept_by_gap.items():
            kept_by_gap.setdefault(canonical_id, source)
        reverted.extend(from_gap.reverted)
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def _best_config(current_best: Mapping[str, Any], optimization_stack: Sequence[Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_args = current_best.get("extra_server_args")
    if current_args:
        config["extra_server_args"] = str(current_args)
    for key in ("extra_envs", "name", "tput", "accuracy"):
        if key in current_best:
            config[key] = current_best[key]
    last = optimization_stack[-1] if optimization_stack else None
    # The last layer's extra_server_args is the cumulative stack; its candidate_extra_server_args is only its delta.
    stack_args = str(last.get("extra_server_args") or "").strip() if isinstance(last, dict) else ""
    if stack_args:
        config["extra_server_args"] = stack_args
    return config


def _what_worked(
    optimization_stack: Sequence[Any],
    gain_per_entry: Sequence[Any],
    provenance: AttemptProvenance,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(optimization_stack):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("variant_name") or entry.get("name") or entry.get("kernel_id") or "")
        row: dict[str, Any] = {
            "name": name,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
            "extra_envs": dict(entry.get("extra_envs") or {}),
            "gain_pct": gain_per_entry[index] if index < len(gain_per_entry) else None,
        }
        # The layer's gap id wins; the variant or kernel name answers for
        # layers recorded before gap ids were stamped.
        gap_id = str(entry.get("gap_canonical_id") or "").strip()
        by_name = provenance.kept_by_name
        source = (provenance.kept_by_gap.get(gap_id) if gap_id else "") or by_name.get(name)
        source = source or by_name.get(str(entry.get("kernel_id") or "")) or ""
        if source:
            row["source"] = source
        rows.append(row)
    return rows


def _what_failed(recent_failures: Sequence[Any], reverted: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in recent_failures[-10:]:
        if isinstance(failure, dict):
            rows.append(
                {
                    "name": str(failure.get("name") or failure.get("action") or ""),
                    "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                }
            )
    rows.extend(reverted)
    return rows


def _session_row(
    state: Any,
    session_name: str,
    optimization_stack: Collection[Any],
    final_throughput: float,
) -> dict[str, Any]:
    validated_stack_len = int(state.cumulative_gain_validated_stack_len or 0)
    actions = [
        str(entry.get("variant_name") or entry.get("name") or entry.get("action") or "").strip()
        for entry in optimization_stack
        if isinstance(entry, dict)
    ]
    return {
        "session_id": str(state.recipe_kb_session_id or session_name),
        "gain_pct": float(state.cumulative_gain_validated or 0.0),
        "stack_len": validated_stack_len or len(optimization_stack),
        "throughput_before": float(state.baseline_tput or 0.0),
        "throughput_after": final_throughput,
        "date": datetime.now(timezone.utc).isoformat(),
        "actions_taken": [name for name in actions if name],
    }


def recipe_attrs_from_state(state: Any, session_name: str) -> dict[str, Any]:
    """Materialise the recipe-shaped view of one session's settled state.

    ``framework_version`` is left unset here because it is manifest-derived; the
    T0 backfill writes it.
    """
    optimization_stack = state.optimization_stack or []
    # A session that does not sediment its attempts reports no provenance.
    provenance = attempt_provenance(state) if state.recipe_sediment_enabled else AttemptProvenance({}, {}, [])
    final_throughput = float(state.current_best.get("tput", 0.0))
    return {
        "best_config": _best_config(state.current_best, optimization_stack),
        "best_throughput": final_throughput,
        "what_worked": _what_worked(optimization_stack, state.gain_per_stack_entry or [], provenance),
        "what_failed": _what_failed(state.last_action_failures or [], provenance.reverted),
        "last_profiled": str(state.cumulative_gain_validated_ts or ""),
        "workload": workload_kb_tags(state),
        "sessions": [_session_row(state, session_name, optimization_stack, final_throughput)],
    }
