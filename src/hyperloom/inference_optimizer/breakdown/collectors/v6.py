"""Additive V6 projections built from the existing V5 evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


from ..kb_timeline import collect_kb_events
from ...session.sbd_v6 import read_timeline_events
from ._common import (
    _dict_rows,
    _first,
    _mapping,
    _optional_bool,
    _parse_iso_unix as _timestamp_number,
    _to_float as _optional_float,
    _to_int as _optional_int,
)
from .v6_stages import project_conc_sweep_event


_SUCCESS_STOP_REASONS = frozenset(
    {
        "target_reached",
        "global_converged",
        "time_exhausted",
        "max_ticks",
        "sweep_done",
    }
)
_ABORTED_STOP_REASONS = frozenset({"signal", "user_stop_requested"})
_MODEL_GATE_STOP_REASONS = frozenset(
    {
        "model_context_window_too_small",
        "model_config_incompatible",
        "unsupported_model_arch",
    }
)
# Structural model fields carried verbatim out of ``state.model_info``. Kept in
# lockstep with the recorder's own list (``recorder/session_metadata.py``) so a
# fragment-backed session and a collector fallback expose the same block.
_ARCHITECTURE_FIELDS = (
    "model_family",
    "model_type",
    "architectures",
    "attention_type",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "torch_dtype",
    "kv_cache_dtype",
    "quantization",
    "is_moe",
    "num_experts",
    "num_experts_per_tok",
    "has_shared_expert",
    "num_shared_experts",
)


def _tool_versions(versions: Any) -> dict[str, dict[str, Any]]:
    """Fold the recorded ``versions`` item stream into a per-tool provenance map.

    Each tool keeps its full recorded row (``root_dir`` / ``commit`` /
    ``version``); a bare string is the shape a legacy session recorded before
    the row carried provenance.
    """
    if not isinstance(versions, dict):
        return {}
    tools: dict[str, dict[str, Any]] = {}
    for name, value in versions.items():
        tool = str(name or "").strip()
        if not tool:
            continue
        if isinstance(value, str):
            tools[tool] = {"tool": tool, "version": value or None}
            continue
        if isinstance(value, dict):
            row = {k: v for k, v in value.items() if v not in (None, "")}
            row.setdefault("tool", tool)
            tools[tool] = row
    return tools


def _architecture(workload: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    """The structural model summary, carried whole rather than digested.

    ``model_class`` is the operator's declaration when present and a dense/moe
    split otherwise; every other field is the parsed ``config.json`` summary as
    ``summarize_model_config`` produced it.
    """
    if not workload and not model_info:
        return {}
    model_class = str(workload.get("model_class") or "").strip()
    if not model_class and model_info:
        model_class = "moe" if bool(model_info.get("is_moe")) else "dense"
    architecture: dict[str, Any] = {"model_class": model_class}
    for field in _ARCHITECTURE_FIELDS:
        if field in model_info:
            architecture[field] = model_info[field]
    return architecture


def langfuse_block(langfuse: dict[str, Any]) -> dict[str, Any]:
    """The trace entrypoint plus the reason a disabled session pushed nothing."""
    config = langfuse.get("config") if isinstance(langfuse.get("config"), dict) else {}
    trace_url = langfuse.get("trace_url")
    if not trace_url:
        host = str(config.get("host") or "").rstrip("/")
        trace_id = str(langfuse.get("trace_id") or "").strip()
        if host and trace_id:
            trace_url = f"{host}/trace/{trace_id}"
    counts = langfuse.get("counts")
    return {
        "enabled": bool(langfuse.get("enabled")),
        "disabled_reason": langfuse.get("disabled_reason") or None,
        "trace_id": langfuse.get("trace_id") or None,
        "session_id": langfuse.get("session_id") or None,
        "trace_url": trace_url or None,
        "counts": {str(k): int(v or 0) for k, v in counts.items()} if isinstance(counts, dict) else {},
    }


def collect_v6_metadata(
    *,
    exported_at_utc: str,
    session: dict[str, Any],
    workload: dict[str, Any],
    model_info: dict[str, Any],
    langfuse: dict[str, Any],
    versions: dict[str, Any],
    state: dict[str, Any],
    warnings: list[str],
    recorded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the V6 ``metadata`` section, preferring recorded facts.

    Every field here is recorded at the moment it is decided (see
    ``recorder/session_metadata.py``); this collector is the fallback for a
    session whose fragments are missing or partial, and ``recorded`` is
    overlaid leaf-by-leaf on top of it so a live-recorded run is never
    downgraded to a re-derived value.

    ``exported_at_utc`` and ``warnings`` are facts about the export itself,
    not the session, so they are never taken from a fragment.

    Args:
        exported_at_utc: When this export ran.
        session: The resolved ``session`` section.
        workload: The resolved ``workload`` section.
        model_info: The parsed model config summary.
        langfuse: The Langfuse push receipt.
        versions: The assembled per-tool version map.
        state: Parsed ``state.json``.
        warnings: The V6 warnings accumulated by this export.
        recorded: The recorder's ``metadata`` fragment, when present.

    Returns:
        The ``metadata`` section.
    """
    recovery = session.get("recovery") if isinstance(session.get("recovery"), dict) else {}
    image = str(session.get("image") or "").strip()
    task_config = {
        "model_name": str(workload.get("model_name") or ""),
        "model_path": str(workload.get("model_path") or ""),
        "framework_name": str(workload.get("framework_name") or ""),
        "framework_version": str(workload.get("framework_version") or ""),
        "gpu_type": str(workload.get("gpu_type") or ""),
        "tp": workload.get("tp"),
        "conc": workload.get("conc"),
        "isl": workload.get("isl"),
        "osl": workload.get("osl"),
        "precision": str(workload.get("precision") or ""),
        "max_model_len": workload.get("max_model_len"),
        "objective": dict(workload.get("objective") or {}),
        "launch_env": dict(state.get("operator_extra_env") or {}),
        "launch_server_args": str(state.get("operator_server_args") or state.get("server_args") or ""),
        "architecture": _architecture(workload, model_info),
    }
    projected = {
        "versions": {
            "framework": str(workload.get("framework_name") or "") or None,
            "framework_version": str(workload.get("framework_version") or "") or None,
            "tools": _tool_versions(versions),
        },
        "session": {
            "session_id": str(session.get("session_id") or ""),
            "claw_session_id": session.get("claw_session_id"),
            "sandbox_user_id": session.get("sandbox_user_id"),
            "created_at_utc": str(session.get("created_at_utc") or ""),
            "start_ts": str(session.get("start_ts") or ""),
            "ended_at_utc": str(session.get("ended_at_utc") or ""),
            "host": str(session.get("host") or ""),
            "session_dir": str(session.get("session_dir") or ""),
            "user_data_path": str(session.get("user_data_path") or ""),
            "code_revision": str(session.get("code_revision") or ""),
            "pid": int(session.get("pid") or 0),
            "image": image or None,
            "image_id": (image.split("/")[-1] or None) if image else None,
            "max_minutes": int(session.get("max_minutes") or 0),
            "elapsed_minutes": float(session.get("elapsed_minutes") or 0.0),
            # Absent a per-leg history the fallback can only report the one
            # window it can measure, so a never-resumed session reads the same
            # either way and a resumed one is under-reported rather than
            # charged the gap between its legs.
            "total_elapsed_minutes": float(session.get("elapsed_minutes") or 0.0),
            "tick_count": int(session.get("tick_count") or 0),
            "recovery": {
                "recovered": bool(recovery.get("recovered")),
                "crash_count": int(recovery.get("crash_count") or 0),
                "crash_timestamps": list(recovery.get("crash_timestamps") or []),
                "degraded_mode": bool(recovery.get("degraded_mode")),
                "resume_pending_revalidation": bool(recovery.get("resume_pending_revalidation")),
                "last_tick_exception": recovery.get("last_tick_exception"),
            },
        },
        "task_config": task_config,
        "langfuse": langfuse_block(langfuse),
    }
    metadata = _overlay_recorded(projected, recorded)
    return {"exported_at_utc": exported_at_utc, **metadata, "warnings": list(warnings)}


def _overlay_recorded(projected: dict[str, Any], recorded: Any) -> dict[str, Any]:
    """Overlay recorded leaves onto the projection, keeping projected fallbacks.

    An empty recorded value is absence of evidence and never overwrites a
    projected one, but it does land on a key the projection has no source for.
    """
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in projected.items()}
    if not isinstance(recorded, dict) or not recorded:
        return merged
    for block, value in recorded.items():
        if not isinstance(value, dict):
            continue
        target = merged.get(block)
        merged[block] = _overlay_leaves(target, value) if isinstance(target, dict) else dict(value)
    return merged


def _overlay_leaves(target: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Leaf-wise overlay of ``recorded`` onto ``target`` (recursing into dicts)."""
    merged = dict(target)
    for key, value in recorded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _overlay_leaves(merged[key], value)
        elif _recorded_leaf(value) or key not in merged:
            merged[key] = value
    return merged


def _recorded_leaf(value: Any) -> bool:
    """Whether a recorded leaf carries evidence (``0`` / ``""`` / ``None`` do not)."""
    if value is None or value == "":
        return False
    if isinstance(value, (list, dict)):
        return bool(value)
    return not (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


def _projected(
    stage: str,
    project: Callable[[], Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Run one stage projector so its failure costs only its own events.

    The exporter already wraps this whole collector, but that granularity is
    too coarse to honor what V6 promises. A single ``_safe_collect`` around the
    lot means one projector raising on a malformed field discards the durable
    ``install`` / ``model_gate`` events read moments earlier and every other
    stage that projected cleanly — so a session that failed at the model gate,
    whose gate event is the only thing worth reporting, can lose it to a
    kernel-stage bug it never reached.

    Args:
        stage (str): Stage name, used to name the projector in the warning.
        project (Callable[[], Any]): Returns one event, a list of events, or
            ``None``.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        list[dict[str, Any]]: The projected events, or ``[]`` on failure.
    """
    try:
        result = project()
    except Exception as exc:  # noqa: BLE001 — one stage must not cost the timeline
        warnings.append(f"v6.timeline.{stage}: projection failed ({type(exc).__name__}: {exc}); stage omitted")
        return []
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    return [event for event in result if isinstance(event, dict)]


def _recorded_types(timeline: list[dict[str, Any]], event_type: str) -> bool:
    """Whether the durable events already hold one of a type.

    Args:
        timeline (list[dict[str, Any]]): The durable events read back.
        event_type (str): The type to look for.

    Returns:
        bool: True when at least one durable event has that type.
    """
    wanted = str(event_type)
    return any(isinstance(event, dict) and str(event.get("type") or "") == wanted for event in timeline)


def collect_v6_timeline(
    session_dir: Path,
    warnings: list[str],
    *,
    state: dict[str, Any] | None = None,
    conc_sweep_summary: Any = None,
    phase_timeline: Any = None,
) -> list[dict[str, Any]]:
    """Load durable events and project what is not yet recorded.

    ``install``, ``model_gate``, ``baseline``, ``roofline``, ``kernel``,
    ``conc_sweep`` and ``framework_agent`` are read back from the durable
    event directory: the first two run before the Coordinator exists, and the
    rest are recorded by the phase or the action that produces them, which
    knows things no projection over ``state.json`` can recover -- when the
    work started, and the thresholds a decision actually ruled on, most
    plainly. ``conc_sweep`` and ``warm_replay`` are also recorded now, so what
    remains here is the fallback for a session recorded before those events
    existed: each is projected only when the durable read found none, from
    V5 sections the exporter has already built -- so both keyword arguments
    are optional: a caller that passes none still gets every durable event.

    Every projection is isolated (see :func:`_projected`). The durable events
    are read first and are never discarded by a later stage's failure.
    """
    timeline = read_timeline_events(session_dir, warnings=warnings)
    state = state if isinstance(state, dict) else {}
    if not _recorded_types(timeline, "conc_sweep"):
        # A sweep the executor recorded already carries everything the
        # projection could rebuild and the decisions it could not, so
        # projecting alongside it would publish the same sweep twice -- the
        # second time worse. The projection is what a session recorded before
        # this event existed still gets.
        timeline.extend(
            _projected(
                "conc_sweep",
                lambda: project_conc_sweep_event(conc_sweep_summary, state, phase_timeline, warnings),
                warnings,
            )
        )
    if not _recorded_types(timeline, "warm_replay"):
        # Same reason as the sweep above: PRELUDE records the replay with the
        # skip code or the gate that decided it, which a projection over
        # ``state.json`` cannot recover. Projecting alongside it would publish
        # the same replay twice.
        timeline.extend(_projected("kb", lambda: collect_kb_events(session_dir, state, warnings), warnings))
    indexed = list(enumerate(timeline))
    indexed.sort(
        key=lambda row: (
            _timestamp_number(_first(row[1].get("start_time"), row[1].get("end_time"))) is None,
            _timestamp_number(_first(row[1].get("start_time"), row[1].get("end_time"))) or 0.0,
            row[0],
        )
    )
    return [event for _, event in indexed]


def _outcome_status(stop_reason: str) -> str:
    if stop_reason in _SUCCESS_STOP_REASONS:
        return "completed"
    if stop_reason in _ABORTED_STOP_REASONS or not stop_reason:
        return "aborted"
    return "failed"


def _stage_reached(
    state: dict[str, Any],
    stop_reason: str,
    timeline: list[dict[str, Any]],
) -> str:
    if stop_reason in _MODEL_GATE_STOP_REASONS:
        return "model_gate"
    phase = str(state.get("phase") or "").strip().upper()
    history = state.get("phase_history")
    if isinstance(history, list):
        for row in reversed(history):
            if isinstance(row, dict) and str(row.get("to_phase") or "").strip():
                phase = str(row.get("to_phase") or "").strip().upper()
                break
    if phase == "PRELUDE":
        if state.get("roofline_snapshots") or state.get("last_roofline") or state.get("roofline_attempts"):
            return "roofline"
        if state.get("last_profile_trace") or state.get("last_profile") or state.get("profile_attempts"):
            return "profile"
        if state.get("warm_replay_attempted") or state.get("warm_replay_outcome") or state.get("warm_replay_pending"):
            return "warm_replay"
        if _recorded_types(timeline, "enablement"):
            return "enablement"
        # A session recorded before the enablement event existed still has to
        # be read off state, by probing the six fields any of which means the
        # lane did something. The event replaces the probe with the fact.
        enablement = state.get("enablement")
        if isinstance(enablement, dict) and any(
            (
                int(enablement.get("attempts") or 0) > 0,
                bool(enablement.get("pending")),
                bool(enablement.get("validation_pending")),
                bool(enablement.get("succeeded")),
                bool(enablement.get("launch_log")),
                bool(enablement.get("inflight_task_id")),
            )
        ):
            return "enablement"
        baseline_tput = state.get("baseline_tput")
        if (
            isinstance(baseline_tput, (int, float))
            and baseline_tput > 0
            or state.get("last_baseline")
            or state.get("baseline_attempts")
            or int(state.get("baseline_failure_streak") or 0) > 0
        ):
            return "baseline"
        if (
            state.get("warm_start_ts")
            or state.get("warm_start_recipe")
            or state.get("warm_start_pitfalls")
            or state.get("warm_start_lessons")
            or state.get("warm_start_context")
        ):
            return "warm_start"
    phase_map = {
        "FRAMEWORK_AGENT": "framework_agent",
        "EXPLORE": "framework_agent",
        "KERNEL_AGENT": (
            "kernel"
            if any(state.get(key) for key in ("last_kernel_opt", "last_fusion", "last_gemm_tuning", "last_collective"))
            else "kernel_agent"
        ),
        "SWEEP": "conc_sweep",
        "CLOSE": "close",
    }
    if phase in phase_map:
        return phase_map[phase]
    if timeline:
        return str(timeline[-1].get("type") or "")
    return "install"


#: The figures ``outcome.baseline`` publishes, named as the ``baseline`` event's
#: own measurement block names them so the read is a straight lift.
_BASELINE_OUTCOME_FIELDS = (
    "throughput_tok_s_per_gpu",
    "accuracy",
    "ttft_mean_ms",
    "e2el_mean_ms",
)

#: Action statuses whose figure the session went on to use. ``degraded`` is a
#: baseline that stands on its cold warmup round because the budget would not
#: hold the hot pass: knowingly depressed, but it is the number every later gain
#: in the session was read against, so it is the number to publish.
_ANCHORING_BASELINE_STATUSES = frozenset({"succeeded", "degraded"})


def _baseline_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the session's anchoring baseline off the ``baseline`` events.

    Three different dispatches reach the baseline executor and each lands an
    action on a ``baseline`` event: the genuine baseline, ``replay_warm_recipe``,
    and the kernel lane's throughput-only probes (integrate re-baseline, stack
    validation) which carry ``kind="baseline"`` literally. Only the first
    anchors the session, so the selection reads the action's own
    ``establishes_quality_ref`` -- the flag the executor set from the dispatch
    kind and ``quality_ref_exempt`` -- rather than re-deciding from the kind.

    Args:
        timeline (list[dict[str, Any]]): The assembled V6 timeline.

    Returns:
        dict[str, Any]: The four baseline figures, each ``None`` when the
            timeline holds no anchoring measurement.
    """
    anchors: list[tuple[str, dict[str, Any]]] = []
    for event in timeline:
        if not isinstance(event, dict) or str(event.get("type") or "") != "baseline":
            continue
        for action in _dict_rows(_mapping(event.get("ext")).get("actions")):
            if not _optional_bool(_mapping(action.get("request")).get("establishes_quality_ref")):
                continue
            if str(action.get("status") or "").strip().lower() not in _ANCHORING_BASELINE_STATUSES:
                continue
            # A baseline re-measured after an enablement fix re-anchors the
            # session, so the latest anchor wins. Ordered on the action's own
            # stamps because the actions array is keyed by task id and carries
            # no chronology of its own.
            anchors.append((str(action.get("end_time") or action.get("start_time") or ""), action))
    if not anchors:
        return dict.fromkeys(_BASELINE_OUTCOME_FIELDS)
    anchors.sort(key=lambda row: row[0])
    measurement = _mapping(anchors[-1][1].get("measurement"))
    return {field: _optional_float(measurement.get(field)) for field in _BASELINE_OUTCOME_FIELDS}


def _validation_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the session's gain attribution off the ``stack`` ledger event.

    Every figure here is read, not computed: the ledger event's ``ext`` was
    assembled from rows the orchestrator recorded as each adoption was accepted
    and as each whole-stack validation was measured. The legacy route
    reconstructed the same figures at export from the ``optimizations``
    section, which itself reconstructed them by pairing up rows across three v4
    streams -- and had to publish eight guard counts to report where the three
    disagreed.

    Args:
        timeline (list[dict[str, Any]]): The assembled V6 timeline.

    Returns:
        dict[str, Any]: The ``outcome.validation`` block. ``available`` is
            ``False`` on a session with no ledger event, which means the run
            never reached a close and not that it adopted nothing -- a run that
            kept nothing still closes its ledger, reporting zero adoptions.
    """
    ledger: dict[str, Any] = {}
    for event in timeline:
        if isinstance(event, dict) and str(event.get("type") or "") == "stack":
            ledger = _mapping(event.get("ext"))
    available = bool(ledger)
    buckets = _mapping(_mapping(ledger.get("adoptions")).get("by_source"))

    def _bucket(*names: str) -> dict[str, Any]:
        rows = [_mapping(buckets.get(name)) for name in names]
        return {
            "total_gain_pct": (
                round(sum(_optional_float(row.get("total_gain_pct")) or 0.0 for row in rows), 6)
                if available
                else None
            ),
            "keep_count": sum(_optional_int(row.get("count")) or 0 for row in rows),
            # Adoptions in this bucket whose contribution could not be measured
            # at all. Zero of these is what makes ``total_gain_pct`` a complete
            # account of the bucket rather than a lower bound.
            "unmeasured_keep_count": sum(_optional_int(row.get("unmeasured")) or 0 for row in rows),
        }

    kernel_backends = _mapping(_mapping(buckets.get("kernel")).get("by_backend"))
    return {
        "attributed_gain_pct": _optional_float(ledger.get("attributed_gain_pct")) or 0.0,
        "unattributed_gain_pct": _optional_float(ledger.get("unattributed_gain_pct")) or 0.0,
        "reconciliation_gap_pct": _optional_float(ledger.get("reconciliation_gap_pct")),
        "validated_total_gain_pct": _optional_float(ledger.get("validated_total_gain_pct")),
        "chain_total_gain_pct": _optional_float(ledger.get("chain_total_gain_pct")),
        "attribution": {
            "available": available,
            "by_source": {
                "warm_replay": _bucket("warm_replay"),
                # V6 folds the old Explore phase into Framework Agent, so the
                # two ledger buckets are combined at this boundary.
                "framework_agent": _bucket("framework_agent", "explore"),
                "kernel": {
                    **_bucket("kernel"),
                    "by_backend": {
                        "geak": _bucket_of(kernel_backends, "geak", available),
                        "forge": _bucket_of(kernel_backends, "forge", available),
                    },
                },
            },
        },
        "guards": dict(_mapping(ledger.get("guards"))),
        "notes": _validation_notes(ledger),
    }


def _bucket_of(backends: dict[str, Any], name: str, available: bool) -> dict[str, Any]:
    """Project one kernel backend's slice of the ledger.

    Args:
        backends (dict[str, Any]): The ledger's per-backend split.
        name (str): The backend to project.
        available (bool): Whether the ledger event exists at all.

    Returns:
        dict[str, Any]: The backend's adoption count and summed contribution.
    """
    row = _mapping(backends.get(name))
    return {
        "total_gain_pct": _optional_float(row.get("total_gain_pct")) if available else None,
        "keep_count": _optional_int(row.get("count")) or 0,
        "unmeasured_keep_count": _optional_int(row.get("unmeasured")) or 0,
    }


def _validation_notes(ledger: dict[str, Any]) -> list[str]:
    """Name what the ledger's own figures say is wrong with it.

    The legacy ``notes`` were a fixed prose string explaining how the ledger was
    built. These are findings: each one is present only when the recorded rows
    show the condition it describes, so an empty list is the meaningful case.

    Args:
        ledger (dict[str, Any]): The stack event's ``ext``.

    Returns:
        list[str]: One note per finding, empty when the ledger reconciles.
    """
    if not ledger:
        return []
    notes: list[str] = []
    guards = _mapping(ledger.get("guards"))
    unmeasured = _optional_int(guards.get("unmeasured")) or 0
    if unmeasured:
        notes.append(f"{unmeasured} adoption(s) have no measurable contribution; attributed gain is a lower bound")
    breaks = _optional_int(guards.get("chain_breaks")) or 0
    if breaks:
        notes.append(
            f"the anchor moved outside an adoption {breaks} time(s); "
            "that movement is the unattributed gain, not a rounding error"
        )
    validations = _mapping(ledger.get("validations"))
    if not _optional_int(validations.get("count")):
        notes.append("no whole-stack validation was measured, so the ledger has nothing to reconcile against")
    elif validations.get("at_head") is False:
        notes.append("the last whole-stack validation predates the final adoptions; the total covers a shorter stack")
    return notes


def collect_v6_outcome(
    *,
    session: dict[str, Any],
    final: dict[str, Any],
    state: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project V5 result sections into the V6 ``outcome`` shape."""
    stop_reason = str(session.get("stop_reason") or "").strip()
    outcome_status = _outcome_status(stop_reason)
    for event in reversed(timeline):
        if not isinstance(event, dict) or str(event.get("type") or "") not in {"install", "model_gate"}:
            continue
        if str(event.get("status") or "").strip().lower() == "failed":
            outcome_status = "failed"
        break
    return {
        "stop_reason": stop_reason,
        "status": outcome_status,
        "stage_reached": _stage_reached(state, stop_reason, timeline),
        "baseline": _baseline_from_timeline(timeline),
        "final": {
            "throughput_tok_s_per_gpu": final.get("throughput_tok_s_per_gpu"),
            "gain_pct": final.get("cumulative_gain_pct_validated", 0.0),
            "action_path": list(final.get("action_path") or []),
            "extra_envs": dict(final.get("extra_envs") or {}),
            "extra_server_args": str(final.get("extra_server_args") or ""),
        },
        "validation": _validation_from_timeline(timeline),
    }


__all__ = [
    "collect_v6_metadata",
    "langfuse_block",
    "collect_v6_outcome",
    "collect_v6_timeline",
]
