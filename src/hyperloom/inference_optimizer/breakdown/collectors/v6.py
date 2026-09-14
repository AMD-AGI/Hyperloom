"""The V6 sections projected from the recorder's own record of the session.

``metadata`` and ``outcome`` are assembled from the durable timeline events,
the close-out's fragment, and the launch inputs that predate the recorder.
Nothing here re-derives a fact the run could have stated: where a figure was
once recovered by walking the session directory, it is now read off the event
that measured it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...session.sbd_v6 import read_timeline_events
from ..stop_reasons import MODEL_GATE_STOP_REASONS, outcome_status as _outcome_status
from ._common import (
    _dict_rows,
    _first,
    _mapping,
    _optional_bool,
    _parse_iso_unix as _timestamp_number,
    _to_float as _optional_float,
    _to_int as _optional_int,
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
            # Probed per tool the moment it is first used and written straight
            # into the recorded ``metadata`` singleton, so there is nothing to
            # re-derive here: the overlay supplies the whole map.
            "tools": {},
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
) -> list[dict[str, Any]]:
    """Read the session's durable events back, in the order they happened.

    Every event type is recorded by the phase or the action that produces it:
    ``install`` and ``model_gate`` run before the Coordinator exists, and the
    rest know things no projection over ``state.json`` could recover -- when
    the work started, and the thresholds a decision actually ruled on, most
    plainly.

    Args:
        session_dir (Path): The session directory holding the event files.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        list[dict[str, Any]]: The events, oldest first, with the ones carrying
            no timestamp kept in read order behind those that do.
    """
    timeline = read_timeline_events(session_dir, warnings=warnings)
    indexed = list(enumerate(timeline))
    indexed.sort(
        key=lambda row: (
            _timestamp_number(_first(row[1].get("start_time"), row[1].get("end_time"))) is None,
            _timestamp_number(_first(row[1].get("start_time"), row[1].get("end_time"))) or 0.0,
            row[0],
        )
    )
    return [event for _, event in indexed]


def _stage_reached(
    state: dict[str, Any],
    stop_reason: str,
    timeline: list[dict[str, Any]],
) -> str:
    if stop_reason in MODEL_GATE_STOP_REASONS:
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
            if any(state.get(key) for key in ("last_kernel_opt", "last_fusion", "last_gemm_tuning"))
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
                round(sum(_optional_float(row.get("total_gain_pct")) or 0.0 for row in rows), 6) if available else None
            ),
            "keep_count": sum(_optional_int(row.get("count")) or 0 for row in rows),
            # Adoptions in this bucket whose contribution could not be measured
            # at all. Zero of these is what makes ``total_gain_pct`` a complete
            # account of the bucket rather than a lower bound.
            "unmeasured_keep_count": sum(_optional_int(row.get("unmeasured")) or 0 for row in rows),
        }

    kernel_backends = _mapping(_mapping(buckets.get("kernel")).get("by_backend"))
    validations = _mapping(ledger.get("validations"))
    settled = _mapping(validations.get("settled"))
    return {
        "attributed_gain_pct": _optional_float(ledger.get("attributed_gain_pct")) or 0.0,
        "unattributed_gain_pct": _optional_float(ledger.get("unattributed_gain_pct")) or 0.0,
        "reconciliation_gap_pct": _optional_float(ledger.get("reconciliation_gap_pct")),
        "validated_total_gain_pct": _optional_float(ledger.get("validated_total_gain_pct")),
        "chain_total_gain_pct": _optional_float(ledger.get("chain_total_gain_pct")),
        "adoption_count": _optional_int(_mapping(ledger.get("adoptions")).get("count")) or 0,
        # Which stack the settled figure was measured on, and when. Both are
        # read off the validation row that produced the figure rather than
        # inferred from the stack's length at export.
        "validated_at_stack_len": _optional_int(settled.get("stack_len")),
        "validated_ts": str(settled.get("ts") or "") or None,
        "measurement_basis": str(settled.get("measurement_basis") or "") or None,
        # The latency the settled measurement reported, carried on the same row
        # as the throughput it was measured beside.
        "ttft_mean_ms": _optional_float(settled.get("ttft_mean_ms")),
        "e2el_mean_ms": _optional_float(settled.get("e2el_mean_ms")),
        "ttft_e2el_source": str(settled.get("ttft_e2el_source") or "") or None,
        "server_launch_flags": str(settled.get("server_launch_flags") or "") or None,
        "workspace": str(settled.get("workspace") or "") or None,
        # The settled figure does not cover the stack that shipped: adoptions
        # landed after it was measured. ``at_head`` is absent on a session with
        # no ledger, which is not the same as a stack that outran its
        # validation, so the negation is only taken when the ledger exists.
        "stack_changed_after_validation": (not bool(validations.get("at_head")) if available else False),
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


#: Below this a reconciliation gap is float noise from re-serialized
#: throughputs, well under any measurement's own repeatability.
_RECONCILIATION_NOISE_PP = 0.01


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
    gap = _optional_float(ledger.get("reconciliation_gap_pct"))
    if gap is not None and abs(gap) > _RECONCILIATION_NOISE_PP:
        # The one finding worth alerting on: the whole and the parts were both
        # measured, and they disagree, so one of the two figures is wrong.
        notes.append(
            f"the whole-stack measurement and the sum of the adoptions differ by {gap:+.2f} pp; "
            "either an adoption is missing from the ledger or its recorded throughputs disagree "
            "with the end-to-end measurement"
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
    close: dict[str, Any],
    state: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the V6 ``outcome`` block off the timeline and the close-out.

    Args:
        session (dict[str, Any]): The session section, read for the stop
            reason.
        close (dict[str, Any]): The ``close`` section, whose ``final_recipe``
            states the configuration the session ended on. The close-out is
            the only author of that fact: the stack ledger keeps an adoption
            row a later revert does not retract, and a session that adopted
            nothing has no ledger event while still ending on its baseline.
        state (dict[str, Any]): Parsed ``state.json``, read for the stage the
            run reached.
        timeline (list[dict[str, Any]]): The assembled V6 timeline.

    Returns:
        dict[str, Any]: The ``outcome`` block.
    """
    stop_reason = str(session.get("stop_reason") or "").strip()
    outcome_status = _outcome_status(stop_reason)
    for event in reversed(timeline):
        if not isinstance(event, dict) or str(event.get("type") or "") not in {"install", "model_gate"}:
            continue
        if str(event.get("status") or "").strip().lower() == "failed":
            outcome_status = "failed"
        break
    validation = _validation_from_timeline(timeline)
    recipe = _mapping(close.get("final_recipe"))
    return {
        "stop_reason": stop_reason,
        "status": outcome_status,
        "stage_reached": _stage_reached(state, stop_reason, timeline),
        "baseline": _baseline_from_timeline(timeline),
        "final": {
            "throughput_tok_s_per_gpu": _optional_float(recipe.get("throughput")),
            # The ledger's own settled figure rather than a second tally of it:
            # the validation row that measured the whole stack is what the
            # session's total means, and asking two sources the same question
            # is how the export came to publish an answer nothing measured.
            "gain_pct": validation.get("validated_total_gain_pct") or 0.0,
            "action_path": [str(step) for step in recipe.get("action_path") or []],
            "extra_envs": dict(_mapping(recipe.get("extra_envs"))),
            "extra_server_args": str(recipe.get("extra_server_args") or ""),
            # Read off the validation row that measured the final stack: the
            # settled figure and the latency and launch beside it come from one
            # benchmark, so pairing them here cannot mismatch.
            **_measured_final(validation, recipe),
        },
        "validation": validation,
    }


def _measured_final(validation: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """Take the final stack's latency and launch off the settled validation.

    Args:
        validation (dict[str, Any]): The ``outcome.validation`` block.
        recipe (dict[str, Any]): The close-out's ``final_recipe``, read for the
            latency pair when the session has no settled validation -- a run
            that never validated its stack still ends on a configuration, and
            the close-out's reading of it is the only one there is.

    Returns:
        dict[str, Any]: The latency pair and the ``invocation`` block, whose
            ``framework_args_source`` names where the flags came from rather
            than how far a disk walk had to go to find them.
    """
    settled = bool(validation.get("validated_ts"))
    latency = validation if settled else recipe
    flags = str(validation.get("server_launch_flags") or "")
    return {
        "ttft_mean_ms": _optional_float(latency.get("ttft_mean_ms")),
        "e2el_mean_ms": _optional_float(latency.get("e2el_mean_ms")),
        "invocation": {
            "framework_args": flags,
            "framework_args_source": "settled_validation" if flags else "unrecorded",
            "workspace": validation.get("workspace"),
        },
    }


__all__ = [
    "collect_v6_metadata",
    "langfuse_block",
    "collect_v6_outcome",
    "collect_v6_timeline",
]
