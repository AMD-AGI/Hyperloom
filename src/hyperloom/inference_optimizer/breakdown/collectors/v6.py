"""Additive V6 projections built from the existing V5 evidence."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json, read_jsonl

from ...session.sbd_v6 import SCHEMA_VERSION_V6, read_timeline_events


_SUCCESS_STOP_REASONS = frozenset(
    {
        "target_reached",
        "global_converged",
        "time_exhausted",
        "max_ticks",
        "sweep_done",
        "conc_sweep_done",
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
_FRAMEWORK_PHASES = frozenset({"FRAMEWORK_AGENT", "EXPLORE"})
_FRAMEWORK_EXIT_REASON_MAP = {
    "explore_no_more_leverage": "optimize_no_more_leverage",
    "plateau_explore": "optimize_no_more_leverage",
    "explore_phase_budget_exhausted": "optimize_phase_budget_exhausted",
    "explore_budget_cap": "optimize_budget_cap",
    "explore_force_exit_low_budget": "optimize_force_exit_low_budget",
}
_AUTHORING_TASK_KINDS = frozenset(
    {
        "explore_apply_retry",
        "framework_authoring",
        "framework_local_explore",
    }
)
_MACRO_CYCLE_RE = re.compile(r"(?:^|\s)macro_cycle\s*=\s*(-?\d+)(?=\s|$)")


def _tool_versions(versions: Any) -> dict[str, str | None]:
    if not isinstance(versions, dict):
        return {}
    tools: dict[str, str | None] = {}
    for name, value in versions.items():
        tool = str(name or "").strip()
        if not tool:
            continue
        if isinstance(value, str):
            tools[tool] = value or None
            continue
        if isinstance(value, dict):
            label = value.get("version") or value.get("commit")
            tools[tool] = str(label) if label not in (None, "") else None
    return tools


def _architecture(workload: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    if not workload and not model_info:
        return {}
    model_class = str(workload.get("model_class") or "").strip()
    if not model_class and model_info:
        model_class = "moe" if bool(model_info.get("is_moe")) else "dense"
    return {
        "model_class": model_class,
        "model_type": str(model_info.get("model_type") or ""),
        "num_hidden_layers": model_info.get("num_hidden_layers"),
        "attention_type": str(model_info.get("attention_type") or ""),
        "num_experts": model_info.get("num_experts"),
    }


def _langfuse_projection(langfuse: dict[str, Any]) -> dict[str, Any]:
    config = langfuse.get("config") if isinstance(langfuse.get("config"), dict) else {}
    trace_url = langfuse.get("trace_url")
    if not trace_url:
        host = str(config.get("host") or "").rstrip("/")
        trace_id = str(langfuse.get("trace_id") or "").strip()
        if host and trace_id:
            trace_url = f"{host}/trace/{trace_id}"
    return {
        "enabled": bool(langfuse.get("enabled")),
        "trace_url": trace_url or None,
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
) -> dict[str, Any]:
    """Project V5 session/config sections into the V6 ``metadata`` shape."""
    recovery = session.get("recovery") if isinstance(session.get("recovery"), dict) else {}
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
    return {
        "exported_at_utc": exported_at_utc,
        "versions": {
            "schema_version": SCHEMA_VERSION_V6,
            "hyperloom": str(session.get("code_revision") or ""),
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
            "max_minutes": int(session.get("max_minutes") or 0),
            "elapsed_minutes": float(session.get("elapsed_minutes") or 0.0),
            "tick_count": int(session.get("tick_count") or 0),
            "recovery": {
                "recovered": bool(recovery.get("recovered")),
                "crash_count": int(recovery.get("crash_count") or 0),
                "degraded_mode": bool(recovery.get("degraded_mode")),
            },
        },
        "task_config": task_config,
        "langfuse": _langfuse_projection(langfuse),
        "warnings": list(warnings),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _dict_value_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value.values() if isinstance(row, dict)] if isinstance(value, dict) else []


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "passed", "succeeded"}:
            return True
        if normalized in {"0", "false", "no", "off", "failed"}:
            return False
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _row_cycle(row: dict[str, Any]) -> int | None:
    for value in (
        row.get("macro_cycle"),
        row.get("cycle"),
        _nested(row, "outputs", "macro_cycle"),
        _nested(row, "outputs", "cycle"),
        _nested(row, "inputs", "macro_cycle"),
        _nested(row, "inputs", "cycle"),
        _nested(row, "metadata", "extras", "macro_cycle"),
        _nested(row, "metadata", "extras", "cycle"),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _row_timestamp(row: dict[str, Any]) -> str:
    return str(
        _first(
            row.get("ts"),
            row.get("ended_at"),
            row.get("completed_at"),
            row.get("started_at"),
        )
        or ""
    )


def _prompt_macro_cycle(value: Any) -> int | None:
    match = _MACRO_CYCLE_RE.search(str(value or ""))
    return _optional_int(match.group(1)) if match else None


def _timestamp_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _operation_name(operation: dict[str, Any]) -> str:
    return str(operation.get("name") or operation.get("kind") or "").strip().lower()


def _operation_task_id(operation: dict[str, Any]) -> str:
    return str(
        _first(
            _nested(operation, "extensions", "task_id"),
            _nested(operation, "outputs", "task_id"),
            _nested(operation, "metadata", "extras", "task_id"),
        )
        or ""
    )


def _candidate_id(value: Any) -> str:
    candidate = _mapping(value)
    return str(
        _first(
            candidate.get("candidate_id"),
            candidate.get("pr_url"),
            candidate.get("url"),
            candidate.get("ref"),
            candidate.get("head_sha"),
        )
        or ""
    )


def _is_framework_operation(operation: dict[str, Any]) -> bool:
    name = _operation_name(operation)
    phase = str(operation.get("phase") or "").strip().upper()
    agent = str(operation.get("agent") or "").strip().lower()
    outputs = _mapping(operation.get("outputs"))
    source_phase = str(outputs.get("source_phase") or "").strip().upper()
    if name == "framework_agent":
        return True
    if name == "explore":
        if phase:
            return phase in _FRAMEWORK_PHASES or source_phase in _FRAMEWORK_PHASES
        if source_phase:
            return source_phase in _FRAMEWORK_PHASES
        return agent in {"framework_agent", "explore"}
    if name in {"integrate", "integrate_patch"}:
        if outputs.get("enablement") or outputs.get("enablement_landing"):
            return False
        return bool(
            agent in {"framework_agent", "explore"}
            or phase in _FRAMEWORK_PHASES
            or outputs.get("framework_agent_authoring")
            or outputs.get("framework_agent_candidate_id")
            or source_phase in _FRAMEWORK_PHASES
        )
    return name.startswith("specialist") and (phase in _FRAMEWORK_PHASES or agent in {"framework_agent", "explore"})


def _new_framework_window(cycle: int, start_time: str = "") -> dict[str, Any]:
    return {
        "cycle": cycle,
        "start_time": start_time,
        "end_time": "",
        "rows": [],
        "exit_row": {},
    }


def _framework_windows(
    state: dict[str, Any],
    recorded_operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    history = _dict_rows(state.get("phase_history"))
    for row in history:
        from_phase = str(row.get("from_phase") or "").strip().upper()
        to_phase = str(row.get("to_phase") or "").strip().upper()
        if not from_phase and not to_phase:
            continue
        cycle = _row_cycle(row)
        if cycle is None:
            cycle = int(state.get("macro_cycle") or 0)
        from_framework = from_phase in _FRAMEWORK_PHASES
        to_framework = to_phase in _FRAMEWORK_PHASES
        if to_framework and not from_framework:
            active = _new_framework_window(cycle, str(row.get("ts") or ""))
            active["rows"].append(row)
            windows.append(active)
            continue
        if from_framework and to_framework:
            if active is None or int(active["cycle"]) != cycle:
                active = _new_framework_window(cycle)
                windows.append(active)
            active["rows"].append(row)
            continue
        if from_framework and not to_framework:
            if active is None or int(active["cycle"]) != cycle:
                active = next(
                    (
                        window
                        for window in reversed(windows)
                        if int(window["cycle"]) == cycle and not window["end_time"]
                    ),
                    None,
                )
            if active is None:
                active = _new_framework_window(cycle)
                windows.append(active)
            active["rows"].append(row)
            active["end_time"] = str(row.get("ts") or "")
            active["exit_row"] = row
            active = None
            continue
    current_phase = str(state.get("phase") or "").strip().upper()
    current_cycle = int(state.get("macro_cycle") or 0)
    if current_phase in _FRAMEWORK_PHASES and not any(
        int(window["cycle"]) == current_cycle and not window["end_time"] for window in windows
    ):
        windows.append(_new_framework_window(current_cycle, str(state.get("phase_started_ts") or "")))

    evidence_rows: list[dict[str, Any]] = [
        operation for operation in recorded_operations if _is_framework_operation(operation)
    ]
    evidence_rows.extend(_dict_rows(state.get("framework_agent_batches")))
    evidence_rows.extend(_dict_rows(state.get("framework_agent_phase_progress")))
    evidence_rows.extend(_dict_rows(state.get("specialist_rounds")))
    evidence_rows.extend(_dict_rows(state.get("framework_config_exploration_results")))
    explore_search = _mapping(state.get("explore_search"))
    tested = explore_search.get("tested")
    if isinstance(tested, dict):
        evidence_rows.extend(row for row in tested.values() if isinstance(row, dict))
    evidence_rows.extend(_dict_rows(explore_search.get("winners_history")))
    known_cycles = {int(window["cycle"]) for window in windows}
    for row in evidence_rows:
        cycle = _row_cycle(row)
        if cycle is None or cycle in known_cycles:
            continue
        windows.append(_new_framework_window(cycle))
        known_cycles.add(cycle)

    windows.sort(
        key=lambda window: (
            int(window["cycle"]),
            _timestamp_number(window.get("start_time")) or float("inf"),
        )
    )
    return windows


def _row_in_window(row: dict[str, Any], window: dict[str, Any], window_count: int) -> bool:
    cycle = _row_cycle(row)
    if cycle is not None and cycle != int(window["cycle"]):
        return False
    row_ts = _timestamp_number(_row_timestamp(row))
    start_ts = _timestamp_number(window.get("start_time"))
    end_ts = _timestamp_number(window.get("end_time"))
    if row_ts is not None and (start_ts is not None or end_ts is not None):
        return (start_ts is None or row_ts >= start_ts) and (end_ts is None or row_ts <= end_ts)
    return cycle is not None or window_count == 1


def _window_operations(
    recorded_operations: list[dict[str, Any]],
    window: dict[str, Any],
    window_count: int,
) -> list[dict[str, Any]]:
    return [
        operation
        for operation in recorded_operations
        if _is_framework_operation(operation) and _row_in_window(operation, window, window_count)
    ]


def _window_state_rows(
    state: dict[str, Any],
    field: str,
    window: dict[str, Any],
    window_count: int,
) -> list[dict[str, Any]]:
    return [row for row in _dict_rows(state.get(field)) if _row_in_window(row, window, window_count)]


def _window_evidence(window: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for row in _dict_rows(window.get("rows")):
        evidence.update(_mapping(row.get("evidence")))
    return evidence


def _operation_value(operations: list[dict[str, Any]], *paths: tuple[str, ...]) -> Any:
    for operation in reversed(operations):
        for path in paths:
            value = _nested(operation, *path)
            if value is not None and value != "":
                return value
    return None


def _framework_policy(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    overrides = _mapping(state.get("plateau_overrides"))
    stack_rebench_enabled = _optional_bool(
        _operation_value(
            operations,
            ("outputs", "stack_rebench_enabled"),
            ("outputs", "enable_stack_rebench"),
            ("inputs", "stack_rebench_enabled"),
            ("inputs", "enable_stack_rebench"),
        )
    )
    return {
        "keep_threshold_pct": _optional_float(
            _first(
                _operation_value(
                    operations,
                    ("outputs", "keep_threshold_pct"),
                    ("inputs", "keep_threshold_pct"),
                    ("decisions", "evidence", "keep_threshold_pct"),
                ),
                evidence.get("keep_threshold_pct"),
            )
        ),
        "stack_stable_threshold_pct": _optional_float(
            _operation_value(
                operations,
                ("outputs", "stack_stable_threshold_pct"),
                ("inputs", "stack_stable_threshold_pct"),
            )
        ),
        "stack_rebench_enabled": stack_rebench_enabled,
        "variant_timeout_sec": _optional_int(
            _first(
                _operation_value(
                    operations,
                    ("outputs", "variant_timeout_sec"),
                    ("inputs", "variant_timeout_sec"),
                ),
                state.get("explore_variant_timeout_sec_override"),
            )
        ),
        "overtime_kill_ratio": _optional_float(
            _first(
                _operation_value(
                    operations,
                    ("outputs", "explore_overtime_kill_ratio"),
                    ("outputs", "overtime_kill_ratio"),
                    ("inputs", "explore_overtime_kill_ratio"),
                ),
                state.get("explore_overtime_kill_ratio"),
            )
        ),
        "force_exit_budget_pct": _optional_float(
            _first(evidence.get("force_exit_budget_pct"), overrides.get("force_exit_budget_pct"))
        ),
        "config_arm": {
            "keep_gain_threshold_pct": _optional_float(
                _first(evidence.get("keep_gain_threshold_pct"), overrides.get("explore_keep_gain_pct"))
            ),
            "empty_streak_threshold": _optional_int(
                _first(evidence.get("empty_streak_threshold"), overrides.get("explore_empty_streak"))
            ),
            "lookback": _optional_int(_first(evidence.get("lookback"), overrides.get("explore_lookback"))),
        },
        "source_arm": {
            "no_keep_streak_threshold": _optional_int(
                _first(evidence.get("source_threshold"), evidence.get("no_keep_streak_threshold"))
            ),
            "discovery_retry_limit": _optional_int(
                _first(evidence.get("retry_limit"), evidence.get("discovery_retry_limit"))
            ),
            "authoring_enabled": _optional_bool(state.get("framework_agent_authoring_enabled")),
        },
    }


def _specialist_rows(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    window: dict[str, Any],
    window_count: int,
) -> list[dict[str, Any]]:
    ordered: list[str] = []
    by_key: dict[str, dict[str, Any]] = {}

    def _upsert(row: dict[str, Any]) -> None:
        key = str(_first(row.get("task_id"), row.get("round_id")) or "")
        if not key:
            key = f"row:{len(ordered)}"
        if key not in by_key:
            ordered.append(key)
            by_key[key] = dict(row)
            return
        merged = dict(row)
        merged.update(by_key[key])
        by_key[key] = merged

    for row in _window_state_rows(state, "specialist_rounds", window, window_count):
        _upsert(row)
    for operation in operations:
        if not _operation_name(operation).startswith("specialist"):
            continue
        row = dict(_mapping(operation.get("inputs")))
        row.update(_mapping(operation.get("outputs")))
        row.setdefault("task_id", _operation_task_id(operation))
        row.setdefault("cycle", _row_cycle(operation))
        row.setdefault("status", operation.get("status"))
        row.setdefault("completed_at", operation.get("ended_at"))
        row.setdefault("source_phase", operation.get("phase"))
        _upsert(row)
    return [by_key[key] for key in ordered]


def _specialist_role(
    row: dict[str, Any],
    candidate_map: dict[str, str],
    source_task_ids: set[str],
) -> str:
    task_id = str(row.get("task_id") or "")
    domain = str(row.get("domain") or "").strip().lower()
    task_kind = str(row.get("task_kind") or "").strip().lower()
    if (
        row.get("candidate_discovery")
        or task_kind == "candidate_discovery"
        or domain == "candidate_discovery_specialist"
    ):
        return "discovery"
    authoring_marker = _optional_bool(row.get("framework_agent_authoring")) is True
    candidate_id = str(_first(row.get("framework_agent_candidate_id"), row.get("candidate_id")) or "")
    if (
        task_id in candidate_map
        or task_id in source_task_ids
        or authoring_marker
        or candidate_id
        or task_kind in _AUTHORING_TASK_KINDS
        or bool(_patch_refs(row))
    ):
        return "authoring"
    return "config"


def _specialist_status(row: dict[str, Any]) -> str:
    raw = str(row.get("status") or "").strip().lower()
    if raw in {"failed", "error", "timed_out", "timeout"} or row.get("error"):
        return "failed"
    proposals = row.get("proposal_set")
    if bool(row.get("empty")) or isinstance(proposals, list) and not proposals:
        return "empty"
    if raw in {"empty", "skipped"}:
        return "empty"
    return "succeeded"


def _proposal_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for proposal in _dict_rows(row.get("proposal_set")):
        name = str(_first(proposal.get("name"), proposal.get("variant_name"), proposal.get("proposal_id")) or "")
        if name and name not in names:
            names.append(name)
    return names


def _config_specialist_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_id": str(_first(row.get("round_id"), row.get("task_id")) or ""),
        "task_id": str(row.get("task_id") or ""),
        "status": _specialist_status(row),
        "domain": str(row.get("domain") or ""),
        "scope": _first(row.get("scope"), None),
        "tags": _string_list(row.get("tags")),
        "gap_canonical_id": _first(row.get("gap_canonical_id"), None),
        "proposal_msg_id": _first(row.get("proposal_msg_id"), None),
        "proposal_names": _proposal_names(row),
        "reason": _first(row.get("reason"), row.get("error"), None),
    }


def _config_source_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        info = {
            "specialist_round_id": _first(row.get("round_id"), row.get("task_id"), None),
            "proposal_msg_id": _first(row.get("proposal_msg_id"), None),
        }
        for proposal in _dict_rows(row.get("proposal_set")):
            for key in (
                proposal.get("fingerprint"),
                proposal.get("name"),
                proposal.get("variant_name"),
                proposal.get("proposal_id"),
            ):
                text = str(key or "").strip()
                if text:
                    index.setdefault(text, info)
    return index


def _config_variant(raw: dict[str, Any], source_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = _mapping(raw.get("metrics"))
    variant = _mapping(raw.get("variant"))
    combined = dict(metrics)
    combined.update(variant)
    combined.update(raw)
    name = str(_first(combined.get("name"), combined.get("variant_name")) or "")
    fingerprint = str(combined.get("fingerprint") or "")
    source = source_index.get(fingerprint) or source_index.get(name) or {}
    outcome = str(combined.get("outcome") or "").strip().upper()
    stack_tput = _optional_float(combined.get("stack_rebench_tput"))
    stack_ran = _optional_bool(combined.get("stack_rebench_ran"))
    if stack_ran is None and (stack_tput is not None or combined.get("stack_rebench_workspace")):
        stack_ran = True
    stack_stable = _optional_bool(combined.get("stack_rebench_stable"))
    if stack_stable is None and stack_ran:
        if outcome == "KEEP_UNSTABLE":
            stack_stable = False
        elif outcome == "KEEP":
            stack_stable = True
    return {
        "name": name,
        "fingerprint": fingerprint,
        "source": {
            "provenance": str(combined.get("provenance") or ""),
            "scope": _first(combined.get("scope"), None),
            "specialist_round_id": _first(combined.get("specialist_round_id"), source.get("specialist_round_id")),
            "proposal_msg_id": _first(combined.get("proposal_msg_id"), source.get("proposal_msg_id")),
            "critic_iteration": _optional_int(combined.get("critic_iteration")),
        },
        "config_delta": {
            "extra_server_args": str(combined.get("extra_server_args") or ""),
            "extra_envs": dict(_mapping(combined.get("extra_envs"))),
            "remove_args": _string_list(combined.get("remove_args")),
            "unset_envs": _string_list(combined.get("unset_envs")),
            "args_mode": _first(combined.get("args_mode"), None),
        },
        "accepted_kernels": _string_list(combined.get("accepted_kernels")),
        "measurement": {
            "base_tput": _optional_float(combined.get("base_tput")),
            "decision_tput": _optional_float(_first(combined.get("decision_tput"), combined.get("tput"))),
            "gain_pct": _optional_float(combined.get("gain_pct")),
            "runtime_sec": _optional_float(combined.get("runtime_sec")),
            "estimated_output_throughput": _optional_float(combined.get("estimated_output_throughput")),
        },
        "accuracy": {
            "required": _optional_bool(_first(combined.get("accuracy_required"), combined.get("require_accuracy"))),
            "reference": _optional_float(_first(combined.get("accuracy_reference"), combined.get("accuracy_baseline"))),
            "value": _optional_float(combined.get("accuracy")),
            "passed": _optional_bool(_first(combined.get("accuracy_pass"), combined.get("accuracy_passed"))),
        },
        "stack_rebench": {
            "ran": stack_ran,
            "tput": stack_tput,
            "stable": stack_stable,
        },
        "outcome": outcome,
        "reason": _first(combined.get("reason"), None),
        "stage": _first(combined.get("stage"), None),
        "failure": {
            "error_class": _first(combined.get("error_class"), None),
            "error_excerpt": _first(combined.get("error_excerpt"), combined.get("error"), None),
        },
        "artifacts": {
            "workspace": _first(combined.get("workspace"), combined.get("single_workspace"), None),
            "stack_rebench_workspace": _first(combined.get("stack_rebench_workspace"), None),
            "server_log_path": _first(combined.get("server_log_path"), None),
            "raw_result_path": _first(combined.get("raw_result_path"), None),
        },
    }


def _round_variant_rows(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    tested = _nested(outputs, "explore_search_update", "tested")
    tested_by_fingerprint = tested if isinstance(tested, dict) else {}
    round_id = str(outputs.get("round_id") or "")
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for outcome in _dict_rows(outputs.get("per_variant_outcomes")):
        fingerprint = str(outcome.get("fingerprint") or "")
        merged = dict(_mapping(tested_by_fingerprint.get(fingerprint)))
        merged.update(outcome)
        key = fingerprint or str(outcome.get("variant_name") or len(ordered))
        seen.add(key)
        ordered.append(merged)
    for fingerprint, tested_row in tested_by_fingerprint.items():
        if not isinstance(tested_row, dict):
            continue
        if round_id and str(tested_row.get("round_id") or "") not in {"", round_id}:
            continue
        key = str(fingerprint or tested_row.get("name") or len(ordered))
        if key in seen:
            continue
        merged = dict(tested_row)
        merged.setdefault("fingerprint", str(fingerprint))
        ordered.append(merged)
    return ordered


def _config_round_from_operation(
    operation: dict[str, Any],
    source_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    outputs = _mapping(operation.get("outputs"))
    inputs = _mapping(operation.get("inputs"))
    last_round = _mapping(_nested(outputs, "explore_search_update", "last_round"))
    input_stack = _mapping(_first(outputs.get("input_stack"), inputs.get("input_stack")))
    variants = [_config_variant(row, source_index) for row in _round_variant_rows(outputs)]
    decision_mode = _first(outputs.get("decision_mode"), inputs.get("decision_mode"))
    if decision_mode is None:
        modes = {
            str(_nested(row, "metrics", "overtime_anchor_kind") or row.get("overtime_anchor_kind") or "")
            for row in _dict_rows(outputs.get("per_variant_outcomes"))
        }
        modes.discard("")
        decision_mode = next(iter(modes)) if len(modes) == 1 else None
    return {
        "round_id": str(
            _first(
                outputs.get("round_id"),
                _nested(operation, "metadata", "extras", "round_id"),
                operation.get("operation_id"),
            )
            or ""
        ),
        "task_id": _operation_task_id(operation),
        "status": "succeeded"
        if str(_first(outputs.get("status"), operation.get("status")) or "").lower()
        in {"succeeded", "success", "completed", "kept", "keep"}
        else "failed",
        "framework": str(
            _first(outputs.get("framework"), _nested(outputs, "workload", "framework"), inputs.get("framework")) or ""
        ),
        "workload_signature": str(
            _first(
                outputs.get("workload_signature"),
                next(
                    (
                        row.get("workload_signature")
                        for row in _mapping(_nested(outputs, "explore_search_update", "tested")).values()
                        if isinstance(row, dict) and row.get("workload_signature")
                    ),
                    None,
                ),
            )
            or ""
        ),
        "throughput_unit": _first(outputs.get("throughput_unit"), inputs.get("throughput_unit"), None),
        "decision_mode": decision_mode,
        "input_stack": {
            "throughput": _optional_float(
                _first(input_stack.get("throughput"), outputs.get("base_tput"), last_round.get("base_tput"))
            ),
            "accuracy": _optional_float(
                _first(input_stack.get("accuracy"), outputs.get("accuracy_baseline"), inputs.get("accuracy_baseline"))
            ),
            "extra_server_args": str(
                _first(
                    input_stack.get("extra_server_args"),
                    outputs.get("base_extra_args"),
                    last_round.get("base_extra_args"),
                    "",
                )
                or ""
            ),
            "extra_envs": dict(_mapping(_first(input_stack.get("extra_envs"), outputs.get("base_extra_envs")))),
            "remove_args": _string_list(_first(input_stack.get("remove_args"), outputs.get("base_remove_args"), [])),
            "unset_envs": _string_list(_first(input_stack.get("unset_envs"), outputs.get("base_unset_envs"), [])),
            "args_mode": _first(input_stack.get("args_mode"), outputs.get("base_args_mode"), None),
        },
        "variants": variants,
    }


def _fallback_config_rounds(
    state: dict[str, Any],
    window: dict[str, Any],
    window_count: int,
    source_index: dict[str, dict[str, Any]],
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    explore_search = _mapping(state.get("explore_search"))
    tested = explore_search.get("tested")
    grouped: dict[str, list[dict[str, Any]]] = {}
    if isinstance(tested, dict):
        for fingerprint, row in tested.items():
            if not isinstance(row, dict) or not _row_in_window(row, window, window_count):
                continue
            round_id = str(row.get("round_id") or "")
            if not round_id or round_id in existing_ids:
                continue
            item = dict(row)
            item.setdefault("fingerprint", str(fingerprint))
            grouped.setdefault(round_id, []).append(item)
    rounds: list[dict[str, Any]] = []
    for round_id, rows in grouped.items():
        measured_outcomes = {"KEEP", "REVERT", "KEEP_UNSTABLE", "KILLED_OVERTIME"}
        rounds.append(
            {
                "round_id": round_id,
                "task_id": "",
                "status": (
                    "succeeded"
                    if any(str(row.get("outcome") or "").upper() in measured_outcomes for row in rows)
                    else "failed"
                ),
                "framework": str(next((row.get("framework") for row in rows if row.get("framework")), "")),
                "workload_signature": str(
                    next((row.get("workload_signature") for row in rows if row.get("workload_signature")), "")
                ),
                "throughput_unit": None,
                "decision_mode": None,
                "input_stack": {
                    "throughput": _optional_float(next((row.get("base_tput") for row in rows), None)),
                    "accuracy": None,
                    "extra_server_args": "",
                    "extra_envs": {},
                    "remove_args": [],
                    "unset_envs": [],
                    "args_mode": None,
                },
                "variants": [_config_variant(row, source_index) for row in rows],
            }
        )
    return rounds


def _config_arm(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    specialist_rows: list[dict[str, Any]],
    specialist_roles: dict[str, str],
    window: dict[str, Any],
    window_count: int,
    evidence: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    config_rows = [
        row
        for row in specialist_rows
        if specialist_roles.get(str(_first(row.get("task_id"), row.get("round_id")) or "")) == "config"
    ]
    specialist_runs = [_config_specialist_run(row) for row in config_rows]
    source_index = _config_source_index(config_rows)
    rounds = [
        _config_round_from_operation(operation, source_index)
        for operation in operations
        if _operation_name(operation) == "explore"
    ]
    existing_ids = {str(row.get("round_id") or "") for row in rounds}
    rounds.extend(_fallback_config_rounds(state, window, window_count, source_index, existing_ids))
    recorded_task_ids = {str(row.get("task_id") or "") for row in rounds}
    for row in _window_state_rows(state, "framework_config_exploration_results", window, window_count):
        task_id = str(row.get("task_id") or "")
        round_id = str(_first(row.get("round_id"), task_id) or "")
        if not round_id or task_id in recorded_task_ids or round_id in existing_ids:
            continue
        rounds.append(
            {
                "round_id": round_id,
                "task_id": task_id,
                "status": "succeeded" if str(row.get("status") or "succeeded").lower() != "failed" else "failed",
                "framework": str(row.get("framework") or ""),
                "workload_signature": str(row.get("workload_signature") or ""),
                "throughput_unit": _first(row.get("throughput_unit"), None),
                "decision_mode": _first(row.get("decision_mode"), None),
                "input_stack": {
                    "throughput": None,
                    "accuracy": None,
                    "extra_server_args": "",
                    "extra_envs": {},
                    "remove_args": [],
                    "unset_envs": [],
                    "args_mode": None,
                },
                "variants": [],
            }
        )
    policy_config = _mapping(policy.get("config_arm"))
    explore_search = _mapping(state.get("explore_search"))
    tested_rows = [
        row for row in _dict_value_rows(explore_search.get("tested")) if _row_in_window(row, window, window_count)
    ]
    winner_rows = [
        row for row in _dict_rows(explore_search.get("winners_history")) if _row_in_window(row, window, window_count)
    ]
    recent_keep_gain = _optional_float(evidence.get("recent_keep_gain_pct"))
    lookback = _optional_int(policy_config.get("lookback"))
    if recent_keep_gain is None and lookback is not None and lookback > 0:
        recent_keep_gain = round(
            sum(_optional_float(row.get("gain_pct")) or 0.0 for row in winner_rows[-lookback:]),
            4,
        )
    empty_streak = _optional_int(evidence.get("empty_streak"))
    if empty_streak is None:
        empty_streak = 0
        for row in reversed(specialist_runs):
            if row["status"] != "empty":
                break
            empty_streak += 1
    tested_this_cycle = _optional_int(evidence.get("tested_this_cycle"))
    if tested_this_cycle is None:
        tested_this_cycle = len(tested_rows)
    triggered_value = _optional_bool(evidence.get("config_arm_plateaued"))
    if triggered_value is None:
        keep_gain_threshold = _optional_float(policy_config.get("keep_gain_threshold_pct"))
        empty_streak_threshold = _optional_int(policy_config.get("empty_streak_threshold"))
        if recent_keep_gain is not None and keep_gain_threshold is not None and empty_streak_threshold is not None:
            exhausted = empty_streak >= empty_streak_threshold or (
                not specialist_runs and tested_this_cycle >= empty_streak_threshold
            )
            triggered_value = recent_keep_gain < keep_gain_threshold and exhausted
    return {
        "plateau": {
            "triggered": triggered_value,
            "recent_keep_gain_pct": recent_keep_gain,
            "empty_streak": empty_streak,
            "tested_this_cycle": tested_this_cycle,
        },
        "specialist_runs": specialist_runs,
        "rounds": rounds,
    }


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    audit = _mapping(candidate.get("audit"))
    raw_verdict = str(_first(candidate.get("verdict"), audit.get("verdict"), candidate.get("applicability")) or "")
    verdict = raw_verdict.strip().lower()
    if verdict not in {"worth_a_bench", "already_present", "not_applicable"}:
        verdict = ""
    route = str(_first(candidate.get("route"), audit.get("recommended_next_step")) or "").strip()
    if route in {"direct_apply", "direct"}:
        route = "direct_framework"
    if route not in {"direct_framework", "author_via_specialist"}:
        route = ""
    return {
        "candidate_id": _candidate_id(candidate),
        "source_ref": str(
            _first(candidate.get("pr_url"), candidate.get("ref"), candidate.get("head_sha"), candidate.get("diff_url"))
            or ""
        ),
        "repo": str(
            _first(candidate.get("repo"), candidate.get("repo_url"), candidate.get("discovered_repo_url")) or ""
        ),
        "title": str(candidate.get("title") or ""),
        "changed_files": _string_list(candidate.get("changed_files")),
        "verdict": verdict,
        "route": route,
        "gap_canonical_id": _first(candidate.get("gap_canonical_id"), None),
        "reason": _first(candidate.get("reason"), audit.get("reason"), candidate.get("rationale"), None),
    }


def _candidate_index(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for batch in _dict_rows(state.get("framework_agent_batches")):
        for candidate in _dict_rows(batch.get("candidates")):
            candidate_id = _candidate_id(candidate)
            if candidate_id:
                index[candidate_id] = candidate
    return index


def _candidate_discovery_runs(
    state: dict[str, Any],
    specialist_rows: list[dict[str, Any]],
    specialist_roles: dict[str, str],
    window: dict[str, Any],
    window_count: int,
) -> list[dict[str, Any]]:
    discovery_rows = [
        row
        for row in specialist_rows
        if specialist_roles.get(str(_first(row.get("task_id"), row.get("round_id")) or "")) == "discovery"
    ]
    used_tasks: set[str] = set()
    timestamped_runs: list[tuple[str, dict[str, Any]]] = []

    def _append_run(run: dict[str, Any], source: dict[str, Any]) -> None:
        timestamped_runs.append((_row_timestamp(source), run))

    for batch in _dict_rows(state.get("framework_agent_batches")):
        batch_candidates = _dict_rows(batch.get("candidates"))
        batch_ids = {_candidate_id(candidate) for candidate in batch_candidates if _candidate_id(candidate)}
        task_id = str(batch.get("task_id") or "")
        matched: dict[str, Any] | None = None
        for row in discovery_rows:
            row_task_id = str(row.get("task_id") or "")
            proposal_ids = {_candidate_id(proposal) for proposal in _dict_rows(row.get("proposal_set"))}
            if task_id and row_task_id == task_id:
                matched = row
                break
            if row_task_id and str(batch.get("batch_id") or "").endswith(row_task_id[:8]):
                matched = row
                break
            if batch_ids and proposal_ids.intersection(batch_ids):
                matched = row
                break
        if matched is not None:
            task_id = task_id or str(matched.get("task_id") or "")
            used_tasks.add(str(matched.get("task_id") or ""))
        elif not _row_in_window(batch, window, window_count):
            continue
        discovered_candidates = _dict_rows((matched or {}).get("proposal_set")) or batch_candidates
        _append_run(
            {
                "task_id": task_id,
                "status": _specialist_status(matched or batch)
                if matched is not None
                else ("succeeded" if batch_candidates else "empty"),
                "batch_id": _first(batch.get("batch_id"), None),
                "gap_canonical_id": _first(
                    batch.get("gap_canonical_id"),
                    (matched or {}).get("gap_canonical_id"),
                    None,
                ),
                "reason": _first(batch.get("reason"), (matched or {}).get("reason"), None),
                "candidates": [_candidate_row(candidate) for candidate in discovered_candidates],
            },
            matched or batch,
        )
    for row in discovery_rows:
        task_id = str(row.get("task_id") or "")
        if task_id in used_tasks:
            continue
        _append_run(
            {
                "task_id": task_id,
                "status": _specialist_status(row),
                "batch_id": _first(row.get("batch_id"), None),
                "gap_canonical_id": _first(row.get("gap_canonical_id"), None),
                "reason": _first(row.get("reason"), row.get("error"), None),
                "candidates": [_candidate_row(candidate) for candidate in _dict_rows(row.get("proposal_set"))],
            },
            row,
        )

    failed_markers = 0
    terminal_retry_rows: list[dict[str, Any]] = []
    for row in _dict_rows(window.get("rows")):
        evidence = _mapping(row.get("evidence"))
        event = str(evidence.get("event") or "").strip()
        reason = str(row.get("reason") or "").strip()
        if event == "framework_agent_discover_failed":
            failed_markers += 1
            _append_run(
                {
                    "task_id": str(_first(evidence.get("task_id"), evidence.get("failed_task_id")) or ""),
                    "status": "failed",
                    "batch_id": _first(evidence.get("batch_id"), None),
                    "gap_canonical_id": _first(evidence.get("gap_canonical_id"), None),
                    "reason": _first(evidence.get("error"), reason, None),
                    "candidates": [],
                },
                row,
            )
        elif event == "framework_agent_phase_done" and reason == "discover_empty_payload":
            _append_run(
                {
                    "task_id": str(evidence.get("task_id") or ""),
                    "status": "empty",
                    "batch_id": _first(evidence.get("batch_id"), None),
                    "gap_canonical_id": _first(evidence.get("gap_canonical_id"), None),
                    "reason": reason,
                    "candidates": [],
                },
                row,
            )
        elif event == "framework_agent_phase_done" and reason == "discover_retries_exhausted":
            terminal_retry_rows.append(row)

    if failed_markers == 0:
        for row in terminal_retry_rows:
            evidence = _mapping(row.get("evidence"))
            _append_run(
                {
                    "task_id": str(_first(evidence.get("task_id"), evidence.get("failed_task_id")) or ""),
                    "status": "failed",
                    "batch_id": _first(evidence.get("batch_id"), None),
                    "gap_canonical_id": _first(evidence.get("gap_canonical_id"), None),
                    "reason": str(row.get("reason") or "discover_retries_exhausted"),
                    "candidates": [],
                },
                row,
            )

    if timestamped_runs and all(_timestamp_number(timestamp) is not None for timestamp, _ in timestamped_runs):
        timestamped_runs.sort(key=lambda item: _timestamp_number(item[0]) or 0.0)
    return [run for _, run in timestamped_runs]


def _patch_refs(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("patch_refs", "patches_written", "patches", "artifacts_written"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                value = _first(value.get("path"), value.get("patch_path"), value.get("target"), value.get("rel_target"))
            text = str(value or "").strip()
            if text and text not in refs:
                refs.append(text)
    for proposal in _dict_rows(row.get("proposal_set")):
        for value in _patch_refs(proposal):
            if value not in refs:
                refs.append(value)
    return refs


def _authoring_runs(
    state: dict[str, Any],
    specialist_rows: list[dict[str, Any]],
    specialist_roles: dict[str, str],
    progress_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_map = {
        str(key): str(value)
        for key, value in _mapping(state.get("framework_agent_specialist_candidate_map")).items()
        if key and value
    }
    progress_by_task = {
        str(row.get("specialist_task_id") or ""): row for row in progress_rows if row.get("specialist_task_id")
    }
    runs: list[dict[str, Any]] = []
    used_tasks: set[str] = set()
    for row in specialist_rows:
        key = str(_first(row.get("task_id"), row.get("round_id")) or "")
        if specialist_roles.get(key) != "authoring":
            continue
        task_id = str(row.get("task_id") or "")
        progress = progress_by_task.get(task_id, {})
        candidate_id = str(
            _first(
                row.get("candidate_id"),
                row.get("framework_agent_candidate_id"),
                candidate_map.get(task_id),
                progress.get("candidate_id"),
            )
            or ""
        )
        reauthor_attempt = _optional_int(
            _first(
                row.get("reauthor_attempt"),
                row.get("apply_retry_attempt"),
                progress.get("reauthor_attempt"),
            )
        )
        kind = (
            "reauthor"
            if reauthor_attempt is not None and reauthor_attempt > 0
            else "local_authoring"
            if candidate_id.startswith("local_explore:") or bool(row.get("framework_local_explore"))
            else "candidate_authoring"
            if candidate_id
            else ""
        )
        runs.append(
            {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "kind": kind,
                "status": _specialist_status(row),
                "reauthor_attempt": reauthor_attempt,
                "specialist_domain": str(row.get("domain") or ""),
                "gap_canonical_id": _first(row.get("gap_canonical_id"), None),
                "patch_refs": _patch_refs(row),
                "reason": _first(row.get("reason"), row.get("error"), progress.get("rationale"), None),
            }
        )
        if task_id:
            used_tasks.add(task_id)
    for progress in progress_rows:
        task_id = str(progress.get("specialist_task_id") or "")
        if not task_id or task_id in used_tasks:
            continue
        candidate_id = str(progress.get("candidate_id") or candidate_map.get(task_id) or "")
        runs.append(
            {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "kind": "local_authoring" if candidate_id.startswith("local_explore:") else "candidate_authoring",
                "status": "failed"
                if str(progress.get("status") or "").lower() in {"dispatch_failed", "author_failed", "recovery_failed"}
                else "empty"
                if str(progress.get("provenance") or "").lower() == "authored_empty"
                else "succeeded",
                "reauthor_attempt": _optional_int(progress.get("reauthor_attempt")),
                "specialist_domain": str(progress.get("domain") or ""),
                "gap_canonical_id": _first(progress.get("gap_canonical_id"), None),
                "patch_refs": _patch_refs(progress),
                "reason": _first(progress.get("rationale"), progress.get("error"), None),
            }
        )
    return runs


def _source_attempt_status(value: Any, *, kept: Any = None) -> str:
    status = str(value or "").strip().lower()
    if status in {"keep", "kept", "promoted", "adopted"} or kept is True:
        return "KEEP"
    if status in {"kept_inert", "keep_inert"}:
        return "KEEP_INERT"
    if status in {"revert", "reverted", "rejected", "accuracy_unavailable_reject"}:
        return "REVERT"
    if status in {"critic_denied", "rejected_by_critic", "needs_review_no_evidence", "reauthor_cap"}:
        return "CRITIC_DENIED"
    if status in {"skipped", "already_present", "not_applicable", "author_empty", "no_patch", "no_patches"}:
        return "SKIPPED"
    if status in {
        "failed",
        "error",
        "apply_failed",
        "bench_reverted",
        "enqueue_failed",
        "dispatch_failed",
        "materialize_failed",
        "no_result_failed",
        "apply_fail_cap",
        "recovery_failed",
        "repeated_review_abort",
    }:
        return "FAILED"
    return ""


def _source_attempt(
    operation: dict[str, Any] | None,
    progress: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    operation = operation or {}
    outputs = _mapping(operation.get("outputs"))
    inputs = _mapping(operation.get("inputs"))
    candidate = _mapping(_first(outputs.get("candidate"), inputs.get("candidate")))
    candidate_id = str(
        _first(
            _candidate_id(candidate),
            outputs.get("framework_agent_candidate_id"),
            inputs.get("framework_agent_candidate_id"),
            _nested(operation, "metadata", "extras", "candidate_id"),
            progress.get("candidate_id"),
        )
        or ""
    )
    known_candidate = candidate_index.get(candidate_id, {})
    if not candidate:
        candidate = known_candidate
    task_id = str(_first(_operation_task_id(operation), progress.get("integrate_task_id")) or "")
    source_task_id = str(
        _first(outputs.get("specialist_task_id"), inputs.get("specialist_task_id"), progress.get("specialist_task_id"))
        or ""
    )
    reauthor_attempt = _optional_int(
        _first(outputs.get("reauthor_attempt"), inputs.get("reauthor_attempt"), progress.get("reauthor_attempt"))
    )
    route = str(
        _first(
            outputs.get("route"),
            inputs.get("route"),
            candidate.get("route"),
            known_candidate.get("route"),
        )
        or ""
    ).strip()
    if route in {"direct", "direct_apply"}:
        route = "direct_framework"
    patch_source = str(_first(outputs.get("patch_source"), inputs.get("patch_source")) or "").strip().lower()
    if patch_source not in {"specialist_authored", "upstream_pr"}:
        patch_source = ""
    provenance = str(_first(outputs.get("provenance"), inputs.get("provenance")) or "").strip().lower()
    if not patch_source and provenance == "upstream_pr":
        patch_source = "upstream_pr"
    authoring_marker = _optional_bool(
        _first(outputs.get("framework_agent_authoring"), inputs.get("framework_agent_authoring"))
    )
    if not patch_source and (route == "direct_framework" or source_task_id and task_id and source_task_id == task_id):
        patch_source = "upstream_pr"
    if not patch_source and (
        route in {"author_via_specialist", "local_authoring", "reauthor"}
        or source_task_id
        and source_task_id != task_id
    ):
        patch_source = "specialist_authored"
    if not patch_source and _operation_name(operation) == "framework_agent":
        patch_source = "upstream_pr"
    if not patch_source and authoring_marker is True and not source_task_id:
        patch_source = "specialist_authored"
    lever_kind = str(_first(outputs.get("lever_kind"), inputs.get("lever_kind")) or "").strip().lower()
    if not lever_kind:
        lever_kind = "upstream_pr" if patch_source == "upstream_pr" else "source_patch" if patch_source else ""
    if lever_kind not in {"config", "source_patch", "upstream_pr", "enablement"}:
        lever_kind = ""
    if not route:
        if reauthor_attempt and reauthor_attempt > 0:
            route = "reauthor"
        elif candidate_id.startswith("local_explore:"):
            route = "local_authoring"
        elif patch_source == "upstream_pr":
            route = "direct_framework"
        elif patch_source == "specialist_authored":
            route = "author_via_specialist"
    parity = _mapping(outputs.get("switch_off_parity"))
    stack_rebench = _mapping(outputs.get("stack_rebench"))
    files = _string_list(_first(outputs.get("target_files"), candidate.get("changed_files"), []))
    applied_artifacts = _dict_rows(outputs.get("artifacts_applied"))
    if not files:
        files = [
            str(_first(row.get("rel_target"), row.get("target")) or "")
            for row in applied_artifacts
            if _first(row.get("rel_target"), row.get("target"))
        ]
    raw_status = _first(outputs.get("status"), progress.get("status"), operation.get("status"))
    return {
        "attempt_id": str(_first(operation.get("operation_id"), task_id, candidate_id) or ""),
        "task_id": task_id or None,
        "candidate_id": candidate_id or None,
        "source_task_id": source_task_id or None,
        "patch_source": patch_source or None,
        "lever_kind": lever_kind or None,
        "route": route or None,
        "status": _source_attempt_status(raw_status, kept=progress.get("kept")),
        "before_tput": _optional_float(_first(outputs.get("base_tput"), progress.get("pre_tput"))),
        "after_tput": _optional_float(_first(outputs.get("output_throughput"), progress.get("post_tput"))),
        "local_gain_pct": _optional_float(
            _first(outputs.get("delta_pct"), outputs.get("gain_pct"), progress.get("gain_pct"))
        ),
        "ts": str(_first(operation.get("ended_at"), progress.get("ts")) or ""),
        "source_ref": _first(
            candidate.get("pr_url"),
            candidate.get("ref"),
            candidate.get("head_sha"),
            candidate.get("diff_url"),
            None,
        ),
        "files": files,
        "reason": _first(outputs.get("reason"), progress.get("rationale"), outputs.get("error"), None),
        "gates": {
            "accuracy_passed": _optional_bool(outputs.get("accuracy_pass")),
            "keep_threshold_pct": _optional_float(outputs.get("keep_threshold_pct")),
            "switch_off_parity_passed": _optional_bool(_first(parity.get("ok"), parity.get("passed"))),
            "stack_rebench_passed": _optional_bool(_first(stack_rebench.get("stable"), stack_rebench.get("ok"))),
        },
        "framework_levers": _dict_rows(outputs.get("framework_levers")),
        "config_delta": {
            "extra_server_args": str(outputs.get("extra_server_args_applied") or ""),
            "extra_envs": dict(
                _mapping(_first(outputs.get("extra_envs_applied"), outputs.get("config_changes_applied")))
            ),
        },
        "artifacts": {
            "patches_applied": _string_list(outputs.get("patches_applied")),
            "source_snapshot": _first(outputs.get("source_snapshot"), None),
            "source_manifest": _first(outputs.get("source_manifest"), None),
            "workspace": _first(outputs.get("workspace"), None),
        },
        "failure": {
            "error_class": _first(outputs.get("error_class"), progress.get("error_class"), None),
            "error": _first(outputs.get("error"), progress.get("error"), None),
        },
    }


def _source_attempts(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = _candidate_index(state)
    progress_by_integrate = {
        str(row.get("integrate_task_id") or ""): row for row in progress_rows if row.get("integrate_task_id")
    }
    progress_by_candidate = {
        str(row.get("candidate_id") or ""): row for row in progress_rows if row.get("candidate_id")
    }
    progress_by_source = {
        str(row.get("specialist_task_id") or ""): row for row in progress_rows if row.get("specialist_task_id")
    }
    used_progress: set[int] = set()
    attempts: list[dict[str, Any]] = []
    for operation in operations:
        if _operation_name(operation) not in {"framework_agent", "integrate", "integrate_patch"}:
            continue
        outputs = _mapping(operation.get("outputs"))
        candidate_id = str(
            _first(
                _candidate_id(outputs.get("candidate")),
                outputs.get("framework_agent_candidate_id"),
                _nested(operation, "metadata", "extras", "candidate_id"),
            )
            or ""
        )
        task_id = _operation_task_id(operation)
        source_task_id = str(outputs.get("specialist_task_id") or "")
        progress = (
            progress_by_integrate.get(task_id)
            or progress_by_candidate.get(candidate_id)
            or progress_by_source.get(source_task_id)
            or {}
        )
        if progress:
            used_progress.add(id(progress))
        attempts.append(_source_attempt(operation, progress, candidates))
    for progress in progress_rows:
        if id(progress) in used_progress or str(progress.get("status") or "").lower() == "cycle_boundary":
            continue
        attempts.append(_source_attempt(None, progress, candidates))
    attempts.sort(key=lambda row: row.get("ts") or "")
    return attempts


def _critic_cycle_indexes(
    session_dir: Path,
    warnings: list[str],
    state: dict[str, Any],
    operations: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    task_cycles: dict[str, int] = {}
    proposal_cycles: dict[str, int] = {}
    candidate_cycles: dict[str, int] = {}

    def _index_row(row: dict[str, Any]) -> None:
        cycle = _row_cycle(row)
        if cycle is None:
            return
        task_id = str(
            _first(
                _operation_task_id(row),
                row.get("task_id"),
                row.get("integrate_task_id"),
                row.get("specialist_task_id"),
            )
            or ""
        )
        if task_id:
            task_cycles.setdefault(task_id, cycle)
        proposal_id = str(
            _first(
                row.get("proposal_msg_id"),
                _nested(row, "inputs", "proposal_msg_id"),
                _nested(row, "outputs", "proposal_msg_id"),
                _nested(row, "metadata", "extras", "proposal_msg_id"),
            )
            or ""
        )
        if proposal_id:
            proposal_cycles.setdefault(proposal_id, cycle)
        candidate_id = str(
            _first(
                row.get("candidate_id"),
                row.get("framework_agent_candidate_id"),
                _nested(row, "inputs", "framework_agent_candidate_id"),
                _nested(row, "outputs", "framework_agent_candidate_id"),
                _candidate_id(_nested(row, "inputs", "candidate")),
                _candidate_id(_nested(row, "outputs", "candidate")),
            )
            or ""
        )
        if candidate_id:
            candidate_cycles.setdefault(candidate_id, cycle)

    for operation in operations:
        _index_row(operation)
    for field in (
        "framework_agent_phase_progress",
        "framework_agent_batches",
        "specialist_rounds",
    ):
        for row in _dict_rows(state.get(field)):
            _index_row(row)
            cycle = _row_cycle(row)
            if cycle is None:
                continue
            for candidate in _dict_rows(row.get("candidates")):
                candidate_id = _candidate_id(candidate)
                if candidate_id:
                    candidate_cycles.setdefault(candidate_id, cycle)

    map_path = session_dir / "reports" / "trace" / "proposal_task_map.jsonl"
    if map_path.is_file():
        map_rows = read_jsonl(
            map_path,
            require_dict=True,
            skip_malformed=True,
            on_error=lambda exc: warnings.append(
                f"timeline.framework_agent.critic: failed to parse {map_path}: {exc!r}"
            ),
        )
        for row in map_rows:
            proposal_id = str(row.get("proposal_msg_id") or "")
            task_id = str(row.get("task_id") or "")
            cycle = task_cycles.get(task_id)
            if proposal_id and cycle is not None:
                proposal_cycles.setdefault(proposal_id, cycle)
    return proposal_cycles, candidate_cycles


def _critic_review_rows(
    session_dir: Path,
    warnings: list[str],
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    window: dict[str, Any],
    window_count: int,
) -> list[dict[str, Any]]:
    root = session_dir / "critic-workdir"
    if not root.is_dir():
        return []
    proposal_cycles, candidate_cycles = _critic_cycle_indexes(session_dir, warnings, state, operations)
    rows: list[dict[str, Any]] = []
    for workdir in sorted(root.iterdir(), key=lambda path: path.name):
        if not workdir.is_dir():
            continue
        loaded: dict[str, dict[str, Any]] = {}
        failed = False
        for name in ("request", "judge_bundle", "review", "emit"):
            path = workdir / f"{name}.json"
            if not path.is_file():
                loaded[name] = {}
                continue
            try:
                loaded[name] = read_json(path, require_dict=True, strict=True)
            except Exception as exc:
                warnings.append(f"timeline.framework_agent.critic: failed to parse {path}: {exc!r}")
                failed = True
                break
        if failed:
            continue
        request = loaded["request"]
        judge = loaded["judge_bundle"]
        review = loaded["review"]
        emit = loaded["emit"]
        review_phase = (
            str(
                _first(
                    judge.get("phase"),
                    _nested(request, "context", "phase"),
                    _nested(judge, "merged_context", "phase"),
                )
                or ""
            )
            .strip()
            .upper()
        )
        if review_phase and review_phase not in _FRAMEWORK_PHASES:
            continue
        proposals = {
            str(proposal.get("msg_id") or ""): proposal
            for proposal in _dict_rows(judge.get("proposals"))
            if proposal.get("msg_id")
        }
        effective = {}
        envelope = _mapping(emit.get("intent_envelope"))
        for intent in _dict_rows(envelope.get("intents")):
            if str(intent.get("intent_type") or "") != "review_verdict":
                continue
            payload = _mapping(intent.get("payload"))
            target = str(payload.get("target_proposal_msg_id") or "")
            if target:
                effective[target] = payload
        for verdict_row in _dict_rows(review.get("review_verdicts")):
            proposal_id = str(verdict_row.get("target_proposal_msg_id") or "")
            proposal = proposals.get(proposal_id, {})
            payload = _mapping(proposal.get("payload"))
            params = _mapping(payload.get("params"))
            candidate = _mapping(_first(payload.get("candidate"), params.get("candidate")))
            action = str(
                _first(proposal.get("action_name"), payload.get("action_name"), payload.get("kind")) or ""
            ).lower()
            candidate_id = str(
                _first(
                    payload.get("framework_agent_candidate_id"),
                    params.get("framework_agent_candidate_id"),
                    _candidate_id(candidate),
                )
                or ""
            )
            if action in {"params", "backends", "explore"}:
                arm = "config"
                target_action = "explore"
            elif action in {"framework_agent", "integrate", "integrate_patch"}:
                arm = "source"
                target_action = "integrate_patch"
            elif action == "specialist":
                task_kind = str(params.get("task_kind") or "").strip().lower()
                source_marker = bool(
                    candidate_id
                    or _optional_bool(params.get("framework_agent_authoring")) is True
                    or _optional_bool(params.get("candidate_discovery")) is True
                    or task_kind in _AUTHORING_TASK_KINDS
                    or bool(_patch_refs(params))
                )
                arm = "source" if source_marker else "config"
                target_action = "specialist"
            else:
                continue
            cycle_row = {
                "cycle": _first(
                    payload.get("cycle"),
                    payload.get("macro_cycle"),
                    params.get("cycle"),
                    params.get("macro_cycle"),
                    proposal.get("cycle"),
                    proposal.get("macro_cycle"),
                    proposal_cycles.get(proposal_id),
                    candidate_cycles.get(candidate_id),
                    _nested(request, "context", "macro_cycle"),
                    _nested(request, "context", "cycle"),
                    _prompt_macro_cycle(request.get("raw_prompt")),
                    _nested(judge, "merged_context", "macro_cycle"),
                    _nested(judge, "merged_context", "cycle"),
                ),
                "ts": _first(verdict_row.get("ts"), emit.get("ts"), review.get("ts")),
            }
            if not _row_in_window(cycle_row, window, window_count):
                continue
            effective_row = _mapping(effective.get(proposal_id))
            risks = [
                {
                    "severity": str(risk.get("severity") or ""),
                    "risk": str(_first(risk.get("risk"), risk.get("summary"), risk.get("reason")) or ""),
                }
                for risk in _dict_rows(verdict_row.get("risks"))
            ]
            followup_task_ids = _string_list(
                _first(effective_row.get("followup_task_ids"), verdict_row.get("followup_task_ids"), [])
            )
            rows.append(
                {
                    "proposal_msg_id": proposal_id,
                    "candidate_id": candidate_id or None,
                    "variant_name": _first(payload.get("variant_name"), params.get("variant_name"), None),
                    "arm": arm,
                    "target_action": target_action,
                    "source": "critic"
                    if str(verdict_row.get("source") or "critic") == "critic"
                    else "critic_unavailable",
                    "verdict": str(verdict_row.get("verdict") or ""),
                    "effective_verdict": str(
                        _first(
                            effective_row.get("verdict"),
                            verdict_row.get("effective_verdict"),
                            verdict_row.get("verdict"),
                        )
                        or ""
                    ),
                    "reasoning": str(verdict_row.get("reasoning") or ""),
                    "confidence": _first(verdict_row.get("confidence"), None),
                    "failure_reason_code": _first(verdict_row.get("failure_reason_code"), None),
                    "required_evidence": _string_list(verdict_row.get("required_evidence")),
                    "risks": risks,
                    "advice_text": _first(verdict_row.get("advice_text"), effective_row.get("advice_text"), None),
                    "alternative_action": _first(verdict_row.get("alternative_action"), None),
                    "followup_task_ids": followup_task_ids,
                    "ts": str(_first(verdict_row.get("ts"), emit.get("ts"), review.get("ts")) or ""),
                    "review_path": (workdir / "review.json").relative_to(session_dir).as_posix(),
                }
            )
    rows.sort(key=lambda row: row.get("ts") or "")
    return rows


def _source_arm(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    specialist_rows: list[dict[str, Any]],
    specialist_roles: dict[str, str],
    progress_rows: list[dict[str, Any]],
    window: dict[str, Any],
    window_count: int,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    consecutive_no_keep = _optional_int(
        _first(evidence.get("source_consecutive_no_keep"), evidence.get("consecutive_no_keep"))
    )
    if consecutive_no_keep is None:
        consecutive_no_keep = 0
        for row in reversed(progress_rows):
            if str(row.get("status") or "").lower() == "cycle_boundary":
                break
            if bool(row.get("kept")) or str(row.get("status") or "").lower() == "kept":
                break
            if str(row.get("status") or "").lower() == "dispatch_failed":
                continue
            consecutive_no_keep += 1
    candidates_exhausted = _optional_bool(
        _first(evidence.get("source_candidates_exhausted"), evidence.get("candidates_exhausted"))
    )
    if candidates_exhausted is None:
        state_cycle = _optional_int(state.get("macro_cycle"))
        if window_count == 1 or state_cycle == int(window["cycle"]):
            candidates_exhausted = _optional_bool(state.get("framework_agent_phase_done"))
    triggered = _optional_bool(evidence.get("source_arm_plateaued"))
    if triggered is None:
        threshold = _optional_int(_first(evidence.get("source_threshold"), evidence.get("threshold")))
        if candidates_exhausted is True:
            triggered = True
        elif threshold is not None:
            triggered = consecutive_no_keep >= threshold
    return {
        "plateau": {
            "triggered": triggered,
            "consecutive_no_keep": consecutive_no_keep,
            "candidates_exhausted": candidates_exhausted,
        },
        "candidate_discovery_runs": _candidate_discovery_runs(
            state,
            specialist_rows,
            specialist_roles,
            window,
            window_count,
        ),
        "authoring_runs": _authoring_runs(state, specialist_rows, specialist_roles, progress_rows),
        "attempts": _source_attempts(state, operations, progress_rows),
    }


def _framework_exit(
    window: dict[str, Any],
    config_arm: dict[str, Any],
    source_arm: dict[str, Any],
) -> dict[str, Any]:
    exit_row = _mapping(window.get("exit_row"))
    evidence = _mapping(exit_row.get("evidence"))
    raw_reason = str(_first(evidence.get("passed_through_reason"), exit_row.get("reason")) or "")
    reason = _FRAMEWORK_EXIT_REASON_MAP.get(raw_reason, raw_reason) or None
    trigger = _first(evidence.get("trigger"), evidence.get("evidence"), None)
    if trigger == "phase_budget_cap":
        trigger = "budget_cap"
    if trigger is None:
        trigger = {
            "optimize_no_more_leverage": "both_arms_plateaued",
            "optimize_phase_budget_exhausted": "phase_budget_exhausted",
            "optimize_budget_cap": "budget_cap",
            "optimize_force_exit_low_budget": "force_exit",
        }.get(str(reason or ""))
    switch_bottleneck = _optional_bool(evidence.get("switch_bottleneck"))
    if switch_bottleneck is None:
        plateau_values = (
            _nested(config_arm, "plateau", "triggered"),
            _nested(source_arm, "plateau", "triggered"),
        )
        if any(value is True for value in plateau_values):
            switch_bottleneck = True
        elif all(value is False for value in plateau_values):
            switch_bottleneck = False
    return {
        "reason": reason,
        "trigger": trigger,
        "hint": _first(evidence.get("hint"), None),
        "switch_bottleneck": switch_bottleneck,
    }


def _framework_failure(window: dict[str, Any]) -> dict[str, Any]:
    exit_row = _mapping(window.get("exit_row"))
    evidence = _mapping(exit_row.get("evidence"))
    reason = str(exit_row.get("reason") or "").lower()
    is_failure = bool(evidence.get("error") or evidence.get("error_class") or reason.endswith("_failed"))
    if is_failure:
        return {
            "failed_task_id": _first(evidence.get("failed_task_id"), evidence.get("task_id"), None),
            "error_class": _first(evidence.get("error_class"), None),
            "error": _first(evidence.get("error"), None),
        }

    rows = _dict_rows(window.get("rows"))
    terminal_discovery_failure = next(
        (
            row
            for row in reversed(rows)
            if _nested(row, "evidence", "event") == "framework_agent_phase_done"
            and str(row.get("reason") or "") == "discover_retries_exhausted"
        ),
        None,
    )
    if terminal_discovery_failure is None:
        return {"failed_task_id": None, "error_class": None, "error": None}

    failed_discovery = next(
        (row for row in reversed(rows) if _nested(row, "evidence", "event") == "framework_agent_discover_failed"),
        terminal_discovery_failure,
    )
    failure_evidence = _mapping(failed_discovery.get("evidence"))
    return {
        "failed_task_id": _first(
            failure_evidence.get("failed_task_id"),
            failure_evidence.get("task_id"),
            None,
        ),
        "error_class": _first(failure_evidence.get("error_class"), None),
        "error": _first(
            failure_evidence.get("error"),
            terminal_discovery_failure.get("reason"),
            None,
        ),
    }


def _framework_event(
    session_dir: Path,
    state: dict[str, Any],
    recorded_operations: list[dict[str, Any]],
    warnings: list[str],
    window: dict[str, Any],
    window_count: int,
) -> dict[str, Any]:
    operations = _window_operations(recorded_operations, window, window_count)
    progress_rows = _window_state_rows(state, "framework_agent_phase_progress", window, window_count)
    candidate_map = {
        str(key): str(value)
        for key, value in _mapping(state.get("framework_agent_specialist_candidate_map")).items()
        if key and value
    }
    source_task_ids = {
        str(row.get("specialist_task_id") or "") for row in progress_rows if row.get("specialist_task_id")
    }
    specialist_rows = _specialist_rows(state, operations, window, window_count)
    specialist_roles = {
        str(_first(row.get("task_id"), row.get("round_id")) or ""): _specialist_role(
            row,
            candidate_map,
            source_task_ids,
        )
        for row in specialist_rows
    }
    evidence = _window_evidence(window)
    policy = _framework_policy(state, operations, evidence)
    config_arm = _config_arm(
        state,
        operations,
        specialist_rows,
        specialist_roles,
        window,
        window_count,
        evidence,
        policy,
    )
    source_arm = _source_arm(
        state,
        operations,
        specialist_rows,
        specialist_roles,
        progress_rows,
        window,
        window_count,
        evidence,
    )
    exit_details = _framework_exit(window, config_arm, source_arm)
    failure = _framework_failure(window)
    has_work = bool(
        config_arm["specialist_runs"]
        or config_arm["rounds"]
        or any(run["candidates"] for run in source_arm["candidate_discovery_runs"])
        or source_arm["authoring_runs"]
        or source_arm["attempts"]
    )
    if any(value is not None for value in failure.values()):
        status = "failed"
    elif not window.get("end_time"):
        status = "degraded"
    elif has_work:
        status = "succeeded"
    else:
        status = "skipped"
    return {
        "type": "framework_agent",
        "kind": "agent",
        "status": status,
        "start_time": str(window.get("start_time") or ""),
        "end_time": str(window.get("end_time") or ""),
        "ext": {
            "macro_cycle": int(window["cycle"]),
            "policy": policy,
            "critic_reviews": _critic_review_rows(
                session_dir,
                warnings,
                state,
                recorded_operations,
                window,
                window_count,
            ),
            "config_arm": config_arm,
            "source_arm": source_arm,
            "exit": exit_details,
            "failure": failure,
        },
    }


def collect_v6_timeline(
    session_dir: Path,
    warnings: list[str],
    *,
    state: dict[str, Any] | None = None,
    recorded_operations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load durable events and project framework work without mutating V5 state."""
    timeline = read_timeline_events(session_dir, warnings=warnings)
    state = state if isinstance(state, dict) else {}
    operations = [row for row in recorded_operations or [] if isinstance(row, dict)]
    windows = _framework_windows(state, operations)
    for window in windows:
        timeline.append(
            _framework_event(
                session_dir,
                state,
                operations,
                warnings,
                window,
                len(windows),
            )
        )
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
        "SWEEP": ("conc_sweep" if state.get("last_conc_sweep") or state.get("last_conc_sweep_watermark") else "sweep"),
        "CLOSE": "close",
    }
    if phase in phase_map:
        return phase_map[phase]
    if timeline:
        return str(timeline[-1].get("type") or "")
    return "install"


def collect_v6_outcome(
    *,
    session: dict[str, Any],
    baseline: dict[str, Any],
    final: dict[str, Any],
    optimizations: dict[str, Any],
    state: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project V5 result sections into the V6 ``outcome`` shape."""
    stop_reason = str(session.get("stop_reason") or "").strip()
    validation = optimizations.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    return {
        "stop_reason": stop_reason,
        "status": _outcome_status(stop_reason),
        "stage_reached": _stage_reached(state, stop_reason, timeline),
        "baseline": {
            "throughput_tok_s_per_gpu": baseline.get("throughput_tok_s_per_gpu"),
            "accuracy": baseline.get("accuracy"),
            "ttft_mean_ms": baseline.get("ttft_mean_ms"),
            "e2el_mean_ms": baseline.get("e2el_mean_ms"),
        },
        "final": {
            "throughput_tok_s_per_gpu": final.get("throughput_tok_s_per_gpu"),
            "gain_pct": final.get("cumulative_gain_pct_validated", 0.0),
            "action_path": list(final.get("action_path") or []),
            "extra_envs": dict(final.get("extra_envs") or {}),
            "extra_server_args": str(final.get("extra_server_args") or ""),
        },
        "validation": {
            "attributed_gain_pct": validation.get("attributed_total_gain_pct", 0.0),
            "unattributed_gain_pct": validation.get("unattributed_gain_pct", 0.0),
            "reconciliation_gap_pct": validation.get("reconciliation_gap_pct"),
            "notes": list(validation.get("notes") or []),
        },
    }


__all__ = [
    "collect_v6_metadata",
    "collect_v6_outcome",
    "collect_v6_timeline",
]
