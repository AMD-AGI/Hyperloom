"""Additive V6 projections built from the existing V5 evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def collect_v6_timeline(
    session_dir: Path,
    warnings: list[str],
    *,
    state: dict[str, Any] | None = None,
    recorded_operations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load durable install and model-gate events without mutating V5 state."""
    del state, recorded_operations
    return read_timeline_events(session_dir, warnings=warnings)


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
