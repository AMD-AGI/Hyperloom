"""Pure read-only collectors for session breakdown.

Each collector reads from disk (session_dir) and returns a typed section
dict. Failures are appended to ``warnings``; collectors never raise.

Data sources (hyperloom architecture):
  - state.json (session state, depth tracker, targets)
  - agents/<id>/manifest.json (per-agent task manifests)
  - agents/<id>/done.json (completion reports)
  - agents/<id>/results.json (incremental results)
  - agents/<id>/process.log (agent stdout/stderr)
  - agents/<id>/*.patch (generated patches)
  - benchmarks/ (benchmark reports and configs)
  - profiling/ (torch traces, TraceLens outputs)
  - watchdog/events.jsonl (watchdog event stream)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect session metadata."""
    created_at = state.get("created_at") or manifest.get("created_at", "")
    ended_at = state.get("ended_at", "")
    elapsed_minutes = 0.0
    if created_at and ended_at:
        try:
            start = datetime.fromisoformat(created_at)
            end = datetime.fromisoformat(ended_at)
            elapsed_minutes = (end - start).total_seconds() / 60.0
        except (ValueError, TypeError):
            pass

    agents_dir = session_dir / "agents"
    agent_count = 0
    if agents_dir.is_dir():
        agent_count = sum(1 for d in agents_dir.iterdir() if d.is_dir())

    return {
        "session_id": state.get("session_id") or manifest.get("session_id", ""),
        "created_at_utc": created_at,
        "ended_at_utc": ended_at,
        "stop_reason": state.get("stop_reason", ""),
        "max_minutes": manifest.get("max_minutes", 0),
        "elapsed_minutes": round(elapsed_minutes, 1),
        "host": platform.node(),
        "pid": os.getpid(),
        "session_dir": str(session_dir),
        "tick_count": state.get("tick_count", 0),
        "agent_count": agent_count,
    }


def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect workload/configuration data."""
    config = state.get("config") or manifest.get("config", {})
    benchmark = config.get("benchmark", {})
    return {
        "framework": benchmark.get("framework", state.get("framework", "")),
        "model_name": config.get("model_name", ""),
        "model_path": config.get("model_path", state.get("model_path", "")),
        "gpu_type": config.get("gpu_type", state.get("gpu_type", "")),
        "tp": config.get("tp"),
        "concurrency": config.get("concurrency"),
        "isl": config.get("isl"),
        "osl": config.get("osl"),
        "precision": config.get("precision", ""),
        "objective": {
            "kind": "gain_pct",
            "value": config.get("target_gain", state.get("target_gain")),
        },
    }


def collect_baseline(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect baseline benchmark results."""
    baseline = state.get("baseline", {})
    if not baseline:
        bench_dir = session_dir / "benchmarks" / "baseline"
        report_path = bench_dir / "benchmark_report.json"
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text())
                return {
                    "throughput_tok_s": float(data.get("output_token_throughput", 0)),
                    "latency_mean_ms": data.get("mean_e2e_latency_ms"),
                    "benchmark_report_path": str(report_path),
                    "timestamp": data.get("timestamp", ""),
                }
            except (json.JSONDecodeError, ValueError) as e:
                warnings.append(f"baseline report parse error: {e}")
        return {}

    return {
        "throughput_tok_s": baseline.get("throughput", baseline.get("throughput_tok_s", 0)),
        "latency_mean_ms": baseline.get("latency_mean_ms"),
        "accuracy": baseline.get("accuracy"),
        "config_path": baseline.get("config_path"),
        "benchmark_report_path": baseline.get("benchmark_report_path"),
        "timestamp": baseline.get("timestamp", ""),
    }


def collect_final(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect final optimization results."""
    current_best = state.get("current_best", {})
    gain = state.get("cumulative_gain", 0.0)
    validated = state.get("cumulative_gain_validated", gain)
    stack = state.get("optimization_stack", [])

    return {
        "throughput_tok_s": current_best.get("throughput"),
        "cumulative_gain_pct": gain,
        "validated": abs(gain - validated) < 0.01 and gain > 0,
        "action_path": [entry.get("action", "") for entry in stack if isinstance(entry, dict)],
        "extra_args": current_best.get("extra_args", ""),
        "extra_envs": current_best.get("extra_envs", {}),
        "timestamp": current_best.get("timestamp", ""),
    }


def collect_agent_timeline(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Collect dispatch/completion timeline for all agents."""
    agents_dir = session_dir / "agents"
    if not agents_dir.is_dir():
        return []

    events: list[dict[str, Any]] = []

    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        agent_id = agent_dir.name
        manifest_path = agent_dir / "manifest.json"
        done_path = agent_dir / "done.json"

        manifest_data: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        done_data: dict[str, Any] = {}
        if done_path.exists():
            try:
                done_data = json.loads(done_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        status = done_data.get("status", "unknown")
        if not done_data and not manifest_data:
            continue

        dispatched_at = manifest_data.get("dispatched_at", "")
        completed_at = done_data.get("completed_at", "")
        runtime_s = 0.0
        if dispatched_at and completed_at:
            try:
                start = datetime.fromisoformat(dispatched_at)
                end = datetime.fromisoformat(completed_at)
                runtime_s = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

        events.append({
            "ts": dispatched_at,
            "agent_id": agent_id,
            "role": manifest_data.get("role", _infer_role_from_id(agent_id)),
            "status": status,
            "task_summary": manifest_data.get("task_description", ""),
            "gpu_ids": manifest_data.get("gpu_ids", []),
            "runtime_s": round(runtime_s, 1),
            "attempt": manifest_data.get("attempt", 1),
            "failure_type": done_data.get("failure_type"),
            "error_snippet": (done_data.get("error") or "")[:500] or None,
        })

    return sorted(events, key=lambda e: e.get("ts", ""))


def collect_capability_summary(
    session_dir: Path,
    state: dict[str, Any],
    agent_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect capability usage summary from state and agent outcomes."""
    caps = state.get("capabilities", {})

    def _cap_entry(name: str, configured: bool) -> dict[str, Any]:
        agents_for_cap = [
            a for a in agent_timeline
            if name.lower() in (a.get("role", "") + a.get("task_summary", "")).lower()
        ]
        attempts = len(agents_for_cap)
        keeps = sum(1 for a in agents_for_cap if a.get("status") == "success")
        return {
            "status": "kept" if keeps > 0 else ("tried" if attempts > 0 else
                     ("not_attempted" if configured else "not_configured")),
            "attempts": attempts,
            "keeps": keeps,
            "best_gain_pct": None,
            "reason": "",
        }

    return {
        "geak": _cap_entry("geak", caps.get("geak", False)),
        "oob": _cap_entry("oob", caps.get("oob", False)),
        "tracelens": _cap_entry("tracelens", caps.get("tracelens", False)),
        "magpie": _cap_entry("magpie", caps.get("magpie", False)),
        "specialist": _cap_entry("specialist", True),
    }


def collect_kernel_invocations(
    session_dir: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect GEAK and OOB kernel optimization invocations."""
    geak_invocations: list[dict[str, Any]] = []
    oob_invocations: list[dict[str, Any]] = []

    kernel_dir = session_dir / "kernel_phase"
    if not kernel_dir.is_dir():
        return geak_invocations, oob_invocations

    for run_dir in sorted(kernel_dir.rglob("runs/*")):
        if not run_dir.is_dir():
            continue
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text())
            backend = data.get("backend", "unknown")
            entry = {
                "kernel_id": data.get("kernel_id", run_dir.name),
                "attempt_id": data.get("attempt_id", ""),
                "ts": data.get("timestamp", ""),
                "backend": backend,
                "kernel_metadata": data.get("kernel_metadata", {}),
                "optimized_files": data.get("optimized_files", []),
                "decision": data.get("decision", ""),
                "micro_speedup": data.get("micro_speedup"),
                "compile_passed": data.get("compile_passed"),
                "correctness_passed": data.get("correctness_passed"),
                "error": data.get("error"),
            }
            if backend in ("geak", "claude"):
                geak_invocations.append(entry)
            else:
                oob_invocations.append(entry)
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"kernel result parse error in {run_dir}: {e}")

    return geak_invocations, oob_invocations


def collect_kernel_lifecycle(
    session_dir: Path,
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Build kernel lifecycle from profiling + invocation data."""
    all_invocations = geak_invocations + oob_invocations

    detected = state.get("detected_kernels", [])
    adopted = state.get("adopted_kernels", [])
    rejected = state.get("rejected_kernels", [])

    kernel_attempts: dict[str, list[dict]] = {}
    for inv in all_invocations:
        kid = inv.get("kernel_id", "")
        kernel_attempts.setdefault(kid, []).append(inv)

    optimized = []
    for kid, attempts in kernel_attempts.items():
        successes = [a for a in attempts if a.get("decision") in ("KEEP", "PARTIAL")]
        best_speedup = max(
            (a.get("micro_speedup") or 0 for a in attempts), default=None
        )
        optimized.append({
            "kernel_id": kid,
            "backend": attempts[-1].get("backend", "") if attempts else "",
            "total_attempts": len(attempts),
            "successful_attempts": len(successes),
            "best_micro_speedup": best_speedup if best_speedup else None,
            "last_decision": attempts[-1].get("decision", "") if attempts else "",
        })

    return {
        "detected": detected,
        "optimized": optimized,
        "adopted": adopted,
        "rejected": rejected,
    }


def collect_profiling(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect profiling summary."""
    prof_dir = session_dir / "profiling"
    if not prof_dir.is_dir():
        return {}

    trace_path = None
    for candidate in prof_dir.rglob("trace.json"):
        trace_path = str(candidate)
        break

    analysis_path = None
    for candidate in prof_dir.rglob("analysis.md"):
        analysis_path = str(candidate)
        break

    top_kernels = state.get("top_kernels", [])
    total_gpu_time = state.get("total_gpu_time_us", 0.0)

    return {
        "total_gpu_time_us": total_gpu_time,
        "hot_kernel_count": len([k for k in top_kernels if k.get("percentage", 0) > 0.05]),
        "top_kernels": top_kernels[:20],
        "trace_path": trace_path,
        "tracelens_analysis_path": analysis_path,
    }


def collect_sweep(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect benchmark sweep results."""
    sweep = state.get("last_sweep", {})
    if sweep:
        return sweep

    sweep_dir = session_dir / "benchmarks" / "sweep"
    if not sweep_dir.is_dir():
        return {}

    variants: list[dict[str, Any]] = []
    for report_path in sorted(sweep_dir.glob("*/benchmark_report.json")):
        try:
            data = json.loads(report_path.read_text())
            variants.append({
                "variant_name": report_path.parent.name,
                "throughput_tok_s": data.get("output_token_throughput"),
                "latency_mean_ms": data.get("mean_e2e_latency_ms"),
                "status": "ok",
            })
        except (json.JSONDecodeError, OSError):
            continue

    if not variants:
        return {}

    best = max(variants, key=lambda v: v.get("throughput_tok_s") or 0)
    return {
        "grid_size": len(variants),
        "best_overall": best,
        "all_variants": variants,
    }


def collect_watchdog_events(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Collect watchdog events from the event log."""
    events_path = session_dir / "watchdog" / "events.jsonl"
    if not events_path.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append({
                    "ts": event.get("timestamp", ""),
                    "severity": event.get("severity", "info"),
                    "category": event.get("category", ""),
                    "summary": event.get("summary", event.get("message", "")),
                    "action_taken": event.get("action_taken"),
                })
            except json.JSONDecodeError:
                continue
    except OSError as e:
        warnings.append(f"watchdog events read error: {e}")

    return events


def collect_attribution(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect gain attribution."""
    stack = state.get("optimization_stack", [])
    if not stack:
        return {"method": "missing", "notes": ["No optimization stack found"]}

    gain_entries: list[dict[str, Any]] = []
    cumulative = 0.0
    for entry in stack:
        if not isinstance(entry, dict):
            continue
        delta = entry.get("gain_pct", 0.0)
        cumulative += delta
        gain_entries.append({
            "ts": entry.get("timestamp", ""),
            "action": entry.get("action", ""),
            "variant_name": entry.get("variant_name"),
            "delta_pct": delta,
            "cum_gain_after": round(cumulative, 2),
        })

    geak_gain = sum(
        e.get("delta_pct", 0) for e in gain_entries
        if "kernel" in (e.get("action") or "").lower()
        or "geak" in (e.get("action") or "").lower()
    )
    total = cumulative or 1.0
    source_breakdown = {
        "geak_pct_of_total": round(geak_gain / total * 100, 1) if total else 0,
        "oob_pct_of_total": 0.0,
        "framework_pct_of_total": 0.0,
        "config_pct_of_total": 0.0,
    }

    return {
        "gain_per_entry": gain_entries,
        "source_breakdown": source_breakdown,
        "method": "reconstructed" if gain_entries else "missing",
        "notes": [],
    }


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect telemetry/GPU monitoring data."""
    traces = [str(p) for p in (session_dir / "profiling").rglob("trace.json")] if (session_dir / "profiling").is_dir() else []
    server_logs = [str(p) for p in session_dir.rglob("server.log")]

    return {
        "gpu_count": len(state.get("gpu_ids", [])) or int(os.environ.get("GPU_COUNT", "0")),
        "gpu_type": state.get("gpu_type", os.environ.get("GPU_TYPE", "")),
        "trace_paths": traces,
        "server_log_paths": server_logs,
    }


def collect_source_files(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Collect source file references for reproducibility."""
    agents_dir = session_dir / "agents"
    manifests = [str(p) for p in agents_dir.rglob("manifest.json")] if agents_dir.is_dir() else []
    logs = [str(p) for p in agents_dir.rglob("process.log")] if agents_dir.is_dir() else []
    patches = [str(p) for p in session_dir.rglob("*.patch")]
    reports = [str(p) for p in session_dir.rglob("benchmark_report.json")]

    return {
        "session_dir": str(session_dir),
        "state_json": str(session_dir / "state.json") if (session_dir / "state.json").exists() else None,
        "manifests": manifests,
        "agent_logs": logs[:50],
        "patches": patches,
        "benchmark_reports": reports,
    }


def _infer_role_from_id(agent_id: str) -> str:
    """Infer agent role from the ID prefix."""
    for prefix in ("kernel", "specialist", "research", "profiler", "framework"):
        if agent_id.startswith(prefix):
            return prefix
    return "agent"
