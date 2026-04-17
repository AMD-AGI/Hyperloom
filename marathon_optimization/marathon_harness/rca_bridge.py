"""Bridge to agentic-rc/run_analysis.py — prepare issue dir, call RCA agent, parse output."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

AGENTIC_RC_PATH = "/shared_nfs/nehaprakriya/agentic-rc"


async def prepare_and_run(
    event: dict[str, Any],
    work_queue_context: dict[str, Any] | None,
    state: Any,
    env: dict[str, str],
) -> Path:
    """Full RCA pipeline: prepare issue dir -> call run_analysis -> return path."""
    session_dir = state.session_dir
    event_id = event.get("id", "unknown")
    issue_dir = Path(session_dir) / "kernel_manager" / "rca_reports" / event_id
    issue_dir.mkdir(parents=True, exist_ok=True)
    (issue_dir / "evidence").mkdir(exist_ok=True)

    _prepare_issue_dir(issue_dir, event, work_queue_context, state)

    try:
        await asyncio.get_event_loop().run_in_executor(None, _run_rca_sync, str(issue_dir), env)
    except Exception as exc:
        log.error("RCA agent failed for %s: %s", event_id, exc)
        _write_fallback_report(issue_dir, event, str(exc))

    return issue_dir


def _prepare_issue_dir(
    issue_dir: Path,
    event: dict[str, Any],
    wq_context: dict[str, Any] | None,
    state: Any,
) -> None:
    """Populate issue dir with files the RCA agent expects."""
    # inference_workload_info.json (required)
    (issue_dir / "inference_workload_info.json").write_text(json.dumps({
        "model": state.model_name,
        "hardware": state.gpu_type,
        "gpu_count": state.gpu_count,
        "framework": state.framework,
        "tp": state.tp,
        "kernel_name": event.get("kernel_name"),
        "event_type": event.get("type"),
        "server_config": getattr(state, "server_config", {}),
    }, indent=2, default=str))

    # crash_metrics.json (required: at least 1 non-workload-info JSON)
    details = event.get("details", {})
    (issue_dir / "crash_metrics.json").write_text(json.dumps({
        "gpu_pct": details.get("gpu_pct"),
        "micro_speedup": details.get("micro_speedup_before_crash"),
        "exit_code": details.get("exit_code"),
        "round_number": details.get("round_number"),
        "strategy_used": details.get("strategy_used"),
        "backend_used": details.get("backend_used"),
        "session_history": details.get("session_history"),
    }, indent=2, default=str))

    # crash_log.log
    crash_log = details.get("crash_log_snippet", "")
    if crash_log:
        (issue_dir / "crash_log.log").write_text(crash_log)
        (issue_dir / "evidence" / "crash_log.txt").write_text(crash_log)

    # kernel context from work queue
    if wq_context:
        (issue_dir / "kernel_context.json").write_text(json.dumps({
            "source_file": wq_context.get("source_file"),
            "strategy": wq_context.get("strategy"),
            "dispatch_analysis": wq_context.get("dispatch_analysis"),
            "trace_shapes": wq_context.get("trace_shapes"),
            "constraints": wq_context.get("constraints"),
            "rca_constraints": wq_context.get("rca_constraints"),
        }, indent=2, default=str))


def _run_rca_sync(issue_dir: str, env: dict[str, str]) -> None:
    """Call run_analysis.run_agent() synchronously (runs in thread executor).

    Falls back gracefully if agentic-rc is unavailable.
    """
    if AGENTIC_RC_PATH not in sys.path:
        sys.path.insert(0, AGENTIC_RC_PATH)

    try:
        from run_analysis import run_agent, build_system_prompt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"agentic-rc not available at {AGENTIC_RC_PATH}: {exc}. "
            f"Falling back to inline diagnosis."
        ) from exc

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(f"openai package not installed: {exc}") from exc

    api_key = env.get("LITELLM_API_KEY", "")
    base_url = env.get("LITELLM_BASE_URL", "")
    model = env.get("LITELLM_MODEL", "claude-opus-4-6")

    if not api_key or not base_url:
        raise ValueError("LITELLM_API_KEY and LITELLM_BASE_URL required for RCA agent")

    client = OpenAI(api_key=api_key, base_url=base_url)
    output_path = str(Path(issue_dir) / "detailed_report.md")
    system_prompt = build_system_prompt(issue_dir, output_path, use_skill=True)
    run_agent(client, model, system_prompt, issue_dir, max_iteration=50)


def _write_fallback_report(issue_dir: Path, event: dict[str, Any], error: str) -> None:
    """Write minimal RCA output when the agent fails."""
    (issue_dir / "rca_summary.json").write_text(json.dumps({
        "event_id": event.get("id"),
        "kernel_name": event.get("kernel_name"),
        "classification": "unknown",
        "root_cause": f"RCA agent failed: {error}",
        "confidence": "low",
        "evidence_quality": "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    (issue_dir / "detailed_report.md").write_text(
        f"# RCA Report (fallback)\n\nAgent failed: {error}\n\n"
        f"Event:\n```json\n{json.dumps(event, indent=2, default=str)}\n```\n"
    )


def parse_rca_output(issue_dir: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Parse rca_summary.json into finding schema for findings.jsonl."""
    rca_path = issue_dir / "rca_summary.json"
    if not rca_path.exists():
        return _fallback_finding(event)

    try:
        rca = json.loads(rca_path.read_text())
    except (json.JSONDecodeError, OSError):
        return _fallback_finding(event)

    classification = rca.get("classification", "unknown")
    return {
        "event_id": event.get("id"),
        "task_id": event.get("task_id"),
        "kernel_name": event.get("kernel_name"),
        "classification": classification,
        "root_cause": rca.get("root_cause", ""),
        "actionable_guidance": {
            "constraint": rca.get("constraint"),
            "approach": rca.get("approach", "oob-rewrite"),
            "avoid": rca.get("avoid", []),
            "compiler_flags": rca.get("compiler_flags"),
            "reference_commit": rca.get("reference_commit"),
            "fix_command": rca.get("fix_command"),
        },
        "rca_report_path": str(issue_dir / "detailed_report.md"),
        "confidence": rca.get("confidence", "medium"),
        "resubmit": classification != "hardware",
        "systemic": rca.get("systemic", False),
        "affects_kernels": rca.get("affects_kernels", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_finding(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("id"),
        "task_id": event.get("task_id"),
        "kernel_name": event.get("kernel_name"),
        "classification": "unknown",
        "root_cause": "RCA investigation incomplete",
        "actionable_guidance": {
            "constraint": None,
            "approach": "oob-rewrite",
            "avoid": [],
            "compiler_flags": None,
            "reference_commit": None,
            "fix_command": None,
        },
        "rca_report_path": "",
        "confidence": "low",
        "resubmit": True,
        "systemic": False,
        "affects_kernels": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
