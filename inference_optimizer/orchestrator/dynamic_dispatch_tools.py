"""Tool definitions and executor for dynamic specialist dispatch.

Provides the orchestration agent with three tools:
  - dispatch_specialists: Launch specialist agents with free-form tasks
  - check_specialists: Poll status of all dispatched specialists
  - collect_specialist_results: Retrieve results from a completed specialist

These tools are registered alongside the existing orchestration tool surface
so the agent can use both the structured domain path (via delegate intents)
and the dynamic free-form path (via these tools).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .dynamic_dispatch import (
    AgentHandle,
    TaskSpec,
    TaskPriority,
    dispatch_batch,
    reap_completed,
)
from .dynamic_dispatch_comms import (
    read_completion,
    read_heartbeat,
    read_agent_results,
    get_agent_status_summary,
    collect_patches,
)

log = logging.getLogger(__name__)

_AGENT_HANDLES: list[AgentHandle] = []


DYNAMIC_DISPATCH_TOOLS = [
    {
        "name": "dispatch_specialists",
        "description": (
            "Dispatch specialist agents to work in parallel on research, "
            "code analysis, KB search, or patch generation tasks. "
            "All specialists are CPU-only and launch immediately. "
            "Returns agent IDs for tracking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": (
                                    "Full natural-language task for the specialist. "
                                    "Be specific: what to investigate, what code to read, "
                                    "what patches to produce, what to report back."
                                ),
                            },
                            "task_summary": {
                                "type": "string",
                                "description": "Short summary for status display (< 100 chars).",
                            },
                            "role": {
                                "type": "string",
                                "default": "specialist",
                                "description": (
                                    "Free-form role label (e.g. 'researcher', 'patcher', "
                                    "'config_explorer', 'profiler')."
                                ),
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["critical", "high", "normal", "low"],
                                "default": "normal",
                            },
                        },
                        "required": ["task_description", "task_summary"],
                    },
                },
                "model": {
                    "type": "string",
                    "description": "Model for specialist agents (default: claude-sonnet-4-6).",
                },
                "timeout_minutes": {
                    "type": "integer",
                    "default": 120,
                    "description": "Per-agent timeout in minutes.",
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "check_specialists",
        "description": (
            "Check status of all dynamically dispatched specialists. "
            "Returns active, completed, and dead agents with heartbeats."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "collect_specialist_results",
        "description": (
            "Collect results from a completed specialist: completion report, "
            "incremental findings, and patches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the completed specialist agent.",
                },
            },
            "required": ["agent_id"],
        },
    },
]


def execute_dynamic_dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    session_dir: str,
) -> str:
    """Execute a dynamic dispatch tool call. Returns string result."""
    try:
        if tool_name == "dispatch_specialists":
            return _exec_dispatch(tool_input, session_dir)
        elif tool_name == "check_specialists":
            return _exec_check(session_dir)
        elif tool_name == "collect_specialist_results":
            return _exec_collect(tool_input, session_dir)
        else:
            return f"Unknown dynamic dispatch tool: {tool_name}"
    except Exception as e:
        import traceback
        return f"Tool error: {e}\n{traceback.format_exc()}"


def _exec_dispatch(tool_input: dict[str, Any], session_dir: str) -> str:
    global _AGENT_HANDLES

    tasks_input = tool_input.get("tasks", [])
    model = tool_input.get("model", os.environ.get("AGENT_MODEL", "claude-sonnet-4-6"))
    timeout_minutes = tool_input.get("timeout_minutes", 120)

    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW,
    }

    tasks: list[TaskSpec] = []
    for t in tasks_input:
        tasks.append(TaskSpec(
            prompt=t["task_description"],
            task_summary=t.get("task_summary", t["task_description"][:100]),
            role=t.get("role", "specialist"),
            priority=priority_map.get(t.get("priority", "normal"), TaskPriority.NORMAL),
            timeout_minutes=timeout_minutes,
        ))

    result = dispatch_batch(tasks, session_dir, model=model)
    _AGENT_HANDLES.extend(result.launched)

    lines = [f"Dispatched {len(result.launched)} specialists.\n"]
    for h in result.launched:
        lines.append(
            f"  LAUNCHED: id={h.agent_id} role={h.role} summary={h.task_summary!r}"
        )
    for e in result.errors:
        lines.append(f"  ERROR: {e}")

    return "\n".join(lines)


def _exec_check(session_dir: str) -> str:
    global _AGENT_HANDLES

    if _AGENT_HANDLES:
        still_running, _ = reap_completed(_AGENT_HANDLES, session_dir)
        _AGENT_HANDLES = still_running

    summary = get_agent_status_summary(session_dir)
    lines = [
        f"Active: {len(summary['active'])}  "
        f"Completed: {len(summary['completed'])}  "
        f"Dead: {len(summary['dead'])}\n"
    ]

    for aid in summary["active"]:
        hb = read_heartbeat(session_dir, aid)
        note = hb.progress_note if hb else "no heartbeat"
        lines.append(f"  ACTIVE  {aid}: {note}")

    for aid in summary["completed"]:
        report = read_completion(session_dir, aid)
        if report:
            lines.append(f"  DONE    {aid}: status={report.status} — {report.summary[:120]}")
        else:
            lines.append(f"  DONE    {aid}: (report unreadable)")

    for aid in summary["dead"]:
        lines.append(f"  DEAD    {aid}: presumed crashed")

    return "\n".join(lines)


def _exec_collect(tool_input: dict[str, Any], session_dir: str) -> str:
    agent_id = tool_input.get("agent_id", "")
    if not agent_id:
        return "Error: agent_id is required"

    agent_dir = Path(session_dir) / "agents" / agent_id
    if not agent_dir.exists():
        return f"Error: agent directory not found: {agent_dir}"

    lines = [f"=== Results for specialist {agent_id} ===\n"]

    report = read_completion(session_dir, agent_id)
    if report:
        lines.append(f"Status: {report.status}")
        lines.append(f"Summary: {report.summary}")
        lines.append(f"Completed: {report.completed_at}")
        if report.config_changes:
            lines.append(f"Config changes: {json.dumps(report.config_changes)}")
        if report.patches_written:
            lines.append(f"Patches: {report.patches_written}")
        if report.new_knowledge:
            lines.append(f"New knowledge: {report.new_knowledge}")
        if report.error:
            lines.append(f"Error: {report.error}")
    else:
        lines.append("No completion report (done.json) found.")

    results = read_agent_results(session_dir, agent_id)
    if results:
        lines.append(f"\n--- Incremental results ({len(results)}) ---")
        for r in results:
            lines.append(f"  [{r.impact}] {r.category}: {r.description[:200]}")

    patches = collect_patches(session_dir, agent_id)
    if patches:
        lines.append(f"\n--- Patches ({len(patches)}) ---")
        for p in patches:
            lines.append(f"  {p.name} ({p.stat().st_size} bytes)")

    knowledge_path = agent_dir / "new_knowledge.md"
    if knowledge_path.exists():
        content = knowledge_path.read_text()[:2000]
        lines.append(f"\n--- New Knowledge ---\n{content}")

    return "\n".join(lines)


def get_dynamic_dispatch_tool_names() -> list[str]:
    """Return the tool names for dynamic dispatch (for PolicyGate whitelisting)."""
    return [t["name"] for t in DYNAMIC_DISPATCH_TOOLS]
