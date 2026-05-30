"""SDK-based orchestrator — agentic loop with dynamic agent dispatch.

Runs the orchestrator as a multi-turn conversation with tool use via the
Anthropic messages API. The orchestrator LLM dynamically decides:
  - WHAT to dispatch (tasks, roles, priorities)
  - WHEN to dispatch (based on available GPUs, session state)
  - HOW MANY agents to run in parallel (respecting GPU pool)

Available tools for the orchestrator:
  - bash:                  Execute commands (benchmarks, server restart, health checks)
  - read_file:             Read files from disk
  - write_file:            Write files to disk
  - dispatch_agents:       Dispatch specialist agents (CPU or GPU)
  - check_agents:          Poll agent status (active, completed, dead)
  - collect_agent_results: Read an agent's completion report and patches
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from hyperloom.comms import (
    read_completion,
    read_heartbeat,
    read_agent_results,
    get_agent_status_summary,
    collect_patches,
)
from hyperloom.dispatch import (
    AgentHandle,
    TaskSpec,
    TaskPriority,
    dispatch_batch,
    reap_completed,
    DispatchResult,
)
from hyperloom.gpu_pool import GPUPool, auto_pool

log = logging.getLogger(__name__)

_AGENT_HANDLES: list[AgentHandle] = []
_GPU_POOL: GPUPool | None = None


def _get_gpu_pool(session_dir: str) -> GPUPool:
    global _GPU_POOL
    if _GPU_POOL is None:
        total = int(os.environ.get("TOTAL_GPUS", "8"))
        _GPU_POOL = auto_pool(session_dir=session_dir, total_gpus=total)
    return _GPU_POOL


# ─── Tool definitions ──────────────────────────────────────────────────────────

ORCHESTRATOR_TOOLS = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command. Use for: running benchmarks, "
            "restarting the server, applying patches, accuracy evals, "
            "and health checks. Do NOT use bash for optimization work — "
            "dispatch specialist agents for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command"},
                "timeout": {"type": "integer", "description": "Timeout seconds (default 900)", "default": 900},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "dispatch_agents",
        "description": (
            "Dispatch specialist agents to work in parallel. "
            "CPU-only agents launch immediately; GPU agents are allocated from the pool. "
            "Returns agent IDs and status for each launched/deferred task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_description": {"type": "string"},
                            "task_summary": {"type": "string"},
                            "needs_gpu": {"type": "boolean", "default": False},
                            "gpu_count": {"type": "integer", "default": 1},
                            "role": {"type": "string", "default": "specialist"},
                            "priority": {"type": "string", "enum": ["critical", "high", "normal", "low"], "default": "normal"},
                            "kb_domains": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["task_description", "task_summary"],
                    },
                },
                "model": {"type": "string"},
                "timeout_minutes": {"type": "integer", "default": 120},
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "check_agents",
        "description": (
            "Check status of all dispatched agents. Returns active, "
            "completed, and dead agents with heartbeats and summaries."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "collect_agent_results",
        "description": (
            "Collect results from a completed agent: completion report, "
            "incremental results, and patches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
            },
            "required": ["agent_id"],
        },
    },
]


# ─── Tool execution ───────────────────────────────────────────────────────────


def execute_tool(tool_name: str, tool_input: dict, session_dir: str) -> str:
    """Execute a tool call from the orchestrator."""
    try:
        if tool_name == "bash":
            return _exec_bash(tool_input, session_dir)
        elif tool_name == "read_file":
            return _exec_read_file(tool_input)
        elif tool_name == "write_file":
            return _exec_write_file(tool_input)
        elif tool_name == "dispatch_agents":
            return _exec_dispatch_agents(tool_input, session_dir)
        elif tool_name == "check_agents":
            return _exec_check_agents(session_dir)
        elif tool_name == "collect_agent_results":
            return _exec_collect_results(tool_input, session_dir)
        else:
            return f"Unknown tool: {tool_name}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {tool_input.get('timeout', 900)}s"
    except Exception as e:
        import traceback
        return f"Tool error: {e}\n{traceback.format_exc()}"


def _exec_bash(tool_input: dict, session_dir: str) -> str:
    cmd = tool_input.get("command", "")
    timeout = tool_input.get("timeout", 120)
    # Run from repo root (parent of session_dir), not session_dir itself
    work_dir = os.path.dirname(os.path.abspath(session_dir))
    try:
        r = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s] Command: {cmd[:200]}"
    output = r.stdout + r.stderr
    if r.returncode != 0:
        output += f"\n[exit code: {r.returncode}]"
    return output[:50000]


def _exec_read_file(tool_input: dict) -> str:
    p = Path(tool_input["path"])
    if p.exists():
        return p.read_text()[:100000]
    return f"File not found: {p}"


def _exec_write_file(tool_input: dict) -> str:
    p = Path(tool_input["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tool_input["content"])
    return f"Wrote {len(tool_input['content'])} chars to {p}"


def _exec_dispatch_agents(tool_input: dict, session_dir: str) -> str:
    global _AGENT_HANDLES

    tasks_input = tool_input.get("tasks", [])
    model = tool_input.get("model", os.environ.get("AGENT_MODEL", "claude-sonnet-4-6"))
    timeout_minutes = tool_input.get("timeout_minutes", 120)
    pool = _get_gpu_pool(session_dir)

    from hyperloom.kb import select_kb, load_kb_content
    from hyperloom.prompt_builder import build_agent_prompt

    tasks: list[TaskSpec] = []
    for t in tasks_input:
        kb_content = ""
        kb_domains = t.get("kb_domains", [])
        if kb_domains:
            try:
                kb_files = select_kb(t["task_description"], domains=kb_domains)
                kb_content = load_kb_content([f.path for f in kb_files])
            except Exception:
                pass

        prompt = build_agent_prompt(
            task=t["task_description"],
            kb_content=kb_content,
            session_dir=session_dir,
        )

        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "normal": TaskPriority.NORMAL,
            "low": TaskPriority.LOW,
        }

        tasks.append(TaskSpec(
            prompt=prompt,
            task_summary=t.get("task_summary", t["task_description"][:100]),
            needs_gpu=t.get("needs_gpu", False),
            gpu_count=t.get("gpu_count", 1),
            role=t.get("role", "specialist"),
            priority=priority_map.get(t.get("priority", "normal"), TaskPriority.NORMAL),
            timeout_minutes=timeout_minutes,
            kb_domains=kb_domains,
        ))

    mcp_config = os.environ.get("HYPERLOOM_MCP_CONFIG")
    result = dispatch_batch(
        tasks, session_dir, gpu_pool=pool,
        model=model, mcp_config_path=mcp_config,
    )
    _AGENT_HANDLES.extend(result.launched)

    lines = [f"Dispatched {len(result.launched)} agents, {len(result.deferred)} deferred.\n"]
    for h in result.launched:
        lines.append(
            f"  LAUNCHED: id={h.agent_id} role={h.role} gpu={h.gpu_ids} "
            f"summary={h.task_summary!r}"
        )
    for d in result.deferred:
        lines.append(f"  DEFERRED: summary={d.task_summary!r} (waiting for GPUs)")
    for e in result.errors:
        lines.append(f"  ERROR: {e}")

    return "\n".join(lines)


def _exec_check_agents(session_dir: str) -> str:
    global _AGENT_HANDLES

    pool = _get_gpu_pool(session_dir)
    if _AGENT_HANDLES:
        still_running, newly_completed = reap_completed(
            _AGENT_HANDLES, session_dir, gpu_pool=pool,
        )
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


def _exec_collect_results(tool_input: dict, session_dir: str) -> str:
    agent_id = tool_input.get("agent_id", "")
    if not agent_id:
        return "Error: agent_id is required"

    agent_dir = Path(session_dir) / "agents" / agent_id
    if not agent_dir.exists():
        return f"Error: agent directory not found: {agent_dir}"

    lines = [f"=== Results for agent {agent_id} ===\n"]

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


# ─── Claude Code CLI orchestrator ─────────────────────────────────────────────


def _build_mcp_config(session_dir: str) -> Path:
    """Create a temporary MCP config file for the dispatch server."""
    config_path = Path(session_dir) / "mcp_config.json"
    server_script = Path(__file__).parent / "mcp_dispatch_server.py"

    config = {
        "mcpServers": {
            "hyperloom-dispatch": {
                "command": "python3",
                "args": [str(server_script)],
                "env": {
                    "HYPERLOOM_SESSION_DIR": str(Path(session_dir).resolve()),
                },
            }
        }
    }

    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def run_claude_code_orchestrator(
    session_dir: str,
    system_prompt: str,
    model: str = "claude-opus-4-7",
    user_prompt: str = "Begin the optimization loop.",
) -> None:
    """Run the orchestrator via Claude Code CLI (claude).

    Uses `claude --dangerously-skip-permissions -p` for a non-interactive
    agentic session with full tool access. Connects to the Hyperloom MCP
    dispatch server so the orchestrator can spawn specialist sub-agents.
    """
    import shutil

    claude_bin = shutil.which("claude")
    if not claude_bin:
        click.echo("ERROR: claude CLI not found. Install: npm install -g @anthropic-ai/claude-code", err=True)
        return

    log_dir = Path(session_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "orchestrator.log"

    mcp_config_path = _build_mcp_config(session_dir)
    click.echo(f"[Claude Code] MCP config: {mcp_config_path}")

    prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    cmd = [
        claude_bin,
        "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions",
        "-p", prompt,
        "--model", model,
        "--max-turns", "200",
        "--output-format", "stream-json",
        "--verbose",
        "--mcp-config", str(mcp_config_path),
    ]

    env = os.environ.copy()
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["IS_SANDBOX"] = "1"

    click.echo(f"[Claude Code] Starting orchestrator (model={model})")
    click.echo(f"[Claude Code] Log: {log_path}")
    click.echo(f"[Claude Code] Working dir: {os.path.dirname(os.path.abspath(session_dir))}")

    with open(log_path, "w") as log_file:
        log_file.write(f"=== Claude Code orchestrator started ===\n")
        log_file.write(f"Model: {model}\n")
        log_file.write(f"Session: {session_dir}\n")
        log_file.write(f"Prompt length: {len(prompt)} chars\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.path.dirname(os.path.abspath(session_dir)),
            text=True,
            bufsize=1,
        )

        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    event = json.loads(line_stripped)
                    etype = event.get("type", "")
                    if etype == "assistant" and "message" in event:
                        msg = event["message"]
                        if isinstance(msg, str) and msg.strip():
                            click.echo(f"[orch] {msg[:300]}")
                    elif etype == "result":
                        result_text = event.get("result", "")
                        if result_text:
                            click.echo(f"[orch] RESULT: {str(result_text)[:300]}")
                    elif etype == "tool_use":
                        tool_name = event.get("tool", event.get("name", ""))
                        click.echo(f"[orch] TOOL: {tool_name}")
                except json.JSONDecodeError:
                    if line_stripped:
                        click.echo(f"[orch] {line_stripped[:200]}")
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            proc.wait()
            log_file.write(f"\n=== Exited with code {proc.returncode} ===\n")

    click.echo(f"[Claude Code] Orchestrator finished (exit={proc.returncode})")


# ─── Main orchestrator loop (raw SDK) ────────────────────────────────────────


def run_sdk_orchestrator(
    session_dir: str,
    system_prompt: str,
    model: str = "claude-sonnet-4-6",
    max_turns: int = 10000,
    user_prompt: str = "Begin the optimization loop. Run baseline, then dispatch agents.",
) -> None:
    """Run the full SDK-based orchestrator agentic loop."""
    try:
        import anthropic
    except ImportError:
        click.echo("ERROR: anthropic package not installed", err=True)
        return

    log_dir = Path(session_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "orchestrator.log"

    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    turn = 0

    click.echo(f"[SDK] Starting agentic loop (model={model}, max_turns={max_turns})")
    click.echo(f"[SDK] Log: {log_path}")

    def _log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    _log(f"=== Orchestrator started at {datetime.now(timezone.utc).isoformat()} ===")
    _log(f"Model: {model}")
    _log(f"Session: {session_dir}")

    max_retries = 5

    while turn < max_turns:
        turn += 1
        _log(f"\n--- Turn {turn} ---")

        stop_file = Path(session_dir) / "STOP"
        if stop_file.exists():
            _log(f"Stopped by STOP file at turn {turn}")
            click.echo(f"[SDK] Stopped by STOP file after {turn} turns")
            break

        response = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=16384,
                    system=system_prompt,
                    messages=messages,
                    tools=ORCHESTRATOR_TOOLS,
                )
                break
            except Exception as e:
                wait = min(30 * (2 ** attempt), 300)
                _log(f"API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    click.echo(f"[SDK] API error, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    click.echo(f"[SDK] All retries exhausted: {e}")

        if response is None:
            _log("Giving up after max retries")
            break

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        has_tool_use = False
        tool_results = []

        for block in assistant_content:
            if block.type == "text":
                _log(f"[ASSISTANT] {block.text[:2000]}")
                if turn <= 5 or turn % 10 == 0:
                    click.echo(f"[SDK] Turn {turn}: {block.text[:200]}")
            elif block.type == "tool_use":
                has_tool_use = True
                _log(f"[TOOL] {block.name}: {str(block.input)[:500]}")

                result = execute_tool(block.name, block.input, session_dir)
                _log(f"[RESULT] {result[:1000]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result[:50000],
                })

        if has_tool_use:
            messages.append({"role": "user", "content": tool_results})
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Continue the optimization loop. Check agent status, "
                    "collect results, integrate patches, and dispatch new agents. "
                    "Do not stop until STOP file exists or time runs out."
                ),
            })

    _log(f"\n=== Session ended. Total turns: {turn} ===")
    click.echo(f"[SDK] Session ended. Total turns: {turn}. Log: {log_path}")
