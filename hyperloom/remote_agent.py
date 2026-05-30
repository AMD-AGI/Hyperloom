"""Standalone agent runner for remote SSH dispatch.

Runs a specialist agent using the Anthropic Python SDK with bash/read/write tools.
Designed to be invoked via SSH when `claude` CLI is not available on the remote node.

Usage:
    python3 -m hyperloom.remote_agent --prompt-file /path/to/prompt.md \
        --model claude-sonnet-4-6 --session-dir /path/to/session \
        --agent-id specialist-123-abc
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _execute_tool(tool_name: str, tool_input: dict, cwd: str) -> str:
    try:
        if tool_name == "bash":
            cmd = tool_input.get("command", "")
            timeout = tool_input.get("timeout", 600)
            r = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout, cwd=cwd,
            )
            output = r.stdout + r.stderr
            if r.returncode != 0:
                output += f"\n[exit code: {r.returncode}]"
            return output[:50000]
        elif tool_name == "read_file":
            p = Path(tool_input["path"])
            return p.read_text()[:100000] if p.exists() else f"File not found: {p}"
        elif tool_name == "write_file":
            p = Path(tool_input["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(tool_input["content"])
            return f"Wrote {len(tool_input['content'])} chars to {p}"
        else:
            return f"Unknown tool: {tool_name}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {tool_input.get('timeout', 600)}s"
    except Exception as e:
        return f"Tool error: {e}"


TOOLS = [
    {
        "name": "bash",
        "description": "Execute a bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 600},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


def run_agent(
    prompt: str,
    model: str,
    session_dir: str,
    agent_id: str,
    log_file: str | None = None,
    max_turns: int = 200,
) -> None:
    """Run a specialist agent with tool use via Anthropic SDK."""
    import anthropic

    client = anthropic.Anthropic()
    agent_dir = Path(session_dir) / "agents" / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(log_file) if log_file else agent_dir / "agent.log"

    def _log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    _log(f"=== Agent {agent_id} started at {datetime.now(timezone.utc).isoformat()} ===")

    messages: list[dict] = [{"role": "user", "content": prompt}]
    turn = 0

    heartbeat_path = agent_dir / "heartbeat.json"

    def _heartbeat(note: str = "") -> None:
        heartbeat_path.write_text(json.dumps({
            "agent_id": agent_id,
            "timestamp": time.time(),
            "status": "running",
            "progress_note": note,
            "iteration": turn,
        }))

    _heartbeat("starting")

    try:
        while turn < max_turns:
            turn += 1
            _heartbeat(f"turn {turn}")

            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=prompt,
                messages=messages,
                tools=TOOLS,
            )

            messages.append({"role": "assistant", "content": response.content})

            has_tool_use = False
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    _log(f"[turn {turn}] {block.text[:1000]}")
                elif block.type == "tool_use":
                    has_tool_use = True
                    result = _execute_tool(block.name, block.input, session_dir)
                    _log(f"[tool] {block.name}: {result[:500]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:50000],
                    })

            if has_tool_use:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        done_data = {
            "agent_id": agent_id,
            "status": "success",
            "summary": f"Completed after {turn} turns",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (agent_dir / "done.json").write_text(json.dumps(done_data, indent=2))
        _log(f"=== Agent completed successfully after {turn} turns ===")

    except Exception as e:
        _log(f"=== Agent failed: {e} ===")
        done_data = {
            "agent_id": agent_id,
            "status": "failed",
            "error": str(e),
            "summary": f"Failed at turn {turn}: {e}",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (agent_dir / "done.json").write_text(json.dumps(done_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Hyperloom remote agent runner")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt file")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text()
    run_agent(
        prompt=prompt,
        model=args.model,
        session_dir=args.session_dir,
        agent_id=args.agent_id,
        log_file=args.log_file,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
