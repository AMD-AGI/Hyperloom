#!/usr/bin/env python3
"""OOB GPU Optimizer REST API CLI client.

Replaces OOB MCP tool calls with direct REST API calls.
No external dependencies beyond Python stdlib.

The OOB server exposes two transports:
- MCP JSON-RPC at POST /  and GET /sse
- REST at POST /tools/{tool_name} with JSON body = MCP args

This CLI uses the REST transport.

Environment variables:
    OOB_API_URL   — OOB service base URL
                    Remote: https://oci-slc.primus-safe.amd.com/control-plane/
                            control-plane-sandbox/agent-mcp-server-zr29p
                    Local:  http://localhost:8003
    OOB_AUTH_KEY  — Bearer token (SaFE ak-xxx for remote; any value for local)

Usage:
    oob_client.py create-task --agent codex --file kernel.py --prompt "..." --workspace-id control-plane-moe
    oob_client.py submit-task TASK_ID
    oob_client.py get-task TASK_ID
    oob_client.py poll-task TASK_ID [--interval 10] [--timeout 600]
    oob_client.py get-outputs TASK_ID
    oob_client.py download-file TASK_ID FILE_PATH [--output-dir .]
    oob_client.py cancel-task TASK_ID
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _env(name: str, required: bool = True) -> str:
    val = os.environ.get(name, "")
    if required and not val:
        for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
            env_file = p / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        val = v.strip().strip('"').strip("'")
                        break
            if val:
                break
        if not val:
            print(f"ERROR: {name} not set. Export it or add to .env", file=sys.stderr)
            sys.exit(1)
    return val


def _api_url() -> str:
    return _env("OOB_API_URL").rstrip("/")


def _auth_key() -> str:
    return _env("OOB_AUTH_KEY")


def _call_tool(tool_name: str, args: dict) -> dict:
    """POST /tools/{tool_name} with JSON body."""
    url = f"{_api_url()}/tools/{tool_name}"
    headers = {
        "Authorization": f"Bearer {_auth_key()}",
        "Content-Type": "application/json",
    }
    data = json.dumps(args).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw.decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(err_body)
        except json.JSONDecodeError:
            detail = err_body
        print(json.dumps({"error": detail, "status_code": e.code}, indent=2), file=sys.stderr)
        sys.exit(1)


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_create_task(args):
    body: dict = {
        "workspace_id": args.workspace_id,
    }
    if args.agent:
        body["agent"] = args.agent
    if args.prompt:
        body["prompt"] = args.prompt
    if args.system_prompt:
        body["system_prompt"] = args.system_prompt
    if args.model:
        body["model"] = args.model
    if args.max_turns is not None:
        body["max_turns"] = args.max_turns
    if args.max_rounds is not None:
        body["max_rounds"] = args.max_rounds
    if args.convergence_threshold is not None:
        body["convergence_threshold"] = args.convergence_threshold
    if args.gpu_count is not None:
        body["gpu_count"] = args.gpu_count
    if args.cpu is not None:
        body["cpu"] = args.cpu
    if args.memory:
        body["memory"] = args.memory
    if args.ephemeral_storage:
        body["ephemeral_storage"] = args.ephemeral_storage
    if args.replicas is not None:
        body["replicas"] = args.replicas
    if args.rdma:
        body["rdma"] = args.rdma
    if args.timeout is not None:
        body["timeout"] = args.timeout
    if args.image:
        body["image"] = args.image

    files = []
    for fpath in (args.file or []):
        p = Path(fpath)
        if not p.exists():
            print(f"ERROR: file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        files.append({"filename": p.name, "content": p.read_text()})
    if files:
        body["files"] = files

    result = _call_tool("agent_create_task", body)
    _print(result)
    task_id = result.get("task_id") or result.get("id", "")
    if task_id:
        print(f"\nTask created: {task_id}", file=sys.stderr)
        print(f"Next: python3 {sys.argv[0]} submit-task {task_id}", file=sys.stderr)


def cmd_submit_task(args):
    result = _call_tool("agent_submit_task", {"task_id": args.task_id})
    _print(result)


def cmd_get_task(args):
    result = _call_tool("agent_get_task", {"task_id": args.task_id})
    _print(result)


def cmd_poll_task(args):
    """Poll task status until terminal state, printing updates."""
    interval = args.interval
    timeout = args.timeout
    start = time.time()
    prev_status = None

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"TIMEOUT after {timeout}s — task still not complete", file=sys.stderr)
            sys.exit(2)

        result = _call_tool("agent_get_task", {"task_id": args.task_id})
        status = result.get("status", "unknown")
        updated = result.get("updated_at", "")

        if status != prev_status:
            print(f"[{elapsed:.0f}s] Status: {status} (updated: {updated})", file=sys.stderr)
            prev_status = status

        if status in ("completed", "failed", "cancelled"):
            _print(result)
            if status == "completed":
                print(f"\nTask completed. Get outputs:", file=sys.stderr)
                print(f"  python3 {sys.argv[0]} get-outputs {args.task_id}", file=sys.stderr)
            elif status == "failed":
                print(f"\nTask failed: {result.get('error') or result.get('error_message', 'unknown')}", file=sys.stderr)
            return

        time.sleep(interval)


def cmd_cancel_task(args):
    result = _call_tool("agent_cancel_task", {"task_id": args.task_id})
    _print(result)


def cmd_get_outputs(args):
    result = _call_tool("agent_get_outputs", {"task_id": args.task_id})
    _print(result)
    files = result.get("files", [])
    if files:
        print(f"\nDownload files:", file=sys.stderr)
        for f in files[:10]:
            path = f.get("path") if isinstance(f, dict) else f
            print(f"  python3 {sys.argv[0]} download-file {args.task_id} \"{path}\"", file=sys.stderr)


def cmd_download_file(args):
    result = _call_tool("agent_download_file", {
        "task_id": args.task_id,
        "file_path": args.file_path,
    })

    content = result.get("content")
    if content is None:
        _print(result)
        return

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(args.file_path).name
        if isinstance(content, str):
            out_path.write_text(content)
        else:
            out_path.write_bytes(content)
        print(f"Saved: {out_path}", file=sys.stderr)
    else:
        if isinstance(content, str):
            print(content)
        else:
            print(f"Binary content ({len(content)} bytes). Use --output-dir to save.", file=sys.stderr)
            sys.exit(1)


# ─── Argument parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oob_client.py",
        description="OOB GPU Optimizer REST API CLI — replaces OOB MCP tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create-task
    p = sub.add_parser("create-task", help="Create a new kernel optimization task")
    p.add_argument("--workspace-id", required=True,
                   help="SaFE workspace ID (required), e.g. control-plane-moe")
    p.add_argument("--agent", choices=["claude", "codex"], default="claude",
                   help="Agent backend (default: claude)")
    p.add_argument("--prompt", help="Optimization instructions")
    p.add_argument("--system-prompt", help="Custom system prompt for the agent")
    p.add_argument("--model", help="LLM model (default: claude-opus-4-6 for claude, gpt-5.3-codex for codex)")
    p.add_argument("--file", action="append", help="Kernel source file(s) to include (repeatable)")
    p.add_argument("--max-turns", type=int, help="Max agent turns (default: 50)")
    p.add_argument("--max-rounds", type=int, help="Max optimization rounds (default: 5)")
    p.add_argument("--convergence-threshold", type=float, help="Stop if speedup improvement < threshold")
    p.add_argument("--gpu-count", type=int, help="Number of GPUs (default: 1)")
    p.add_argument("--cpu", type=int, help="CPU cores (default: 4)")
    p.add_argument("--memory", help="Memory limit, e.g. 16Gi (default: 16Gi)")
    p.add_argument("--ephemeral-storage", help="Ephemeral storage, e.g. 50Gi")
    p.add_argument("--replicas", type=int, help="Replicas (default: 1)")
    p.add_argument("--rdma", help="RDMA resource, e.g. 1k")
    p.add_argument("--timeout", type=int, help="Max execution time in seconds (default: 1800)")
    p.add_argument("--image", help="Docker image override")
    p.set_defaults(func=cmd_create_task)

    # submit-task
    p = sub.add_parser("submit-task", help="Submit a pending task for execution")
    p.add_argument("task_id", help="Task ID")
    p.set_defaults(func=cmd_submit_task)

    # get-task
    p = sub.add_parser("get-task", help="Get task details and status")
    p.add_argument("task_id", help="Task ID")
    p.set_defaults(func=cmd_get_task)

    # poll-task
    p = sub.add_parser("poll-task", help="Poll task until completion")
    p.add_argument("task_id", help="Task ID")
    p.add_argument("--interval", type=int, default=10, help="Poll interval in seconds (default: 10)")
    p.add_argument("--timeout", type=int, default=600, help="Max wait in seconds (default: 600)")
    p.set_defaults(func=cmd_poll_task)

    # cancel-task
    p = sub.add_parser("cancel-task", help="Cancel a pending or running task")
    p.add_argument("task_id", help="Task ID")
    p.set_defaults(func=cmd_cancel_task)

    # get-outputs
    p = sub.add_parser("get-outputs", help="List output files from a completed task")
    p.add_argument("task_id", help="Task ID")
    p.set_defaults(func=cmd_get_outputs)

    # download-file
    p = sub.add_parser("download-file", help="Download a file from task outputs")
    p.add_argument("task_id", help="Task ID")
    p.add_argument("file_path", help="File path within task outputs")
    p.add_argument("--output-dir", help="Directory to save file (prints to stdout if omitted)")
    p.set_defaults(func=cmd_download_file)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
