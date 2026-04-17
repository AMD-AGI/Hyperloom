#!/usr/bin/env python3
"""GEAK REST API CLI client.

Replaces GEAK MCP tool calls with direct REST API calls.
No external dependencies beyond Python stdlib.

Environment variables:
    GEAK_API_URL   — GEAK service base URL (e.g. https://host/control-plane/.../geak-agent-wvsbv)
    GEAK_AUTH_KEY  — Bearer token for authentication (ak-xxx)

Usage:
    python3 geak_client.py create-task --input-type file --file kernel.py --prompt "Optimize for MI355X" --step-limit 100
    python3 geak_client.py submit-task TASK_ID
    python3 geak_client.py poll-task TASK_ID --interval 30 --timeout 1800
    python3 geak_client.py get-outputs TASK_ID
    python3 geak_client.py download-file TASK_ID FILE_PATH --output-dir ./out
    python3 geak_client.py list-tasks --status running --limit 10
    python3 geak_client.py cancel-task TASK_ID
    python3 geak_client.py get-model-config
    python3 geak_client.py set-model-config --model-class litellm --model-name "openai/claude-opus-4-6" --model-kwargs '{"api_base":"...","api_key":"..."}'
    python3 geak_client.py delete-model-config
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _env(name: str, required: bool = True) -> str:
    val = os.environ.get(name, "")
    if required and not val:
        # Try loading from .env in current or parent dirs
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
    return _env("GEAK_API_URL").rstrip("/")


def _auth_key() -> str:
    return _env("GEAK_AUTH_KEY")


def _request(method: str, path: str, body: dict | None = None, stream: bool = False) -> dict | bytes:
    url = f"{_api_url()}{path}"
    headers = {
        "Authorization": f"Bearer {_auth_key()}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            if stream:
                return raw
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return json.loads(raw)
            # Text fallback
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
    body: dict = {"input_type": args.input_type}

    if args.input_type == "file":
        files = []
        for fpath in (args.file or []):
            p = Path(fpath)
            if not p.exists():
                print(f"ERROR: file not found: {fpath}", file=sys.stderr)
                sys.exit(1)
            files.append({"filename": p.name, "content": p.read_text()})
        if not files:
            print("ERROR: --file required for input_type=file", file=sys.stderr)
            sys.exit(1)
        body["files"] = files
    elif args.input_type == "repo":
        if not args.repo_url:
            print("ERROR: --repo-url required for input_type=repo", file=sys.stderr)
            sys.exit(1)
        repo = {"url": args.repo_url}
        if args.repo_branch:
            repo["branch"] = args.repo_branch
        body["repo"] = repo

    if args.prompt:
        body["prompt"] = args.prompt
    if args.workspace_id:
        body["workspace_id"] = args.workspace_id

    config: dict = {}
    if args.step_limit:
        config["agent"] = {"step_limit": args.step_limit}
    if config:
        body["config"] = config

    runtime: dict = {}
    if args.gpu_count is not None:
        runtime["gpu_count"] = args.gpu_count
    if args.image:
        runtime["image"] = args.image
    if runtime:
        body["runtime"] = runtime

    result = _request("POST", "/api/v1/tasks", body)
    _print(result)
    task_id = result.get("id", "")
    if task_id:
        print(f"\nTask created: {task_id}", file=sys.stderr)
        print(f"Next: python3 {sys.argv[0]} submit-task {task_id}", file=sys.stderr)


def cmd_submit_task(args):
    result = _request("POST", f"/api/v1/tasks/{args.task_id}/submit")
    _print(result)


def cmd_get_task(args):
    result = _request("GET", f"/api/v1/tasks/{args.task_id}")
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

        result = _request("GET", f"/api/v1/tasks/{args.task_id}")
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
                print(f"\nTask failed: {result.get('error_message', 'unknown')}", file=sys.stderr)
            return

        time.sleep(interval)


def cmd_list_tasks(args):
    params = []
    if args.status:
        params.append(f"status={args.status}")
    if args.limit:
        params.append(f"limit={args.limit}")
    qs = f"?{'&'.join(params)}" if params else ""
    result = _request("GET", f"/api/v1/tasks{qs}")
    _print(result)


def cmd_cancel_task(args):
    result = _request("POST", f"/api/v1/tasks/{args.task_id}/cancel")
    _print(result)


def cmd_get_outputs(args):
    result = _request("GET", f"/api/v1/tasks/{args.task_id}/outputs")
    _print(result)
    files = result.get("files", [])
    if files:
        print(f"\nDownload files:", file=sys.stderr)
        for f in files:
            print(f"  python3 {sys.argv[0]} download-file {args.task_id} \"{f['path']}\"", file=sys.stderr)


def cmd_download_file(args):
    path_encoded = urllib.parse.quote(args.file_path, safe="")
    raw = _request("GET", f"/api/v1/tasks/{args.task_id}/download?path={path_encoded}", stream=True)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(args.file_path).name
        out_path.write_bytes(raw)
        print(f"Saved: {out_path}", file=sys.stderr)
    else:
        # Print text content to stdout
        try:
            text = raw.decode("utf-8")
            print(text)
        except UnicodeDecodeError:
            print(f"Binary file ({len(raw)} bytes). Use --output-dir to save.", file=sys.stderr)
            sys.exit(1)


def cmd_get_model_config(args):
    result = _request("GET", "/api/v1/config/model")
    _print(result)


def cmd_set_model_config(args):
    body = {
        "model_class": args.model_class,
        "model_name": args.model_name,
        "model_kwargs": json.loads(args.model_kwargs) if args.model_kwargs else {},
    }
    result = _request("PUT", "/api/v1/config/model", body)
    _print(result)


def cmd_delete_model_config(args):
    result = _request("DELETE", "/api/v1/config/model")
    if isinstance(result, dict) and result.get("success"):
        print("Model config deleted.", file=sys.stderr)
    else:
        _print(result)


# ─── Argument parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geak_client.py",
        description="GEAK REST API CLI — replaces GEAK MCP tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create-task
    p = sub.add_parser("create-task", help="Create a new optimization task")
    p.add_argument("--input-type", required=True, choices=["file", "repo"])
    p.add_argument("--file", action="append", help="Kernel source file(s) to include (repeatable)")
    p.add_argument("--repo-url", help="Git repo URL (for input_type=repo)")
    p.add_argument("--repo-branch", help="Git branch")
    p.add_argument("--prompt", help="Optimization instructions")
    p.add_argument("--step-limit", type=int, help="Max agent steps (recommend 100)")
    p.add_argument("--gpu-count", type=int, help="Number of GPUs")
    p.add_argument("--image", help="Docker image")
    p.add_argument("--workspace-id", help="SaFE workspace ID (default: control-plane-moe)")
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
    p.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    p.add_argument("--timeout", type=int, default=1800, help="Max wait in seconds (default: 1800)")
    p.set_defaults(func=cmd_poll_task)

    # list-tasks
    p = sub.add_parser("list-tasks", help="List tasks")
    p.add_argument("--status", choices=["pending", "running", "completed", "failed", "cancelled"])
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_list_tasks)

    # cancel-task
    p = sub.add_parser("cancel-task", help="Cancel a running task")
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

    # get-model-config
    p = sub.add_parser("get-model-config", help="Get user's default model configuration")
    p.set_defaults(func=cmd_get_model_config)

    # set-model-config
    p = sub.add_parser("set-model-config", help="Set user's default model configuration")
    p.add_argument("--model-class", required=True, help="Model class (e.g. litellm)")
    p.add_argument("--model-name", required=True, help="Model name (e.g. openai/claude-opus-4-6)")
    p.add_argument("--model-kwargs", help="JSON string of model params")
    p.set_defaults(func=cmd_set_model_config)

    # delete-model-config
    p = sub.add_parser("delete-model-config", help="Delete user's default model configuration")
    p.set_defaults(func=cmd_delete_model_config)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
