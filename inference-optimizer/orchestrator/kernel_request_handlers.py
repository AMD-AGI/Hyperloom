"""Conductor-side handlers for Kernel-agent REQUEST kinds.

DESIGN v0.6 §7.2 says the Kernel agent is responder-only: it answers
Orchestration's ``request{target_agent='kernel', kind=...}`` with a
``response`` intent. In v0.6 this can happen two ways:

  1. **LLM responder** — the request is mirrored into Kernel's inbox; on
     the next reactor pass the Kernel LLM emits a ``response`` intent.
  2. **Programmatic handler** — the Conductor recognises the ``kind`` and
     runs a deterministic Python callable instead of routing to the LLM.

P2-2 wires (2) for the well-defined kinds that map onto the existing
Hyperloom/kernel-agent shell tools (`tracelens_analysis.py` /
`kernel_optimization.py`). The LLM still handles unknown kinds (or any
kind we explicitly want LLM judgment on).

Why programmatic for these kinds:

* tracelens / GEAK runs are heavy multi-second-to-multi-minute shell
  workflows; routing them through an extra Codex/Claude turn just to
  spawn the subprocess wastes a turn and an LLM call.
* The Hyperloom/kernel-agent tools already encapsulate the full
  protocol (input validation, retry, structured JSON output).
* Result determinism: RESPONSE payload comes straight from the tool's
  JSON output — easier to write tests against, easier to debug.

Handler signature::

    async def handler(payload: dict, *, session_dir: Path) -> dict:
        # returns the dict that becomes RESPONSE.payload['result']

Dispatch table is exposed via :data:`KERNEL_REQUEST_HANDLERS` so callers
can monkey-patch in tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


# Where Hyperloom/kernel-agent's shell tools live. Override-able for tests.
HYPERLOOM_KERNEL_AGENT_ROOT = Path(
    os.environ.get(
        "HYPERLOOM_KERNEL_AGENT_ROOT",
        "/wekafs/xiaofei/Hyperloom/kernel-agent",
    )
)


HandlerResult = dict[str, Any]
HandlerFn = Callable[..., Awaitable[HandlerResult]]


# ---------------------------------------------------------------------------
async def _run_subprocess(cmd: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """asyncio-friendly wrapper around blocking subprocess.run.

    Conductor reactor stays responsive while the shell tool runs.
    """
    def _run() -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, env=env,
        )

    proc = await asyncio.to_thread(_run)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
async def select_kernels_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir.

    Required payload:
        trace_input: path to a torch_trace dir or single .trace.json.gz file
                     (typically from a previous profile_executor result)

    Optional payload:
        top_k:           default 10
        model_name:      default ''
        framework:       default 'sglang'
        target_platform: default 'MI300X'
        dry_run:         default False (testing)
        budget_minutes:  default 5

    Returns::

        {
          "status": "ok" | "failed",
          "hot_kernels": [...],
          "trace_report_path": "...",
          "cli_log_path": "...",
          "details": {...},  # raw tool output
        }
    """
    trace_input = payload.get("trace_input") or payload.get("trace_dir")
    if not trace_input:
        return {"status": "failed", "error": "missing 'trace_input' in payload"}

    workspace_path = (
        payload.get("workspace_path")
        or str(session_dir / "kernel-agent-workspace")
    )
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(HYPERLOOM_KERNEL_AGENT_ROOT / "tools" / "tracelens_analysis.py"),
        "--trace-input", str(trace_input),
        "--session-id", str(payload.get("session_id") or session_dir.name),
        "--top-k", str(payload.get("top_k", 10)),
        "--workspace-path", workspace_path,
    ]
    if payload.get("model_name"):
        cmd += ["--model-name", str(payload["model_name"])]
    if payload.get("framework"):
        cmd += ["--framework", str(payload["framework"])]
    if payload.get("target_platform"):
        cmd += ["--target-platform", str(payload["target_platform"])]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 5)) * 60

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    return _shape_tool_result(rc, stdout, stderr)


# ---------------------------------------------------------------------------
async def run_optimization_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's kernel_optimization.py on one kernel.

    Required payload:
        kernel_id: str

    Optional payload:
        backends:        comma-separated 'geak,claude,codex' (auto-pick if empty)
        budget_minutes:  default 30
        source_file:     path to original kernel source (for context)
        candidates_path: path to JSON describing candidates (optional)
        dry_run:         default False (testing)

    Returns the tool's JSON output verbatim under ``result``.
    """
    kernel_id = payload.get("kernel_id")
    if not kernel_id:
        return {"status": "failed", "error": "missing 'kernel_id' in payload"}

    workspace_path = (
        payload.get("workspace_path")
        or str(session_dir / "kernel-agent-workspace")
    )
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(HYPERLOOM_KERNEL_AGENT_ROOT / "tools" / "kernel_optimization.py"),
        "--kernel-id", str(kernel_id),
        "--session-id", str(payload.get("session_id") or session_dir.name),
        "--workspace-path", workspace_path,
    ]
    if payload.get("backends"):
        cmd += ["--backends", str(payload["backends"])]
    if payload.get("source_file"):
        cmd += ["--source-file", str(payload["source_file"])]
    if payload.get("candidates_path"):
        cmd += ["--candidates-path", str(payload["candidates_path"])]
    if payload.get("benchmark_file"):
        cmd += ["--benchmark-file", str(payload["benchmark_file"])]
    if payload.get("test_harness_path"):
        cmd += ["--test-harness-path", str(payload["test_harness_path"])]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 30)) * 60

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    return _shape_tool_result(rc, stdout, stderr)


# ---------------------------------------------------------------------------
def _shape_tool_result(rc: int, stdout: str, stderr: str) -> HandlerResult:
    """Wrap a Hyperloom kernel-agent tool's exit + stdout into our schema.

    The tools always print a single JSON object on stdout, even on
    failure (with `"status": "failed"` and a diagnostic field). Prefer
    that structured payload; fall back to a synthesized one only when
    stdout couldn't be parsed.
    """
    parsed = _parse_tool_stdout(stdout)
    if parsed:
        # Trust the tool's own status if it set one; otherwise infer
        # from rc.
        if "status" not in parsed:
            parsed["status"] = "ok" if rc == 0 else "failed"
        if rc != 0:
            parsed.setdefault("returncode", rc)
            if stderr.strip():
                parsed.setdefault("stderr_tail", stderr[-2000:])
        return parsed
    return {
        "status": "failed" if rc != 0 else "ok",
        "returncode": rc,
        "error": (stderr or stdout)[-2000:],
    }


def _parse_tool_stdout(stdout: str) -> dict[str, Any]:
    """Tool stdout SHOULD be a single JSON object; survive other shapes."""
    text = stdout.strip()
    if not text:
        return {}
    # Try whole stdout as JSON first.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    # Fall back: scan for the last JSON object on its own line.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return {"raw_stdout_tail": text[-2000:]}


# ---------------------------------------------------------------------------
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "select_kernels":   select_kernels_handler,
    "run_optimization": run_optimization_handler,
    # apply_patch / integrate are P2-4 — left as LLM passthrough today
}


def has_handler(kind: str) -> bool:
    return kind in KERNEL_REQUEST_HANDLERS


def get_handler(kind: str) -> HandlerFn | None:
    return KERNEL_REQUEST_HANDLERS.get(kind)


__all__ = [
    "HYPERLOOM_KERNEL_AGENT_ROOT",
    "KERNEL_REQUEST_HANDLERS",
    "get_handler",
    "has_handler",
    "run_optimization_handler",
    "select_kernels_handler",
]
