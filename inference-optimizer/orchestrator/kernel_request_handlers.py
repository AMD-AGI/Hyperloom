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
async def integrate_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Mirrors DESIGN v0.6 §16 integrate action. The full real-world flow
    is: ``patch_inductor.py --target-file <kernel.py> --patch <new.py>``
    → restart sglang with the patched cache → baseline → compare. P2-4
    keeps the **patch step as a stub** (we record patch metadata and
    skip the apply) but does run a real re-baseline via
    :class:`BaselineExecutor`, so the KEEP/REVERT signal is honest
    against whatever the patch actually changed (or didn't, in stub mode).

    Required payload:
        base_tput:    float — what we're comparing against

    Optional payload:
        patch_path:        path to the rewritten kernel file
        target_file:       inductor cache file to patch (informational)
        kernel_id:         label used in result + bus events
        config_path:       Magpie YAML for the re-baseline run
        extra_sglang_args: extra flags layered onto the Magpie envs
        keep_threshold_pct: KEEP if gain > X% (default 1.0)
        budget_minutes:    re-baseline timeout (default 20)

    Returns::

        {
          "status":  "ok" | "failed",
          "decision": "KEEP" | "REVERT" | "NEEDS_REVIEW",
          "base_tput":   float,
          "new_tput":    float,
          "gain_pct":    float,
          "kernel_id":   str | None,
          "patch_path":  str | None,
          "report_path": str (re-baseline benchmark_report.json),
          "workspace":   str (re-baseline magpie workspace),
        }
    """
    from .action_executors.baseline import BaselineExecutor
    from .sub_agent_runner import ExecutorContext
    from .task_registry import Task

    base_tput = float(payload.get("base_tput", 0.0))
    if base_tput <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }

    # NOTE: real patch apply (patch_inductor.py + sglang restart) lives
    # in P3+. Here we just log the apply intent so the bus has a paper
    # trail, then run re-baseline.
    patch_path = payload.get("patch_path")
    kernel_id = payload.get("kernel_id")
    log.info(
        "integrate_handler: stub-applying patch_path=%s kernel_id=%s "
        "(real patch_inductor invocation deferred to P3)",
        patch_path, kernel_id,
    )

    keep_threshold_pct = float(payload.get("keep_threshold_pct", 1.0))
    extra_args = str(payload.get("extra_sglang_args", "") or "").strip()

    # Build a Task wrapper around BaselineExecutor (which expects an
    # ExecutorContext with a Task in it). The "extra_sglang_args" hand-
    # off goes via the task params even though baseline_executor doesn't
    # use them yet — kept for forward compat (P3 will inject EXTRA_SGLANG_ARGS).
    workspace = session_dir / "integrate" / (kernel_id or "anon")
    workspace.mkdir(parents=True, exist_ok=True)
    fake_task = Task(
        task_id=f"integrate-{kernel_id or 'anon'}",
        kind="baseline",
        state="running",
        params={
            "config_path": payload.get("config_path"),
            "output_dir":  str(workspace),
            "timeout_sec": int(payload.get("budget_minutes", 20)) * 60,
            "extra_sglang_args": extra_args,
        },
        idempotency_key=f"integrate-{kernel_id or 'anon'}-rebaseline",
    )
    ctx = ExecutorContext(task=fake_task, lease=None)

    try:
        bench_result = await BaselineExecutor()(ctx)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_class": "rebaseline_exception",
            "error": repr(exc),
            "kernel_id": kernel_id,
            "patch_path": patch_path,
        }

    if bench_result.get("status") != "succeeded":
        return {
            "status": "failed",
            "error": "re-baseline did not succeed",
            "decision": "REVERT",
            "rebaseline_detail": bench_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
        }

    new_tput = float(bench_result.get("output_throughput") or 0.0)
    gain_pct = ((new_tput - base_tput) / base_tput * 100.0) if base_tput > 0 else 0.0
    decision = (
        "KEEP" if gain_pct > keep_threshold_pct
        else ("REVERT" if gain_pct < -keep_threshold_pct
              else "NEEDS_REVIEW")
    )
    return {
        "status":      "ok",
        "decision":    decision,
        "kernel_id":   kernel_id,
        "patch_path":  patch_path,
        "base_tput":   base_tput,
        "new_tput":    new_tput,
        "gain_pct":    gain_pct,
        "report_path": bench_result.get("report_path"),
        "workspace":   bench_result.get("workspace"),
    }


# ---------------------------------------------------------------------------
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "select_kernels":   select_kernels_handler,
    "run_optimization": run_optimization_handler,
    "integrate":        integrate_handler,
    "apply_patch":      integrate_handler,   # alias — same flow
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
    "integrate_handler",
    "run_optimization_handler",
    "select_kernels_handler",
]
