"""Coordinator-side handlers for Kernel-agent REQUEST kinds.

DESIGN v0.6 §7.2 says the Kernel agent is responder-only: it answers
Orchestration's ``request{target_agent='kernel', kind=...}`` with a
``response`` intent. In v0.6 this can happen two ways:

  1. **LLM responder** — the request is mirrored into Kernel's inbox; on
     the next reactor pass the Kernel LLM emits a ``response`` intent.
  2. **Programmatic handler** — the Coordinator recognises the ``kind`` and
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
import importlib.util
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


# Where Hyperloom/kernel-agent's shell tools live. Env is set by
# inference_optimizer/scripts/install.sh -> pod-local kernel-agent env.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"
HYPERLOOM_KERNEL_AGENT_ROOT = (
    Path(os.environ[_KERNEL_AGENT_ROOT_ENV])
    if os.environ.get(_KERNEL_AGENT_ROOT_ENV)
    else None
)


HandlerResult = dict[str, Any]
HandlerFn = Callable[..., Awaitable[HandlerResult]]

_RUNTIME_GENERATED_SOURCE_MARKERS = (
    "/tmp/torchinductor",
    "/torchinductor_",
    "/.cache/torch/inductor",
    "/.triton/cache",
    "/triton/cache",
)
_COMPILE_GENERATED_NAME_MARKERS = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "torchinductor",
    "inductor",
)
_REUSABLE_SOURCE_ROOTS = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
)
_APPLY_TOOL_MODULE: Any | None = None
_DEFAULT_KERNEL_BACKEND_ORDER = ("claude", "codex", "geak")
_DEFAULT_KERNEL_BATCH_PARALLEL = 3


def _kernel_agent_root_error() -> str | None:
    if HYPERLOOM_KERNEL_AGENT_ROOT is None:
        return (
            f"{_KERNEL_AGENT_ROOT_ENV} is not set; run "
            "inference_optimizer/scripts/install.sh and source /workspace/hyperloom/runtime/kernel-agent.env.sh"
        )
    if not HYPERLOOM_KERNEL_AGENT_ROOT.is_dir():
        return f"{_KERNEL_AGENT_ROOT_ENV} does not exist: {HYPERLOOM_KERNEL_AGENT_ROOT}"
    return None


def _kernel_agent_tool_path(tool_name: str) -> Path:
    err = _kernel_agent_root_error()
    if err:
        raise RuntimeError(err)
    assert HYPERLOOM_KERNEL_AGENT_ROOT is not None
    path = HYPERLOOM_KERNEL_AGENT_ROOT / "tools" / tool_name
    if not path.is_file():
        raise RuntimeError(f"kernel-agent tool not found: {path}")
    return path


def _is_runtime_generated_kernel(name: str, source_file: str) -> bool:
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        return not any(root in lower_file for root in _REUSABLE_SOURCE_ROOTS)
    return False


def _load_candidate_metadata(payload: dict) -> dict[str, Any]:
    """Find candidate metadata for the requested kernel_id if available."""
    if isinstance(payload.get("candidate"), dict):
        return payload["candidate"]
    candidates_path = payload.get("candidates_path")
    kernel_id = str(payload.get("kernel_id") or "")
    if not candidates_path or not kernel_id:
        return {}
    try:
        data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    kernels = data.get("hot_kernels") if isinstance(data, dict) else None
    if not isinstance(kernels, list):
        return {}
    for item in kernels:
        if not isinstance(item, dict):
            continue
        if str(item.get("kernel_id") or "") == kernel_id:
            return item
    return {}


def _validate_reusable_native_kernel(payload: dict) -> HandlerResult | None:
    """Reject compile-generated or otherwise non-reusable kernel targets."""
    candidate = _load_candidate_metadata(payload)
    kernel_id = str(payload.get("kernel_id") or "")
    name = str(candidate.get("name") or payload.get("kernel_name") or kernel_id)
    source_file = str(
        payload.get("source_file")
        or candidate.get("source_file")
        or ""
    )
    reusable = candidate.get("reusable_native_kernel")
    if reusable is False:
        return {
            "status": "failed",
            "error_class": "non_reusable_kernel",
            "error": "kernel-opt only accepts reusable native kernel sources",
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
            "reason": candidate.get("optimization_notes")
                      or "candidate marked reusable_native_kernel=false",
        }
    if not source_file:
        return {
            "status": "failed",
            "error_class": "missing_native_source",
            "error": "kernel-opt requires a resolved stable source_file",
            "kernel_id": kernel_id,
            "kernel_name": name,
        }
    if _is_runtime_generated_kernel(name, source_file):
        return {
            "status": "failed",
            "error_class": "runtime_generated_kernel",
            "error": (
                "refusing to optimize torch.compile/Inductor runtime-generated "
                "kernel; result would not be reusable"
            ),
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
        }
    lower_file = source_file.lower()
    if not any(root in lower_file for root in _REUSABLE_SOURCE_ROOTS):
        return {
            "status": "failed",
            "error_class": "unstable_source_path",
            "error": "source_file is not under a known reusable framework source root",
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
        }
    payload.setdefault("source_file", source_file)
    return None


def _load_apply_tool() -> Any:
    global _APPLY_TOOL_MODULE
    if _APPLY_TOOL_MODULE is not None:
        return _APPLY_TOOL_MODULE
    path = _kernel_agent_tool_path("apply_kernel_patch.py")
    spec = importlib.util.spec_from_file_location("hyperloom_apply_kernel_patch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load apply_kernel_patch.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _APPLY_TOOL_MODULE = module
    return module


def _artifact_paths_from_payload(payload: dict) -> list[str]:
    raw = payload.get("artifact_paths") or payload.get("compiled_artifact_paths") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _maybe_apply_kernel_patch(
    payload: dict,
    *,
    session_dir: Path,
    kernel_id: str | None,
) -> HandlerResult:
    patch_path = str(payload.get("patch_path") or "").strip()
    target_file = str(
        payload.get("target_file")
        or payload.get("source_file")
        or ""
    ).strip()
    if not patch_path or not target_file:
        return {
            "status": "skipped",
            "reason": "missing patch_path or target_file/source_file",
        }
    from ..session_paths import patches_dir
    kid = str(kernel_id or payload.get("kernel_id") or "")
    backup_root = payload.get("backup_root") or (
        patches_dir(session_dir, kid or "anon") / "backup"
    )
    tool = _load_apply_tool()
    return tool.apply_kernel_patch(
        patch_path=patch_path,
        target_file=target_file,
        backup_root=backup_root,
        kernel_id=kid,
        artifact_paths=_artifact_paths_from_payload(payload),
        rebuild_command=payload.get("rebuild_command"),
        rebuild_timeout_sec=int(payload.get("rebuild_timeout_sec", 1800)),
        skip_rebuild=bool(payload.get("skip_rebuild", False)),
        allow_unknown_target=bool(payload.get("allow_unknown_target", False)),
        dry_run=bool(payload.get("dry_run_patch", False)),
    )


def _maybe_revert_kernel_patch(apply_result: HandlerResult) -> HandlerResult:
    if apply_result.get("status") != "ok" or not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    tool = _load_apply_tool()
    return tool.revert_kernel_patch(apply_result["manifest_path"])


def _find_selected_kernel_source(state: Any, kernel_id: str) -> str:
    kernels = (
        (state.last_select_kernels or {}).get("hot_kernels_top15")
        or (state.last_select_kernels or {}).get("hot_kernels")
        or []
    )
    for item in kernels:
        if not isinstance(item, dict):
            continue
        if str(item.get("kernel_id") or "") == kernel_id:
            return str(item.get("source_file") or "")
    return ""


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id.

    Orchestration often knows only ``kernel_id`` after a successful
    ``run_optimization``. The concrete artifact path lives in
    ``last_kernel_opt`` and the source target lives in ``last_select_kernels``.
    Resolve them here so integrate applies the optimized source before
    re-baselining; never silently run an E2E benchmark without applying a patch.
    """
    from .shared_state import SharedState

    resolved = dict(payload)
    kernel_id = str(resolved.get("kernel_id") or "")
    state = SharedState.load_or_init(session_dir)
    last_kernel = state.last_kernel_opt or {}

    if kernel_id and str(last_kernel.get("kernel_id") or "") == kernel_id:
        if not resolved.get("patch_path"):
            artifact = (
                last_kernel.get("best_artifact_path")
                or last_kernel.get("patch_path")
                or last_kernel.get("optimized_path")
            )
            if artifact:
                resolved["patch_path"] = str(artifact)
        if not resolved.get("source_file") and last_kernel.get("source_file"):
            resolved["source_file"] = str(last_kernel["source_file"])

    if kernel_id and not (resolved.get("target_file") or resolved.get("source_file")):
        source = _find_selected_kernel_source(state, kernel_id)
        if source:
            resolved["source_file"] = source

    patch_path = str(resolved.get("patch_path") or "").strip()
    target_file = str(
        resolved.get("target_file")
        or resolved.get("source_file")
        or ""
    ).strip()
    if not patch_path or not target_file:
        missing = []
        if not patch_path:
            missing.append("patch_path")
        if not target_file:
            missing.append("target_file/source_file")
        return resolved, {
            "status": "failed",
            "error_class": "missing_integration_inputs",
            "error": "integrate requires an optimized artifact and target source before E2E",
            "decision": "REVERT",
            "kernel_id": kernel_id or None,
            "patch_path": patch_path or None,
            "target_file": target_file or None,
            "missing": missing,
            "last_kernel_opt": {
                k: last_kernel.get(k)
                for k in ("kernel_id", "best_artifact_path", "patch_path", "source_file")
                if k in last_kernel
            },
        }
    return resolved, None


# ---------------------------------------------------------------------------
async def _run_subprocess(cmd: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """asyncio-friendly wrapper around blocking subprocess.run.

    Coordinator reactor stays responsive while the shell tool runs.
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
        target_platform: defaults to payload target_platform, then SharedState.gpu_type
        roofline_json:   optional path from a separate pmc_roofline action
        dry_run:         default False (testing)
        budget_minutes:  default 60

    Returns::

        {
          "status": "ok" | "failed",
          "hot_kernels": [...],
          "trace_report_path": "...",        # tracelens_report.json (structured)
          "analysis_report_path": "...",      # analysis.md from TraceLens v0.3
                                              #   orchestrator (markdown final
                                              #   stakeholder report — pass to
                                              #   GEAK so it can ground its
                                              #   actions on the same Detailed
                                              #   Analysis prose Hyperloom
                                              #   parsed for hot_kernels[]).
          "cli_log_path": "...",
          "details": {...},  # raw tool output
        }
    """
    trace_input = payload.get("trace_input") or payload.get("trace_dir")
    if not trace_input:
        return {"status": "failed", "error": "missing 'trace_input' in payload"}
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    workspace_path = (
        payload.get("workspace_path")
        or str(session_dir / "kernel-agent-workspace")
    )
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    # Backfill workload context from SharedState so downstream
    # tracelens_analysis.py / TraceLens skill receive the correct
    # framework / platform / model / analysis_mode instead of defaulting to
    # "" / MI355X / "default" when Orchestration omits them in the payload.
    from .shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    framework = (payload.get("framework") or state.framework or "").strip()
    target_platform = (
        payload.get("target_platform") or state.gpu_type or ""
    ).strip()
    model_name = (
        payload.get("model_name")
        or state.model_name
        or state.model_path
        or ""
    ).strip()
    analysis_mode = (payload.get("analysis_mode") or "").strip()
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("tracelens_analysis.py")),
        "--trace-input", str(trace_input),
        "--session-id", str(payload.get("session_id") or session_dir.name),
        "--top-k", str(payload.get("top_k", 10)),
        "--workspace-path", workspace_path,
    ]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if framework:
        cmd += ["--framework", str(framework)]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if analysis_mode:
        cmd += ["--analysis-mode", str(analysis_mode)]
    capture_folder = (
        payload.get("capture_folder")
        or payload.get("graph_capture_path")
        or payload.get("capture_folder_path")
    )
    if capture_folder:
        cmd += ["--capture-folder", str(capture_folder)]
    if payload.get("roofline_json"):
        cmd += ["--roofline-json", str(payload["roofline_json"])]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    result = _shape_tool_result(rc, stdout, stderr)
    artifacts = result.get("artifact_paths") if isinstance(result, dict) else None
    if isinstance(artifacts, dict) and artifacts.get("kernel_candidates"):
        result["candidates_path"] = artifacts["kernel_candidates"]
    # Surface analysis.md path at the handler boundary so the Coordinator can
    # forward it to GEAK without having to dig through artifact_paths.
    if isinstance(result, dict):
        report_path = result.get("analysis_report_path")
        if not report_path and isinstance(artifacts, dict):
            report_path = (
                artifacts.get("tracelens_agent_report")
                or artifacts.get("trace_report_path")
            )
        if report_path:
            result["analysis_report_path"] = str(report_path)
        # PR-A §3: surface tracelens/summary.json — the per-run audit
        # sidecar listing reusable tasks vs skipped kernels with reasons.
        # Coordinator / SharedState can show this in the prompt summary
        # so operators see at a glance whether GEAK was offered the
        # kernels they expected.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
    return result


# ---------------------------------------------------------------------------
async def run_optimization_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run kernel optimization.

    When candidate metadata is available, this handler upgrades legacy
    single-kernel requests into a batch over all reusable native kernels. Each
    kernel is optimized concurrently, while backends are tried sequentially per
    kernel in the preferred order: Claude first, then Codex, then GEAK.
    """
    if payload.get("_single_kernel"):
        return await _run_optimization_single(payload, session_dir=session_dir)
    candidates = _batch_kernel_candidates(payload)
    if len(candidates) <= 1:
        single_payload = dict(payload)
        if candidates and not single_payload.get("kernel_id"):
            single_payload["kernel_id"] = candidates[0].get("kernel_id")
        single_payload["_single_kernel"] = True
        return await _run_optimization_single(single_payload, session_dir=session_dir)
    return await _run_optimization_batch(payload, candidates, session_dir=session_dir)


def _backend_order(payload: dict) -> list[str]:
    raw = payload.get("backend_order") or os.environ.get("KERNEL_OPT_BACKEND_ORDER")
    if raw:
        order = [item.strip() for item in str(raw).split(",") if item.strip()]
    else:
        # Ignore legacy payload["backends"] here. Older Orchestration prompts
        # often send backends="claude"; batch scheduling must still exercise
        # the full fallback ladder.
        order = list(_DEFAULT_KERNEL_BACKEND_ORDER)
    allowed = {"claude", "codex", "geak"}
    return [backend for backend in order if backend in allowed]


def _batch_kernel_candidates(payload: dict) -> list[dict[str, Any]]:
    candidates_path = payload.get("candidates_path")
    if not candidates_path:
        return []
    try:
        data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    kernels = data.get("hot_kernels") or data.get("hot_kernels_top15") or []
    if not isinstance(kernels, list):
        return []
    reusable_ids = data.get("reusable_native_kernel_ids") or []
    reusable_id_set = {str(item) for item in reusable_ids if item}
    selected: list[dict[str, Any]] = []
    for item in kernels:
        if not isinstance(item, dict):
            continue
        kernel_id = str(item.get("kernel_id") or "")
        if not kernel_id:
            continue
        if reusable_id_set and kernel_id not in reusable_id_set:
            continue
        if item.get("reusable_native_kernel") is not True:
            continue
        if not item.get("source_file"):
            continue
        selected.append(item)
    return selected


async def _run_kernel_backend_sequence(
    base_payload: dict,
    candidate: dict[str, Any],
    *,
    session_dir: Path,
) -> HandlerResult:
    kernel_id = str(candidate.get("kernel_id") or base_payload.get("kernel_id") or "")
    attempts: list[dict[str, Any]] = []
    best: HandlerResult | None = None
    for backend in _backend_order(base_payload):
        child = dict(base_payload)
        child["_single_kernel"] = True
        child["kernel_id"] = kernel_id
        child["backends"] = backend
        child["candidate"] = candidate
        child.setdefault("source_file", candidate.get("source_file"))
        result = await _run_optimization_single(child, session_dir=session_dir)
        attempts.append({
            "backend": backend,
            "status": result.get("status"),
            "kernel_id": result.get("kernel_id"),
            "proposal": result.get("proposal"),
            "verification": result.get("verification"),
            "best_artifact_path": result.get("best_artifact_path"),
            "error": result.get("error"),
        })
        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        if best is None or float(verification.get("micro_speedup") or 0.0) > float(
            (best.get("verification") or {}).get("micro_speedup") or 0.0
        ):
            best = result
        if result.get("status") == "ok" and proposal.get("decision") == "KEEP":
            break
    if best is None:
        best = {
            "status": "failed",
            "kernel_id": kernel_id,
            "error": "no backend attempts were run",
        }
    best = dict(best)
    best["backend_fallback_attempts"] = attempts
    best["batch_kernel_id"] = kernel_id
    return best


async def _run_optimization_batch(
    payload: dict,
    candidates: list[dict[str, Any]],
    *,
    session_dir: Path,
) -> HandlerResult:
    max_parallel = int(
        payload.get("max_parallel")
        or os.environ.get("KERNEL_OPT_MAX_PARALLEL", _DEFAULT_KERNEL_BATCH_PARALLEL)
    )
    max_parallel = max(1, max_parallel)
    sem = asyncio.Semaphore(max_parallel)

    async def _guarded(candidate: dict[str, Any]) -> HandlerResult:
        async with sem:
            return await _run_kernel_backend_sequence(
                payload, candidate, session_dir=session_dir,
            )

    results = await asyncio.gather(*(_guarded(c) for c in candidates))
    best = max(
        results,
        key=lambda r: (
            1 if (r.get("proposal") or {}).get("decision") == "KEEP" else 0,
            float((r.get("verification") or {}).get("micro_speedup") or 0.0),
        ),
        default=None,
    )
    if best is None:
        return {
            "status": "failed",
            "error": "batch optimization produced no results",
            "batch_results": [],
        }
    out = dict(best)
    out["batch_mode"] = True
    out["batch_kernel_ids"] = [str(c.get("kernel_id")) for c in candidates]
    out["backend_order"] = _backend_order(payload)
    out["max_parallel"] = max_parallel
    out["batch_results"] = results
    return out


async def _run_optimization_single(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's kernel_optimization.py on one kernel.

    Required payload:
        kernel_id: str

    Optional payload:
        backends:        comma-separated 'geak,claude,codex' (auto-pick if empty)
        budget_minutes:  default 60
        source_file:     path to original kernel source (for context)
        candidates_path: path to JSON describing candidates (optional)
        extra_sglang_args: SGLang runtime flags for GEAK metadata (optional)
        enable_rag:      default True; false disables GEAK RAG tools
        enable_xs_memory: default True; false disables GEAK cross-session memory
        dry_run:         default False (testing)

    Returns the tool's JSON output verbatim under ``result``.
    """
    kernel_id = payload.get("kernel_id")
    if not kernel_id:
        return {"status": "failed", "error": "missing 'kernel_id' in payload"}
    guard = _validate_reusable_native_kernel(payload)
    if guard is not None:
        return guard
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    workspace_path = (
        payload.get("workspace_path")
        or str(session_dir / "kernel-agent-workspace")
    )
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    from .shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    target_platform = (
        payload.get("target_platform") or state.gpu_type or ""
    ).strip()
    if target_platform:
        os.environ["TARGET_GPU_TYPE"] = target_platform

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("kernel_optimization.py")),
        "--kernel-id", str(kernel_id),
        "--session-id", str(payload.get("session_id") or session_dir.name),
        "--workspace-path", workspace_path,
    ]
    if payload.get("backends"):
        cmd += ["--backends", str(payload["backends"])]
    if payload.get("source_file"):
        cmd += ["--source-file", str(payload["source_file"])]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if payload.get("extra_sglang_args"):
        cmd += ["--extra-sglang-args", str(payload["extra_sglang_args"])]
    if payload.get("candidates_path"):
        cmd += ["--candidates-path", str(payload["candidates_path"])]
    if payload.get("benchmark_file"):
        cmd += ["--benchmark-file", str(payload["benchmark_file"])]
    if payload.get("test_harness_path"):
        cmd += ["--test-harness-path", str(payload["test_harness_path"])]
    if payload.get("micro_speedup") is not None:
        cmd += ["--micro-speedup", str(payload["micro_speedup"])]
    if payload.get("e2e_gain_pct") is not None:
        cmd += ["--e2e-gain-pct", str(payload["e2e_gain_pct"])]
    if payload.get("correctness_passed") is not None:
        cmd += [
            "--correctness-passed",
            "true" if bool(payload["correctness_passed"]) else "false",
        ]
    if payload.get("accuracy_passed") is not None:
        cmd += [
            "--accuracy-passed",
            "true" if bool(payload["accuracy_passed"]) else "false",
        ]
    if payload.get("enable_rag") is False:
        cmd += ["--disable-rag"]
    if payload.get("enable_xs_memory") is False:
        cmd += ["--disable-xs-memory"]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    # Give kernel_optimization.py time to handle its own backend timeout and
    # salvage partial artifacts. The backend receives budget_minutes as its
    # hard wall-clock; killing this wrapper at the exact same second loses
    # optimized_versions/ and report paths.
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60 + 180

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

    Mirrors DESIGN v0.6 §16 integrate action: apply an optimized kernel
    artifact, re-run the active Magpie baseline config, then KEEP only
    if the measured E2E throughput clears the threshold. Compiled kernels
    are backed up as source plus existing .so/.co/.hsaco artifacts before
    rebuild so non-KEEP decisions can restore quickly without a rebuild.

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
    from .action_executors.benchmark_result import is_valid_measurement
    from .sub_agent_runner import RunnerContext
    from .task_registry import Task

    base_tput = float(payload.get("base_tput", 0.0))
    if base_tput <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }

    payload, missing_inputs = _resolve_integrate_payload(
        payload, session_dir=session_dir,
    )
    if missing_inputs is not None:
        return missing_inputs

    patch_path = payload.get("patch_path")
    kernel_id = payload.get("kernel_id")
    apply_result = _maybe_apply_kernel_patch(
        payload, session_dir=session_dir, kernel_id=kernel_id,
    )
    log.info("integrate_handler: apply_result=%s", apply_result)
    if apply_result.get("status") == "failed":
        return {
            "status": "failed",
            "error": "kernel patch apply failed",
            "decision": "REVERT",
            "apply_result": apply_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
        }
    if apply_result.get("status") != "ok":
        return {
            "status": "failed",
            "error_class": "patch_not_applied",
            "error": "kernel patch was not applied; refusing to run E2E benchmark",
            "decision": "REVERT",
            "apply_result": apply_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": payload.get("target_file") or payload.get("source_file"),
        }

    keep_threshold_pct = float(payload.get("keep_threshold_pct", 1.0))
    extra_args = str(payload.get("extra_sglang_args", "") or "").strip()

    # Build a Task wrapper around BaselineExecutor (which expects an
    # RunnerContext with a Task in it). The "extra_sglang_args" hand-
    # off goes via the task params even though baseline_executor doesn't
    # use them yet — kept for forward compat (P3 will inject EXTRA_SGLANG_ARGS).
    from ..session_paths import runs_dir
    fake_task_id = f"integrate-{kernel_id or 'anon'}"
    workspace = runs_dir(session_dir, "integrate", fake_task_id)
    workspace.mkdir(parents=True, exist_ok=True)
    fake_task = Task(
        task_id=fake_task_id,
        kind="baseline",
        state="running",
        params={
            "config_path": payload.get("config_path"),
            "output_dir":  str(workspace),
            "timeout_sec": int(payload.get("budget_minutes", 20)) * 60,
            "extra_sglang_args": extra_args,
        },
        idempotency_key=f"{fake_task_id}-rebaseline",
    )
    ctx = RunnerContext(task=fake_task, lease=None)

    try:
        bench_result = await BaselineExecutor(session_dir=session_dir)(ctx)
    except Exception as exc:  # noqa: BLE001
        revert_result = _maybe_revert_kernel_patch(apply_result)
        return {
            "status": "failed",
            "error_class": "rebaseline_exception",
            "error": repr(exc),
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "apply_result": apply_result,
            "revert_result": revert_result,
        }

    if not is_valid_measurement(bench_result):
        revert_result = _maybe_revert_kernel_patch(apply_result)
        return {
            "status": "failed",
            "error": "re-baseline did not succeed",
            "decision": "REVERT",
            "rebaseline_detail": bench_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "apply_result": apply_result,
            "revert_result": revert_result,
        }

    new_tput = float(bench_result.get("output_throughput") or 0.0)
    gain_pct = ((new_tput - base_tput) / base_tput * 100.0) if base_tput > 0 else 0.0
    decision = (
        "KEEP" if gain_pct > keep_threshold_pct
        else ("REVERT" if gain_pct < -keep_threshold_pct
              else "NEEDS_REVIEW")
    )
    revert_result = (
        {"status": "skipped", "reason": "KEEP decision"}
        if decision == "KEEP"
        else _maybe_revert_kernel_patch(apply_result)
    )
    return {
        "status":      "ok",
        "decision":    decision,
        "kernel_id":   kernel_id,
        "patch_path":  patch_path,
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "base_tput":   base_tput,
        "new_tput":    new_tput,
        "gain_pct":    gain_pct,
        "report_path": bench_result.get("report_path"),
        "workspace":   bench_result.get("workspace"),
        "extra_sglang_args": extra_args,
        "apply_result": apply_result,
        "revert_result": revert_result,
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
