"""Coordinator-side handlers for Kernel-agent REQUEST kinds.

 says the Kernel agent is responder-only: it answers
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
import shlex
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


# Where Hyperloom/kernel-agent's shell tools live. Env is set by
# inference_optimizer/scripts/install.sh -> pod-local kernel-agent env.
#
# Why lazy: in May 2026 the R1 N14 GPU run stalled for 1h 12min because
# this module was imported BEFORE the cli preflight had a chance to
# source $USER_DATA_PATH/runtime/kernel-agent.env.sh (the launcher only
# pre-sourced the user-level .env with 3 vars). A frozen module-level
# snapshot meant HYPERLOOM_KERNEL_AGENT_ROOT was permanently None even
# after preflight injected it into os.environ. Reading via a function
# at each call site lets cli.py's late env injection win; the snapshot
# constant is preserved (re-exported below) for backward compat.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"


def _kernel_agent_root_from_env() -> Path | None:
    raw = os.environ.get(_KERNEL_AGENT_ROOT_ENV)
    if not raw:
        return None
    return Path(raw)


# Backward-compat re-export (NOT used by internal logic — kept as a
# module-level alias for any external caller still doing `from
# kernel_request_handlers import HYPERLOOM_KERNEL_AGENT_ROOT`).
# Internal logic must use `_kernel_agent_root_from_env()` so a late
# env injection still wins.
HYPERLOOM_KERNEL_AGENT_ROOT = _kernel_agent_root_from_env()


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
    "/opt/venv/lib/python3.12/site-packages/aiter/",
    "/opt/venv/lib/python3.12/site-packages/sglang/",
    "/opt/venv/lib/python3.12/site-packages/vllm/",
    # Production vLLM wheel install layout (system dist-packages). Keep
    # this in sync with ``kernel-agent/tools/tracelens_analysis.py`` so
    # both the kernel-agent classifier and the orchestrator-side gate
    # in ``run_optimization_handler`` agree on what counts as a
    # reusable framework source.
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
    # atom layout (atom_plan/phase2_open_kernel_agent/2.5). The
    # editable install lives under ``/app/ATOM/atom/`` on disk but
    # ``_is_runtime_generated_kernel`` and ``run_optimization_handler``
    # lower-case their inputs before the ``startswith`` / substring
    # check, so the prefix below is stored lower-case for the match
    # to fire. ``framework_paths._DEFAULT_SOURCE_ROOTS`` carries the
    # canonical-case ``/app/ATOM/atom/`` because PolicyGate's
    # ``_path_in_allowlist`` is case-sensitive against the real
    # filesystem path. Keep this block in sync with
    # ``kernel-agent/tools/tracelens_analysis.py``
    # ``_REUSABLE_SOURCE_ROOTS`` (cross-cutting guard
    # ``test_atom_present_in_tracelens_reusable_roots``).
    "/app/atom/atom/",
    "/opt/venv/lib/python3.10/site-packages/atom/",
    "/opt/venv/lib/python3.12/site-packages/atom/",
    "/usr/local/lib/python3.12/dist-packages/atom/",
    "/usr/local/lib/python3.10/dist-packages/atom/",
)
_APPLY_TOOL_MODULE: Any | None = None
_DEFAULT_KERNEL_BACKEND_ORDER = ("geak", "claude", "codex", "cursor")
# Soft upper bound on concurrent ``_run_kernel_backend_sequence`` coroutines
# inside ``_run_optimization_batch``. The real GPU scheduling happens one
# layer below: GEAK / OOB submitters register their work as
# ``ray.remote(num_gpus=...)`` tasks, so Ray serializes any oversubscription
# against the cluster's actual GPU resources (typical MI300X / MI355X node
# = 8 GPU). This default mirrors that node size so a single
# ``run_optimization`` request can fan out to one GEAK/OOB attempt per GPU
# without the asyncio semaphore artificially capping below Ray's view.
# Pre-PR-X default was 3, which throttled even small batches (e.g. A3B's
# 3 kernel units were already at the cap, and larger TraceLens outputs
# silently serialized behind sem). Override via
# ``KERNEL_OPT_MAX_PARALLEL`` env (>=1) on nodes with fewer GPUs or when
# the LLM API gateway becomes the new bottleneck.
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
_DEFAULT_OOB_BUDGET_MINUTES = 60.0
_DEFAULT_GEAK_BUDGET_MINUTES = 90.0
_CANDIDATE_ENV_KEYS = {
    "CONC",
    "ISL",
    "OSL",
    "TP",
    "NUM_PROMPTS",
    "NUM_WARMUPS",
    "MAX_MODEL_LEN",
    "RANDOM_RANGE_RATIO",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
}
_CANDIDATE_ENV_PREFIXES = (
    "SGLANG_",
    "VLLM_",
    "AITER_",
    "TRITON_",
    "HIPBLASLT_",
    "PYTORCH_TUNABLEOP_",
)
_SENSITIVE_ENV_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _kernel_agent_root_error() -> str | None:
    root = _kernel_agent_root_from_env()
    if root is None:
        return (
            f"{_KERNEL_AGENT_ROOT_ENV} is not set; run "
            "inference_optimizer/scripts/install.sh and source $KERNEL_AGENT_ENV "
            "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
        )
    if not root.is_dir():
        return f"{_KERNEL_AGENT_ROOT_ENV} does not exist: {root}"
    return None


def _kernel_agent_tool_path(tool_name: str) -> Path:
    err = _kernel_agent_root_error()
    if err:
        raise RuntimeError(err)
    root = _kernel_agent_root_from_env()
    assert root is not None
    path = root / "tools" / tool_name
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


def _coerce_runtime_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        try:
            return float(stripped) if "." in stripped else value
        except ValueError:
            return value
    return value


def _candidate_env_allowed(key: str) -> bool:
    upper = key.upper()
    if any(part in upper for part in _SENSITIVE_ENV_PARTS):
        return False
    return key in _CANDIDATE_ENV_KEYS or any(
        key.startswith(prefix) for prefix in _CANDIDATE_ENV_PREFIXES
    )


def _split_server_args(raw: str) -> list[str]:
    try:
        return shlex.split(raw) if raw else []
    except ValueError:
        log.warning("failed to parse materialized server args; preserving raw string")
        return []


def _load_materialized_workload_metadata(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read materialized workload config %s: %s", path, exc)
        return {}
    bench = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    framework = str(bench.get("framework") or "").strip().lower()
    server_key = "EXTRA_VLLM_ARGS" if framework == "vllm" else "EXTRA_SGLANG_ARGS"
    server_args = str(envs.get(server_key) or "").strip()
    workload = {
        out_key: _coerce_runtime_value(envs[src_key])
        for out_key, src_key in (
            ("tp", "TP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("num_prompts", "NUM_PROMPTS"),
            ("num_warmups", "NUM_WARMUPS"),
            ("max_model_len", "MAX_MODEL_LEN"),
            ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        )
        if src_key in envs
    }
    runtime_args = {
        "materialized_config": str(path),
        "framework": framework or None,
        "model": bench.get("model"),
        "precision": bench.get("precision"),
        "server_args": server_args,
        "server_args_argv": _split_server_args(server_args),
        "workload": workload,
    }
    return {
        "env_vars": {
            str(key): str(value)
            for key, value in envs.items()
            if _candidate_env_allowed(str(key))
        },
        "runtime_args": {
            key: value for key, value in runtime_args.items()
            if value not in (None, "", {})
        },
    }


def _enrich_candidate_runtime_metadata(
    candidates: Any,
    metadata: dict[str, Any],
) -> None:
    if not isinstance(candidates, list) or not metadata:
        return
    env_vars = metadata.get("env_vars") if isinstance(metadata.get("env_vars"), dict) else {}
    runtime_args = (
        metadata.get("runtime_args") if isinstance(metadata.get("runtime_args"), dict) else {}
    )
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_env = item.setdefault("env_vars", {})
        if isinstance(item_env, dict):
            for key, value in env_vars.items():
                item_env.setdefault(key, value)
        item_args = item.setdefault("runtime_args", {})
        if isinstance(item_args, dict):
            for key, value in runtime_args.items():
                item_args.setdefault(key, value)


def _enrich_candidate_trace_report(candidates: Any, report_path: str) -> None:
    if not isinstance(candidates, list) or not report_path:
        return
    for item in candidates:
        if isinstance(item, dict):
            item.setdefault("trace_report_path", report_path)


def _enrich_candidates_artifact(
    candidates_path: str,
    metadata: dict[str, Any],
    *,
    trace_report_path: str = "",
) -> None:
    if not candidates_path:
        return
    path = Path(candidates_path)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read candidates artifact %s: %s", path, exc)
        return
    if not isinstance(data, dict):
        return
    if metadata:
        _enrich_candidate_runtime_metadata(data.get("hot_kernels"), metadata)
        _enrich_candidate_runtime_metadata(data.get("hot_kernels_top15"), metadata)
    if trace_report_path:
        data.setdefault("trace_report_path", trace_report_path)
        artifact_paths = data.setdefault("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            artifact_paths.setdefault("trace_report_path", trace_report_path)
        _enrich_candidate_trace_report(data.get("hot_kernels"), trace_report_path)
        _enrich_candidate_trace_report(
            data.get("hot_kernels_top15"), trace_report_path,
        )
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        (state.last_trace_analyze or {}).get("hot_kernels_top15")
        or (state.last_trace_analyze or {}).get("hot_kernels")
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
    ``last_kernel_opt`` and the source target lives in ``last_trace_analyze``.
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

    # Multi-KEEP queue fallback (PR-B B-5):
    # ``last_kernel_opt`` only ever holds the strongest pending KEEP.
    # When the queue drains a second/third KEEP whose kernel_id != that
    # of ``last_kernel_opt``, the block above doesn't fire and we'd
    # bail out with ``missing_integration_inputs``. Pull patch_path /
    # source_file out of the per-kernel ledger so any queued KEEP can
    # integrate.
    if (
        kernel_id
        and not resolved.get("patch_path")
    ):
        attempt = (state.kernel_opt_attempts or {}).get(kernel_id) or {}
        if attempt.get("last_artifact_path"):
            resolved["patch_path"] = str(attempt["last_artifact_path"])
        if not resolved.get("source_file") and attempt.get("last_source_file"):
            resolved["source_file"] = str(attempt["last_source_file"])

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
        from .action_executors._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
        )
        if is_multi_node():
            addr = ray_gcs_address_from_state()
            if addr:
                env.setdefault("RAY_ADDRESS", addr)
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, env=env,
        )

    proc = await asyncio.to_thread(_run)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
async def trace_analyze_handler(
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
        roofline_json:   optional pre-computed roofline JSON path; the
                         orchestrator no longer auto-produces this (the
                         retired ``pmc_roofline`` action), but operators
                         can still inject one manually when an external
                         tool generated it
        dry_run:         default False (testing)
        budget_minutes:  default 60

    Returns::

        {
          "status": "ok" | "failed",
          "hot_kernels": [...],
          "trace_report_path": "...",        # analysis.md from TraceLens v0.3
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

    # Tool output lands at ``<workspace_path>/kernel-agent/runs/<session_id>/``
    # (the suffix is hardcoded inside ``tracelens_analysis.py``). Pass the
    # session root so the artefacts settle at
    # ``<session_dir>/kernel-agent/runs/...`` — a sibling of
    # ``<session_dir>/kernel-agent-workspace/<kernel_id>/`` (which the
    # tool also reads/writes for cross-call GEAK/OOB artefacts). Both
    # locations now live under ``$USER_DATA_PATH`` for unified monitoring.
    workspace_path = payload.get("workspace_path") or str(session_dir)
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

    # Load the materialized baseline workload metadata once. Used twice:
    # (1) here to feed CONC / OSL / RANDOM_RANGE_RATIO into the splitter
    #     CLI flags (`--split-conc` / `--split-osl` / `--split-r`) so
    #     TraceLens.TraceUtils.split_inference_trace_annotation picks the
    #     correct steady-state window — without these the splitter falls
    #     back to in-trace heuristics that can yield 0 chunks
    #     (`trace_split_no_steady_state`) and collapse the whole
    #     select_kernels / kernel_opt / integrate chain.
    # (2) downstream below to enrich result.hot_kernels and the
    #     kernel_candidates artifact with the same runtime context.
    metadata = _load_materialized_workload_metadata(state.baseline_config_path)
    workload = (
        metadata.get("runtime_args", {}).get("workload", {})
        if isinstance(metadata, dict) else {}
    )

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

    # Splitter workload hints. Priority chain: payload (explicit
    # operator/critic override) > materialized baseline metadata > drop
    # the flag entirely so tracelens_analysis.py keeps its existing env
    # fallback (TRACELENS_SPLIT_* / CONC / OSL / RANDOM_RANGE_RATIO).
    # Without these, the splitter has historically had to guess the
    # mixed-window selection's PD ratio from heuristics; on workloads
    # where heuristics miss, all three steady-state windows come back
    # empty and `select_kernels` returns
    # ``status=failed error=trace_split_no_steady_state``, blocking the
    # entire kernel-optimization chain (select_kernels -> kernel_opt ->
    # integrate -> operator_tuning -> deep_kernel_analysis).
    split_conc = payload.get("split_conc") or workload.get("conc")
    if split_conc not in (None, ""):
        cmd += ["--split-conc", str(split_conc).strip()]
    split_osl = payload.get("split_osl") or workload.get("osl")
    if split_osl not in (None, ""):
        cmd += ["--split-osl", str(split_osl).strip()]
    split_r = payload.get("split_r") or workload.get("random_range_ratio")
    if split_r not in (None, ""):
        cmd += ["--split-r", str(split_r).strip()]

    capture_folder = (
        payload.get("capture_folder")
        or payload.get("graph_capture_path")
        or payload.get("capture_folder_path")
    )
    if capture_folder:
        cmd += ["--capture-folder", str(capture_folder)]
    # N25: forward TraceLens splitter steady-state mode (mixed /
    # decode_only / prefilldecode). Set via payload OR env so the
    # coordinator can re-issue roofline with a different mode after a
    # steady_state_chunk_missing / steady_state_chunk_empty warning
    # lands (e.g. SOLAR-10.7B TP=1 mixed-window degenerates to PD=0
    # with all forward inside CUDA graph + no rocprofiler Dispatch
    # Task aggregate; prefilldecode chunk carries the real GEMM /
    # attention workload).
    steady_state_mode = (
        payload.get("steady_state_mode")
        or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
    )
    steady_state_mode = str(steady_state_mode).strip()
    if steady_state_mode:
        cmd += ["--steady-state-mode", steady_state_mode]
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
        report_path = result.get("trace_report_path")
        if not report_path and isinstance(artifacts, dict):
            report_path = artifacts.get("trace_report_path")
        if report_path:
            result["trace_report_path"] = str(report_path)
            _enrich_candidate_trace_report(
                result.get("hot_kernels"), str(report_path),
            )
        # PR-A §3 (#206): surface tracelens/summary.json — the per-run
        # audit sidecar listing reusable tasks vs skipped kernels with
        # reasons, so operators can see at a glance whether GEAK was
        # offered the kernels they expected.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])

        # A failed TraceLens run is a hard trace-quality / integration
        # failure, not a valid "empty candidates" signal. Keep
        # ``status=failed`` so the Coordinator does not continue down a
        # misleading params/backends path as if kernel analysis had
        # completed. Still attach a structured warning so operators can
        # inspect the root cause from SharedState / event logs.
        if (
            result.get("status") == "failed"
            and "trace_split_no_steady_state" not in str(result.get("error") or "")
        ):
            failure_warning: dict[str, Any] = {
                "code": "tracelens_analysis_failed",
                "severity": "warning",
                "message": (
                    "TraceLens analysis failed; refusing to treat this as a "
                    "successful empty-kernel result. See ``stderr_tail`` / "
                    "``error`` for the upstream failure."
                ),
            }
            for key in ("returncode", "rc", "error", "stderr_tail", "raw_stdout_tail"):
                if key in result and result[key] not in (None, ""):
                    failure_warning[key] = result[key]
            health = list(result.get("trace_health_warnings") or [])
            health.append(failure_warning)
            result["trace_health_warnings"] = health
            result["hot_kernels"] = []
            result.setdefault("orchestrator_error", failure_warning.get("error", ""))

        # T3 / T4: guarantee ``trace_health_warnings`` is always a list
        # at the handler boundary so downstream code can iterate without
        # a ``None``-guard. Empty list = steady-state ("nothing wrong").
        result.setdefault("trace_health_warnings", [])

        # ``metadata`` was loaded once at the top of the handler so both
        # the splitter CLI hints and the downstream candidate enrichment
        # see the same materialized baseline workload state.
        _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
        candidates_path = result.get("candidates_path")
        if isinstance(candidates_path, str):
            _enrich_candidates_artifact(
                candidates_path,
                metadata,
                trace_report_path=str(report_path or ""),
            )
    return result


# ---------------------------------------------------------------------------
async def run_optimization_handler(
    payload: dict,
    *,
    session_dir: Path,
    record_partial: Callable[[dict], None] | None = None,
) -> HandlerResult:
    """Run kernel optimization.

    When candidate metadata is available, this handler upgrades legacy
    single-kernel requests into a batch over all reusable native kernels. Each
    kernel is optimized concurrently, while backends are tried sequentially per
    kernel in the preferred order: Claude → Codex → Cursor → GEAK.

    ``record_partial`` (optional) is a synchronous callback invoked the
    instant each batch sub-result completes -- before ``asyncio.gather``
    wait-all returns. The Coordinator passes
    :meth:`Coordinator._record_kernel_opt_partial` here so SharedState
    sees KEEP/REVERT decisions on the next tick even while slow GEAK
    siblings are still running. Single-kernel runs ignore it (the same
    end-of-handler ``record_kernel_opt`` path covers them).
    """
    if payload.get("_single_kernel"):
        return await _run_optimization_single(payload, session_dir=session_dir)
    candidates = _batch_kernel_candidates(payload, session_dir=session_dir)
    if len(candidates) <= 1:
        single_payload = dict(payload)
        if candidates and not single_payload.get("kernel_id"):
            single_payload["kernel_id"] = candidates[0].get("kernel_id")
        single_payload["_single_kernel"] = True
        return await _run_optimization_single(single_payload, session_dir=session_dir)
    return await _run_optimization_batch(
        payload, candidates,
        session_dir=session_dir,
        record_partial=record_partial,
    )


def _geak_budget_minutes(payload: dict) -> float:
    return float(
        payload.get("geak_budget_min")
        or os.environ.get("HYPERLOOM_GEAK_BUDGET_MIN", _DEFAULT_GEAK_BUDGET_MINUTES)
    )


def _optimization_budget_minutes(payload: dict) -> float:
    """Wall-clock budget mirrored by the kernel_optimization.py wrapper."""
    oob_budget = float(payload.get("budget_minutes", _DEFAULT_OOB_BUDGET_MINUTES))
    geak_budget = _geak_budget_minutes(payload)
    backend = str(payload.get("backends") or "").strip().lower()
    if backend == "geak":
        return geak_budget
    if backend in {"claude", "codex", "cursor"}:
        return oob_budget
    # Empty / multi-backend payloads may still run GEAK first in the ladder.
    return max(oob_budget, geak_budget)


def _optimization_wrapper_timeout_sec(payload: dict) -> int:
    # +180s grace so kernel_optimization.py can salvage partial artifacts.
    return int(_optimization_budget_minutes(payload) * 60) + 180


def _backend_order(payload: dict) -> list[str]:
    raw = payload.get("backend_order") or os.environ.get("KERNEL_OPT_BACKEND_ORDER")
    if raw:
        order = [item.strip() for item in str(raw).split(",") if item.strip()]
        explicit = True
    else:
        # Ignore legacy payload["backends"] here. Older Orchestration prompts
        # often send backends="claude"; batch scheduling must still exercise
        # the full fallback ladder. The default ladder mirrors
        # ``kernel_optimization.choose_backends`` so single-kernel and batch
        # paths agree on the policy (GEAK FIRST per #144 last comment Layer 1
        # — high-priority handoff, Claude/Codex follow as fallbacks if GEAK
        # times out or rejects). Cursor only joins the ladder when the
        # operator has provisioned ``CURSOR_API_KEY``; see filter below.
        order = list(_DEFAULT_KERNEL_BACKEND_ORDER)
        explicit = False
    allowed = {"claude", "codex", "cursor", "geak"}
    selected = [backend for backend in order if backend in allowed]
    # When the operator has not provisioned CURSOR_API_KEY, drop cursor from
    # the auto-derived ladder so we don't waste a fallback slot on a 401.
    # Explicit `payload["backend_order"]` / KERNEL_OPT_BACKEND_ORDER still
    # wins (respect intent; failure surfaces clearly in the attempt log).
    if not explicit and not os.environ.get("CURSOR_API_KEY", "").strip():
        selected = [b for b in selected if b != "cursor"]
    return selected


def _in_flight_kernel_ids(session_dir: Path) -> set[str]:
    """Scan the kernel-agent run dir for status files in ``state=running``.

    Used by :func:`_batch_kernel_candidates` to skip kernels that are
    still being optimized by a prior batch's subprocess (Qwen3-30B-
    A3B-Base 164405Z saw five concurrent ``kernel_optimization.py``
    processes for the same k002/k004 because the LLM kept proposing
    fresh ``run_optimization`` requests while subprocesses from
    earlier batches were still alive).
    """
    in_flight: set[str] = set()
    sid = session_dir.name
    status_dir = session_dir / "kernel-agent" / "runs" / sid / "status" / "kernel_optimization"
    if not status_dir.is_dir():
        return in_flight
    for p in status_dir.glob("ko-*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("state") or "").lower() != "running":
            continue
        kid = ""
        for line in d.get("last_lines") or []:
            if isinstance(line, str) and line.startswith("kernel_id="):
                kid = line.split("=", 1)[1].strip()
                break
        if not kid:
            kid = str(d.get("kernel_id") or "")
        if kid:
            in_flight.add(kid)
    return in_flight


def _batch_kernel_candidates(
    payload: dict,
    *,
    session_dir: Path | None = None,
) -> list[dict[str, Any]]:
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

    # PR-C filters: build the "live" exclusion sets up front so both
    # the task_group fallback and the legacy per-kernel pass can honor
    # them. Without session_dir (legacy tests / dry-run paths) the
    # filters degrade to empty sets and behaviour matches PR-A/B.
    rejected_kernel_ids: set[str] = set()
    attempts_by_kid: dict[str, dict] = {}
    in_flight: set[str] = set()
    max_attempts = 1
    min_gpu_pct = 0.0
    try:
        max_attempts = max(1, int(os.environ.get(
            "INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_ATTEMPTS", "1",
        )))
    except (TypeError, ValueError):
        max_attempts = 1
    # PR-I: default min_gpu_pct must match SharedState.untried_hot_
    # reusable_kernels' default (3.0). Earlier code defaulted to 0.0
    # here, so the LLM saw an empty "untried" queue (gate >=3%) but
    # _batch_kernel_candidates would still dispatch <3% candidates
    # picked up via task_group fallback (e.g. rmsnorm group's k006 at
    # gpu_pct=1.3% in Qwen3-30B-A3B-Base session 20260523T035235Z's
    # third batch round). Mirroring the SharedState default keeps the
    # two layers in sync and avoids tiny kernels eating 30-90 min of
    # ladder wall-clock for no E2E gain.
    from .shared_state import _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
    try:
        min_gpu_pct = float(os.environ.get(
            "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
            _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
        ))
    except (TypeError, ValueError):
        min_gpu_pct = _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
    if session_dir is not None:
        try:
            from .shared_state import SharedState
            state = SharedState.load_or_init(session_dir)
            rejected_kernel_ids = set(state.rejected_kernel_ids or [])
            attempts_by_kid = dict(state.kernel_opt_attempts or {})
            in_flight = _in_flight_kernel_ids(session_dir)
        except Exception:
            log.exception(
                "_batch_kernel_candidates: failed to load SharedState "
                "from %s; PR-C filters disabled this dispatch",
                session_dir,
            )

    def _is_live(kid: str, current_source: str = "") -> bool:
        """A kernel_id is live (eligible for batch) iff it is NOT
        rejected, NOT in-flight, and has fewer than max_attempts
        recorded attempts AGAINST THE CURRENT CANDIDATE'S source_file.

        ``max_attempts = 1`` (default) means: any prior attempt against
        the same source_file -> not live, but a prior attempt against a
        DIFFERENT source_file is ignored. This is what lets PR-K's
        launcher → device source promotion unlock a fresh attempt:
        when ``aiter::ck_moe_stage1`` was first dispatched against the
        python wrapper ``aiter/ops/moe_op.py`` and PARTIAL'd, a
        subsequent dispatch with ``current_source`` pointing at the
        promoted device file ``csrc/.../gemm_moe_ck2stages.cu`` is
        treated as a fresh target with its own quota. Without this,
        the wrapper's first failed attempt would lock the entire
        task_group as ``group_exhausted`` even though the device path
        had never been tried (Qwen3-30B-A3B-Base session
        20260523T162026Z burned 2 hours on this).

        Falls back to the cumulative ``attempts`` counter when:
          * ``current_source`` is empty (legacy callers / synthetic
            test fixtures that don't carry source_file);
          * the entry was written by a v1 ``record_kernel_opt`` that
            predated ``attempts_per_source`` (resumed state.json from
            before this PR).
        Both fallbacks preserve the pre-PR-K behaviour byte-for-byte.
        """
        if kid in rejected_kernel_ids:
            return False
        if kid in in_flight:
            return False
        entry = attempts_by_kid.get(kid) or {}
        if current_source:
            per_source = entry.get("attempts_per_source")
            if isinstance(per_source, dict):
                src_attempts = int(per_source.get(current_source, 0))
                return src_attempts < max_attempts
        if int(entry.get("attempts", 0)) >= max_attempts:
            return False
        return True

    # PR-B §1: collapse kernels that share a source function into a
    # single dispatch via ``task_groups[]``. Each group emits exactly
    # one GEAK / Codex / Claude request keyed off ``primary_kernel_id``,
    # and the full row list lives on ``item["task_group"]`` so
    # ``build_prompt`` can render multi-row benchmark cases. Kernels
    # whose launcher path wasn't parseable (analysis.md with empty
    # Kernel Path, raw-trace path, csv fallback) fall through to the
    # legacy per-kernel path below — aggregation is purely additive,
    # never lossy.
    task_groups = data.get("task_groups") or []
    if not isinstance(task_groups, list):
        task_groups = []
    kernel_by_id: dict[str, dict[str, Any]] = {
        str(k.get("kernel_id") or ""): k
        for k in kernels
        if isinstance(k, dict) and k.get("kernel_id")
    }
    grouped_kernel_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}  # kid -> reason, for debug logging
    for group in task_groups:
        if not isinstance(group, dict):
            continue
        member_ids = [str(k) for k in (group.get("kernel_ids") or []) if k]
        if not member_ids:
            continue
        # All members marked across the group so the legacy loop never
        # picks them up regardless of which member we end up dispatching
        # (or whether we skip the group entirely).
        grouped_kernel_ids.update(member_ids)
        # Only members marked reusable_native_kernel survive (vendor /
        # aten:: / runtime-generated were filtered upstream by
        # ``classify_patchability``); the group's primary may itself
        # have been rejected, in which case we fall back to the next
        # live reusable member of the same AST function (same source,
        # equivalent leverage). When EVERY member is rejected /
        # in-flight / exhausted, the whole group skips.
        primary = str(group.get("primary_kernel_id") or "")
        primary_cand = kernel_by_id.get(primary)
        primary_live = (
            primary_cand is not None
            and primary_cand.get("reusable_native_kernel") is True
            and bool(primary_cand.get("source_file"))
            and _is_live(primary, str(primary_cand.get("source_file") or ""))
        )
        if not primary_live:
            primary_cand = next(
                (
                    kernel_by_id[m]
                    for m in member_ids
                    if m in kernel_by_id
                    and kernel_by_id[m].get("reusable_native_kernel") is True
                    and kernel_by_id[m].get("source_file")
                    and _is_live(m, str(kernel_by_id[m].get("source_file") or ""))
                ),
                None,
            )
            if primary_cand is None:
                # Every reusable member exhausted -> nothing to dispatch
                # for this group this round.
                for m in member_ids:
                    skipped.setdefault(m, "group_exhausted")
                continue
        if not primary_cand.get("source_file"):
            continue
        try:
            picked_pct = float(primary_cand.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            picked_pct = 0.0
        if picked_pct < min_gpu_pct:
            for m in member_ids:
                skipped.setdefault(m, f"below_min_gpu_pct={min_gpu_pct}")
            continue
        # Shallow copy + attach group so the kernel_optimization.py
        # subprocess sees ``candidate["task_group"]`` and can render
        # benchmark cases.
        item = dict(primary_cand)
        item["task_group"] = group
        selected.append(item)

    # Legacy per-kernel pass for any reusable kernel that wasn't
    # absorbed into a task_group above (no parseable launcher path).
    for item in kernels:
        if not isinstance(item, dict):
            continue
        kernel_id = str(item.get("kernel_id") or "")
        if not kernel_id:
            continue
        if kernel_id in grouped_kernel_ids:
            continue
        if reusable_id_set and kernel_id not in reusable_id_set:
            continue
        if item.get("reusable_native_kernel") is not True:
            continue
        if not item.get("source_file"):
            continue
        if not _is_live(kernel_id, str(item.get("source_file") or "")):
            skipped[kernel_id] = "not_live"
            continue
        try:
            row_pct = float(item.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            row_pct = 0.0
        if row_pct < min_gpu_pct:
            skipped[kernel_id] = f"below_min_gpu_pct={min_gpu_pct}"
            continue
        selected.append(item)

    if skipped:
        log.info(
            "batch candidates filtered: %d selected, skipped=%s",
            len(selected), skipped,
        )
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
        # PR-F: prefer a KEEP verdict over a higher-micro non-KEEP. The
        # ladder runs GEAK first; GEAK frequently returns NEEDS_REVIEW
        # at e.g. 1.3x because it has no correctness gate, while a
        # subsequent Claude/Codex attempt may deliver a real KEEP at
        # 1.17x with full correctness. Before PR-F the higher-micro
        # NEEDS_REVIEW won the best-selection contest, the ladder
        # broke on KEEP but returned the wrong result -- the actual
        # KEEP patch was silently discarded (Qwen3-30B-A3B-Base
        # 20260523T035235Z k004: codex KEEP @1.17x lost to geak
        # NEEDS_REVIEW @1.3x, never reached integrate).
        # Mirror the batch handler's max-key in
        # ``_run_optimization_batch`` so ladder + batch agree.
        new_keep = (
            result.get("status") == "ok"
            and proposal.get("decision") == "KEEP"
        )
        new_micro = float(verification.get("micro_speedup") or 0.0)
        if best is None:
            best = result
        else:
            best_proposal = (best.get("proposal") or {})
            best_keep = (
                best.get("status") == "ok"
                and best_proposal.get("decision") == "KEEP"
            )
            best_micro = float(
                (best.get("verification") or {}).get("micro_speedup") or 0.0
            )
            if (new_keep, new_micro) > (best_keep, best_micro):
                best = result
        if new_keep:
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
    # Preserve source_file on the aggregated best so the streaming
    # record callback in _run_optimization_batch can group by file
    # without re-reading kernel_candidates.json.
    if not best.get("source_file"):
        cand_src = candidate.get("source_file") if isinstance(candidate, dict) else None
        if cand_src:
            best["source_file"] = str(cand_src)
    return best


async def _run_optimization_batch(
    payload: dict,
    candidates: list[dict[str, Any]],
    *,
    session_dir: Path,
    record_partial: Callable[[dict], None] | None = None,
) -> HandlerResult:
    """Fan ``run_optimization`` out across reusable native kernels.

    If ``record_partial`` is provided, every sub-attempt streams its
    result into SharedState the moment :func:`_run_kernel_backend_sequence`
    returns -- *before* ``asyncio.gather`` wait-all unblocks. This is
    what lets a fast KEEP land on the integrate queue without waiting
    for a slow GEAK sibling to time out (Qwen3-30B-A3B-Base session
    20260522T093903Z burned 3 hours on this).
    """
    max_parallel = int(
        payload.get("max_parallel")
        or os.environ.get("KERNEL_OPT_MAX_PARALLEL", _DEFAULT_KERNEL_BATCH_PARALLEL)
    )
    max_parallel = max(1, max_parallel)
    sem = asyncio.Semaphore(max_parallel)

    async def _guarded(candidate: dict[str, Any]) -> HandlerResult:
        cand_kid = str(candidate.get("kernel_id") or "") if isinstance(candidate, dict) else ""
        cand_src = (
            str(candidate.get("source_file") or "")
            if isinstance(candidate, dict) else ""
        )
        async with sem:
            try:
                result = await _run_kernel_backend_sequence(
                    payload, candidate, session_dir=session_dir,
                )
            except Exception as exc:  # noqa: BLE001
                # A sub-task exception (network blip, GEAK crash, ...)
                # must NOT propagate out of asyncio.gather while sibling
                # tasks are still in flight. With the default
                # ``return_exceptions=False``, gather would re-raise on
                # first exception while siblings keep running in the
                # background -- the Coordinator would then unblock
                # mid-batch, potentially dispatch an integrate, and
                # collide with still-running kernel_opt subprocesses on
                # the GPU. We turn every sub-task failure into a
                # structured failed result so gather behaves as wait-all
                # regardless of sub-task outcomes, and so the streaming
                # ``record_partial`` callback still has a kernel_id to
                # ledger against (preserving rejection / retire logic).
                log.exception(
                    "kernel-opt sub-task crashed for kernel_id=%s; "
                    "wrapping as failed result so gather wait-all holds",
                    cand_kid or "?",
                )
                result = {
                    "status": "failed",
                    "kernel_id": cand_kid,
                    "source_file": cand_src,
                    "error_class": "subtask_exception",
                    "error": repr(exc),
                }
        # Stamp the candidate's source_file onto the sub-result so
        # SharedState's same-source-file conflict guard
        # (``_source_files_in_optimization_stack``) can detect when two
        # KEEPs target the same file. ``_run_kernel_backend_sequence``
        # already preserves ``source_file`` from the candidate payload,
        # but defensively re-stamp here in case a backend dropped it.
        if isinstance(result, dict) and not result.get("source_file") and cand_src:
            result["source_file"] = cand_src
        if record_partial is not None:
            try:
                record_partial(result)
            except Exception:  # noqa: BLE001
                # Per-sub-attempt callback failures must not abort the
                # batch -- log and continue. The final aggregation below
                # still runs, and the Coordinator's post-gather
                # ``record_kernel_opt`` skips dedup only when batch_mode
                # is set (so the lost streaming write is recoverable on
                # next batch).
                log.exception(
                    "record_partial callback failed for kernel_id=%s",
                    (result or {}).get("kernel_id") if isinstance(result, dict) else None,
                )
        return result

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
        backends:        comma-separated 'geak,claude,codex,cursor' (auto-pick if empty)
        budget_minutes:  default 60 (OOB backends)
        geak_budget_min: default 90 (GEAK only; also ``HYPERLOOM_GEAK_BUDGET_MIN``)
        source_file:     path to original kernel source (for context)
        candidates_path: path to JSON describing candidates (optional)
        extra_server_args: SGLang runtime flags for GEAK metadata (optional)
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

    # Same convention as :func:`trace_analyze_handler`: pass the session
    # root so ``kernel_optimization.py`` lands its run artefacts at
    # ``<session_dir>/kernel-agent/runs/<session_id>/`` while still reading
    # ``<session_dir>/kernel-agent-workspace/<kernel_id>/`` for the
    # cross-call GEAK/OOB cache. Both subtrees live under
    # ``$USER_DATA_PATH``.
    workspace_path = payload.get("workspace_path") or str(session_dir)
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
    if payload.get("extra_server_args"):
        cmd += ["--extra-sglang-args", str(payload["extra_server_args"])]
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
    geak_budget_min = _geak_budget_minutes(payload)
    backend = str(payload.get("backends") or "").strip().lower()
    if backend == "geak" or not backend:
        cmd += ["--geak-budget-min", str(geak_budget_min)]
    if payload.get("budget_minutes") is not None:
        cmd += ["--budget-minutes", str(payload["budget_minutes"])]
    # Give kernel_optimization.py time to handle its own backend timeout and
    # salvage partial artifacts. GEAK defaults to 90 min; OOB defaults to 60.
    timeout_sec = _optimization_wrapper_timeout_sec(payload)

    from .action_executors._multi_node_env import is_multi_node

    if is_multi_node():
        from inference_optimizer.multi_node.cli import (
            kill_inference_for_kernel_agent_best_effort,
        )

        await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    result = _shape_tool_result(rc, stdout, stderr)
    # Stamp source_file / kernel_id from the payload onto the parsed
    # tool result so the multi-KEEP integrate queue
    # (``SharedState.next_pending_keep_kernel_id``) can group same-file
    # KEEPs and the streaming-record callback can record the source
    # without re-resolving from candidates_path. kernel_optimization.py
    # already prints kernel_id, but in failure modes (timeout, crash)
    # it may be missing; payload always has it because we just passed
    # it on the CLI above.
    if isinstance(result, dict):
        if not result.get("kernel_id") and payload.get("kernel_id"):
            result["kernel_id"] = str(payload["kernel_id"])
        if not result.get("source_file") and payload.get("source_file"):
            result["source_file"] = str(payload["source_file"])
    return result


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

    Mirrors integrate action: apply an optimized kernel
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
        extra_server_args: extra flags layered onto the Magpie envs
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
    # ``payload`` arrives via the integrate_patch sub-agent envelope;
    # route the read through the Phase 4 compat helper so a legacy
    # ``extra_sglang_args`` envelope still resolves (with a single
    # DeprecationWarning logged via stacklevel=3).
    from ..compat.payload_aliases import read_extra_server_args
    extra_args = read_extra_server_args(payload).strip()

    # Build a Task wrapper around BaselineExecutor (which expects an
    # RunnerContext with a Task in it). The "extra_server_args" hand-
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
            "extra_server_args": extra_args,
        },
        idempotency_key=f"{fake_task_id}-rebaseline",
    )
    ctx = RunnerContext(task=fake_task, lease=None)

    # Multi-node: apply_kernel_patch has just fanned the new source
    # files to every pod. sglang must be FULLY restarted (not resume-
    # pathed) so it re-imports the patched modules; otherwise the
    # re-baseline measures the pre-patch process and integrate decisions
    # become noise. We do the restart HERE (not inside BaselineExecutor)
    # and set ctx.extra["mn_round_restarted"] so BaselineExecutor does
    # NOT restart a second time. force_full_restart=True scopes the env
    # override (MULTI_NODE_RESTART_RESUME_RUNNING=0) for this call only;
    # subsequent non-integrate rounds keep their resume savings.
    from .action_executors._multi_node_env import is_multi_node
    if is_multi_node():
        from .action_executors._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )
        try:
            await restart_server_for_round(
                extra_server_args=extra_args,
                framework=os.environ.get("FRAMEWORK") or None,
                model_path=(
                    str(payload.get("model_path") or "").strip()
                    or os.environ.get("MODEL_PATH") or None
                ),
                tp=int(os.environ.get("TP") or 0) or None,
                ep=int(os.environ.get("EP") or 0) or None,
                force_full_restart=True,
            )
            ctx.extra = {**(getattr(ctx, "extra", None) or {}),
                         "mn_round_restarted": True}
        except ServerRestartFailed as exc:
            revert_result = _maybe_revert_kernel_patch(apply_result)
            return {
                "status": "failed",
                "error_class": "mn_server_restart_failed_post_patch",
                "error": str(exc),
                "kernel_id": kernel_id,
                "patch_path": patch_path,
                "apply_result": apply_result,
                "revert_result": revert_result,
                "decision": "REVERT",
            }

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
        "extra_server_args": extra_args,
        "apply_result": apply_result,
        "revert_result": revert_result,
    }


# ---------------------------------------------------------------------------
# F1-2 (roofline composite) + main M4 — back-compat alias.
#
# Hyperloom main renamed ``select_kernels_handler`` to
# ``trace_analyze_handler`` (the function does TraceLens analysis +
# kernel selection in a single pass, so the new name is more accurate).
# The M4 merge adopts main's canonical ``trace_analyze_handler`` name;
# the back-compat alias below keeps the ~30 legacy callsites that import
# ``select_kernels_handler`` working unchanged.
select_kernels_handler = trace_analyze_handler

KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "select_kernels":   trace_analyze_handler,
    # ``trace_analyze`` dispatch routes to the same handler as
    # ``select_kernels`` — RooflineExecutor (F1-2) calls the function
    # directly, but explicit dispatch entries keep the action-table
    # symmetric for future PolicyGate / audit code that keys on the
    # request kind.
    "trace_analyze":    trace_analyze_handler,
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
    "trace_analyze_handler",
]
