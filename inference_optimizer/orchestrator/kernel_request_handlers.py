# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator-side handlers for Kernel-agent REQUEST kinds.

A request is served by an LLM responder or a programmatic handler.

Handler signature::

    async def handler(payload: dict, *, session_dir: Path) -> dict:

Dispatch table is exposed via :data:`KERNEL_REQUEST_HANDLERS` for test monkey-patching.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)
_BACKGROUND_ROCPROF_TASKS: set[asyncio.Task[Any]] = set()


# Where the kernel-agent shell tools live; read lazily so cli.py's late env injection wins.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"


def _kernel_agent_root_from_env() -> Path | None:
    raw = os.environ.get(_KERNEL_AGENT_ROOT_ENV)
    if not raw:
        return None
    return Path(raw)


# Backward-compat re-export (NOT used internally; internal logic must use _kernel_agent_root_from_env() so late env injection wins).
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
# Shape sources trusted for kernel-opt dispatch (``torch_trace`` from TraceLens; ``tuning_csv`` reserved for a profiled sweep).
_ALLOWED_SHAPE_PROVENANCE = frozenset({"torch_trace", "tuning_csv"})
def _reusable_source_roots() -> tuple[str, ...]:
    """Framework install roots for patchability checks (from :func:`framework_paths.resolve_patch_target_roots`; emits a lower-case variant per root for case-insensitive matching)."""
    from .framework_paths import resolve_patch_target_roots

    roots = resolve_patch_target_roots()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    # PR #668: FlyDSL kernel checkout(s) for moe_flydsl_* candidates.
    for env_key in ("DSL2_ROOT", "FLYDSL_ROOT"):
        val = (os.environ.get(env_key, "") or "").strip()
        if val:
            cand = (val.rstrip("/") + "/").lower()
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    for default in ("/wekafs/yunkai/flydsl/", "/sgl-workspace/flydsl/"):
        if default not in seen:
            seen.add(default)
            out.append(default)
    return tuple(out)
_APPLY_TOOL_MODULE: Any | None = None
# GEAK FIRST per SKILL.md §"choose_backends" "Default ladder"; Cursor last (dropped when CURSOR_API_KEY is unset).
_DEFAULT_KERNEL_BACKEND_ORDER = ("geak", "claude", "codex", "cursor")
# Soft cap on concurrent kernel-backend coroutines (legacy MI300X 8-GPU fallback; pin with KERNEL_OPT_MAX_PARALLEL).
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
_DEFAULT_OOB_BUDGET_MINUTES = 60.0
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 3 * 60 * 60


@functools.lru_cache(maxsize=1)
def _default_geak_budget_minutes() -> float:
    """Default per-GEAK-attempt budget tracking ``$GEAK_RUN_MODE`` (quick→70, full→130). PR #301: mirrors kernel-agent tool defaults."""
    raw = (os.environ.get("GEAK_RUN_MODE") or "").strip().lower()
    return 70.0 if raw == "quick" else 130.0


@functools.lru_cache(maxsize=1)
def _default_kernel_batch_parallel() -> int:
    """Adaptive batch fanout: ``min(cap, visible_gpus // per_task_gpus)`` (cached)."""
    try:
        import torch  # local import: torch driver init can be expensive
        n_gpus = int(torch.cuda.device_count() or 0)
    except Exception:  # noqa: BLE001 -- torch missing / driver init failure
        return _DEFAULT_KERNEL_BATCH_PARALLEL
    if n_gpus <= 0:
        return _DEFAULT_KERNEL_BATCH_PARALLEL
    try:
        per_task = int(os.environ.get("KERNEL_AGENT_NUM_GPUS", "0") or 0)
    except (TypeError, ValueError):
        per_task = 0
    if per_task <= 0:
        per_task = 1
    return max(1, min(_DEFAULT_KERNEL_BATCH_PARALLEL, n_gpus // per_task))


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
    "FLYDSL_",
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
        return not any(root in lower_file for root in _reusable_source_roots())
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
    # Route through the per-framework env-name source of truth so atom reads ``EXTRA_ATOM_ARGS`` instead of dropping flags via a sglang/vllm default.
    from .action_executors._grid_runner import server_args_env_name
    server_key = server_args_env_name(framework)
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
    if not any(root in lower_file for root in _reusable_source_roots()):
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


def _allow_empty_kernel_shape(payload: dict) -> bool:
    """Escape hatch (default off) via ``payload['allow_empty_kernel_shape']`` or ``HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE=1``."""
    if bool(payload.get("allow_empty_kernel_shape")):
        return True
    return str(
        os.environ.get("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", "")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _validate_kernel_shape_and_paths(
    payload: dict, *, session_dir: Path,
) -> HandlerResult | None:
    """Reject a kernel-opt dispatch with no trace-anchored shape or a missing source/workspace path (would burn budget with no anchor; guides back to ``trace_analyze``)."""
    # ``dry_run`` exercises the plumbing without a backend, so no GPU budget and fake fixture paths need not exist.
    if bool(payload.get("dry_run")):
        return None
    candidate = _load_candidate_metadata(payload)
    kernel_id = str(payload.get("kernel_id") or "")
    name = str(candidate.get("name") or payload.get("kernel_name") or kernel_id)

    shapes = candidate.get("shapes")
    if not isinstance(shapes, list):
        shapes = []
    provenance = str(
        candidate.get("shape_provenance")
        or payload.get("shape_provenance")
        or ""
    ).strip()
    if not shapes and not _allow_empty_kernel_shape(payload):
        return {
            "status": "failed",
            "error_class": "empty_kernel_shape",
            "error": (
                "selected kernel candidate has no trace-anchored shape; "
                "re-run trace_analyze to capture shapes before optimizing "
                "(or pass --allow-empty-kernel-shape to override)"
            ),
            "kernel_id": kernel_id,
            "kernel_name": name,
            "shape_provenance": provenance,
        }
    if provenance and provenance not in _ALLOWED_SHAPE_PROVENANCE:
        return {
            "status": "failed",
            "error_class": "untrusted_shape_provenance",
            "error": (
                f"shape_provenance={provenance!r} is not a trusted source; "
                f"expected one of {sorted(_ALLOWED_SHAPE_PROVENANCE)}"
            ),
            "kernel_id": kernel_id,
            "kernel_name": name,
            "shape_provenance": provenance,
        }

    source_file = str(
        payload.get("source_file") or candidate.get("source_file") or ""
    ).strip()
    if source_file and not Path(source_file).exists():
        return {
            "status": "failed",
            "error_class": "missing_source_path",
            "error": f"kernel source path does not exist: {source_file}",
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
        }
    workspace_path = str(
        payload.get("workspace_path") or session_dir or ""
    ).strip()
    if workspace_path and not Path(workspace_path).exists():
        return {
            "status": "failed",
            "error_class": "missing_workspace_path",
            "error": f"kernel workspace path does not exist: {workspace_path}",
            "kernel_id": kernel_id,
            "kernel_name": name,
        }
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


def _fill_integrate_defaults_from_state(
    payload: dict, *, session_dir: Path,
) -> dict:
    """Pull ``base_tput`` / ``config_path`` / ``extra_server_args`` defaults from SharedState.

    Runs before the ``base_tput > 0`` hard-check in ``integrate_handler`` for
    bare ``{"kernel_id": ...}`` payloads. Always returns a shallow copy; never
    raises on a missing snapshot.
    """
    from .shared_state import SharedState

    resolved = dict(payload)
    state = SharedState.load_or_init(session_dir)

    if float(resolved.get("base_tput", 0.0) or 0.0) <= 0:
        bt = float(getattr(state, "baseline_tput", 0.0) or 0.0)
        if bt > 0:
            resolved["base_tput"] = bt

    if not resolved.get("config_path"):
        cfg = getattr(state, "baseline_config_path", "") or ""
        if cfg:
            resolved["config_path"] = cfg

    # Field renamed ``extra_sglang_args`` -> ``extra_server_args``; read canonical first with a legacy fallback, write canonical.
    current_best = getattr(state, "current_best", None) or {}
    if not resolved.get("extra_server_args") and isinstance(current_best, dict):
        cb_args = (
            current_best.get("extra_server_args")
            or current_best.get("extra_sglang_args")
            or ""
        )
        if cb_args:
            resolved["extra_server_args"] = cb_args

    return resolved


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id (artifact in ``last_kernel_opt``, source in ``last_trace_analyze``)."""
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

    # Multi-KEEP queue fallback: ``last_kernel_opt`` holds only the strongest pending KEEP, so pull patch_path/source_file from the per-kernel ledger for other queued KEEPs.
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


async def _run_subprocess(cmd: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """asyncio-friendly wrapper around blocking subprocess.run (keeps the reactor responsive)."""
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


def _normalize_precision(value: Any) -> str:
    return str(value or "").strip().lower()


def _gemm_tuning_timeout_sec(payload: dict) -> int:
    raw = payload.get("timeout_sec") or os.environ.get(
        "HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC",
        "",
    )
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = _DEFAULT_GEMM_TUNING_TIMEOUT_SEC
    return max(60, value)


def _gemm_tuning_workspace(payload: dict, *, session_dir: Path) -> Path:
    raw = payload.get("workspace_path")
    if raw:
        return Path(raw)
    suffix = str(payload.get("task_id") or payload.get("request_id") or "").strip()
    if not suffix:
        suffix = f"request_{int(time.time())}"
    return Path(session_dir) / "runs" / "gemm_tuning" / suffix


def _write_gemm_tuning_benchmark_script(
    *,
    workspace: Path,
    model_path: str,
    framework: str,
    gpu_type: str,
    tp: int,
    conc: int,
    isl: int,
    osl: int,
) -> Path:
    """Create an isolated benchmark wrapper for GEAK GEMM tuning (distinct port + no global ``pgrep sglang`` cleanup, so it can't kill the main optimizer's server)."""
    inferencex_path = os.environ.get("INFERENCEX_PATH") or "/hyperloom/InferenceX"
    runner = f"{inferencex_path}/benchmarks/{framework}_{gpu_type}.sh"
    path = workspace / "geak_gemm_benchmark.sh"
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
export MODEL={shlex.quote(model_path)}
export TP={int(tp)}
export CONC={int(conc)}
export ISL={int(isl)}
export OSL={int(osl)}
export RANDOM_RANGE_RATIO="${{RANDOM_RANGE_RATIO:-1}}"
export NUM_PROMPTS="${{NUM_PROMPTS:-320}}"
export NUM_WARMUPS="${{NUM_WARMUPS:-8}}"
export RUN_EVAL="${{RUN_EVAL:-false}}"
export RESULT_DIR="${{RESULT_DIR:-$PWD/gemm_benchmark_result}}"
export RESULT_FILENAME="${{RESULT_FILENAME:-bench_serving.json}}"
export PORT="${{PORT:-18888}}"
export PATH="/opt/node20/bin:/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export INFERENCEX_PATH={shlex.quote(inferencex_path)}
mkdir -p "$RESULT_DIR"
cd "$INFERENCEX_PATH"
exec {shlex.quote(runner)}
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


async def run_gemm_tuning_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run GEAK's FP8 block-scale GEMM tuning workflow (separate from ``run_optimization``; tunes GEMM dispatch before source-level rewrites)."""
    from .shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    precision = _normalize_precision(payload.get("precision") or state.precision)
    if precision != "fp8":
        return {
            "status": "skipped",
            "decision": "REVERT",
            "error_class": "fp8_only_action",
            "error": f"GEAK GEMM tuning only applies to FP8 workloads (precision={precision or '(unset)'})",
            "precision": precision,
        }
    framework = str(payload.get("framework") or state.framework or "sglang").strip().lower()
    if framework != "sglang":
        return {
            "status": "skipped",
            "decision": "REVERT",
            "error_class": "unsupported_framework",
            "error": f"GEAK GEMM tuning first version supports SGLang only (framework={framework or '(unset)'})",
            "framework": framework,
            "precision": precision,
        }
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    workspace = _gemm_tuning_workspace(payload, session_dir=session_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    model_path = str(payload.get("model_path") or state.model_path or os.environ.get("MODEL_PATH") or "").strip()
    if not model_path:
        return {"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"}
    tp = int(payload.get("tp") or state.tp or os.environ.get("TP") or 1)
    conc = int(payload.get("conc") or state.conc or os.environ.get("CONC") or 0)
    isl = int(payload.get("isl") or state.isl or os.environ.get("ISL") or 0)
    osl = int(payload.get("osl") or state.osl or os.environ.get("OSL") or 0)
    gpu_type = str(payload.get("gpu_type") or state.gpu_type or os.environ.get("GPU_TYPE") or "").strip().lower()
    benchmark_script = str(
        payload.get("benchmark_script")
        or os.environ.get("GEAK_GEMM_BENCHMARK_SCRIPT")
        or ""
    ).strip()
    if not benchmark_script:
        if not gpu_type:
            gpu_type = "mi355x"
        benchmark_script = str(_write_gemm_tuning_benchmark_script(
            workspace=workspace,
            model_path=model_path,
            framework=framework,
            gpu_type=gpu_type,
            tp=tp,
            conc=conc,
            isl=isl,
            osl=osl,
        ))
    geak_config = str(payload.get("config") or os.environ.get("GEAK_CONFIG") or "").strip()
    baseline_tput = payload.get("baseline_tput")
    if baseline_tput is None:
        baseline_tput = state.baseline_tput

    input_json = workspace / "gemm_tuning_input.json"
    input_payload = {
        "cwd": str(workspace),
        "model_path": model_path,
        "benchmark_script": benchmark_script,
        "framework": framework,
        "precision": precision,
        "gpu_type": gpu_type,
        "tp": tp,
        "conc": conc,
        "isl": isl,
        "osl": osl,
        "baseline_tput": float(baseline_tput or 0.0),
    }
    if geak_config:
        input_payload["config"] = geak_config
    if payload.get("dry_run"):
        input_payload["dry_run"] = True
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("gemm_tuning.py")),
        "--input-json", str(input_json),
    ]

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=_gemm_tuning_timeout_sec(payload))
    result = _shape_tool_result(rc, stdout, stderr)
    result.setdefault("workspace", str(workspace))
    result.setdefault("precision", precision)
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)
    result.setdefault("benchmark_script", benchmark_script)
    return result


async def trace_analyze_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir.

    Required payload:
        trace_input: path to a torch_trace dir or single .trace.json.gz file.

    Returns ``{status, hot_kernels, trace_report_path (analysis.md), cli_log_path, details}``.
    """
    trace_input = payload.get("trace_input") or payload.get("trace_dir")
    if not trace_input:
        return {"status": "failed", "error": "missing 'trace_input' in payload"}
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    # Pass the session root so artefacts settle at ``<session_dir>/kernel-agent/runs/...`` (the suffix is hardcoded in the tool).
    workspace_path = payload.get("workspace_path") or str(session_dir)
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    # Backfill workload context from SharedState so the tool gets the right framework/platform/model/analysis_mode when Orchestration omits them.
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

    # Load materialized baseline workload metadata once: feeds splitter CLI flags (--split-*) so the steady-state window is correct, and enriches hot_kernels downstream.
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

    # Splitter workload hints. Priority: payload override > baseline metadata > drop the flag (tool keeps its env fallback). Missing hints can cause trace_split_no_steady_state.
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
    # Forward TraceLens splitter steady-state mode (mixed/decode_only/prefilldecode) via payload or env, so the coordinator can re-issue after a steady_state_chunk warning.
    steady_state_mode = (
        payload.get("steady_state_mode")
        or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
    )
    steady_state_mode = str(steady_state_mode).strip()
    if steady_state_mode:
        cmd += ["--steady-state-mode", steady_state_mode]
    # ``--roofline-json`` CLI param retired with the ``pmc_roofline`` action; a stale payload key is silently ignored.
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    result = _shape_tool_result(rc, stdout, stderr)
    artifacts = result.get("artifact_paths") if isinstance(result, dict) else None
    if isinstance(artifacts, dict) and artifacts.get("kernel_candidates"):
        result["candidates_path"] = artifacts["kernel_candidates"]
    # Surface analysis.md path at the handler boundary so the Coordinator forwards it to GEAK without digging through artifact_paths.
    if isinstance(result, dict):
        report_path = result.get("trace_report_path")
        if not report_path and isinstance(artifacts, dict):
            report_path = artifacts.get("trace_report_path")
        if report_path:
            result["trace_report_path"] = str(report_path)
            _enrich_candidate_trace_report(
                result.get("hot_kernels"), str(report_path),
            )
        # Surface tracelens/summary.json — the per-run audit sidecar of reusable vs skipped kernels.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
        if isinstance(artifacts, dict) and artifacts.get("kernel_roofline"):
            result["kernel_roofline_path"] = str(artifacts["kernel_roofline"])

        # A failed TraceLens run is a hard failure, not "empty candidates"; keep status=failed and attach a structured warning.
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

        # Guarantee ``trace_health_warnings`` is always a list (empty = nothing wrong).
        result.setdefault("trace_health_warnings", [])

        _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
        candidates_path = result.get("candidates_path")
        if isinstance(candidates_path, str):
            _enrich_candidates_artifact(
                candidates_path,
                metadata,
                trace_report_path=str(report_path or ""),
            )
    return result


def _validate_trace_analyze_inputs(
    payload: dict, *, session_dir: Path,
) -> HandlerResult | None:
    """Confirm the run_optimization payload references a valid trace_analyze."""
    candidates_path = str(payload.get("candidates_path") or "").strip()
    if candidates_path and not Path(candidates_path).exists():
        return {
            "status": "failed",
            "error_class": "missing_candidates_artifact",
            "error": (
                "run_optimization requires a candidates_path that exists "
                "on disk; re-run trace_analyze to regenerate it"
            ),
            "candidates_path": candidates_path,
        }
    if candidates_path:
        return None
    if (
        payload.get("dry_run")
        or payload.get("source_file")
        or isinstance(payload.get("candidate"), dict)
    ):
        return None
    try:
        from .shared_state import SharedState
        state = SharedState.load_or_init(session_dir)
    except Exception:  # noqa: BLE001 — best-effort read
        return None
    last = state.last_trace_analyze or {}
    cached = str(last.get("candidates_path") or "").strip()
    if not cached:
        return {
            "status": "failed",
            "error_class": "missing_trace_analyze",
            "error": (
                "run_optimization requires a prior trace_analyze: the "
                "payload supplied no candidates_path / source_file / "
                "candidate, and SharedState has no cached "
                "last_trace_analyze.candidates_path. Issue request "
                "kind='trace_analyze' first."
            ),
        }
    return None


async def run_optimization_handler(
    payload: dict,
    *,
    session_dir: Path,
    record_partial: Callable[[dict], None] | None = None,
) -> HandlerResult:
    """Run kernel optimization.

    With candidate metadata, upgrades single-kernel requests into a concurrent
    batch over all reusable native kernels. ``record_partial`` (optional) streams
    each batch sub-result into SharedState before gather wait-all returns.
    """
    data_guard = _validate_trace_analyze_inputs(payload, session_dir=session_dir)
    if data_guard is not None:
        return data_guard
    if payload.get("_single_kernel"):
        return await _run_optimization_single(payload, session_dir=session_dir)
    candidates = _batch_kernel_candidates(payload, session_dir=session_dir)
    if len(candidates) <= 1:
        single_payload = dict(payload)
        if candidates:
            # Reconcile the (often hallucinated) LLM kernel_id against the real candidate id so the CLI doesn't KeyError.
            single_payload["kernel_id"] = _reconcile_kernel_id(
                single_payload.get("kernel_id"), candidates,
            )
        else:
            # No routable hot candidate: canonicalize an aliased id against the full set so the rejection lands on the real k00x, not a hallucinated alias.
            canon = _resolve_candidate_id(
                single_payload.get("kernel_id"),
                _all_kernel_candidates(payload),
            )
            if canon:
                single_payload["kernel_id"] = canon
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
        or os.environ.get("HYPERLOOM_GEAK_BUDGET_MIN")
        or _default_geak_budget_minutes()
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
        # Ignore legacy payload["backends"]; the default ladder (GEAK first) mirrors ``kernel_optimization.choose_backends`` so single/batch agree.
        order = list(_DEFAULT_KERNEL_BACKEND_ORDER)
        explicit = False
    allowed = {"claude", "codex", "cursor", "geak"}
    selected = [backend for backend in order if backend in allowed]
    # Drop cursor from the auto-derived ladder when CURSOR_API_KEY is unset (explicit order still wins).
    if not explicit and not os.environ.get("CURSOR_API_KEY", "").strip():
        selected = [b for b in selected if b != "cursor"]
    return selected


def _in_flight_kernel_ids(session_dir: Path) -> set[str]:
    """Scan the kernel-agent run dir for ``state=running`` status files, so :func:`_batch_kernel_candidates` skips kernels still in flight from a prior batch."""
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


def _normalize_kernel_id(value: str) -> str:
    """Fold hallucinated ``kn``/``rn`` prefixes onto the real ``k`` numbering (mirrors ``kernel_optimization._normalize_kernel_id``)."""
    s = str(value or "").strip().lower()
    for prefix in ("kn", "rn"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return "k" + s[len(prefix):]
    return s


def _reconcile_kernel_id(
    requested: Any, candidates: list[dict[str, Any]],
) -> str:
    """Resolve the LLM kernel_id to a real candidate id (exact kernel_id/name, then normalized; only a missing id falls back to the first candidate)."""
    req = str(requested or "")
    if req:
        for cand in candidates:
            cid = str(cand.get("kernel_id") or "")
            if cid == req or str(cand.get("name") or "") == req:
                return cid or req
        target = _normalize_kernel_id(req)
        for cand in candidates:
            cid = str(cand.get("kernel_id") or "")
            if _normalize_kernel_id(cid) == target:
                return cid
        log.warning(
            "kernel_id %r did not match any candidate %s; leaving unchanged",
            req, [str(c.get("kernel_id") or "") for c in candidates],
        )
        return req
    fallback = str(candidates[0].get("kernel_id") or "")
    return fallback


def _resolve_candidate_id(
    requested: Any, candidates: list[dict[str, Any]],
) -> str:
    """Return the canonical ``k00x`` id for ``requested`` or ``""`` (like ``find_candidate`` but with no first-candidate fallback; a pure hallucination returns ``""``)."""
    req = str(requested or "")
    if not req:
        return ""
    for cand in candidates:
        if str(cand.get("kernel_id") or "") == req:
            return req
    name_matches = [
        cand
        for cand in candidates
        if str(cand.get("name") or "") == req
        and cand.get("reusable_native_kernel") is not False
        and cand.get("source_file")
    ]
    if len(name_matches) == 1:
        return str(name_matches[0].get("kernel_id") or "")
    target = _normalize_kernel_id(req)
    for cand in candidates:
        if _normalize_kernel_id(str(cand.get("kernel_id") or "")) == target:
            return str(cand.get("kernel_id") or "")
    return ""


def _all_kernel_candidates(payload: dict) -> list[dict[str, Any]]:
    """Load every candidate (``hot_kernels`` ∪ ``skipped_kernels``) so id canonicalization resolves even when hot_kernels is empty."""
    candidates_path = payload.get("candidates_path")
    if not candidates_path:
        return []
    try:
        data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for key in ("hot_kernels", "kernel_candidates", "skipped_kernels"):
        value = data.get(key)
        if isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    return out


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

    # Build the "live" exclusion sets up front for both passes (empty without session_dir).
    rejected_kernel_ids: set[str] = set()
    attempts_by_kid: dict[str, dict] = {}
    in_flight: set[str] = set()
    max_attempts = 1
    try:
        max_attempts = max(1, int(os.environ.get(
            "INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_ATTEMPTS", "1",
        )))
    except (TypeError, ValueError):
        max_attempts = 1
    # min_gpu_pct must mirror SharedState.untried_hot_reusable_kernels' 3.0 default so the two layers agree and tiny kernels don't eat ladder wall-clock.
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
        """A kernel_id is live (batch-eligible) iff NOT rejected, NOT in-flight, and < max_attempts recorded against the CURRENT source_file (PR-K per-source counting)."""
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

    # Collapse kernels sharing a source function into one dispatch via ``task_groups[]`` (keyed off ``primary_kernel_id``); unparseable kernels fall through to the legacy per-kernel pass.
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
        # Mark all members so the legacy loop never re-picks them.
        grouped_kernel_ids.update(member_ids)
        # Only reusable_native members survive; fall back to the next live reusable member when the primary is rejected, else skip the whole group.
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
                # Every reusable member exhausted -> nothing to dispatch this round.
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
        # Shallow copy + attach group so the subprocess sees ``candidate["task_group"]``.
        item = dict(primary_cand)
        item["task_group"] = group
        selected.append(item)

    # Legacy per-kernel pass for reusable kernels not absorbed into a task_group.
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
        # Prefer a KEEP verdict over a higher-micro non-KEEP (GEAK's gate-less NEEDS_REVIEW shouldn't beat a real Claude/Codex KEEP); mirrors the batch handler's max-key.
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
    # Preserve source_file on the aggregated best so the streaming callback can group by file without re-reading the candidates artifact.
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
    """Fan ``run_optimization`` out across reusable native kernels (``record_partial`` streams each sub-attempt into SharedState before gather wait-all unblocks)."""
    max_parallel = int(
        payload.get("max_parallel")
        or os.environ.get("KERNEL_OPT_MAX_PARALLEL")
        or _default_kernel_batch_parallel()
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
                # Wrap a sub-task failure as a structured result so gather stays wait-all (a raised exception would unblock mid-batch and collide with running siblings on the GPU).
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
        # Re-stamp source_file onto the sub-result so the same-source-file conflict guard can detect two KEEPs on one file (defensive; the sequence already preserves it).
        if isinstance(result, dict) and not result.get("source_file") and cand_src:
            result["source_file"] = cand_src
        if record_partial is not None:
            try:
                record_partial(result)
            except Exception:  # noqa: BLE001
                # Callback failure must not abort the batch; the post-gather record path recovers the lost streaming write.
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

    Required payload: ``kernel_id``. Returns the tool's JSON output verbatim.
    """
    kernel_id = payload.get("kernel_id")
    if not kernel_id:
        return {"status": "failed", "error": "missing 'kernel_id' in payload"}
    guard = _validate_reusable_native_kernel(payload)
    if guard is not None:
        return guard
    shape_guard = _validate_kernel_shape_and_paths(
        payload, session_dir=session_dir,
    )
    if shape_guard is not None:
        return shape_guard
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    # Pass the session root (same convention as trace_analyze_handler) so artefacts land under ``<session_dir>/kernel-agent/runs/...``.
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
    extra_args = str(
        payload.get("extra_server_args")
        or payload.get("extra_sglang_args")
        or ""
    ).strip()
    if extra_args:
        cmd += ["--extra-sglang-args", extra_args]
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
    if payload.get("test_command"):
        cmd += ["--test-command", str(payload["test_command"])]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    geak_budget_min = _geak_budget_minutes(payload)
    backend = str(payload.get("backends") or "").strip().lower()
    if backend == "geak" or not backend:
        cmd += ["--geak-budget-min", str(geak_budget_min)]
    if payload.get("budget_minutes") is not None:
        cmd += ["--budget-minutes", str(payload["budget_minutes"])]
    # Allow the tool to handle its own backend timeout and salvage partial artifacts.
    timeout_sec = _optimization_wrapper_timeout_sec(payload)

    from .action_executors._multi_node_env import is_multi_node

    if is_multi_node():
        from inference_optimizer.multi_node.cli import (
            kill_inference_for_kernel_agent_best_effort,
        )

        await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    result = _shape_tool_result(rc, stdout, stderr)
    # Stamp source_file / kernel_id from the payload onto the result so the multi-KEEP integrate queue can group same-file KEEPs (the tool may omit them on timeout/crash).
    if isinstance(result, dict):
        if not result.get("kernel_id") and payload.get("kernel_id"):
            result["kernel_id"] = str(payload["kernel_id"])
        if not result.get("source_file") and payload.get("source_file"):
            result["source_file"] = str(payload["source_file"])
    return result


def _shape_tool_result(rc: int, stdout: str, stderr: str) -> HandlerResult:
    """Wrap a kernel-agent tool's exit + stdout into our schema (prefer the tool's own JSON, synthesize only on parse failure)."""
    parsed = _parse_tool_stdout(stdout)
    if parsed:
        # Trust the tool's own status; otherwise infer from rc.
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
def _lookup_kernel_roofline_name(session_dir: Path, kernel_id: str) -> str:
    """Resolve the TraceLens/device kernel name for a roofline sidecar row."""
    sidecar_path = session_dir / "reports" / "kernel_roofline.json"
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows = payload.get("kernels") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if isinstance(row, dict) and str(row.get("kernel_id") or "") == str(kernel_id):
            return str(row.get("name") or row.get("matched_kernel_name") or "").strip()
    return ""


def _record_after_kernel_opt_rocprof_status(
    *,
    session_dir: Path,
    kernel_id: str,
    status: str,
    reason: str = "",
    json_path: str = "",
    txt_path: str = "",
    log: Any = None,
) -> None:
    """Best-effort sidecar status update for skipped/failed after-opt rocprof."""
    try:
        ko_tool = _kernel_agent_tool_path("kernel_optimization.py")
        ko_dir = ko_tool.parent
        import sys as _sys
        if str(ko_dir) not in _sys.path:
            _sys.path.insert(0, str(ko_dir))
        from kernel_optimization import _update_kernel_roofline_sidecar  # type: ignore[import-not-found]  # noqa: PLC0415
        _update_kernel_roofline_sidecar(
            workspace_path=str(session_dir),
            kernel_id=kernel_id,
            rocprof_json_path=json_path,
            rocprof_txt_path=txt_path,
            log_path=None,
            rocprof_status=status,
            rocprof_reason=reason,
            phase="after_kernel_opt",
        )
    except Exception as exc:
        if log is not None:
            log.warning("integrate: after_kernel_opt sidecar status update failed: %s", exc)


def _rocprof_timeout_sec() -> int:
    try:
        return max(60, int(os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800")))
    except (TypeError, ValueError):
        return 1800


def _rocprof_profile_command(test_command: str) -> str:
    if "--correctness" not in test_command:
        return test_command
    if "/unittest/harness_" not in test_command and " harness_" not in test_command:
        return test_command
    return test_command.replace("--correctness", "--profile", 1)


async def _run_after_kernel_opt_rocprof(
    *,
    kernel_id: str,
    session_dir: Path,
    log: Any,
) -> dict[str, Any]:
    """Best-effort: after an integrate KEEP, run rocprof on the now-patched kernel.

    Resolves ``test_command`` from ``SharedState.kernel_opt_attempts`` or
    ``last_kernel_opt``, launches ``rocprof_roofline.py`` as a subprocess, and
    calls ``_update_kernel_roofline_sidecar`` with ``phase='after_kernel_opt'``.

    Always returns a small status dict; never raises.
    """
    rocprof_env = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE", "1").strip().lower()
    if rocprof_env in {"0", "false", "no", "off"}:
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="skipped",
            reason="disabled_by_env",
            log=log,
        )
        return {"status": "skipped", "reason": "disabled_by_env"}

    try:
        from .shared_state import SharedState
        state = SharedState.load_or_init(session_dir)
        attempt = (state.kernel_opt_attempts or {}).get(kernel_id) or {}
        test_command = str(attempt.get("test_command") or "").strip()
        if not test_command:
            lko = state.last_kernel_opt or {}
            if str(lko.get("kernel_id") or "") == kernel_id:
                bp = lko.get("backend_paths") or {}
                test_command = str(bp.get("test_command") or "").strip()
        # Derive workdir from last_source_file (mirrors before-opt logic); fall back to session_dir.
        run_workdir: Path = session_dir
        source_file = str(attempt.get("last_source_file") or "").strip()
        if source_file:
            sf = Path(source_file)
            if sf.is_file():
                run_workdir = sf.parent
            elif sf.is_dir():
                run_workdir = sf
    except Exception as exc:
        reason = f"state_load_error: {type(exc).__name__}"
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="skipped",
            reason=reason,
            log=log,
        )
        return {"status": "skipped", "reason": reason}

    if not test_command:
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="skipped",
            reason="no_test_command_in_state",
            log=log,
        )
        return {"status": "skipped", "reason": "no_test_command_in_state"}

    try:
        tool = _kernel_agent_tool_path("rocprof_roofline.py")
    except Exception:
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="skipped",
            reason="rocprof_roofline_tool_unavailable",
            log=log,
        )
        return {"status": "skipped", "reason": "rocprof_roofline_tool_unavailable"}

    out_dir = session_dir / "kernel-agent" / "rocprof_after_kernel_opt" / kernel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "after.json"
    out_txt = out_dir / "after.txt"
    timeout_sec = _rocprof_timeout_sec()
    profiling_command = _rocprof_profile_command(test_command)

    cmd = [
        "python3", str(tool),
        "--workdir", str(run_workdir),
        "--cmd", profiling_command,
        "--out-json", str(out_json),
        "--out-txt", str(out_txt),
        "--timeout-sec", str(timeout_sec),
    ]
    target_kernel = _lookup_kernel_roofline_name(session_dir, kernel_id)
    if target_kernel:
        cmd.extend(["--target-kernel", target_kernel])
    log.info("integrate: running after_kernel_opt rocprof for %s", kernel_id)
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec + 30)
    except Exception as exc:
        log.warning("integrate: after_kernel_opt rocprof subprocess error: %s", exc)
        reason = f"{type(exc).__name__}: {exc}"
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="failed",
            reason=reason,
            json_path=str(out_json),
            txt_path=str(out_txt),
            log=log,
        )
        return {"status": "failed", "reason": reason}

    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    status = "ok" if rc == 0 and payload.get("status") == "ok" else payload.get("status", "failed")
    log.info("integrate: after_kernel_opt rocprof status=%s for %s", status, kernel_id)

    # Mirror into reports/kernel_roofline.json
    try:
        ko_tool = _kernel_agent_tool_path("kernel_optimization.py")
        ko_dir = ko_tool.parent
        import sys as _sys
        if str(ko_dir) not in _sys.path:
            _sys.path.insert(0, str(ko_dir))
        from kernel_optimization import _update_kernel_roofline_sidecar  # type: ignore[import-not-found] # noqa: PLC0415
        _update_kernel_roofline_sidecar(
            workspace_path=str(session_dir),
            kernel_id=kernel_id,
            rocprof_json_path=str(out_json),
            rocprof_txt_path=str(out_txt),
            log_path=None,
            rocprof_status=status,
            phase="after_kernel_opt",
        )
    except Exception as exc:
        log.warning("integrate: after_kernel_opt sidecar update failed: %s", exc)

    return {
        "status": status,
        "json_path": str(out_json),
        "txt_path": str(out_txt),
    }


def _schedule_after_kernel_opt_rocprof(
    *,
    kernel_id: str,
    session_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    rocprof_env = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE", "1").strip().lower()
    if rocprof_env in {"0", "false", "no", "off"}:
        _record_after_kernel_opt_rocprof_status(
            session_dir=session_dir,
            kernel_id=kernel_id,
            status="skipped",
            reason="disabled_by_env",
            log=log,
        )
        return {"status": "skipped", "reason": "disabled_by_env"}

    _record_after_kernel_opt_rocprof_status(
        session_dir=session_dir,
        kernel_id=kernel_id,
        status="scheduled",
        reason="background_task",
        log=log,
    )
    task = asyncio.create_task(_run_after_kernel_opt_rocprof(
        kernel_id=kernel_id,
        session_dir=session_dir,
        log=log,
    ))
    _BACKGROUND_ROCPROF_TASKS.add(task)

    def _done(done_task: asyncio.Task[Any]) -> None:
        _BACKGROUND_ROCPROF_TASKS.discard(done_task)
        try:
            done_task.result()
        except Exception as exc:  # noqa: BLE001 — best-effort background task
            log.warning("integrate: after_kernel_opt rocprof background failed: %s", exc)

    task.add_done_callback(_done)
    return {"status": "scheduled", "reason": "background_task"}
async def integrate_handler(
    payload: dict, *, session_dir: Path,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Applies an optimized kernel artifact, re-runs the active Magpie baseline,
    and KEEPs only when measured E2E throughput clears the threshold (source +
    artifacts are backed up first so non-KEEP can restore without a rebuild).

    Required payload: ``base_tput``. Optional: patch_path, target_file,
    kernel_id, config_path, extra_server_args, keep_threshold_pct (1.0),
    budget_minutes (20). Returns ``{status, decision, base_tput, new_tput,
    gain_pct, kernel_id, patch_path, report_path, workspace}``.
    """
    from .action_executors.baseline import BaselineExecutor
    from .action_executors.benchmark_result import is_valid_measurement
    from .shared_state import SharedState
    from .sub_agent_runner import RunnerContext
    from .task_registry import Task

    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare {kernel_id} payload isn't failed with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)

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
    # Route through the compat helper so a legacy ``extra_sglang_args`` envelope still resolves.
    from ..compat.payload_aliases import read_extra_server_args
    extra_args = read_extra_server_args(payload).strip()

    # Wrap BaselineExecutor in a Task/RunnerContext; extra_server_args goes via task params (forward compat).
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

    # GH #458: aiter cpp_itfs / runtime-compiled kernels recompile at runtime (cache dir hashes params not source, so pristine+patched collide). Set AITER_REBUILD=1 for the re-baseline server so aiter wipes its BUILD_DIR and recompiles the patched kernel. Scoped to cpp_itfs applies and ALWAYS restored.
    cpp_itfs_backup = apply_result.get("cpp_itfs_cache_backup") or {}
    force_aiter_rebuild = bool(cpp_itfs_backup.get("is_cpp_itfs"))
    _prev_aiter_rebuild = os.environ.get("AITER_REBUILD")
    if force_aiter_rebuild:
        os.environ["AITER_REBUILD"] = "1"

    def _restore_aiter_rebuild_env() -> None:
        if not force_aiter_rebuild:
            return
        if _prev_aiter_rebuild is None:
            os.environ.pop("AITER_REBUILD", None)
        else:
            os.environ["AITER_REBUILD"] = _prev_aiter_rebuild

    # Multi-node: force a FULL sglang restart so it re-imports the patched modules (a resume would measure the pre-patch process); ctx.extra["mn_round_restarted"] stops a double restart in BaselineExecutor, force_full_restart scopes MULTI_NODE_RESTART_RESUME_RUNNING=0 to this call only.
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
            _restore_aiter_rebuild_env()
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
    finally:
        # Restore AITER_REBUILD on every path once the re-baseline server has
        # been launched, so the env override never leaks past this integrate.
        _restore_aiter_rebuild_env()

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

    # GH #458 (point 2): don't score a stale binary. For cpp_itfs targets the
    # served kernel is runtime-compiled, so a re-baseline that reused a stale
    # params-hashed lib.so would silently measure the PRE-patch kernel (the
    # observed -0.17% on a real +2.5% paged_attention win). Before trusting
    # gain_pct, assert a real rebuild landed: apply moved the cache aside, so
    # a fresh <build_dir>/<md_name>_*/lib.so newer than the invalidation is
    # proof the patched kernel was (re)compiled and served. If not, flag for
    # review instead of emitting a KEEP/REVERT on a possibly-stale measure.
    #
    # Single-node only: in multi-node the served cache lives on the serving
    # pod, not this sandbox, so AITER_REBUILD=1 on the pod restart is the
    # mechanism and the sandbox-local check is skipped to avoid false aborts.
    # verify_cpp_itfs_rebuilt() returns verified=True for non-cpp_itfs targets
    # so this gate is a strict no-op off the cpp_itfs path.
    rebuild_check: HandlerResult = {"verified": True, "status": "skipped"}
    if force_aiter_rebuild and not is_multi_node():
        rebuild_check = _load_apply_tool().verify_cpp_itfs_rebuilt(cpp_itfs_backup)
        if not rebuild_check.get("verified", True):
            revert_result = _maybe_revert_kernel_patch(apply_result)
            return {
                "status": "failed",
                "error_class": "cpp_itfs_rebuild_not_verified",
                "error": (
                    "re-baseline did not produce a freshly-built cpp_itfs "
                    "lib.so; refusing to score a possibly-stale binary"
                ),
                "decision": "NEEDS_REVIEW",
                "kernel_id": kernel_id,
                "patch_path": patch_path,
                "target_file": payload.get("target_file") or payload.get("source_file"),
                "apply_result": apply_result,
                "revert_result": revert_result,
                "rebuild_check": rebuild_check,
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

    # After KEEP, schedule rocprof so integrate returns without waiting
    # up to the profiling timeout.
    rocprof_after_info: dict[str, Any] = {}
    if decision == "KEEP" and kernel_id:
        rocprof_after_info = _schedule_after_kernel_opt_rocprof(
            kernel_id=kernel_id,
            session_dir=session_dir,
            log=log,
        )

    result: dict[str, Any] = {
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
        "rebuild_check": rebuild_check,
    }
    if rocprof_after_info:
        result["rocprof_after_kernel_opt"] = rocprof_after_info
    return result


# Kernel-agent programmatic dispatch table (LLM-driven requests routed via ``Coordinator._handle_request``).
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "trace_analyze":    trace_analyze_handler,
    "run_gemm_tuning":  run_gemm_tuning_handler,
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
    "run_gemm_tuning_handler",
    "run_optimization_handler",
    "trace_analyze_handler",
]
