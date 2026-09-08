# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator-side programmatic handlers for kernel REQUEST kinds."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from hyperloom.agents.kernel.tools._capture_shapes import (
    is_capture_fragment as _shared_is_capture_fragment,
)
from hyperloom.common import codex_session, llm_config
from hyperloom.common.env import env_bool, forge_explicitly_enabled, is_truthy
from hyperloom.common.git_safety import safe_directory_args
from hyperloom.common.io import append_jsonl
from hyperloom.orchestrator.roles.agent_role import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)

from ..actions.stop_attribution import stopped_by_the_run_class
from .lane_budget import (
    LANE_FUSION,
    LANE_GEMM,
    allocate as _allocate_lane_budgets,
    gemm_per_tuner_timeout_sec,
)
from .patch_landing import bundle_belongs_to
from .patch_lifecycle import cleanup_verdict as _cleanup_verdict
from ..trace.task_progress import heartbeat_while_output_flows

from ._recorder_trace import trace_recording_skipped

# Re-exported: callers patch these at ``request_handlers.<name>``.
from ._kernel_decisions import (
    _honest_flag as _honest_flag,
    _entry_by_kernel_id as _entry_by_kernel_id,
    index_attempts_by_kernel_id as index_attempts_by_kernel_id,
    _resolve_kernel_patch_identity as _resolve_kernel_patch_identity,
    kernel_patch_key as kernel_patch_key,
    find_rejected_kernel_patch as find_rejected_kernel_patch,
    record_kernel_integrate_result as record_kernel_integrate_result,
    record_gemm_tuning as record_gemm_tuning,
    _kernel_ids_in_optimization_stack as _kernel_ids_in_optimization_stack,
    _source_files_in_optimization_stack as _source_files_in_optimization_stack,
    _kernel_ids_with_integrate_attempts as _kernel_ids_with_integrate_attempts,
    integrate_attempt_count_for_kernel as integrate_attempt_count_for_kernel,
    _kernel_trace_impact_pct as _kernel_trace_impact_pct,
    next_pending_keep_kernel_id as next_pending_keep_kernel_id,
    pending_keep_kernel_ids as pending_keep_kernel_ids,
    has_keep_pending_integrate as has_keep_pending_integrate,
    kernel_opt_attempts_count as kernel_opt_attempts_count,
    untried_hot_reusable_kernels as untried_hot_reusable_kernels,
    is_collective_candidate as is_collective_candidate,
    SUPPORTED_COLLECTIVE_OPS as SUPPORTED_COLLECTIVE_OPS,
    enqueue_nominated_patch as enqueue_nominated_patch,
)
from .nomination_result import parse_outcome as parse_outcome


log = logging.getLogger(__name__)

# Recognized trace-analysis routes.
_VALID_ANALYSIS_ROUTES = frozenset({"bypass", "agent"})
STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT = 1.0
# A patch whose correctness was only established against a reference kernel; serving accuracy is what settles it.
_FRAMEWORK_APPLYBACK_ARTIFACT_KIND = "framework_applyback"
_INTEGRATE_ACCURACY_VALIDATION_TIER = "integrate_e2e_accuracy"

# Mirrors the completion ceiling the inferencex eval shim installs; kept in sync so the feasibility check reasons
# about the budget the eval will really ask for.
_EVAL_DEFAULT_MAX_TOKENS = 4096


def _vram_guarded_server_args(extra_args: str) -> str:
    """Optionally cap ``--gpu-memory-utilization`` for the integrate re-baseline."""
    if not _honest_flag("HL_INTEGRATE_VRAM_GUARD"):
        return extra_args
    # ``--gpu-memory-utilization`` is vLLM-only; apply the cap only for vLLM.
    framework = (os.environ.get("FRAMEWORK") or "").strip().lower()
    if framework != "vllm":
        return extra_args
    if "gpu-memory-utilization" in (extra_args or ""):
        return extra_args
    try:
        cap = float(os.environ.get("HL_INTEGRATE_VRAM_UTIL_CAP", "0.90") or 0.90)
    except (TypeError, ValueError):
        cap = 0.90
    cap = min(max(cap, 0.1), 0.99)
    addition = f"--gpu-memory-utilization {cap:g}"
    return f"{extra_args} {addition}".strip() if extra_args else addition


def _confirm_source_imported(source_file: str, workspace: str | Path | None) -> bool | None:
    """Best-effort confirm the patched source was actually imported/compiled."""
    if not source_file or not workspace:
        return None
    ws = Path(workspace)
    logs = [p for p in (ws / "server.log", ws.parent / "server.log") if p.exists()]
    if not logs:
        try:
            logs = sorted(ws.rglob("server.log"))[:1]
        except Exception:
            logs = []
    if not logs:
        return None
    stem = Path(source_file).stem
    if not stem:
        return None
    try:
        text = logs[0].read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if stem not in text:
        return False
    # Confirmed only when the basename co-occurs with an import/compile cue.
    for line in text.splitlines():
        if stem in line and re.search(r"import|load|compil|build|\.py", line, re.IGNORECASE):
            return True
    # Present but not in an obvious import context.
    return None


def _confirm_sources_imported(
    source_files: list[str],
    workspace: str | Path | None,
) -> tuple[bool | None, dict[str, bool | None]]:
    """Confirm every file a patch wrote was exercised by the served process."""
    ordered = list(dict.fromkeys(path for path in source_files if str(path or "").strip()))
    if not ordered:
        return None, {}
    per_file = {path: _confirm_source_imported(path, workspace) for path in ordered}
    verdicts = list(per_file.values())
    if all(verdict is True for verdict in verdicts):
        return True, per_file
    if all(verdict is False for verdict in verdicts):
        return False, per_file
    return None, per_file


# Backends whose stdout log we mine for token usage.
_TOKEN_TRACED_KERNEL_BACKENDS: frozenset[str] = frozenset({"forge"})


# Kernel-agent shell tools root; read lazily so late env injection wins.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"


def _kernel_agent_root_from_env() -> Path | None:
    """Read the kernel-agent install root from the environment at call time."""
    raw = os.environ.get(_KERNEL_AGENT_ROOT_ENV)
    if not raw:
        return None
    return Path(raw)


HandlerResult = dict[str, Any]
HandlerFn = Callable[..., Awaitable[HandlerResult]]

_RUNTIME_GENERATED_SOURCE_MARKERS = (  # nosec B108 - marker strings, not filesystem writes.
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


def _reusable_source_roots() -> tuple[str, ...]:
    """Framework install roots for the runtime-generated kernel classifier."""
    from ..framework.paths import resolve_patch_target_roots

    roots = resolve_patch_target_roots()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return tuple(out)


_APPLY_TOOL_MODULE: Any | None = None
# forge is the only per-kernel backend.
_DEFAULT_KERNEL_PHASE_BACKEND_ORDER = ("geak",)
# Soft cap on concurrent kernel-backend coroutines (pin with KERNEL_OPT_MAX_PARALLEL).
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
# forge-loop holds back a finalize reserve of half this window, so the figure here buys only half as much search as it
# reads.
_DEFAULT_BACKEND_BUDGET_MINUTES = 90.0
# Outer subprocess cap for the whole GEMM-tuning run (all shapes/tuners); sized for large models with many GEMM
# shapes.
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 5 * 60 * 60
_FORGE_FUSION_WRAPPER_TIMEOUT_GRACE_SEC = 30


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
    """Validate that the kernel-agent install root is configured and present."""
    root = _kernel_agent_root_from_env()
    if root is None:
        return (
            f"{_KERNEL_AGENT_ROOT_ENV} is not set; run "
            "src/hyperloom/inference_optimizer/assets/install.sh and source $KERNEL_AGENT_ENV "
            "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
        )
    if not root.is_dir():
        return f"{_KERNEL_AGENT_ROOT_ENV} does not exist: {root}"
    return None


def _resolve_tracelens_root() -> Path:
    """Resolve the TraceLens checkout, independent of inherited env."""
    from hyperloom.inference_optimizer.session import paths

    return paths.tracelens_root()


def _tracelens_root_error(root: Path) -> str | None:
    """Validate that the resolved TraceLens root is a usable git checkout."""
    if not root.is_dir():
        return (
            f"TraceLens root not found: {root}; run "
            "src/hyperloom/agents/kernel/scripts/install.sh "
            "or set TRACELENS_ROOT to an existing checkout"
        )
    if not (root / ".git").exists():
        return (
            f"TraceLens root incomplete (not a git checkout): {root}; "
            "run src/hyperloom/agents/kernel/scripts/install.sh "
            "or set TRACELENS_ROOT to a valid checkout"
        )
    return None


def _maybe_selfheal_tracelens_root(root: Path, *, log: Any = None) -> None:
    """Rebuild the pod-local TraceLens checkout if it vanished mid-run."""
    from hyperloom.inference_optimizer.session import paths

    # The installer-managed checkout is <deps_cache_root>/TraceLens or the per-revision
    # <deps_cache_root>/TraceLens@<sha>; both are healable.
    try:
        cache_root = paths.deps_cache_root().resolve()
        root_resolved = Path(root).resolve()
    except OSError:
        return
    is_default = root_resolved.parent == cache_root and (
        root_resolved.name == "TraceLens" or root_resolved.name.startswith("TraceLens@")
    )
    if not is_default:
        return  # explicit non-default override: never auto-clone
    try:
        tool = _kernel_agent_tool_path("tracelens_analysis.py")
        tools_dir = str(tool.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import tracelens_analysis as _tla  # type: ignore[import-not-found]

        heal_log = getattr(log, "warning", None) or (lambda *_a, **_k: None)
        heal_log("trace_analyze: TraceLens root %s missing; attempting self-heal", root)
        _tla._ensure_tracelens_checkout(root, log_path=Path(os.devnull))
    except Exception as exc:  # noqa: BLE001  # heal is best-effort; validation reports the real error
        _log = getattr(log, "warning", None)
        if _log:
            _log("trace_analyze: TraceLens self-heal failed: %s", exc)


def _kernel_agent_tool_path(tool_name: str) -> Path:
    """Resolve the absolute path to a kernel-agent shell tool."""
    err = _kernel_agent_root_error()
    if err:
        raise RuntimeError(err)
    root = _kernel_agent_root_from_env()
    assert root is not None
    path = root / "tools" / tool_name
    if not path.is_file():
        raise RuntimeError(f"kernel-agent tool not found: {path}")
    return path


def _coerce_runtime_value(value: Any) -> Any:
    """Best-effort coercion of a string runtime value to ``int`` or ``float``."""
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
    """Decide whether an env var may be forwarded as candidate metadata."""
    upper = key.upper()
    if any(part in upper for part in _SENSITIVE_ENV_PARTS):
        return False
    return key in _CANDIDATE_ENV_KEYS or any(key.startswith(prefix) for prefix in _CANDIDATE_ENV_PREFIXES)


def _split_server_args(raw: str) -> list[str]:
    """Tokenize a raw server-args string into an argv list."""
    try:
        return shlex.split(raw) if raw else []
    except ValueError:
        log.warning("failed to parse materialized server args; preserving raw string")
        return []


def _load_materialized_workload_metadata(config_path: str) -> dict[str, Any]:
    """Extract runtime workload context from a materialized Magpie YAML config."""
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
    # Per-framework env-name source of truth (e.g. atom reads ``EXTRA_ATOM_ARGS``).
    from ..actions.executors._grid_runner import server_args_env_name

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
        "env_vars": {str(key): str(value) for key, value in envs.items() if _candidate_env_allowed(str(key))},
        "runtime_args": {key: value for key, value in runtime_args.items() if value not in (None, "", {})},
    }


def _enrich_candidate_runtime_metadata(
    candidates: Any,
    metadata: dict[str, Any],
) -> None:
    """Backfill runtime env/args metadata onto each candidate kernel in place."""
    if not isinstance(candidates, list) or not metadata:
        return
    env_vars = metadata.get("env_vars") if isinstance(metadata.get("env_vars"), dict) else {}
    runtime_args = metadata.get("runtime_args") if isinstance(metadata.get("runtime_args"), dict) else {}
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
    """Stamp the TraceLens report path onto each candidate kernel in place."""
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
    """Rewrite the on-disk candidates artifact with enriched metadata."""
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
            data.get("hot_kernels_top15"),
            trace_report_path,
        )
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_apply_tool() -> Any:
    """Lazily import and cache the kernel-agent ``apply_kernel_patch.py`` module."""
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
    """Normalize compiled-artifact paths from a payload into a list of strings."""
    raw = payload.get("artifact_paths") or payload.get("compiled_artifact_paths") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _final_content_snapshot(
    *,
    patch_path: str,
    snapshot_dir: str | None,
    repo_root: str | None,
) -> str | None:
    """Return a snapshot dir holding the patch's FINAL bytes, materializing if needed."""
    if not (patch_path.endswith(".patch") and repo_root):
        return snapshot_dir
    try:
        descriptors = _load_apply_tool().parse_patch_manifest(
            Path(patch_path).read_text(encoding="utf-8", errors="replace")
        )
        writes = [str(d.get("path") or "") for d in descriptors if d.get("op") == "write"]
    except Exception:  # noqa: BLE001 — an unreadable patch is apply's error to report.
        return snapshot_dir
    if not writes:
        return snapshot_dir
    if snapshot_dir and all((Path(snapshot_dir) / rel).exists() for rel in writes):
        return snapshot_dir
    try:
        return materialize_unified_patch_snapshot(
            patch_path=patch_path,
            repo_root=repo_root,
            snapshot_dir=Path(patch_path).parent / "integrate_snapshot",
        )
    except Exception:  # noqa: BLE001 — fall back so apply surfaces the real failure.
        log.exception("integrate: could not materialize a final-content snapshot for %s", patch_path)
        return snapshot_dir


def _maybe_apply_kernel_patch(
    payload: dict,
    *,
    session_dir: Path,
    kernel_id: str | None,
) -> HandlerResult:
    """Apply a kernel patch via the kernel-agent ``apply_kernel_patch`` tool."""
    patch_path = str(payload.get("patch_path") or "").strip()
    target_file = str(payload.get("target_file") or payload.get("source_file") or "").strip()
    if not patch_path or not target_file:
        return {
            "status": "skipped",
            "reason": "missing patch_path or target_file/source_file",
        }
    from hyperloom.inference_optimizer.session.session_paths import fs_safe_id, patches_dir

    kid = str(kernel_id or payload.get("kernel_id") or "")
    # Same fold as the integrate workspace: a fusion sibling keys this dir by its ``llm:<recipe>`` operator name,
    # which ``mkdir`` rejects on some filesystems.
    backup_root = payload.get("backup_root") or (patches_dir(session_dir, fs_safe_id(kid)) / "backup")
    tool = _load_apply_tool()
    # Snapshot mode: a snapshot dir of byte-exact final files lands atomically.
    snapshot_dir = str(payload.get("snapshot_dir") or "").strip() or None
    repo_root = str(payload.get("kernel_repo") or payload.get("repo") or "").strip() or None
    snapshot_dir = _final_content_snapshot(
        patch_path=patch_path,
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
    )
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
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
        producer_manifest=(str(payload.get("producer_manifest") or "").strip() or None),
    )


def _checkpoint_collective_apply(
    checkpoint_path: str,
    apply_result: HandlerResult,
) -> None:
    """Persist an applied-patch checkpoint before Collective E2E measurement."""
    if not checkpoint_path or apply_result.get("status") != "ok":
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(apply_result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def materialize_unified_patch_snapshot(
    *,
    patch_path: str | Path,
    repo_root: str | Path,
    snapshot_dir: str | Path | None = None,
) -> str:
    """Materialize final file contents for apply_kernel_patch snapshot mode."""
    patch = Path(patch_path).resolve()
    root = Path(repo_root).resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"patch_path does not exist: {patch}")
    if not root.is_dir():
        raise FileNotFoundError(f"kernel repo does not exist: {root}")

    tool = _load_apply_tool()
    patch_text = patch.read_text(encoding="utf-8", errors="replace")
    descriptors = tool.parse_patch_manifest(patch_text)
    if not descriptors:
        raise ValueError(f"patch has no file operations: {patch}")

    # Paths the patch CREATES: these must be produced by ``git apply``, never pre-seeded with a base, or apply fails
    # "already exists".
    _new_file_paths = {
        str(desc.get("path") or "") for desc in descriptors if desc.get("op") == "write" and desc.get("is_new")
    }

    snap = Path(snapshot_dir) if snapshot_dir is not None else patch.parent / "fusion_snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)

    for desc in descriptors:
        rel = Path(str(desc.get("path") or ""))
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe patch path: {rel}")
        dst = snap / rel
        base = subprocess.run(
            ["git", *safe_directory_args(["-C", str(root), "show", f"HEAD:{rel.as_posix()}"])],
            capture_output=True,
            timeout=60,
        )
        if base.returncode == 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(base.stdout)
        elif rel.as_posix() not in _new_file_paths:
            # ``git show HEAD:`` failed and this is a MODIFY (not a create): non-git repo_root (e.g. vLLM/sglang under
            # site-packages/ dist-packages) or an untracked-but-present file.
            src = root / rel
            if not src.is_file():
                # Neither git HEAD nor the on-disk layout has the base.
                raise FileNotFoundError(
                    f"patch base missing for {rel.as_posix()}: not in git HEAD and not on disk under {root}"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    # ``git apply <path>`` rejects an otherwise valid final hunk when the patch artifact lacks a trailing newline
    # (observed in legacy KB records).
    normalized_patch_text = patch_text if patch_text.endswith(("\n", "\r")) else f"{patch_text}\n"
    # Pin the work tree to ``snap``.
    apply_env = {
        **os.environ,
        "GIT_DIR": str(snap / ".git_materialize"),
        "GIT_WORK_TREE": str(snap),
        "GIT_CEILING_DIRECTORIES": str(snap.parent),
    }
    proc = subprocess.run(
        ["git", "apply", "--unsafe-paths", "-"],
        cwd=snap,
        input=normalized_patch_text,
        capture_output=True,
        text=True,
        timeout=60,
        env=apply_env,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"could not materialize patch snapshot: {msg[:500]}")

    for desc in descriptors:
        if desc.get("op") == "write" and not (snap / str(desc["path"])).is_file():
            raise RuntimeError(f"snapshot missing final content for {desc['path']}")
    return str(snap)


def _maybe_revert_kernel_patch(apply_result: HandlerResult) -> HandlerResult:
    """Revert a kernel patch using its apply manifest."""
    if not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    try:
        return _load_apply_tool().revert_kernel_patch(apply_result["manifest_path"])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_class": "patch_revert_exception",
            "error": repr(exc),
            "manifest_path": str(apply_result["manifest_path"]),
        }


def _maybe_finalize_kernel_patch(
    apply_result: HandlerResult,
) -> HandlerResult:
    """Delete patch backups after a KEEP becomes durable."""
    if apply_result.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": "patch apply did not complete",
        }
    if not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    try:
        return _load_apply_tool().finalize_kernel_patch(apply_result["manifest_path"])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_class": "patch_finalize_exception",
            "error": repr(exc),
            "manifest_path": str(apply_result["manifest_path"]),
        }


def _find_selected_kernel_source(state: Any, kernel_id: str) -> str:
    """Look up a kernel's source file from the last trace-analyze result."""
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


class AmbiguousIntegrationTarget(Exception):
    """A bare ``kernel_id`` named several pending integration records."""


def _fill_integrate_defaults_from_state(
    payload: dict,
    *,
    session_dir: Path,
) -> dict:
    """Pull ``base_tput`` / ``config_path`` / ``extra_server_args`` defaults from SharedState."""
    from ..state.shared_state import SharedState, resolve_grading_anchor_tput

    resolved = dict(payload)
    state = SharedState.load_or_init(session_dir)

    integration_id = str(resolved.get("integration_id") or "")
    pending_records = state.pending_kernel_integration_records()
    pending_record = next(
        (record for record in pending_records if str(record.get("integration_id") or "") == integration_id),
        None,
    )
    if pending_record is None and resolved.get("kernel_id"):
        requested_kernel_id = str(resolved.get("kernel_id") or "")
        requested_task_key = str(resolved.get("task_group_key") or "")
        candidates = [
            record
            for record in pending_records
            if str(record.get("kernel_id") or "") == requested_kernel_id
            and (not requested_task_key or str(record.get("task_group_key") or "") == requested_task_key)
        ]
        if len(candidates) > 1:
            # Siblings of one nomination round share a kernel_id, so picking any of them would bind the verdict to a
            # record nobody named.
            raise AmbiguousIntegrationTarget(
                f"integrate refused: kernel_id={requested_kernel_id!r} matches "
                f"{len(candidates)} pending integration records; an explicit "
                "integration_id is required to bind the KEEP/REVERT verdict. "
                f"candidates={sorted(str(record.get('integration_id') or '') for record in candidates)!r}"
            )
        pending_record = candidates[0] if candidates else None
    if pending_record is not None:
        resolved.setdefault(
            "integration_id",
            str(pending_record.get("integration_id") or ""),
        )
        resolved.setdefault(
            "kernel_id",
            str(pending_record.get("kernel_id") or ""),
        )
        resolved.setdefault(
            "task_group_key",
            str(pending_record.get("task_group_key") or ""),
        )
        resolved.setdefault(
            "identity_route",
            str(pending_record.get("identity_route") or ""),
        )
        resolved.setdefault(
            "artifact_kind",
            str(pending_record.get("artifact_kind") or ""),
        )
        resolved.setdefault(
            "integration_validation_status",
            str(pending_record.get("integration_validation_status") or ""),
        )
        # Fusion siblings carry three facts the generic drain cannot infer.
        if str(pending_record.get("source") or "") == "forge_fusion":
            resolved.setdefault("source", "forge_fusion")
            resolved.setdefault("action_label", str(pending_record.get("action_label") or "fusion"))
            fusion_env_flags = pending_record.get("fusion_env_flags")
            if isinstance(fusion_env_flags, dict) and fusion_env_flags:
                requested = resolved.get("extra_envs")
                requested = dict(requested) if isinstance(requested, dict) else {}
                # The sibling's own flags win over anything already requested.
                resolved["extra_envs"] = {**requested, **{str(k): str(v) for k, v in fusion_env_flags.items()}}
            if "keep_threshold_pct" not in resolved and pending_record.get("keep_threshold_pct") is not None:
                try:
                    resolved["keep_threshold_pct"] = float(pending_record.get("keep_threshold_pct"))
                except (TypeError, ValueError):
                    pass

    current_best = getattr(state, "current_best", None) or {}

    if float(resolved.get("base_tput", 0.0) or 0.0) <= 0:
        # ``extra_server_args`` below is filled from current_best, so the candidate must be graded against that recipe
        # too.
        bt = resolve_grading_anchor_tput(state)
        if bt > 0:
            resolved["base_tput"] = bt

    if not resolved.get("config_path"):
        cfg = getattr(state, "baseline_config_path", "") or ""
        if cfg:
            resolved["config_path"] = cfg

    if not resolved.get("extra_server_args") and isinstance(current_best, dict):
        cb_args = current_best.get("extra_server_args") or ""
        if cb_args:
            resolved["extra_server_args"] = cb_args
    if isinstance(current_best, dict):
        current_envs = current_best.get("extra_envs")
        current_envs = dict(current_envs) if isinstance(current_envs, dict) else {}
        requested_envs = resolved.get("extra_envs")
        requested_envs = dict(requested_envs) if isinstance(requested_envs, dict) else {}
        if current_envs or requested_envs:
            # The candidate stacks onto current_best.
            resolved["extra_envs"] = {
                **current_envs,
                **requested_envs,
            }

    kernel_id = str(resolved.get("kernel_id") or "")
    if kernel_id:
        attempt = _entry_by_kernel_id(state, kernel_id) or {}
        if not resolved.get("task_group_key"):
            task_group_key = str(attempt.get("task_group_key") or "")
            if task_group_key:
                resolved["task_group_key"] = task_group_key
        # Defense-in-depth mirror of _queue_kernel_keep()'s refusal to queue a vendor-playbook KEEP for
        # auto-integration (PR #1191 review finding #1): this also catches an LLM-initiated integrate request that
        # names the kernel_id directly, bypassing the pending-queue lookup above via
        # _resolve_kernel_patch_identity()'s last_kernel_opt.best_artifact_path backfill.
        if attempt.get("vendor_playbook_deploy_blocked"):
            resolved["_vendor_playbook_deploy_blocked"] = True
        elif (
            isinstance(state.last_kernel_opt, dict)
            and str(state.last_kernel_opt.get("kernel_id") or "") == kernel_id
            and state.last_kernel_opt.get("vendor_playbook_deploy_blocked")
        ):
            resolved["_vendor_playbook_deploy_blocked"] = True

    return resolved


def _fill_integrate_snapshot_from_bundle(resolved: dict, bundle: Any) -> None:
    """Backfill integrate inputs from a recorded multi-file artifact bundle."""
    if not isinstance(bundle, dict) or bundle.get("type") != "patch_snapshot":
        return
    if not bundle_belongs_to(bundle, resolved.get("integration_id")):
        # Cross-sibling bundle -- refuse rather than silently mixing write sets.
        return
    if not resolved.get("snapshot_dir") and bundle.get("snapshot_dir"):
        resolved["snapshot_dir"] = str(bundle["snapshot_dir"])
    if not resolved.get("patch_path") and bundle.get("patch_path"):
        resolved["patch_path"] = str(bundle["patch_path"])
    if not resolved.get("kernel_repo") and bundle.get("repo_root"):
        resolved["kernel_repo"] = str(bundle["repo_root"])
    if not resolved.get("producer_manifest") and bundle.get("producer_manifest"):
        resolved["producer_manifest"] = str(bundle["producer_manifest"])
    if not resolved.get("patch_write_paths"):
        write_paths = [str(path) for path in (bundle.get("write_paths") or []) if str(path or "").strip()]
        if write_paths:
            resolved["patch_write_paths"] = write_paths


def _fill_integrate_provenance(
    resolved: dict,
    *,
    framework_applyback: Any,
    integration_validation_status: Any,
) -> None:
    """Backfill artifact provenance for an integrate resolved from a ledger entry."""
    if not resolved.get("artifact_kind") and isinstance(framework_applyback, dict):
        kind = str(framework_applyback.get("artifact_kind") or "")
        if kind:
            resolved["artifact_kind"] = kind
    if not resolved.get("integration_validation_status"):
        status = str(integration_validation_status or "")
        if status:
            resolved["integration_validation_status"] = status


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id (artifact in ``last_kernel_opt``, source in ``last_trace_analyze``)."""
    from ..state.shared_state import SharedState

    resolved = dict(payload)
    kernel_id = str(resolved.get("kernel_id") or "")
    state = SharedState.load_or_init(session_dir)
    last_kernel = state.last_kernel_opt or {}
    integration_id = str(resolved.get("integration_id") or "")
    pending_record = next(
        (
            record
            for record in state.pending_kernel_integration_records()
            if str(record.get("integration_id") or "") == integration_id
        ),
        None,
    )
    if pending_record is not None:
        kernel_id = str(pending_record.get("kernel_id") or kernel_id)
        resolved["kernel_id"] = kernel_id
        resolved["integration_id"] = str(pending_record.get("integration_id") or integration_id)
        resolved.setdefault(
            "task_group_key",
            str(pending_record.get("task_group_key") or ""),
        )
        resolved.setdefault(
            "identity_route",
            str(pending_record.get("identity_route") or ""),
        )
        # Provenance travels with the artifact so the serving verdict can tell a reference-only apply-back from one
        # already proven in place.
        resolved.setdefault(
            "artifact_kind",
            str(pending_record.get("artifact_kind") or ""),
        )
        resolved.setdefault(
            "integration_validation_status",
            str(pending_record.get("integration_validation_status") or ""),
        )
        _fill_integrate_snapshot_from_bundle(
            resolved,
            pending_record.get("artifact_bundle"),
        )
        if not resolved.get("snapshot_dir") and pending_record.get("snapshot_dir"):
            resolved["snapshot_dir"] = str(pending_record["snapshot_dir"])
        if not resolved.get("patch_path"):
            resolved["patch_path"] = str(
                pending_record.get("deploy_patch_path") or pending_record.get("artifact_path") or ""
            )
        if not resolved.get("kernel_repo") and pending_record.get("deploy_repo_root"):
            resolved["kernel_repo"] = str(pending_record["deploy_repo_root"])
        if not resolved.get("source_file") and pending_record.get("source_file"):
            resolved["source_file"] = str(pending_record["source_file"])

    if kernel_id and str(last_kernel.get("kernel_id") or "") == kernel_id:
        # Snapshot deploy: prefer the original patch + snapshot dir so the whole multi-file patch lands atomically.
        _fill_integrate_snapshot_from_bundle(resolved, last_kernel.get("best_artifact_bundle"))
        if not resolved.get("snapshot_dir") and last_kernel.get("deploy_snapshot_dir"):
            resolved["snapshot_dir"] = str(last_kernel["deploy_snapshot_dir"])
            if last_kernel.get("deploy_patch_path") and not resolved.get("patch_path"):
                resolved["patch_path"] = str(last_kernel["deploy_patch_path"])
            if last_kernel.get("deploy_repo_root") and not resolved.get("kernel_repo"):
                resolved["kernel_repo"] = str(last_kernel["deploy_repo_root"])
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
        _fill_integrate_provenance(
            resolved,
            framework_applyback=last_kernel.get("framework_applyback"),
            integration_validation_status=last_kernel.get("integration_validation_status"),
        )

    # Multi-KEEP queue fallback: pull patch_path/source_file from the per-kernel ledger for KEEPs other than the
    # strongest pending one.
    if kernel_id:
        attempt = _entry_by_kernel_id(state, kernel_id) or {}
        _fill_integrate_snapshot_from_bundle(resolved, attempt.get("last_artifact_bundle"))
        if not resolved.get("snapshot_dir") and attempt.get("last_snapshot_dir"):
            resolved["snapshot_dir"] = str(attempt["last_snapshot_dir"])
            if attempt.get("last_deploy_patch_path") and not resolved.get("patch_path"):
                resolved["patch_path"] = str(attempt["last_deploy_patch_path"])
            if attempt.get("last_deploy_repo_root") and not resolved.get("kernel_repo"):
                resolved["kernel_repo"] = str(attempt["last_deploy_repo_root"])
        if not resolved.get("patch_path") and attempt.get("last_artifact_path"):
            resolved["patch_path"] = str(attempt["last_artifact_path"])
        if not resolved.get("source_file") and attempt.get("last_source_file"):
            resolved["source_file"] = str(attempt["last_source_file"])
        _fill_integrate_provenance(
            resolved,
            framework_applyback=attempt.get("last_framework_applyback"),
            integration_validation_status=attempt.get("last_integration_validation_status"),
        )

    if kernel_id and not (resolved.get("target_file") or resolved.get("source_file")):
        source = _find_selected_kernel_source(state, kernel_id)
        if source:
            resolved["source_file"] = source

    patch_path = str(resolved.get("patch_path") or "").strip()
    target_file = str(resolved.get("target_file") or resolved.get("source_file") or "").strip()
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


def _tool_label(cmd: list[str]) -> str:
    """Name the tool a command runs, for the progress note."""
    for arg in cmd:
        text = str(arg)
        if text.endswith(".py"):
            return Path(text).stem
    return Path(str(cmd[0])).name if cmd else "subprocess"


async def _run_subprocess(
    cmd: list[str],
    *,
    timeout_sec: int,
) -> tuple[int, str, str]:
    """Run a bounded subprocess without blocking the reactor."""
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(float(timeout_sec))
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be finite and positive")

    def _run(on_output: Callable[[], None]) -> tuple[int, str, str]:
        """Run the command synchronously in a worker thread."""
        env = os.environ.copy()
        from ..actions.executors._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
            infera_ssh_env_from_state,
        )
        from ..actions.executors._subprocess_kill import run_with_session_kill

        if is_multi_node():
            # Infera backend: route GEAK GPU work to a pod over SSH (no Ray). infera_ssh_env_from_state() returns {}
            # for RayJob/single-node, so the RAY_ADDRESS path below is unchanged for those.
            ssh_env = infera_ssh_env_from_state()
            if ssh_env:
                env.update(ssh_env)
            addr = "" if ssh_env else ray_gcs_address_from_state()
            if addr:
                env.setdefault("RAY_ADDRESS", addr)
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        # The heartbeat around this call is only as honest as the child's flushing: block-buffered on a pipe, it looks
        # dead between flushes.
        env.setdefault("PYTHONUNBUFFERED", "1")
        # run_with_session_kill reaps the whole descendant tree on every exit path.
        cp = run_with_session_kill(
            cmd,
            env=env,
            timeout=timeout_sec,
            text=True,
            on_output=on_output,
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""

    async with heartbeat_while_output_flows(unit="kernel_tool", label=_tool_label(cmd)) as activity:
        return await asyncio.to_thread(_run, activity.note)


def _normalize_precision(value: Any) -> str:
    """Normalize a precision label to a trimmed lower-case string."""
    return str(value or "").strip().lower()


def _lane_budget(
    state: Any,
    lane: str,
    *,
    gemm_target_costs_sec: tuple[int, ...] = (),
):
    """Derive one lane's share of the phase's remaining time."""
    remaining = _lane_remaining_minutes(state)
    return _allocate_lane_budgets(remaining, gemm_target_costs_sec=gemm_target_costs_sec)[lane]


def _lane_remaining_minutes(state: Any) -> float | None:
    """Minutes a lane may plan against: the tighter of session and phase."""
    from ..phases.machine_state import phase_budget_remaining_seconds

    remaining_fn = getattr(state, "remaining_minutes", None)
    session = remaining_fn() if callable(remaining_fn) else None
    if session is None:
        return None
    phase_sec = phase_budget_remaining_seconds(state)
    if phase_sec is None:
        return float(session)
    return min(float(session), float(phase_sec) / 60.0)


def _gemm_tuning_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the GEMM-tuning subprocess timeout in seconds."""
    raw = payload.get("timeout_sec") or os.environ.get(
        "HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC",
        "",
    )
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else _DEFAULT_GEMM_TUNING_TIMEOUT_SEC
    return max(60, value)


def _gemm_router_targets(
    *,
    model_path: str,
    framework: str,
    precision: str,
    quant_type: str,
    gpu_type: str,
    kernel_signature_log: str,
    has_untuned_csv: bool,
    has_shapes_json: bool,
    has_tunableop_input: bool,
) -> tuple[tuple[str, int], ...]:
    """Ask the forge router which tuners it would run and what each costs."""
    try:
        from kernelforge.gemm_tune.model_analyzer import analyze_model  # noqa: PLC0415
        from kernelforge.gemm_tune.router import select_tuners  # noqa: PLC0415

        specs = select_tuners(
            analyze_model(model_path),
            framework=framework,
            precision=precision,
            quant_type=quant_type,
            gpu_type=gpu_type,
            kernel_signature_log=kernel_signature_log or None,
            has_untuned_csv=has_untuned_csv,
            has_shapes_json=has_shapes_json,
            has_tunableop_input=has_tunableop_input,
        )
    except Exception:  # noqa: BLE001 - an unavailable router must not fail the run
        log.debug("GEMM: could not consult the tuner router for lane cost estimates", exc_info=True)
        return ()
    return tuple((str(spec.name), max(0, int(spec.estimated_minutes * 60))) for spec in specs if spec.should_run)


def _forge_fusion_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the forge-fusion subprocess timeout in seconds."""
    raw = (
        payload.get("timeout")
        or payload.get("timeout_sec")
        or os.environ.get(
            "FORGE_FUSION_TIMEOUT",
            "",
        )
    )
    try:
        value = int(float(raw))
    except (OverflowError, TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else 7200
    return max(1, value)


def _forge_fusion_wrapper_timeout_sec(timeout_sec: int) -> int:
    """Give the wrapper time to reap its child tree and emit the timeout sentinel."""
    return max(1, int(timeout_sec)) + _FORGE_FUSION_WRAPPER_TIMEOUT_GRACE_SEC


def _positive_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _fusion_session_serve_args(
    state: object,
    payload: dict,
    *,
    framework: str,
    model_path: str,
) -> dict[str, int]:
    """TP / KV block size / max-model-len the serving smoke must match."""
    tp = _positive_int(payload.get("tp") or getattr(state, "tp", 0))
    max_model_len = _positive_int(payload.get("max_model_len") or getattr(state, "max_model_len", 0))
    block_size = _positive_int(payload.get("block_size"))
    if block_size <= 0 and "vllm" in (framework or "").strip().lower():
        from hyperloom.inference_optimizer.model_config_utils import (  # noqa: PLC0415
            _sparse_kv_block_size,
        )

        block_size = _positive_int(_sparse_kv_block_size(model_path))
    args: dict[str, int] = {}
    if tp:
        args["tp"] = tp
    if block_size:
        args["block_size"] = block_size
    if max_model_len:
        args["max_model_len"] = max_model_len
    return args


def _gemm_tuning_workspace(payload: dict, *, session_dir: Path) -> Path:
    """Resolve the workspace directory for a GEMM-tuning run."""
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
# Shape capture consumes throughput only, so it never pays for an accuracy eval.
export RUN_EVAL="false"
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


def _resolve_gemm_tuning_backend(payload: dict) -> str:
    """Resolve GEMM tuning backend under the forge-explicit-only invariant."""
    return "forge" if forge_explicitly_enabled() else "geak"


def _parse_forge_gemm_sentinel(stdout: str) -> dict[str, Any] | None:
    """Parse FORGE_GEMM_TUNE_RESULT_BEGIN/END sentinel block from stdout."""
    m = re.search(
        r"FORGE_GEMM_TUNE_RESULT_BEGIN\s*\n(.*?)\nFORGE_GEMM_TUNE_RESULT_END",
        stdout,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _read_forge_result_json(workspace: Path) -> dict[str, Any]:
    """Read forge's on-disk ``result.json`` from the tuning workspace."""
    try:
        path = workspace / "result.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {}


def _derive_gemm_skip_reason(tuners_skipped: Any) -> str:
    """Join forge per-tuner skip reasons into one concise human-readable string."""
    if not isinstance(tuners_skipped, list):
        return ""
    parts: list[str] = []
    for entry in tuners_skipped:
        if not isinstance(entry, dict):
            continue
        reason = str(entry.get("skip_reason") or "").strip()
        if not reason:
            continue
        tuner = str(entry.get("tuner") or "").strip()
        parts.append(f"{tuner}: {reason}" if tuner else reason)
    return "; ".join(parts)


_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC = 30


def _forge_gemm_tune_probe_cmd() -> list[str]:
    """Return the exact interpreter and CLI prefix used by GEMM tuning."""
    return [sys.executable, "-m", "kernelforge.cli", "gemm-tune", "--help"]


def _forge_gemm_tune_available() -> bool:
    """Check exactly what ``_build_cmd`` will run, in the interpreter it runs in."""
    try:
        proc = subprocess.run(
            _forge_gemm_tune_probe_cmd(),
            capture_output=True,
            text=True,
            timeout=_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "forge-gemm-tune preflight timed out after %ss in %s",
            _FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
            sys.executable,
        )
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("forge-gemm-tune preflight could not start in %s: %s", sys.executable, exc)
        return False
    if proc.returncode == 0:
        return True
    log.info(
        "forge-gemm-tune preflight failed (rc=%s): %s",
        proc.returncode,
        ((proc.stderr or proc.stdout or "").strip()[-400:] or "(no output)"),
    )
    return False


def _resolve_aiter_root_for_forge() -> str:
    """Resolve AITER's source root, including split ``aiter_meta`` wheels."""
    explicit = os.environ.get("AITER_ROOT_DIR", "").strip()
    if explicit:
        return explicit
    try:
        spec = importlib.util.find_spec("aiter_meta")
    except (ModuleNotFoundError, ValueError):
        spec = None
    locations = getattr(spec, "submodule_search_locations", None) or []
    for location in locations:
        root = Path(location)
        if (root / "csrc").is_dir():
            return str(root)
    return ""


def _resolve_forge_precision_and_quant(state, payload: dict) -> tuple[str, str]:
    """Resolve the actual runtime precision and quant_type for forge tuning."""
    from .roofline_ceiling import _parse_server_arg, resolve_runtime_workload

    framework = str(payload.get("framework") or getattr(state, "framework", "") or "").strip().lower()

    if payload.get("precision"):
        precision = _normalize_precision(payload["precision"])
        quant_type = str(payload.get("quant_type") or "auto").strip()
        if precision == "fp8" and quant_type.lower() == "auto":
            model_path = str(payload.get("model_path") or getattr(state, "model_path", "") or "").strip()
            gpu_type = str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or "").strip()
            quant_type = _resolve_fp8_quant_type(model_path, gpu_type, framework)
        return precision, quant_type

    # Resolve from actual server args (baseline yaml + current_best overlay).
    current_best = getattr(state, "current_best", None) or {}
    try:
        server_args = resolve_runtime_workload(state, arm="current_best").server_args
    except Exception:  # noqa: BLE001 - best-effort fallback for partial state/test doubles
        server_args = ""
        if isinstance(current_best, dict):
            server_args = str(current_best.get("extra_server_args") or "")
    extra_envs = dict(current_best.get("extra_envs") or {}) if isinstance(current_best, dict) else {}
    ref_envs = dict(getattr(state, "reference_envs", None) or {})
    per_token_signal = is_truthy(extra_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")) or is_truthy(
        ref_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")
    )

    quantization_arg = _parse_server_arg(server_args, "--quantization").lower()

    if quantization_arg == "fp8":
        precision = "fp8"
        # Hand forge the fp8 GEMM path the model runs: explicit per-token env wins, else the checkpoint's static
        # format, else "auto".
        if per_token_signal:
            quant_type = "per_token"
        else:
            model_path = str(payload.get("model_path") or getattr(state, "model_path", "") or "").strip()
            gpu_type = str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or "").strip()
            quant_type = _resolve_fp8_quant_type(model_path, gpu_type, framework)
        return precision, quant_type

    if quantization_arg in ("fp4", "mxfp4"):
        return quantization_arg, "fp4"

    # Fall back to session precision.
    precision = _normalize_precision(state.precision)
    if not precision:
        precision = "bf16"
    quant_type = str(payload.get("quant_type") or "auto").strip()
    if precision == "fp8" and quant_type.lower() == "auto":
        model_path = str(payload.get("model_path") or getattr(state, "model_path", "") or "").strip()
        gpu_type = str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or "").strip()
        quant_type = _resolve_fp8_quant_type(model_path, gpu_type, framework)
    return precision, quant_type


#: An aiter dispatch line, hit or miss. Either one proves the process actually
#: routed a GEMM through aiter, which is what makes a server log usable as a
#: shape source; a log without one is silent about shapes no matter how recent
#: or how well its workspace matches.
_AITER_DISPATCH_MARKER = "shape is M:"
#: The MoE half of the same question. The resolved log is not only a dense-shape
#: source: ``kernelforge.gemm_tune.router`` reads it for MoE stage coverage and
#: 1-stage ASM detection, and those parse ``[fused_moe]`` dispatch lines, which
#: aiter prints from a different code path than the dense ``shape is M:`` ones.
#: Requiring the dense marker alone would reject a log that is fully informative
#: about MoE routing -- on a fleet where every model under tuning is MoE, that
#: is the common case, and the router would silently fall back to "tune CK 2-stage
#: unconditionally".
_AITER_MOE_DISPATCH_MARKERS = (b"[fused_moe]", b"Mxfp4 MoE backend")
#: Only the M of a dispatch line. Reading M straight off the serving log is the
#: one token source grounded in what the model actually ran -- forge's fallback
#: derives ``--tokens`` from ``conc`` alone, and real fleet logs reach M=15842,
#: far outside anything that derivation produces.
_AITER_M_RE = re.compile(rb"shape is M:(\d+),")
#: Read logs in chunks: a fleet server.log is ~17MB and the evidence question
#: is usually answered in the first few KB.
_LOG_SCAN_CHUNK = 1 << 20


#: Longest a dispatch prefix can be, so a match straddling a chunk boundary is
#: carried into the next read. ``shape is M:`` plus its digits is far shorter.
_LOG_SCAN_OVERLAP = 64


def _scan_serving_log_m(path) -> dict[int, int]:
    """Count the M values of the dense aiter dispatch lines in a serving log."""
    counts: dict[int, int] = {}
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    break
                buf = carry + chunk
                last_end = 0
                for match in _AITER_M_RE.finditer(buf):
                    last_end = match.end()
                    value = int(match.group(1))
                    if value > 0:
                        counts[value] = counts.get(value, 0) + 1
                carry = buf[max(last_end, len(buf) - _LOG_SCAN_OVERLAP) :]
    except (OSError, ValueError):
        return {}
    return counts


def _log_has_aiter_evidence(path) -> bool:
    """True when the log carries at least one aiter dispatch line, dense or MoE."""
    markers = (_AITER_DISPATCH_MARKER.encode(), *_AITER_MOE_DISPATCH_MARKERS)
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    return False
                buf = carry + chunk
                if any(marker in buf for marker in markers):
                    return True
                carry = buf[-_LOG_SCAN_OVERLAP:]
    except OSError:
        return False


def _tokens_from_serving_log(path, limit: int = 16, reserve_largest: int = 4) -> str:
    """Derive forge's ``--tokens`` from the M values the server actually saw."""
    counts = _scan_serving_log_m(path)
    if not counts:
        return ""
    # Never let the reservation crowd out the frequency ranking: at most a quarter of the budget goes to "largest",
    # and always at least one slot.
    reserve = min(max(reserve_largest, 0), max(1, limit // 4))
    picked: list[int] = sorted(counts, reverse=True)[:reserve]
    for value, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(picked) >= limit:
            break
        if value not in picked:
            picked.append(value)
    return ",".join(str(m) for m in sorted(picked))


def _resolve_trace_shape_manifest(state, session_dir: Path) -> str:
    """Find the newest TraceShapeManifest this session produced."""
    # Deduplicate: state.session_dir is usually the same path we were handed, and an empty session then paid for two
    # full-tree walks to find nothing.
    seen_roots: set[str] = set()
    roots: list[Path] = []
    for raw in (session_dir, Path(str(getattr(state, "session_dir", "") or session_dir))):
        if raw is None or not Path(raw).is_dir():
            continue
        key = str(Path(raw).resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(Path(raw))
    for root in roots:
        best: tuple[float, str] | None = None
        for found in Path(root).glob("**/trace_shape_manifest.json"):
            try:
                mtime = found.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, str(found))
        if best is not None:
            return best[1]
    return ""


def _resolve_forge_server_log(state, session_dir: Path) -> str:
    """Find the server log matching the current runtime configuration."""

    def _find_server_log_near(workspace_str: str) -> str | None:
        if not workspace_str:
            return None
        ws = Path(workspace_str)
        # Direct hit (server started in this exact dir).
        direct = ws / "server.log"
        if direct.is_file() and _log_has_aiter_evidence(direct):
            return str(direct)
        # Sibling warmup_round — benchmark dirs sit under {run_hash}/{warmup_round|measure_round}/{benchmark_dir}/
        parent = ws.parent  # e.g. measure_round/
        if parent.name in ("warmup_round", "measure_round"):
            run_hash_dir = parent.parent
        else:
            run_hash_dir = parent
        warmup = run_hash_dir / "warmup_round"
        if warmup.is_dir():
            candidates: list[tuple[float, str]] = []
            for child in warmup.iterdir():
                sl = child / "server.log"
                if sl.is_file():
                    try:
                        candidates.append((sl.stat().st_mtime, str(sl)))
                    except OSError:
                        continue
            candidates.sort(reverse=True)
            for _mtime, candidate in candidates:
                if _log_has_aiter_evidence(candidate):
                    return candidate
        return None

    current_best = getattr(state, "current_best", None) or {}
    if isinstance(current_best, dict):
        found = _find_server_log_near(str(current_best.get("workspace") or "").strip())
        if found:
            return found

    last_baseline = getattr(state, "last_baseline", None) or {}
    if isinstance(last_baseline, dict):
        found = _find_server_log_near(str(last_baseline.get("workspace") or "").strip())
        if found:
            return found

    # Fallback: the whole runs/ tree, newest first.
    runs_dir = session_dir / "runs"
    candidates_by_age: list[tuple[float, Path]] = []
    if runs_dir.is_dir():
        for candidate_log in runs_dir.glob("**/server.log"):
            try:
                candidates_by_age.append((candidate_log.stat().st_mtime, candidate_log))
            except OSError:
                continue
        candidates_by_age.sort(key=lambda item: item[0], reverse=True)
        for _mtime, candidate_log in candidates_by_age:
            if _log_has_aiter_evidence(candidate_log):
                return str(candidate_log)

    # Separate the two ways this fails.
    if candidates_by_age:
        log.warning(
            "GEMM: %d server.log file(s) under %s but none contain aiter dispatch "
            "lines (dense %r or MoE %s), so there is no runtime shape source. "
            "Serving runs need AITER_LOG_TUNED_CONFIG enabled for shapes to be "
            "observable",
            len(candidates_by_age),
            runs_dir,
            _AITER_DISPATCH_MARKER,
            " / ".join(repr(m.decode()) for m in _AITER_MOE_DISPATCH_MARKERS),
        )
    return ""


def _is_forge_compatible_shapes_json(path: Path) -> bool:
    """Validate that a shapes JSON file matches forge's expected format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("shapes", [])
        if not isinstance(data, list) or not data:
            return False
        sample = data[0]
        if not isinstance(sample, dict):
            return False
        # Must have M/N/K keys (case-insensitive).
        keys = {k.upper() for k in sample}
        return {"M", "N", "K"}.issubset(keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _profile_shapes_are_fresh(state: Any) -> bool:
    """Return whether the latest profile matches the active workload/config."""
    return bool(state.profile_trace_matches_workload())


def _resolve_forge_shapes(
    state,
    session_dir: Path,
    *,
    require_fresh_profile: bool = False,
    precision: str = "",
) -> str:
    """Find TraceLens shapes JSON if available and in forge-compatible format."""
    if require_fresh_profile and not _profile_shapes_are_fresh(state):
        log.info(
            "Forge GEMM shapes: latest TraceLens profile does not match the "
            "active workload/config; ignoring its shape artifacts"
        )
        return ""
    last_trace = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(last_trace, dict):
        return ""

    candidates: list[str] = []

    # Prefer explicit artifact fields when TraceLens exposes them.
    for key in ("shapes_json", "shapes_path"):
        raw = str(last_trace.get(key) or "").strip()
        if raw:
            candidates.append(raw)
    artifact_paths = last_trace.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        for key in ("shapes_json", "shapes", "gemm_shapes_json"):
            raw = str(artifact_paths.get(key) or "").strip()
            if raw:
                candidates.append(raw)
    # Fallback: check beside candidates_path.
    candidates_path_str = last_trace.get("candidates_path") or ""
    if candidates_path_str:
        cand_file = Path(candidates_path_str)
        if cand_file.is_file():
            shapes_file = cand_file.parent / "shapes.json"
            candidates.append(str(shapes_file))

    # Extract the GEMM shapes observed by the latest TraceLens analysis.
    def _extracted() -> str:
        return _extract_gemm_shapes_from_candidates(
            str(last_trace.get("candidates_path") or ""),
            session_dir,
            precision=precision,
        )

    if _canonical_dtype(precision):
        scoped = _extracted()
        if scoped:
            return scoped

    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and _is_forge_compatible_shapes_json(p):
            return str(p)

    return _extracted()


# TraceLens has spelled the tensor separator <br>, <br/> and <BR/> over time.
_BR_SPLIT_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)


def _canonical_dtype(raw: str) -> str:
    """Fold a precision name or traced dtype token onto one canonical family."""
    token = str(raw or "").strip().lower().removeprefix("torch.")
    if not token:
        return ""
    if token.startswith(("fp4", "mxfp4", "float4")) or "e2m1" in token:
        return "fp4"
    if token.startswith(("fp8", "float8")) or token == "f8" or "e4m3" in token or "e5m2" in token:
        return "fp8"
    if token.startswith(("bf16", "bfloat16")) or token == "b16":
        return "bf16"
    if token.startswith(("fp16", "float16")) or token in {"f16", "half"}:
        return "fp16"
    return ""


def _extract_gemm_shapes_from_candidates(candidates_path_str: str, session_dir: Path, *, precision: str = "") -> str:
    """Extract M,N,K from kernel_candidates.json hot_kernels input_shapes."""
    import json as _json
    import re as _re

    if not candidates_path_str:
        return ""
    cand_file = Path(candidates_path_str)
    if not cand_file.is_file():
        return ""

    try:
        data = _json.loads(cand_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    hot_kernels = data.get("hot_kernels", [])
    if not isinstance(hot_kernels, list):
        return ""

    # Tolerate whitespace after the comma ("(1024, 5120)") and any leading token before the tuple; TraceLens formats
    # vary. .search() rather than .match() so a leading dtype/name does not defeat it.
    dim_pattern = _re.compile(r"\((\d+)\s*,\s*(\d+)\)")
    # TraceLens renders the dtype right after the dims: "(64,3072) fp8".
    dtype_pattern = _re.compile(r"\)\s*([A-Za-z][A-Za-z0-9_.]*)")
    wanted_dtype = _canonical_dtype(precision)

    def _dtype_matches(a_text: str) -> bool:
        """Whether the A tensor's traced dtype is the family being tuned."""
        if not wanted_dtype:
            return True
        found = dtype_pattern.search(a_text)
        return bool(found) and _canonical_dtype(found.group(1)) == wanted_dtype

    def _mnk(a_text: str, b_text: str) -> tuple[int, int, int] | None:
        """Derive (M, N, K) from the A ``(M,K)`` and B tensor texts."""
        if not _dtype_matches(a_text):
            return None
        m0 = dim_pattern.search(a_text)
        m1 = dim_pattern.search(b_text)
        if not m0 or not m1:
            return None
        M, K = int(m0.group(1)), int(m0.group(2))
        b0, b1 = int(m1.group(1)), int(m1.group(2))
        # B is stored either (N,K) or (K,N); pick the orientation whose contracted dim matches K, else keep the legacy
        # first-dim reading.
        N = b0 if b1 == K else (b1 if b0 == K else b0)
        # ``N == 1`` is a matrix-vector head (e.g. a scalar projection), not a tunable GEMM tile; it would otherwise
        # sort first on call count and burn a tuning slot.
        return (M, N, K) if min(M, K) > 0 and N > 1 else None

    # ``weight`` is the observed call count: decode GEMMs are invoked far more often than prefill ones, so ordering by
    # it puts the throughput-dominant shapes first and they still get tuned when the tuner runs out of budget.
    weights: dict[tuple[int, int, int], int] = {}
    order: dict[tuple[int, int, int], int] = {}

    def _record(key: tuple[int, int, int] | None, weight: int) -> None:
        if key is None:
            return
        if key not in order:
            order[key] = len(order)
        weights[key] = max(weights.get(key, 0), weight)

    for kernel in hot_kernels:
        name = str(kernel.get("name", ""))
        if "gemm" not in name.lower():
            continue
        input_shapes = kernel.get("input_shapes", [])
        if not isinstance(input_shapes, list):
            continue
        entries = [e for e in input_shapes if isinstance(e, dict) and e.get("shape")]

        # Legacy format: one entry carries every tensor, "<br>"-joined.
        matched_joined = False
        for entry in entries:
            parts = [p.strip() for p in _BR_SPLIT_RE.split(str(entry["shape"])) if p.strip()]
            if len(parts) < 2:
                continue
            matched_joined = True
            _record(_mnk(parts[0], parts[1]), int(entry.get("call_num") or 0))
        if matched_joined or len(entries) < 2:
            continue

        # Current format: one entry per tensor, so A and B are the first two.
        weight = max(int(e.get("call_num") or 0) for e in entries[:2])
        _record(_mnk(str(entries[0]["shape"]), str(entries[1]["shape"])), weight)

    if not weights:
        return ""

    # Most-called first; ties keep discovery order so output stays deterministic.
    ranked = sorted(weights, key=lambda key: (-weights[key], order[key]))
    shapes = [{"M": M, "N": N, "K": K} for M, N, K in ranked]

    out_path = cand_file.parent / "traced_gemm_shapes.json"
    try:
        out_path.write_text(_json.dumps(shapes, indent=2), encoding="utf-8")
    except OSError:
        return ""

    log.info(
        "extracted %d unique GEMM shapes from kernel_candidates.json -> %s",
        len(shapes),
        out_path,
    )
    return str(out_path)


# Map the resolved (precision, quant_type) to the aiter untuned-GEMM CSV the specialist phase records; fp8 "auto"
# resolves to blockscale (forge default).
_FORGE_UNTUNED_CSV_BY_QUANT: dict[str, str] = {
    "auto": "a8w8_blockscale_untuned_gemm.csv",
    "blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "block_scale": "a8w8_blockscale_untuned_gemm.csv",
    "a8w8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "fp8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "per_token": "a8w8_untuned_gemm.csv",
    "per_tensor": "a8w8_untuned_gemm.csv",
    "a8w8": "a8w8_untuned_gemm.csv",
    "w8a8": "a8w8_untuned_gemm.csv",
    "w8a8_fp8": "a8w8_untuned_gemm.csv",
    "fp8_w8a8": "a8w8_untuned_gemm.csv",
    "bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "a8w8_bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "blockscale+bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "fp4": "a4w4_blockscale_untuned_gemm.csv",
    "mxfp4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4_blockscale": "a4w4_blockscale_untuned_gemm.csv",
}


def _csv_has_data_rows(path: Path) -> bool:
    """Return True when ``path`` is a CSV carrying at least one data row."""
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if "M" not in header.upper():
                return False
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _csv_k_values(path: Path) -> set[int]:
    """Return the distinct integer ``K`` (contraction-dim) values in a CSV."""
    ks: set[int] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split(",")
            cols = {name.strip().upper(): i for i, name in enumerate(header)}
            kidx = cols.get("K")
            if kidx is None:
                return ks
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= kidx:
                    continue
                try:
                    ks.add(int(float(parts[kidx])))
                except ValueError:
                    continue
    except OSError:
        return ks
    return ks


def _read_model_config(model_path: str) -> dict | None:
    """Load a HF ``config.json`` as a dict; ``None`` when unavailable/unreadable."""
    if not model_path:
        return None
    # ``model_path`` may be an HF repo id; resolve to the local weights dir (shared resolver) so the config read works
    # for repo-id launches.
    from hyperloom.inference_optimizer.model_config_utils import (
        resolve_local_model_dir,
    )

    cfg = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _model_hidden_size(model_path: str) -> int | None:
    """Read ``hidden_size`` from a HF ``config.json``; ``None`` when unavailable."""
    data = _read_model_config(model_path)
    if data is None:
        return None
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg_dict in candidates:
        for key in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
            val = cfg_dict.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def _resolve_fp8_quant_type(model_path: str, gpu_type: str = "", framework: str = "") -> str:
    """Pick the fp8 dense tuner quant_type from the checkpoint's static format."""
    data = _read_model_config(model_path)
    if data is None:
        return "auto"
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    is_blockscale = False
    for cfg_dict in candidates:
        qc = cfg_dict.get("quantization_config")
        if isinstance(qc, dict):
            if qc.get("weight_block_size"):
                is_blockscale = True
                break
            method = str(qc.get("quant_method") or qc.get("fmt") or "").lower()
            if "block" in method:
                is_blockscale = True
                break
    if is_blockscale:
        if _is_gfx950(gpu_type) and framework.lower() == "sglang":
            return "blockscale_bpreshuffle"
        return "blockscale"
    return "per_token"


_GFX950_GPU_TYPES = frozenset({"mi355x", "gfx950"})


def _is_gfx950(gpu_type: str) -> bool:
    """True when gpu_type resolves to gfx950 (CDNA4 / MI355X)."""
    key = (gpu_type or "").strip().lower()
    if key in _GFX950_GPU_TYPES:
        return True
    if not key or key == "auto":
        return _is_gfx950_rocminfo()
    return False


@functools.lru_cache(maxsize=1)
def _is_gfx950_rocminfo() -> bool:
    """Cached rocminfo probe for gfx950 arch."""
    try:
        out = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
        return "gfx950" in out.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _csv_matches_model(csv_path: Path, model_path: str) -> bool:
    """Return True when an untuned CSV plausibly belongs to ``model_path``."""
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


def _resolve_forge_untuned_csv(session_dir: Path, precision: str, quant_type: str, model_path: str = "") -> str:
    """Find an aiter untuned-GEMM CSV in a specialist worktree."""
    precision = (precision or "").strip().lower()
    quant_type = (quant_type or "").strip().lower()

    fname = _FORGE_UNTUNED_CSV_BY_QUANT.get(quant_type)
    if fname is None:
        log.warning(
            "Forge GEMM shapes: unknown quant_type=%r for precision=%r; not guessing an untuned CSV",
            quant_type,
            precision,
        )
        return ""

    from hyperloom.inference_optimizer.session.session_paths import runs_root

    specialist_dir = runs_root(session_dir) / "specialist"
    if not specialist_dir.is_dir():
        return ""

    best: Path | None = None
    best_mtime = -1.0
    for csv_path in specialist_dir.glob(f"*/worktree/aiter/configs/{fname}"):
        if not _csv_has_data_rows(csv_path):
            continue
        if not _csv_matches_model(csv_path, model_path):
            continue
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = csv_path

    return str(best) if best is not None else ""


def _path_is_existing_file(value: str) -> bool:
    """Safe ``Path.is_file()`` that never raises on an over-long pathname."""
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _normalize_tokens(value: Any) -> str:
    """Return a clean comma-separated token string for forge's ``--tokens``."""
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip().strip("[](){}")
        if not text:
            return ""
        items = [p for p in text.split(",")]
    out: list[str] = []
    for it in items:
        s = str(it).strip().strip("'\"")
        if not s:
            continue
        try:
            out.append(str(int(float(s))))
        except (TypeError, ValueError):
            continue
    return ",".join(out)


def _normalize_forge_shapes_json(value: Any, workspace: Path) -> str:
    """Return a usable shapes-JSON *file path*, materializing inline content."""
    if value in (None, ""):
        return ""

    # Already-parsed inline content.
    if isinstance(value, (list, dict)):
        parsed: Any = value
    else:
        text = str(value).strip()
        if not text:
            return ""
        if _path_is_existing_file(text):
            return text
        # Inline JSON content (possibly Python-repr with single quotes).
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                try:
                    import ast

                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return ""
        else:
            # Non-JSON string that is not an existing file.
            return ""

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        out = workspace / "forge_shapes.json"
        out.write_text(json.dumps(parsed), encoding="utf-8")
        return str(out)
    except (OSError, TypeError, ValueError):
        return ""


# Forge tuner families whose deliverable is an aiter tuned-GEMM CSV, i.e. the ones whose rows are resolved through
# aiter's padded (M, N, K) lookup.
_AITER_CSV_TUNER_FRAMEWORKS = ("sglang", "vllm-aiter")


#: Wall-clock the aiter CK tuner needs per shape, measured on gfx950 at
#: ``--mp 1`` (12 shapes / 1462s in production, ~135s each). Used to size the
#: shape budget so a wider ladder cannot push the tuner past its timeout and
#: return nothing at all.
_AITER_TUNE_SEC_PER_SHAPE = 150


def _align_forge_shapes_for_aiter(
    shapes_json: str,
    *,
    forge_framework: str,
    workspace: Path,
    budget_sec: int = 0,
    mp: int = 1,
) -> tuple[str, dict[str, Any] | None]:
    """Re-key profiled GEMM shapes onto the M values aiter actually looks up."""
    if forge_framework not in _AITER_CSV_TUNER_FRAMEWORKS:
        return shapes_json, None
    if not env_bool("HYPERLOOM_GEMM_ALIGN_SHAPES", True):
        return shapes_json, None

    from .gemm_shape_coverage import align_shapes_to_aiter_keys, load_shapes_json, write_shapes_json

    observed = load_shapes_json(shapes_json)
    if not observed:
        return shapes_json, None
    try:
        max_shapes = int(os.environ.get("HYPERLOOM_GEMM_ALIGN_MAX_SHAPES") or 64)
    except (TypeError, ValueError):
        max_shapes = 64
    max_shapes = max(1, max_shapes)
    if budget_sec > 0:
        try:
            per_shape = int(os.environ.get("HYPERLOOM_GEMM_TUNE_SEC_PER_SHAPE") or _AITER_TUNE_SEC_PER_SHAPE)
        except (TypeError, ValueError):
            per_shape = _AITER_TUNE_SEC_PER_SHAPE
        # Reserve a third of the window for JIT builds and the report step.
        affordable = int(budget_sec * 0.66 * max(1, mp) // max(1, per_shape))
        max_shapes = min(max_shapes, max(len(observed), affordable))
    aligned, report = align_shapes_to_aiter_keys(observed, max_shapes=max_shapes)
    if not aligned or report.get("unchanged"):
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    try:
        out = write_shapes_json(aligned, workspace / "forge_shapes.aiter_aligned.json")
    except OSError:
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    log.info(
        "Forge GEMM shapes: re-keyed %d observed shape(s) onto %d aiter lookup key(s) (observed M=%s -> aligned M=%s)",
        report.get("observed"),
        report.get("aligned"),
        report.get("observed_m"),
        report.get("aligned_m"),
    )
    return out, {**report, "applied": True, "source_shapes_json": shapes_json}


_VLLM_BLOCK_FP8_TRACE_OPS = (
    "w8a8_triton_block_scaled_mm",
    "rocm_aiter_gemm_a8w8_blockscale",
    "rocm_aiter_triton_gemm_a8w8_blockscale",
)


def _is_vllm_block_fp8(precision: str, quant_type: str) -> bool:
    """Return whether vLLM runs the block-scaled FP8 linear kernel path."""
    return precision == "fp8" and quant_type.strip().lower() in {
        "blockscale",
        "block_scale",
        "a8w8_blockscale",
        "fp8_blockscale",
    }


#: Markers that identify which kernels aiter is serving, read off a server log.
#: ``bf16_tuned_gemm.csv`` means dense linears resolve through
#: ``aiter/tuned_gemm.py`` (Forge's ``sglang_dense_bf16`` writes that table via
#: ``AITER_CONFIG_GEMM_BF16``); the fused-MoE markers mean the MoE layers run on
#: aiter's CK kernels (Forge's ``fmoe_ck``, via ``AITER_CONFIG_FMOE``) rather
#: than vLLM's Triton ``fused_moe``.
_AITER_SERVING_MARKERS = {
    "bf16_dense": ("bf16_tuned_gemm.csv",),
    "fused_moe": ("[aiter] [fused_moe]", "Mxfp4 MoE backend"),
}

#: aiter logs every fused-MoE problem it dispatches as a 14-field tuple. The
#: wording before it varies -- measured across 2948 real lines there are three
#: forms, and one of them interposes its own parenthesised kernel names:
#:
#:   [fused_moe] using 2stage default for ('gfx950', 256, 256, 4096, ...)
#:   [fused_moe] no tuned FlyDSL config for ('gfx950', 256, 256, 4096, ...)
#:   [fused_moe] using 2stage (kernelName1='...', kernelName2='...') for ('gfx950', ...)
#:
#: so the tuple is anchored on `` for (`` rather than on the wording. The field
#: order matches aiter's untuned CSV columns after dropping gfx and cu_num, which
#: the runtime supplies itself.
_AITER_FUSED_MOE_TUPLE_RE = re.compile(
    r"\[fused_moe\].*? for \("
    r"'(?P<gfx>[^']*)', "
    r"(?P<cu_num>\d+), (?P<token>\d+), (?P<model_dim>\d+), (?P<inter_dim>\d+), "
    r"(?P<expert>\d+), (?P<topk>\d+), "
    r"'(?P<act_type>[^']*)', '(?P<dtype>[^']*)', "
    r"'(?P<q_dtype_a>[^']*)', '(?P<q_dtype_w>[^']*)', '(?P<q_type>[^']*)', "
    r"(?P<use_g1u1>True|False), (?P<doweight_stage1>True|False)\)"
)

#: Which dtypes fall in each of aiter's width buckets. Mirrors ``bit16_list`` /
#: ``bit8_list`` / ``bit4_list`` in
#: ``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py``.
_AITER_BIT16_DTYPES = frozenset({"bfloat16", "float16"})
_AITER_BIT8_DTYPES = frozenset({"float8_e4m3fn", "float8_e4m3fnuz", "int8"})
_AITER_BIT4_DTYPES = frozenset({"float4_e2m1fn_x2", "uint32", "int4"})

#: The fields that identify one MoE problem, ignoring the token count (which the
#: tuner sweeps) and cu_num/gfx (which the runtime supplies).
_FMOE_SHAPE_FIELDS = (
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)


def _aiter_moe_dtype_pair_supported(q_dtype_a: str, q_dtype_w: str) -> bool:
    """Return whether aiter's CK MoE codegen has a kernel family for this pair."""
    act = q_dtype_a.replace("torch.", "")
    weight = q_dtype_w.replace("torch.", "")
    if act in _AITER_BIT16_DTYPES and weight in _AITER_BIT16_DTYPES:
        return True
    if act in _AITER_BIT8_DTYPES and weight in _AITER_BIT8_DTYPES:
        return True
    # The a8w4 family is FP8-only on the activation side; INT8 does not qualify.
    if act.startswith("float8") and weight in _AITER_BIT4_DTYPES:
        return True
    return act in _AITER_BIT4_DTYPES and weight in _AITER_BIT4_DTYPES


def _aiter_fused_moe_dispatch_keys(server_log: str) -> list[dict[str, str]]:
    """Return the distinct MoE problems a server log shows aiter dispatching."""
    if not server_log:
        return []
    try:
        text = Path(server_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    seen: dict[tuple[str, ...], dict[str, str]] = {}
    for match in _AITER_FUSED_MOE_TUPLE_RE.finditer(text):
        fields = match.groupdict()
        identity = tuple(fields[name] for name in _FMOE_SHAPE_FIELDS)
        if identity not in seen:
            seen[identity] = fields
    return list(seen.values())


def _aiter_ck_moe_tuner_supports(server_log: str) -> bool:
    """Return whether aiter's CK MoE tuner can tune anything the server dispatched."""
    if not server_log:
        return False
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    if not keys:
        # MoE evidence without a parseable problem tuple: let Forge decide.
        return True
    return any(_aiter_moe_dtype_pair_supported(key["q_dtype_a"], key["q_dtype_w"]) for key in keys)


#: Header aiter's MoE tuner expects for its untuned input CSV.
_FMOE_UNTUNED_CSV_HEADER = (
    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
)

#: forge wordings that carry nothing deployable, each distinct from an honest
#: ``no_improvement``. ``build_report`` checks ``has_candidate`` first, so a run
#: holding a usable env reports ``candidate`` even when a sibling crashed --
#: these arrive only with nothing to deploy.
_FORGE_BARREN_MICRO_DECISIONS = (
    "failed",
    "empty_output",
    "partial_failure",
    "partial_output",
)


def _fmoe_token_list(tokens: Any) -> list[int]:
    """Positive token counts to sweep, keyed off whatever the caller sends."""
    if isinstance(tokens, str):
        raw: list[str] = [part.strip() for part in tokens.split(",")]
    elif isinstance(tokens, (list, tuple, set, frozenset)):
        raw = [str(item).strip() for item in tokens]
    elif tokens is None:
        raw = []
    else:
        raw = [str(tokens).strip()]

    out: set[int] = set()
    for item in raw:
        if not item:
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.add(value)
    return sorted(out) or [1]


def _write_fmoe_untuned_csv_from_log(
    server_log: str,
    tokens: Any,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    """Turn the MoE problems observed in ``server_log`` into a tuning input CSV."""
    report: dict[str, Any] = {
        "observed": 0,
        "tunable": 0,
        "dropped_unsupported": [],
        "keys": [],
    }
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    report["observed"] = len(keys)
    if not keys:
        return "", report

    tunable: list[dict[str, str]] = []
    for key in keys:
        pair = (key["q_dtype_a"], key["q_dtype_w"])
        if _aiter_moe_dtype_pair_supported(*pair):
            tunable.append(key)
            report["keys"].append({name: key[name] for name in _FMOE_SHAPE_FIELDS})
        else:
            combo = f"{pair[0]}/{pair[1]}"
            if combo not in report["dropped_unsupported"]:
                report["dropped_unsupported"].append(combo)
    report["tunable"] = len(tunable)
    if not tunable:
        return "", report

    token_list = _fmoe_token_list(tokens)
    lines = [_FMOE_UNTUNED_CSV_HEADER]
    for key in tunable:
        for token in token_list:
            lines.append(
                f"{token},{key['model_dim']},{key['inter_dim']},"
                f"{key['expert']},{key['topk']},{key['act_type']},{key['dtype']},"
                f"{key['q_dtype_a']},{key['q_dtype_w']},{key['q_type']},"
                f"{1 if key['use_g1u1'] == 'True' else 0},"
                f"{1 if key['doweight_stage1'] == 'True' else 0}"
            )

    csv_path = workspace / "untuned_fmoe_from_runtime.csv"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        # A full disk or a read-only workspace must cost the MoE tuner its input, not the whole tuning run: the dense
        # tuners take their shapes from elsewhere and can still produce something useful.
        report["write_error"] = f"{type(exc).__name__}: {exc}"
        log.warning("Forge GEMM shapes: cannot write %s: %s", csv_path, exc)
        return "", report
    log.info(
        "Forge GEMM shapes: derived %d MoE problem(s) x %d token(s) from %s%s",
        len(tunable),
        len(token_list),
        server_log,
        (f"; dropped {report['dropped_unsupported']} as untunable by aiter" if report["dropped_unsupported"] else ""),
    )
    return str(csv_path), report


def _aiter_serving_evidence(server_log: str) -> set[str]:
    """Return which aiter kernel families a server log shows in use."""
    found: set[str] = set()
    if not server_log:
        return found
    try:
        with open(server_log, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for family, markers in _AITER_SERVING_MARKERS.items():
                    if family not in found and any(marker in line for marker in markers):
                        found.add(family)
                if len(found) == len(_AITER_SERVING_MARKERS):
                    break
    except OSError:
        return set()
    return found


def _forge_framework_for_vllm(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    tunableop_input: str,
    aiter_bf16_dense: bool = False,
    aiter_fused_moe: bool = False,
) -> str:
    """Route vLLM runs served by aiter to Forge's AITER tuner family."""
    if framework != "vllm" or tunableop_input:
        return framework
    if _is_vllm_block_fp8(precision, quant_type):
        return "vllm-aiter"
    if aiter_bf16_dense and precision in ("bf16", "fp16"):
        return "vllm-aiter"
    if aiter_fused_moe:
        return "vllm-aiter"
    return framework


def _resolve_vllm_aiter_routing(
    *,
    model_path: str,
    server_log: str,
    tp: int,
) -> dict[str, bool]:
    """Resolve the aiter-routing flags for a vLLM run from runtime evidence."""
    flags = {"aiter_bf16_dense": False, "aiter_fused_moe": False}
    evidence = _aiter_serving_evidence(server_log)
    if not evidence:
        return flags

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path) or {}
    if not summary:
        return flags
    is_moe = bool(summary.get("is_moe"))

    # Dense BF16 routing is for dense checkpoints; a MoE model's dense side rides along with its MoE routing instead.
    flags["aiter_bf16_dense"] = "bf16_dense" in evidence and not is_moe

    if "fused_moe" in evidence and is_moe and _aiter_ck_moe_tuner_supports(server_log):
        # Only route MoE when aiter's CK fused-MoE can actually serve this checkpoint at this TP -- otherwise the
        # tuner has no reachable target.
        from hyperloom.inference_optimizer.cli.model_gate import (
            model_supports_aiter_ck_fused_moe,
        )

        flags["aiter_fused_moe"] = model_supports_aiter_ck_fused_moe(model_path, tp)

    _warn_if_moe_routing_is_coarser_than_the_log(server_log, flags)
    return flags


def _warn_if_moe_routing_is_coarser_than_the_log(server_log: str, flags: dict[str, bool]) -> None:
    """Say so when one log shows both MoE backends and routing picks one."""
    if not flags.get("aiter_fused_moe"):
        return
    try:
        from kernelforge.gemm_tune.evidence import parse_log_file
    except ImportError:
        # Same reasoning as apply_verification._parse: kernelforge is in this wheel, so a miss is a broken install,
        # and a bare return makes the missing routing warning indistinguishable from a clean run.
        log.warning(
            "kernelforge.gemm_tune is not importable, so the aiter/vLLM MoE "
            "routing check is skipped -- it ships with Hyperloom, so this means "
            'an incomplete install; reinstall with pip install -e ".[forge]"'
        )
        return
    try:
        moe = (parse_log_file(server_log).get("dispatch") or {}).get("moe") or {}
    except Exception:  # noqa: BLE001 - a reporting aid must not break routing
        return
    if moe.get("impl") == "mixed" or moe.get("vllm_config_hit"):
        log.warning(
            "gemm routing: %s shows both aiter CK and vLLM Triton MoE dispatch "
            "(impl=%s, stages=%s); routing sends the whole run to the aiter "
            "tuner family, so the token range Triton serves goes untuned",
            server_log,
            moe.get("impl"),
            moe.get("stages_seen"),
        )


def _vllm_block_fp8_profile_capture_required(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    shapes_json: str,
    tunableop_input: str,
    dry_run: bool,
) -> bool:
    """Return whether block-FP8 needs a profiled runtime-shape capture pass."""
    if (
        framework != "vllm"
        or not _is_vllm_block_fp8(precision, quant_type)
        or dry_run
        or shapes_json
        or tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    return not is_multi_node()


def _trace_event_block_fp8_shape(event: Any) -> tuple[int, int, int] | None:
    """Extract one (M, N, K) tuple from a profiled block-FP8 linear event."""
    if not isinstance(event, dict):
        return None
    name = str(event.get("name") or "").lower()
    if not any(marker in name for marker in _VLLM_BLOCK_FP8_TRACE_OPS):
        return None
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    dims = args.get("Input Dims") or args.get("Input dims") or args.get("input_shapes")
    if not isinstance(dims, list) or len(dims) < 2:
        return None
    a_dims, b_dims = dims[0], dims[1]
    if not isinstance(a_dims, list) or not isinstance(b_dims, list) or len(a_dims) < 2 or len(b_dims) < 2:
        return None
    try:
        m = int(a_dims[-2])
        k = int(a_dims[-1])
        if int(b_dims[-1]) == k:
            n = int(b_dims[-2])
        elif int(b_dims[-2]) == k:
            n = int(b_dims[-1])
        else:
            return None
    except (TypeError, ValueError):
        return None
    return (m, n, k) if min(m, n, k) > 0 else None


def _extract_vllm_block_fp8_profile_shapes(
    trace_input: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[str, int]:
    """Convert Kineto block-FP8 events into Forge's structured shapes JSON."""
    import gzip

    def _is_capture_sidecar(path: Path) -> bool:
        # Shared classifier, so a layout the kernel-agent routes demote is also kept out of the shape harvest here;
        # the exact-``capture_traces`` test this replaced missed ``graph_capture_profile/``.
        return _shared_is_capture_fragment(path, trace_input if trace_input.is_dir() else trace_input.parent)

    shapes: set[tuple[int, int, int]] = set()
    # ``Path("")`` normalizes to ``Path(".")``, which would otherwise walk the whole process CWD and harvest shapes
    # from unrelated traces.
    if str(trace_input) in ("", "."):
        return "", 0
    if trace_input.is_file():
        trace_paths = [] if _is_capture_sidecar(trace_input) else [trace_input]
    elif trace_input.is_dir():
        trace_paths = [
            path
            for path in sorted(trace_input.rglob("*.json")) + sorted(trace_input.rglob("*.json.gz"))
            if not _is_capture_sidecar(path)
        ]
    else:
        return "", 0
    for path in trace_paths:
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                    data = json.load(stream)
            else:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        events = data.get("traceEvents") if isinstance(data, dict) else None
        if not isinstance(events, list):
            continue
        for event in events:
            shape = _trace_event_block_fp8_shape(event)
            if shape is not None:
                shapes.add(shape)
    if not shapes:
        return "", 0
    destination = output_dir or (trace_input if trace_input.is_dir() else trace_input.parent)
    out = destination / "forge_shapes.json"
    payload = [{"M": m, "N": n, "K": k} for m, n, k in sorted(shapes)]
    try:
        destination.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return "", 0
    return str(out), len(payload)


def _reuse_vllm_block_fp8_roofline_shapes(
    state: Any,
    *,
    workspace: Path,
    current_workload: dict[str, Any] | None = None,
) -> HandlerResult | None:
    """Reuse block-FP8 runtime shapes from the latest Roofline profile trace."""
    last_trace_analyze = getattr(state, "last_trace_analyze", None)
    if not isinstance(last_trace_analyze, dict):
        return None
    source_trace = str(last_trace_analyze.get("steady_state_trace") or "").strip()
    if not source_trace:
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline has no selected "
            "steady-state trace; running a standard Roofline fallback"
        )
        return None
    profile_trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    analyzed_trace = str(last_trace_analyze.get("trace_input") or "").strip()
    try:
        profile_trace_id = str(Path(profile_trace).expanduser().resolve(strict=False))
        analyzed_trace_id = str(Path(analyzed_trace).expanduser().resolve(strict=False))
    except OSError:
        profile_trace_id = profile_trace
        analyzed_trace_id = analyzed_trace
    if not profile_trace or not analyzed_trace or profile_trace_id != analyzed_trace_id:
        log.info(
            "vLLM block-FP8 shape capture: steady-state trace provenance does "
            "not match the latest profile; running a standard Roofline fallback"
        )
        return None
    if str(getattr(state, "last_profile_status", "") or "").strip().lower() != "succeeded":
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline profile is not successful; "
            "running a standard Roofline fallback"
        )
        return None
    profile_workload = getattr(state, "last_profile_workload", None)
    expected_workload = current_workload or state.current_profile_workload_context()
    recorded_workload = profile_workload if isinstance(profile_workload, dict) else {}
    if recorded_workload != expected_workload:
        mismatches = sorted(
            key
            for key in set(recorded_workload) | set(expected_workload)
            if recorded_workload.get(key) != expected_workload.get(key)
        )
        log.info(
            "vLLM block-FP8 shape capture: Roofline workload mismatch (%s); running a standard Roofline fallback",
            ", ".join(mismatches) or "missing profile workload metadata",
        )
        return None
    shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
        Path(source_trace),
        output_dir=workspace,
    )
    if shape_count == 0:
        log.info(
            "vLLM block-FP8 shape capture: Roofline trace %s contains no reusable "
            "block-FP8 shapes; running a standard Roofline fallback",
            source_trace,
        )
        return None
    log.info(
        "vLLM block-FP8 shape capture: reusing %d shape(s) from Roofline trace %s",
        shape_count,
        source_trace,
    )
    return {
        "status": "ok",
        "shapes_json": shapes_json,
        "shape_capture_workspace": str(workspace),
        "shape_count": shape_count,
        "capture_mode": "roofline_profile_reuse",
        "source_profile_trace": source_trace,
    }


def _vllm_dense_shape_capture_required(
    *,
    framework: str,
    model_path: str,
    shapes_json: str,
    tunableop_input: str,
    dry_run: bool,
) -> bool:
    """Return whether Forge needs an automatic TunableOp recording pass."""
    if (
        framework != "vllm"
        or dry_run
        or shapes_json
        or tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    if is_multi_node():
        return False

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path)
    if not summary or bool(summary.get("is_moe")):
        return False
    try:
        hidden_size = int(summary.get("hidden_size") or 0)
        intermediate_size = int(summary.get("intermediate_size") or 0)
    except (TypeError, ValueError):
        return False
    return hidden_size > 0 and intermediate_size > 0


def _pick_shape_capture_port() -> int:
    """Pick a free local port distinct from the production serving port."""
    import socket

    for _ in range(5):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port != 8888:
            return port
    return 18888


def _resolve_shape_capture_port(value: Any) -> int:
    """Resolve an isolated capture port and reject the production port."""
    if value in (None, ""):
        return _pick_shape_capture_port()
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid shape_capture_port: {value!r}") from exc
    if port <= 0 or port > 65535 or port == 8888:
        raise ValueError(f"shape_capture_port must be 1..65535 and not 8888: {port}")
    return port


def _is_tunableop_untuned_row(line: str) -> bool:
    """Recognize a native PyTorch TunableOp offline-input row."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("Validator"):
        return False
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) < 2 or "TunableOp" not in fields[0] or not fields[1]:
        return False
    dimensions = [int(value) for value in re.findall(r"\d+", fields[1])]
    return sum(value > 0 for value in dimensions) >= 3


def _merge_tunableop_untuned_files(base_path: Path) -> int:
    """Merge per-device TunableOp recordings into one deterministic input."""
    rows: list[str] = []
    seen: set[str] = set()
    pattern = f"{base_path.stem}*{base_path.suffix}"
    for path in sorted(base_path.parent.glob(pattern)):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if _is_tunableop_untuned_row(row) and row not in seen:
                seen.add(row)
                rows.append(row)
    if not rows:
        return 0

    tmp_path = base_path.with_suffix(f"{base_path.suffix}.tmp")
    try:
        tmp_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        os.replace(tmp_path, base_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            log.debug(
                "shape capture: failed to remove temporary TunableOp file %s",
                tmp_path,
                exc_info=True,
            )
        return 0
    return len(rows)


async def _capture_vllm_tunableop_shapes(
    *,
    state: Any,
    session_dir: Path,
    payload: dict,
    workspace: Path,
) -> HandlerResult:
    """Record real vLLM GEMMs using TunableOp or a block-FP8 profiler trace."""
    from ..actions.executors.baseline import BaselineExecutor
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    capture_dir = workspace / "shape_capture" / f"attempt-{time.time_ns()}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    untuned_base = capture_dir / "tunableop_untuned.csv"
    results_base = capture_dir / "tunableop_results.csv"
    profile_mode = payload.get("_shape_capture_mode") == "block_fp8_profile"

    config_path = str(payload.get("config_path") or getattr(state, "baseline_config_path", "") or "").strip()
    if not profile_mode and (not config_path or not Path(config_path).is_file()):
        return {
            "status": "failed",
            "decision": "REVERT",
            "requires_e2e_validation": False,
            "error_class": "shape_capture_failed",
            "error": "vLLM TunableOp shape capture requires an existing baseline_config_path",
            "shape_capture_workspace": str(capture_dir),
        }

    current_best = getattr(state, "current_best", None)
    current_best = current_best if isinstance(current_best, dict) else {}
    inherited_envs = dict(current_best.get("extra_envs") or {})
    inherited_envs.update(dict(payload.get("extra_envs") or {}))
    capture_envs = {
        str(key): str(value)
        for key, value in inherited_envs.items()
        if profile_mode or not str(key).startswith(("PYTORCH_TUNABLEOP_", "HL_TUNABLEOP_"))
    }
    if not profile_mode:
        try:
            capture_port = _resolve_shape_capture_port(payload.get("shape_capture_port"))
        except ValueError as exc:
            return {
                "status": "failed",
                "decision": "REVERT",
                "requires_e2e_validation": False,
                "error_class": "shape_capture_failed",
                "error": str(exc),
                "shape_capture_workspace": str(capture_dir),
            }
        capture_envs.update(
            {
                "PORT": str(capture_port),
                "RUN_EVAL": "false",
            }
        )
        capture_envs.update(
            {
                "HL_TUNABLEOP_MODE": "",
                "HL_TUNABLEOP_FILE": "",
                "HL_TUNABLEOP_VERBOSE": "",
                "PYTORCH_TUNABLEOP_ENABLED": "1",
                "PYTORCH_TUNABLEOP_TUNING": "0",
                "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
                "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(untuned_base),
                "PYTORCH_TUNABLEOP_FILENAME": str(results_base),
            }
        )
    for env_name, state_name in (
        ("TP", "tp"),
        ("CONC", "conc"),
        ("ISL", "isl"),
        ("OSL", "osl"),
        ("MAX_MODEL_LEN", "max_model_len"),
    ):
        value = payload.get(state_name)
        if value in (None, ""):
            value = capture_envs.get(env_name)
        if value in (None, ""):
            value = getattr(state, state_name, 0)
        try:
            resolved = int(value or 0)
        except (TypeError, ValueError):
            resolved = 0
        if resolved > 0:
            capture_envs[env_name] = str(resolved)

    try:
        timeout_sec = int(
            payload.get("shape_capture_timeout_sec")
            or os.environ.get("HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC")
            or 1800
        )
    except (TypeError, ValueError):
        timeout_sec = 1800
    timeout_sec = max(60, timeout_sec)

    task_id = f"{str(payload.get('task_id') or workspace.name)}-shape-capture"
    extra_server_args = (
        str(payload.get("extra_server_args") or "")
        if "extra_server_args" in payload
        else str(current_best.get("extra_server_args") or "")
    )
    inherited_unset = payload.get("unset_envs", current_best.get("unset_envs")) or []
    if isinstance(inherited_unset, str):
        capture_unset_envs = [inherited_unset]
    else:
        capture_unset_envs = [str(key) for key in inherited_unset]
    if not profile_mode:
        capture_unset_envs.extend(
            [
                "HL_TUNABLEOP_MODE",
                "HL_TUNABLEOP_FILE",
                "HL_TUNABLEOP_VERBOSE",
                "PYTORCH_TUNABLEOP_ENABLED",
                "PYTORCH_TUNABLEOP_TUNING",
                "PYTORCH_TUNABLEOP_RECORD_UNTUNED",
                "PYTORCH_TUNABLEOP_UNTUNED_FILENAME",
                "PYTORCH_TUNABLEOP_FILENAME",
            ]
        )
    inherited_remove = payload.get("remove_args", current_best.get("remove_args")) or []
    if isinstance(inherited_remove, str):
        capture_remove_args = [inherited_remove]
    else:
        capture_remove_args = [str(arg) for arg in inherited_remove]
    if not profile_mode:
        capture_remove_args.append("--port")
    task_params: dict[str, Any] = {
        "output_dir": str(capture_dir),
        "framework": "vllm",
        "model_path": str(payload.get("model_path") or getattr(state, "model_path", "") or ""),
        "gpu_type": str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or ""),
        "extra_server_args": extra_server_args,
        "extra_envs": capture_envs,
        "remove_args": capture_remove_args,
        "unset_envs": capture_unset_envs,
        "args_mode": str(payload.get("args_mode") or current_best.get("args_mode") or "append"),
    }
    if profile_mode:
        task_params["workspace_path"] = str(capture_dir / "tracelens")
        last_baseline = getattr(state, "last_baseline", None)
        if isinstance(last_baseline, dict):
            benchmark_script = str(last_baseline.get("benchmark_script") or "").strip()
            if benchmark_script:
                task_params["benchmark_script"] = benchmark_script
    else:
        from ..actions.executors.baseline import SBD_INNER_STEP_PARAM

        task_params.update(
            {
                "config_path": config_path,
                "timeout_sec": timeout_sec,
                "disable_run_eval": True,
                "baseline_double_run": False,
                # Shape capture is a sub-step of the KERNEL phase's own event, not a dispatched measurement, so it
                # leaves no baseline event.
                SBD_INNER_STEP_PARAM: True,
            }
        )
    task = Task(
        task_id=task_id,
        kind="gemm_shape_capture",
        state="running",
        params=task_params,
        idempotency_key=f"{task_id}-run",
    )
    ctx = RunnerContext(task=task, lease=None)
    import copy

    if profile_mode:
        capture_state = state
    else:
        capture_state = copy.deepcopy(state)
        capture_state.baseline_eager_fallback = False
    ctx.extra = {
        "shared_state": capture_state,
        "session_dir": session_dir,
        "workspace": capture_dir,
    }

    try:
        if profile_mode:
            from ..actions.executors.roofline import RooflineExecutor

            benchmark_result = await RooflineExecutor(
                shared_state=capture_state,
            )(ctx)
        else:
            benchmark_result = await BaselineExecutor(
                session_dir=session_dir,
                shared_state=capture_state,
            )(ctx)
    except Exception as exc:  # noqa: BLE001 - convert capture launch faults to a stable result
        return {
            "status": "failed",
            "decision": "REVERT",
            "requires_e2e_validation": False,
            "error_class": "shape_capture_failed",
            "error": f"vLLM TunableOp shape capture raised {exc!r}",
            "shape_capture_workspace": str(capture_dir),
        }

    if not isinstance(benchmark_result, dict):
        benchmark_result = {}
    if profile_mode:
        steady_state_trace = str(benchmark_result.get("steady_state_trace") or "").strip()
        if steady_state_trace:
            shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
                Path(steady_state_trace),
                output_dir=capture_dir,
            )
        else:
            shapes_json, shape_count = "", 0
        if benchmark_result.get("status") == "succeeded" and shape_count > 0:
            return {
                "status": "ok",
                "shapes_json": shapes_json,
                "shape_capture_workspace": str(capture_dir),
                "shape_count": shape_count,
                "capture_mode": "block_fp8_profile",
                "source_profile_trace": steady_state_trace,
            }
        benchmark_error = str(benchmark_result.get("error") or benchmark_result.get("error_class") or "").strip()
        detail = f": {benchmark_error}" if benchmark_error else ""
        return {
            "status": "failed",
            "decision": "REVERT",
            "requires_e2e_validation": False,
            "error_class": "shape_capture_failed",
            "error": f"vLLM block-FP8 profile capture produced no structured GEMM shapes{detail}",
            "shape_capture_workspace": str(capture_dir),
            "shape_count": shape_count,
            "capture_mode": "block_fp8_profile",
        }

    row_count = _merge_tunableop_untuned_files(untuned_base)
    if benchmark_result.get("status") != "succeeded" or row_count == 0:
        try:
            untuned_base.unlink(missing_ok=True)
        except OSError:
            log.debug(
                "shape capture: failed to remove incomplete TunableOp recording %s",
                untuned_base,
                exc_info=True,
            )
        benchmark_error = str(benchmark_result.get("error") or benchmark_result.get("error_class") or "").strip()
        detail = f": {benchmark_error}" if benchmark_error else ""
        return {
            "status": "failed",
            "decision": "REVERT",
            "requires_e2e_validation": False,
            "error_class": "shape_capture_failed",
            "error": f"vLLM TunableOp shape capture produced no complete workload recording{detail}",
            "shape_capture_workspace": str(capture_dir),
            "shape_count": row_count,
        }

    return {
        "status": "ok",
        "tunableop_input": str(untuned_base),
        "shape_capture_workspace": str(capture_dir),
        "shape_count": row_count,
    }


async def _run_forge_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Deterministic GEMM tuning via the ``kernelforge gemm-tune`` CLI."""
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    # Importing kernelforge.cli is deliberately isolated in a subprocess, but that subprocess may still take until the
    # bounded timeout to fail.
    if not await asyncio.to_thread(_forge_gemm_tune_available):
        return {
            "status": "failed",
            "error_class": "forge_gemm_tune_not_found",
            "error": (
                "forge-gemm-tune is not runnable in this interpreter: "
                f"'{sys.executable} -m kernelforge.cli gemm-tune --help' failed. "
                "kernelforge ships with this distribution, so this means a "
                "partial install: reinstall with 'pip install -e .[forge]'."
                f" (interpreter: {sys.executable!r})"
            ),
            "backend": "forge",
        }

    # Resolve precision from actual runtime, not just session-level state.
    precision, quant_type = _resolve_forge_precision_and_quant(state, payload)
    framework = str(payload.get("framework") or state.framework or "sglang").strip().lower()

    workspace = _gemm_tuning_workspace(payload, session_dir=session_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    raw_model_path = str(payload.get("model_path") or state.model_path or os.environ.get("MODEL_PATH") or "").strip()
    if not raw_model_path:
        return {"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"}
    from hyperloom.common.model_paths import resolve_serving_model_path
    from hyperloom.inference_optimizer.model_config_utils import (
        resolve_local_model_dir,
    )

    # Bootstrap already walked HL_MODEL_BASE and the hub cache to decide what to serve; probing only the hub cache
    # here would reject a repo id that the running server resolved fine.
    resolved_model_dir = resolve_local_model_dir(resolve_serving_model_path(raw_model_path) or raw_model_path)
    if resolved_model_dir is None:
        # Forge needs the config on disk to derive shapes, so it cannot run -- but not running one tuning backend is a
        # skip, not a session failure.
        return {
            "status": "skipped",
            "error_class": "model_path_unavailable",
            "skip_reason": (
                f"Model path {raw_model_path!r} is neither an existing local "
                "directory nor an available Hugging Face cache snapshot"
            ),
            "backend": "forge",
        }
    resolved_model_path = str(resolved_model_dir)

    tp = int(payload.get("tp") or state.tp or os.environ.get("TP") or 1)
    conc = int(payload.get("conc") or state.conc or os.environ.get("CONC") or 64)
    gpu_type = str(payload.get("gpu_type") or state.gpu_type or os.environ.get("GPU_TYPE") or "mi300x").strip().lower()
    tokens = _normalize_tokens(payload.get("tokens"))
    # Default mp = all visible GPUs.
    from ..policy.gate import detect_gpu_count

    detected_gpus = detect_gpu_count() or tp
    mp = int(payload.get("mp") or os.environ.get("FORGE_GEMM_TUNE_MP") or detected_gpus)

    # Resolve server log for 1-stage ASM detection.
    kernel_sig_log = str(payload.get("kernel_signature_log") or "").strip()
    if not kernel_sig_log:
        # Off the event loop: this walks runs/ and byte-scans server logs that measure ~17MB apiece on the fleet.
        kernel_sig_log = await asyncio.to_thread(_resolve_forge_server_log, state, session_dir)

    # Explicit operator/benchmark input wins.
    shapes_json = _normalize_forge_shapes_json(payload.get("shapes_json"), workspace)
    untuned_csv = str(payload.get("untuned_csv") or "").strip()
    if untuned_csv and not _path_is_existing_file(untuned_csv):
        # Guard against inline content / stale paths.
        untuned_csv = ""
    if not shapes_json and not untuned_csv and framework != "vllm":
        shapes_json = _resolve_forge_shapes(
            state,
            session_dir,
            require_fresh_profile=True,
            precision=precision,
        )
        if not shapes_json:
            untuned_csv = _resolve_forge_untuned_csv(
                session_dir,
                precision,
                quant_type,
                resolved_model_path,
            )

    # forge's own fallback derives --tokens from ``conc``, which is a guess about M.
    if not tokens and kernel_sig_log:
        tokens = _normalize_tokens(await asyncio.to_thread(_tokens_from_serving_log, kernel_sig_log))
        if tokens:
            log.info("GEMM: derived --tokens=%s from observed M in %s", tokens, kernel_sig_log)

    # MoE shapes come from the runtime, never from inference.
    moe_untuned_csv = str(payload.get("moe_untuned_csv") or "").strip()
    if moe_untuned_csv and not _path_is_existing_file(moe_untuned_csv):
        moe_untuned_csv = ""
    moe_key_report: dict[str, Any] = {}
    if not moe_untuned_csv:
        moe_untuned_csv, moe_key_report = _write_fmoe_untuned_csv_from_log(kernel_sig_log, tokens, workspace)

    tunableop_input = str(payload.get("tunableop_input") or "").strip()
    forge_framework = _forge_framework_for_vllm(
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        tunableop_input=tunableop_input,
        **_resolve_vllm_aiter_routing(
            model_path=resolved_model_path,
            server_log=kernel_sig_log,
            tp=tp,
        ),
    )
    shape_capture: HandlerResult | None = None
    block_fp8_profile_capture = _vllm_block_fp8_profile_capture_required(
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        shapes_json=shapes_json,
        tunableop_input=tunableop_input,
        dry_run=bool(payload.get("dry_run")),
    )
    if block_fp8_profile_capture:
        # Decode steps replay inside a CUDA Graph and therefore emit no Kineto *op* events, so every profile-derived
        # block-FP8 shape set structurally carries prefill M only -- measured on a real capture, the decode-only trace
        # split yields zero block-FP8 events while the prefill splits yield M=2095.
        traced_shapes = _resolve_forge_shapes(
            state,
            session_dir,
            require_fresh_profile=True,
            precision=precision,
        )
        if traced_shapes:
            shapes_json = traced_shapes
            untuned_csv = ""
            block_fp8_profile_capture = False
    if block_fp8_profile_capture:
        shape_capture = _reuse_vllm_block_fp8_roofline_shapes(
            state,
            workspace=workspace,
            current_workload=state.current_profile_workload_context(payload),
        )
        if shape_capture is not None:
            shapes_json = str(shape_capture["shapes_json"])
            untuned_csv = ""
            block_fp8_profile_capture = False
    tunableop_capture = (
        # Keyed on the routed framework: a run handed to the AITER tuner family has no use for a TunableOp recording
        # pass, and paying for one costs a full extra server boot.
        _vllm_dense_shape_capture_required(
            framework=forge_framework,
            model_path=resolved_model_path,
            shapes_json=shapes_json,
            tunableop_input=tunableop_input,
            dry_run=bool(payload.get("dry_run")),
        )
        and not block_fp8_profile_capture
    )
    if block_fp8_profile_capture or tunableop_capture:
        capture_payload = dict(payload)
        if block_fp8_profile_capture:
            capture_payload["_shape_capture_mode"] = "block_fp8_profile"
        shape_capture = await _capture_vllm_tunableop_shapes(
            state=state,
            session_dir=session_dir,
            payload=capture_payload,
            workspace=workspace,
        )
        if shape_capture.get("status") != "ok":
            shape_capture.setdefault("backend", "forge")
            shape_capture.setdefault("engine", "forge")
            shape_capture.setdefault("workspace", str(workspace))
            shape_capture.setdefault("precision", precision)
            shape_capture.setdefault("framework", framework)
            shape_capture.setdefault("model_path", raw_model_path)
            return shape_capture
        tunableop_input = str(shape_capture.get("tunableop_input") or "").strip()
        captured_shapes = str(shape_capture.get("shapes_json") or "").strip()
        if captured_shapes:
            shapes_json = captured_shapes
            # Forge dense tuners prefer untuned_csv over shapes_json.
            untuned_csv = ""

    # forge prefers the manifest over shapes_json as a dense-shape source, and an explicit demand.json over
    # re-deriving demand from the serving log.
    shapes_manifest = str(payload.get("shapes_manifest") or "").strip()
    if not shapes_manifest:
        # Scavenge one from the session only when nothing more specific was produced for THIS run. forge ranks the
        # manifest at priority 0 on the premise that it was explicitly supplied; a manifest found by walking the
        # session tree carries no such promise -- it can come from an earlier run at a different precision or with
        # different server args, and there is no consistency check to catch that.
        if untuned_csv or shapes_json:
            log.debug(
                "GEMM: not scavenging a trace shape manifest; this run already has "
                "a workload-matched dense-shape source (%s)",
                "untuned_csv" if untuned_csv else "shapes_json",
            )
        else:
            # Off the event loop for the same reason: a ``**/`` walk of a session tree that holds thousands of run
            # artifacts.
            shapes_manifest = await asyncio.to_thread(_resolve_trace_shape_manifest, state, session_dir)
    if shapes_manifest and not _path_is_existing_file(shapes_manifest):
        shapes_manifest = ""
    demand_json = str(payload.get("demand_json") or "").strip()
    if demand_json and not _path_is_existing_file(demand_json):
        demand_json = ""

    # The lane's share, priced on the router's own per-tuner estimates.
    gemm_targets = await asyncio.to_thread(
        _gemm_router_targets,
        model_path=resolved_model_path,
        framework=forge_framework,
        precision=precision,
        quant_type=quant_type,
        gpu_type=gpu_type,
        kernel_signature_log=kernel_sig_log,
        has_untuned_csv=bool(untuned_csv),
        has_shapes_json=bool(shapes_json or shapes_manifest or demand_json),
        has_tunableop_input=bool(tunableop_input),
    )
    gemm_lane = _lane_budget(
        state,
        LANE_GEMM,
        gemm_target_costs_sec=tuple(cost for _, cost in gemm_targets),
    )
    timeout = _gemm_tuning_timeout_sec(
        payload,
        lane_budget_sec=gemm_lane.budget_sec if gemm_lane.is_fundable else 0,
    )
    try:
        requested_tuners = int(payload.get("max_tuners") or 0)
    except (TypeError, ValueError):
        requested_tuners = 0
    # An explicit payload value stays an operator/test escape hatch; a named ``tuner`` already narrows the set to one,
    # so no ceiling is needed then.
    if str(payload.get("tuner") or "").strip():
        gemm_tuner_ceiling = 0
    else:
        gemm_tuner_ceiling = requested_tuners if requested_tuners > 0 else gemm_lane.max_targets
    session_max_min = float(getattr(state, "max_minutes", 0) or 0)
    shape_alignment: dict[str, Any] | None = None
    if shapes_json:
        shapes_json, shape_alignment = _align_forge_shapes_for_aiter(
            shapes_json,
            forge_framework=forge_framework,
            workspace=workspace,
            budget_sec=timeout,
            mp=mp,
        )

    input_payload = {
        "model_path": resolved_model_path,
        "framework": forge_framework,
        "precision": precision,
        "quant_type": quant_type,
        "gpu_type": gpu_type,
        "tp": tp,
        "conc": conc,
        "mp": mp,
        "output_dir": str(workspace),
        # Passing the same value to both made the producer's own min(per_tuner, remaining) an identity, so the first
        # tuner could consume the entire session and every later one was skipped for lack of time.
        "timeout": gemm_per_tuner_timeout_sec(timeout),
        # Bounds the whole session across all tuners.
        "global_timeout": timeout,
        "skip_gpu_check": True,
        "tokens": tokens,
        "untuned_csv": untuned_csv,
        "moe_untuned_csv": moe_untuned_csv,
        "shapes_json": shapes_json,
        "shapes_manifest": shapes_manifest,
        "demand_json": demand_json,
        "tunableop_input": tunableop_input,
        "kernel_signature_log": kernel_sig_log,
        "tuner": str(payload.get("tuner") or ""),
        # How many routed tuners the lane's share pays for.
        **({"max_tuners": gemm_tuner_ceiling} if gemm_tuner_ceiling > 0 else {}),
        # Exhaustive search when budget allows (>= 24h) and mp >= 4.
        "thorough": bool(session_max_min >= 1440 and mp >= 4),
    }
    input_json = workspace / "forge_gemm_tuning_input.json"
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        sys.executable,
        str(_kernel_agent_tool_path("forge_gemm_tuning.py")),
        "--input-json",
        str(input_json),
    ]
    aiter_root = _resolve_aiter_root_for_forge()
    if aiter_root:
        cmd = ["env", f"AITER_ROOT_DIR={aiter_root}", *cmd]

    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout)
        result = _parse_forge_gemm_sentinel(stdout)
        if result is None:
            result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess; shape a failed result.
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout}s: {cmd_repr[:1500]}",
        }

    result.setdefault("backend", "forge")
    # Tag the tuning engine so the breakdown attributes this run to forge.
    result.setdefault("engine", "forge")
    result.setdefault("workspace", str(workspace))
    result.setdefault("precision", precision)
    result.setdefault("framework", framework)
    result.setdefault("tuning_framework", forge_framework)
    result.setdefault("model_path", raw_model_path)
    if moe_key_report:
        # Kept even when nothing was tunable: "no MoE problem was observed" and "the observed pair is one aiter cannot
        # tune" lead to different actions, and neither is visible from the tuner's own status.
        result.setdefault("moe_key_source", moe_key_report)
    if shape_alignment is not None:
        result.setdefault("shape_alignment", shape_alignment)
    if shape_capture is not None:
        result.setdefault(
            "shape_capture",
            {
                "status": "ok",
                "workspace": shape_capture.get("shape_capture_workspace"),
                "tunableop_input": tunableop_input,
                "shapes_json": shapes_json,
                "shape_count": shape_capture.get("shape_count"),
                "capture_mode": shape_capture.get("capture_mode", "tunableop"),
                "source_profile_trace": shape_capture.get("source_profile_trace"),
            },
        )

    # Surface why forge skipped: merge per-tuner skip reasons from the on-disk result.json and derive a top-level
    # skip_reason.
    if not result.get("tuners_skipped"):
        disk_skipped = _read_forge_result_json(workspace).get("tuners_skipped")
        if disk_skipped:
            result["tuners_skipped"] = disk_skipped
    if not result.get("skip_reason"):
        reason = _derive_gemm_skip_reason(result.get("tuners_skipped"))
        if reason:
            result["skip_reason"] = reason

    # Surface crashed tuners. forge lists every failure in ``failed_tuners`` regardless of the overall decision, but
    # this array was previously dropped here -- so a dense tuner winning made a MoE tuner's crash invisible, and a
    # KEEP read as "no headroom elsewhere" when siblings had in fact hard-failed.
    if not result.get("failed_tuners"):
        disk_failed = _read_forge_result_json(workspace).get("failed_tuners")
        if disk_failed:
            result["failed_tuners"] = disk_failed
    _failed_tuners = result.get("failed_tuners")
    if isinstance(_failed_tuners, list) and _failed_tuners:
        for _f in _failed_tuners:
            if not isinstance(_f, dict):
                continue
            log.warning(
                "forge gemm tuner %s failed (%s): %s",
                _f.get("tuner") or "?",
                _f.get("error_class") or "?",
                _f.get("error") or "",
            )

    # The breakdown and the stack read the envelope, not the jsonl audit row, so a tuner's own error class has to
    # surface here too.
    _tuner_rows = result.get("tuners_run")
    if not isinstance(_tuner_rows, list):
        _tuner_rows = []
    if not result.get("error_class"):
        for _t in _tuner_rows:
            if isinstance(_t, dict) and _t.get("error_class"):
                result["error_class"] = str(_t["error_class"])
                break
    if not result.get("error"):
        for _t in _tuner_rows:
            if isinstance(_t, dict) and _t.get("error"):
                result["error"] = str(_t["error"])
                break

    # Bridge forge schema → coordinator schema: a "candidate" micro_decision with recommended_env becomes
    # decision="KEEP" + extra_envs.
    micro = str(result.get("micro_decision") or "").strip().lower()
    if micro == "candidate" and result.get("recommended_env"):
        result.setdefault("decision", "KEEP")
        # Make the tuned CSV durable + recipe-portable (mirrors integrate_patch's source-layer snapshot): copy it into
        # the serving aiter config dir, repoint the env there, and snapshot it so the KEEP survives with the recipe
        # instead of referencing the ephemeral tuner-workspace path.
        _durable_envs, _snap_dir = _persist_forge_gemm_csv_durably(
            dict(result["recommended_env"]),
            model_path=raw_model_path,
            session_dir=session_dir,
        )
        result.setdefault("extra_envs", _durable_envs)
        if _snap_dir:
            result.setdefault("source_snapshot", _snap_dir)
        # Derive best_speedup from tuners_run when absent.
        if "best_speedup" not in result:
            best = 1.0
            for t in result.get("tuners_run") or []:
                if isinstance(t, dict):
                    sp = float(t.get("best_micro_speedup") or 1.0)
                    if sp > best:
                        best = sp
            if best > 1.0:
                result["best_speedup"] = best
        # Micro-only result: E2E validation still needed.
        result.setdefault("requires_e2e_validation", True)
    elif micro in ("no_improvement", "skipped"):
        # Left unadorned on purpose: the wordings below are only legible against it.
        result.setdefault("decision", "REVERT")
    elif micro in _FORGE_BARREN_MICRO_DECISIONS:
        result.setdefault("decision", "REVERT")
        if micro == "failed":
            result.setdefault("status", "failed")
        result.setdefault("error_class", f"forge_{micro}")

    return result


def _persist_forge_gemm_csv_durably(extra_envs: dict, *, model_path: str, session_dir: Path) -> tuple[dict, str]:
    """Make forge GEMM tuned CSVs durable + recipe-portable."""
    # Below model_configs/, out of reach of aiter's non-recursive auto-merge glob.
    _FORGE_DURABLE_SUBDIR = "hyperloom"
    _forge_durable_env_stems = {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm",
        "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm",
        "AITER_CONFIG_FMOE": "tuned_fmoe",
    }
    slug = (
        "".join(c if (c.isalnum() or c in "._-") else "_" for c in Path(model_path).name).strip("_").lower() or "model"
    )

    pending: list[tuple[str, str, Path]] = []
    for env_key, stem in _forge_durable_env_stems.items():
        src_csv = str(extra_envs.get(env_key) or "").strip()
        if not src_csv or not Path(src_csv).is_file():
            continue
        rel = f"configs/model_configs/{_FORGE_DURABLE_SUBDIR}/{stem}_{slug}.csv"
        pending.append((env_key, rel, Path(src_csv)))
    if not pending:
        return extra_envs, ""

    # Step 1 -- commit durable copies + env repoints.
    try:
        import importlib.util

        spec = importlib.util.find_spec("aiter")
        if spec is None or not spec.origin:
            return extra_envs, ""
        aiter_pkg = Path(spec.origin).resolve().parent
        updated = dict(extra_envs)
        rel_paths: list[str] = []
        for env_key, rel, src_path in pending:
            dst = aiter_pkg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            updated[env_key] = str(dst)
            rel_paths.append(rel)
    except Exception:  # noqa: BLE001 — durability is best-effort; never break the KEEP
        log.exception("forge gemm CSV durable-copy failed; keeping workspace path")
        return extra_envs, ""

    # Step 2 -- recipe-portability snapshot.
    snap_dir = ""
    try:
        from ..source_snapshot import snapshot_source_layer

        snap = snapshot_source_layer(
            framework_root=aiter_pkg,
            base_sha=None,
            rel_paths=rel_paths,
            dest_dir=Path(session_dir) / "optimization_stack" / "src" / f"forge_gemm_{slug}",
            provenance="kernelforge.gemm_tune",
            extra={
                "env_keys": [env_key for env_key, _, _ in pending],
                "model": slug,
            },
        )
        snap_dir = str((snap or {}).get("snapshot_dir") or "")
    except Exception:  # noqa: BLE001 — snapshot is best-effort; the repoint above stands
        log.exception("forge gemm CSV snapshot failed; durable copy + repoint kept")
    return updated, snap_dir


async def _run_geak_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Legacy GEAK GEMM tuning wrapper."""
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    precision = _normalize_precision(payload.get("precision") or state.precision)
    framework = str(payload.get("framework") or state.framework or "sglang").strip().lower()
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
        payload.get("benchmark_script") or os.environ.get("GEAK_GEMM_BENCHMARK_SCRIPT") or ""
    ).strip()
    if not benchmark_script:
        if not gpu_type:
            gpu_type = "mi355x"
        benchmark_script = str(
            _write_gemm_tuning_benchmark_script(
                workspace=workspace,
                model_path=model_path,
                framework=framework,
                gpu_type=gpu_type,
                tp=tp,
                conc=conc,
                isl=isl,
                osl=osl,
            )
        )
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
        "env": {"E2E_METRIC": "output"},
    }
    if geak_config:
        input_payload["config"] = geak_config
    elif not payload.get("dry_run"):
        return {
            "status": "skipped",
            "decision": "REVERT",
            "backend": "geak",
            "engine": "geak",
            "error_class": "legacy_geak_config_missing",
            "error": (
                "GEAK GEMM tuning requires GEAK_CONFIG. "
                "Forge fallback is disabled unless KERNEL_OPT_BACKEND_ORDER=forge."
            ),
            "workspace": str(workspace),
            "precision": precision,
            "framework": framework,
            "model_path": model_path,
            "benchmark_script": benchmark_script,
        }
    if payload.get("dry_run"):
        input_payload["dry_run"] = True
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        "env",
        "E2E_METRIC=output",
        "python3",
        str(_kernel_agent_tool_path("gemm_tuning.py")),
        "--input-json",
        str(input_json),
    ]

    _gemm_timeout = _gemm_tuning_timeout_sec(payload)
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=_gemm_timeout)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {_gemm_timeout}s: {cmd_repr[:1500]}",
        }
    result.setdefault("backend", "geak")
    result.setdefault("engine", "geak")
    result.setdefault("workspace", str(workspace))
    result.setdefault("precision", precision)
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)
    result.setdefault("benchmark_script", benchmark_script)
    return result


async def run_gemm_tuning_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run GEMM tuning via GEAK, or forge only when explicitly enabled."""
    backend = _resolve_gemm_tuning_backend(payload)
    log.info("run_gemm_tuning: backend=%s", backend)
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_gemm_tuning_operation(
            session_dir,
            payload={**payload, "gemm_tuning_backend": backend},
        )
    except Exception:  # noqa: BLE001
        log.debug("gemm v4 start recording failed", exc_info=True)

    if backend == "forge":
        result = await _run_forge_gemm_tuning(payload, session_dir=session_dir)
    else:
        result = await _run_geak_gemm_tuning(payload, session_dir=session_dir)
    result.setdefault("task_id", payload.get("task_id"))
    result.setdefault("macro_cycle", payload.get("macro_cycle"))
    _trace_gemm_tuning_run(result, session_dir=session_dir)
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_gemm_tuning_operation(
            session_dir,
            payload={**payload, "gemm_tuning_backend": backend},
            result=result,
        )
    except Exception:  # noqa: BLE001
        log.debug("gemm v4 result recording failed", exc_info=True)
    return result


# forge-fusion (autonomous kernel fusion)
_FORGE_FUSION_RESULT_RE = re.compile(r"FORGE_FUSION_RESULT_BEGIN\s*\n(.*?)\nFORGE_FUSION_RESULT_END", re.DOTALL)


def _forge_fusion_available() -> bool:
    """Check that KernelForge's fusion pipeline is importable."""
    try:
        return importlib.util.find_spec("kernelforge.fusion") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _parse_forge_fusion_sentinel(stdout: str) -> dict[str, Any] | None:
    """Parse the FORGE_FUSION_RESULT_BEGIN/END sentinel block from stdout."""
    m = _FORGE_FUSION_RESULT_RE.search(stdout)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _resolve_fusion_decode_trace(state, payload: dict) -> str:
    """Reuse the PRELUDE/roofline decode trace for fusion discovery."""

    def _trace_file(path_str: str) -> str:
        path = Path(path_str)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            return ""
        candidates = sorted(
            list(path.glob("*.trace.json.gz")) + list(path.glob("*.trace.json")) + list(path.glob("*.json.gz")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(candidates[0]) if candidates else ""

    explicit = str(payload.get("trace_path") or "").strip()
    if explicit:
        resolved = _trace_file(explicit)
        if resolved:
            return resolved
    trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    if trace:
        resolved = _trace_file(trace)
        if resolved:
            return resolved
    return ""


def _active_forge_fusion_env_flags(state: Any) -> dict[str, str]:
    """Return active env flags only when forge-fusion itself is current_best."""
    current_best = getattr(state, "current_best", None) or {}
    if not isinstance(current_best, dict):
        return {}
    if str(current_best.get("action") or "") != "fusion":
        return {}
    envs = current_best.get("extra_envs") if isinstance(current_best, dict) else {}
    if not isinstance(envs, dict):
        return {}
    active: dict[str, str] = {}
    for key, val in envs.items():
        name = str(key)
        value = str(val)
        if "_FUSED" not in name.upper():
            continue
        if value.strip().lower() in ("", "0", "false", "no", "off", "none"):
            continue
        active[name] = value
    return active


def _resolve_forge_agent(
    payload: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the Forge agent backend and model as one decision."""
    source = env if env is not None else os.environ
    openai_only = llm_config.is_openai_only(source)
    anthropic_only = llm_config.is_anthropic_only(source)
    has_openai = llm_config.has_openai_side(source)
    has_anthropic = llm_config.has_anthropic_side(source)
    if not has_openai and not has_anthropic:
        raise RuntimeError("no LLM provider is configured for forge")

    explicit_backend = str(payload.get("agent_backend") or "").strip().lower()
    if explicit_backend and explicit_backend not in {"claude", "codex"}:
        raise ValueError(f"agent_backend={payload.get('agent_backend')!r} is invalid; choose 'claude' or 'codex'")

    if explicit_backend:
        agent_backend = explicit_backend
    elif openai_only:
        agent_backend = "codex"
    elif anthropic_only:
        agent_backend = "claude"
    else:
        # Dual-configured deployments retain this agentic role's Claude default.
        agent_backend = "claude"

    default_model = DEFAULT_CODEX_MODEL if agent_backend == "codex" else DEFAULT_CLAUDE_MODEL
    llm_model = llm_config.resolve_forge_llm_model(
        agent_backend,
        env=source,
        explicit=str(payload.get("llm_model") or ""),
        default=default_model,
    )
    return agent_backend, llm_model


def _resolve_forge_fusion_sandbox_mode(
    payload: Mapping[str, Any],
    *,
    agent_backend: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve and validate the sandbox policy recorded for forge-fusion."""
    explicit = str(payload.get("agent_sandbox_mode") or "").strip()
    if agent_backend == "claude" and not explicit:
        return codex_session.DEFAULT_CODEX_SANDBOX_MODE
    return codex_session.resolve_codex_sandbox_mode(
        sandbox_mode=explicit,
        env=dict(env) if env is not None else None,
    )


async def _run_forge_fusion(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Autonomous kernel fusion via the forge-fusion CLI."""
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    active_fusion_flags = _active_forge_fusion_env_flags(state)
    if active_fusion_flags:
        return {
            "status": "complete",
            "backend": "forge",
            "engine": "forge_fusion",
            "micro_decision": "already_active",
            "decision": "REVERT",
            "kept": False,
            "requires_e2e_validation": False,
            "active_env_flags": active_fusion_flags,
            "reason": (
                "current_best is already a forge-fusion KEEP; "
                "skip forge-fusion to avoid rerunning the same adopted source patch"
            ),
            "source": "forge_fusion",
        }

    if not _forge_fusion_available():
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "forge_fusion_not_found",
            "error": ("KernelForge fusion pipeline not found. Install via 'pip install <KernelForge>[claude,codex]'."),
            "decision": "REVERT",
            "kept": False,
        }

    model_path = str(payload.get("model_path") or state.model_path or os.environ.get("MODEL_PATH") or "").strip()
    if not model_path:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "model_path_missing",
            "error": "model_path is required",
            "decision": "REVERT",
            "kept": False,
        }

    trace_path = _resolve_fusion_decode_trace(state, payload)
    if not trace_path:
        return {
            "status": "skipped",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "decode_trace_missing",
            "error": (
                "no decode trace available for fusion discovery "
                "(state.last_profile_trace empty; run profile/roofline first)"
            ),
            "decision": "REVERT",
            "kept": False,
        }

    framework = str(payload.get("framework") or state.framework or "sglang").strip().lower()
    gpu = str(payload.get("gpu") or "0").strip()
    try:
        agent_backend, llm_model = _resolve_forge_agent(payload)
    except (RuntimeError, ValueError) as exc:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": ("llm_provider_unconfigured" if isinstance(exc, RuntimeError) else "invalid_agent_backend"),
            "error": str(exc),
            "decision": "REVERT",
            "kept": False,
        }
    try:
        agent_sandbox_mode = _resolve_forge_fusion_sandbox_mode(
            payload,
            agent_backend=agent_backend,
        )
    except RuntimeError as exc:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "invalid_agent_sandbox_mode",
            "error": str(exc),
            "decision": "REVERT",
            "kept": False,
        }
    max_turns = int(payload.get("max_turns") or os.environ.get("FORGE_FUSION_MAX_TURNS") or 100)
    # The lane's share of the phase; a zero share means none could be derived, and the module default is safer for an
    # unattended lane than a one-second session.
    fusion_lane = _lane_budget(state, LANE_FUSION)
    timeout = _forge_fusion_timeout_sec(payload, lane_budget_sec=fusion_lane.budget_sec)
    try:
        requested_recipes = int(payload.get("max_recipes") or 0)
    except (TypeError, ValueError):
        requested_recipes = 0
    # An explicit payload value stays an operator/test escape hatch.
    fusion_recipe_ceiling = requested_recipes if requested_recipes > 0 else fusion_lane.max_targets

    workspace = session_dir / "runs" / "fusion" / str(payload.get("task_id") or "kernel_entry_fusion")
    workspace.mkdir(parents=True, exist_ok=True)

    input_payload = {
        "trace_path": trace_path,
        "model_path": model_path,
        "framework": framework,
        "output_dir": str(workspace),
        "discover_mode": str(payload.get("discover_mode") or "llm"),
        "agent_backend": agent_backend,
        "llm_model": llm_model,
        "agent_sandbox_mode": agent_sandbox_mode,
        "max_turns": max_turns,
        "gpu": gpu,
        "timeout": timeout,
        # Multi-patch (one independent sibling per recipe) is the default; the combine escape hatch (a single merged
        # patch) must be requested explicitly.
        "fuse_all_confirmed": bool(payload.get("fuse_all_confirmed", False)),
        # How many recipes the lane's share pays for.
        **({"max_recipes": fusion_recipe_ceiling} if fusion_recipe_ceiling > 0 else {}),
        "verbose": bool(payload.get("verbose", False)),
        **_fusion_session_serve_args(state, payload, framework=framework, model_path=model_path),
    }
    input_json = workspace / "forge_fusion_input.json"
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")

    cmd = ["python3", str(_kernel_agent_tool_path("forge_fusion.py")), "--input-json", str(input_json)]

    wrapper_timeout = _forge_fusion_wrapper_timeout_sec(timeout)
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=wrapper_timeout)
        result = _parse_forge_fusion_sentinel(stdout)
        if result is None:
            result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        from hyperloom.agents.kernel.tools.forge_fusion import (  # noqa: PLC0415
            salvage_forge_fusion_from_workspace,
        )

        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        timeout_error = f"TimeoutExpired after {wrapper_timeout}s: {cmd_repr[:1500]}"
        salvaged = salvage_forge_fusion_from_workspace(str(workspace))
        if salvaged:
            result = {
                **salvaged,
                "error_class": "subprocess_timeout",
                "error": timeout_error,
            }
        else:
            result = {
                "status": "failed",
                "backend": "forge",
                "engine": "forge_fusion",
                "error_class": "subprocess_timeout",
                "error": timeout_error,
                "decision": "REVERT",
                "kept": False,
            }

    result.setdefault("backend", "forge")
    result.setdefault("engine", "forge_fusion")
    result.setdefault("workspace", str(workspace))
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)
    result.setdefault("agent_backend", agent_backend)
    result.setdefault("llm_model", llm_model)
    result.setdefault("agent_sandbox_mode", agent_sandbox_mode)
    result.setdefault("source", "forge_fusion")
    return result


async def run_fusion_handler(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Run autonomous kernel fusion via forge-fusion (serving-validated)."""
    return await _run_forge_fusion(payload, session_dir=session_dir)


def _parse_forge_collective_sentinel(stdout: str) -> dict[str, Any]:
    """Parse and validate the collective wrapper result sentinel."""
    match = re.search(
        r"FORGE_COLLECTIVE_RESULT_BEGIN\s*\n(.*?)\nFORGE_COLLECTIVE_RESULT_END",
        stdout,
        re.DOTALL,
    )
    if not match:
        raise ValueError("collective wrapper emitted no result sentinel")
    try:
        parsed = json.loads(match.group(1))
    except (TypeError, ValueError) as exc:
        raise ValueError("collective wrapper emitted malformed result JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("collective wrapper result must be a JSON object")
    if parsed.get("engine") != "forge_collective":
        raise ValueError("collective wrapper result has invalid engine")
    if not isinstance(parsed.get("status"), str) or not parsed["status"]:
        raise ValueError("collective wrapper result has invalid status")
    decision = parsed.get("decision")
    if decision not in {"KEEP", "REVERT"}:
        raise ValueError("collective wrapper result has invalid decision")
    if not isinstance(parsed.get("kept"), bool):
        raise ValueError("collective wrapper result has invalid kept")
    if not isinstance(parsed.get("requires_e2e_validation"), bool):
        raise ValueError("collective wrapper result has invalid requires_e2e_validation")
    if parsed["kept"] != (decision == "KEEP"):
        raise ValueError("collective wrapper result has inconsistent decision")
    if parsed["requires_e2e_validation"] != parsed["kept"]:
        raise ValueError("collective wrapper result has inconsistent E2E gate")
    return parsed


def _enriched_kernel_candidates(state: Any) -> list[dict[str, Any]]:
    """Load full candidate rows from the latest trace artifact."""
    analysis = getattr(state, "last_trace_analyze", None)
    if analysis is None:
        return []
    if not isinstance(analysis, dict):
        raise ValueError("last_trace_analyze must be a mapping")
    if not analysis:
        return []
    path = str(analysis.get("candidates_path") or "").strip()
    if not path:
        raise ValueError("latest trace analysis has no candidates_path")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid collective candidate artifact: {path}") from exc
    rows = payload.get("hot_kernels") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"candidate artifact has no valid hot_kernels list: {path}")
    return rows


def collective_analysis_key(state: Any) -> str:
    """Return a stable identity for the latest trace analysis."""
    analysis = getattr(state, "last_trace_analyze", None)
    if analysis is None:
        return ""
    if not isinstance(analysis, dict):
        raise ValueError("last_trace_analyze must be a mapping")
    if not analysis:
        return ""
    path = str(analysis.get("candidates_path") or "").strip()
    if path:
        return path
    encoded = json.dumps(analysis, sort_keys=True).encode("utf-8")
    return f"inline:{hashlib.sha256(encoded).hexdigest()}"


def _validate_collective_candidate(
    candidate: dict[str, Any],
    *,
    index: int | None = None,
) -> None:
    """Validate fields required by the collective driver."""
    label = f"collective candidate[{index}]" if index is not None else "collective candidate"
    required_strings = ("kernel_id", "source_file", "source_function")
    missing = [
        field for field in required_strings if not isinstance(candidate.get(field), str) or not candidate[field].strip()
    ]
    for field in ("input_shapes", "input_dtypes"):
        value = candidate.get(field)
        if not isinstance(value, list) or not value:
            missing.append(field)
    if missing:
        raise ValueError(f"{label} is missing {', '.join(missing)}")
    gpu_pct = candidate.get("gpu_pct")
    if (
        isinstance(gpu_pct, bool)
        or not isinstance(gpu_pct, (int, float))
        or not math.isfinite(float(gpu_pct))
        or gpu_pct < 0
    ):
        raise ValueError(f"{label}.gpu_pct must be finite and non-negative")


def select_collective_candidate(state: Any) -> dict[str, Any] | None:
    """Pick the hottest source-resolved traced collective."""
    eligible: list[dict[str, Any]] = []
    for index, entry in enumerate(_enriched_kernel_candidates(state)):
        if entry.get("reusable_native_kernel") is not True:
            continue
        contract = entry.get("kernel_contract")
        if not isinstance(contract, dict) or str(contract.get("kind") or "") != "collective":
            continue
        if str(contract.get("collective_op") or "") not in SUPPORTED_COLLECTIVE_OPS:
            continue
        try:
            _validate_collective_candidate(entry, index=index)
        except ValueError as exc:
            log.warning("Skipping unusable collective candidate: %s", exc)
            continue
        eligible.append(entry)
    if not eligible:
        return None
    resolved = [entry for entry in eligible if str(entry.get("candidate_source") or "").strip() == "nccl_summary"]
    pool = resolved or eligible

    def _gpu_pct(entry: dict[str, Any]) -> float:
        """Return a sortable GPU-time share for one collective candidate."""
        return float(entry["gpu_pct"])

    return max(pool, key=_gpu_pct)


def _forge_loop_constant(module: str, name: str, fallback: float) -> float:
    """Read a forge-loop budget bound, falling back when KernelForge is off the path."""
    try:
        return float(getattr(importlib.import_module(module), name))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        log.warning(
            "forge-loop constant %s.%s unreadable (%s); planning against the fallback %s. "
            "KernelForge ships with Hyperloom, so this is a rename or a broken install.",
            module,
            name,
            exc,
            fallback,
        )
        return fallback


# Session time held back for the E2E integrate round plus reporting.
_COLLECTIVE_BUDGET_RESERVE_MIN = 45.0
_COLLECTIVE_PREP_GRACE_SEC = int(_forge_loop_constant("kernelforge.loop.task_preparer", "PREPARE_MAX_WALL_SEC", 3000))
# Wrapper grace to export the patch and restore the repository.
_COLLECTIVE_FINALIZE_GRACE_SEC = 300
# forge-loop rejects a campaign shorter than its own minimum.
_COLLECTIVE_MIN_CAMPAIGN_SEC = int(_forge_loop_constant("kernelforge.cli", "MIN_MAX_HOURS", 1.0) * 3600)
# Mirrors forge_collective.DEFAULT_TIMEOUT_SEC for a session with no deadline.
_COLLECTIVE_UNBOUNDED_WRAPPER_SEC = 14400


def _collective_revert_result(
    error_class: str,
    error: str,
    *,
    status: str = "failed",
    **fields: Any,
) -> HandlerResult:
    """Build a non-KEEP collective handler result."""
    result: HandlerResult = {
        "status": status,
        "backend": "forge",
        "engine": "forge_collective",
        "error_class": error_class,
        "error": error,
        "decision": "REVERT",
        "kept": False,
        "requires_e2e_validation": False,
    }
    result.update(fields)
    return result


def _collective_budget(state: Any, requested_hours: Any, timeout_sec: int) -> tuple[float | None, int]:
    """Derive campaign and wrapper budgets from remaining session time."""
    remaining_fn = getattr(state, "remaining_minutes", None)
    remaining = remaining_fn() if callable(remaining_fn) else None
    wall_limits: list[int] = []
    if remaining is not None:
        if isinstance(remaining, bool):
            raise ValueError("remaining session minutes must be numeric")
        remaining = float(remaining)
        if not math.isfinite(remaining) or remaining < 0:
            raise ValueError(f"remaining session minutes must be finite and non-negative: {remaining}")
        wall_limits.append(
            max(
                0,
                int((remaining - _COLLECTIVE_BUDGET_RESERVE_MIN) * 60),
            )
        )
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, int) or timeout_sec < 0:
        raise ValueError(f"collective timeout must be a non-negative integer: {timeout_sec!r}")
    if timeout_sec > 0:
        wall_limits.append(timeout_sec)
    if requested_hours is None and not wall_limits:
        # Unbounded session: no budget to divide, so the reserve and minimum-campaign contracts below cannot apply.
        log.info(
            "collective budget: unbounded session, defaulting the wrapper to %ds",
            _COLLECTIVE_UNBOUNDED_WRAPPER_SEC,
        )
        return None, _COLLECTIVE_UNBOUNDED_WRAPPER_SEC

    wall_limit = min(wall_limits) if wall_limits else 0
    campaign_capacity = wall_limit - _COLLECTIVE_PREP_GRACE_SEC - _COLLECTIVE_FINALIZE_GRACE_SEC
    if requested_hours is None:
        campaign_sec = campaign_capacity
    else:
        if isinstance(requested_hours, bool):
            raise ValueError("collective max_hours must be numeric")
        requested = float(requested_hours)
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError(f"collective max_hours must be finite and positive: {requested_hours!r}")
        requested_sec = int(requested * 3600)
        if requested_sec < _COLLECTIVE_MIN_CAMPAIGN_SEC:
            return None, 0
        campaign_sec = min(requested_sec, campaign_capacity) if wall_limits else requested_sec
    if campaign_sec < _COLLECTIVE_MIN_CAMPAIGN_SEC:
        return None, 0
    # Truncate to two decimals so the hours we hand forge-loop never round up past the budget they were derived from.
    hours = math.floor(campaign_sec / 3600 * 100) / 100.0
    required = _COLLECTIVE_PREP_GRACE_SEC + int(hours * 3600) + _COLLECTIVE_FINALIZE_GRACE_SEC
    return hours, required


async def _run_forge_collective(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Run the strict collective Forge lane and parse its result contract."""
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    candidate_value = payload.get("candidate")
    if candidate_value is not None and (not isinstance(candidate_value, dict) or not candidate_value):
        return _collective_revert_result(
            "invalid_collective_candidate",
            "candidate must be a non-empty object",
        )
    try:
        candidate = dict(candidate_value) if isinstance(candidate_value, dict) else select_collective_candidate(state)
    except (OSError, TypeError, ValueError) as exc:
        return _collective_revert_result(
            "invalid_collective_candidate_artifact",
            str(exc),
            status="skipped",
            analysis_key=collective_analysis_key(state),
        )
    if not candidate:
        return _collective_revert_result(
            "no_collective_candidate",
            ("no rewritable collective in the latest trace analysis (nccl/rccl are vendor binaries and never qualify)"),
            status="skipped",
            analysis_key=collective_analysis_key(state),
        )

    contract = candidate.get("kernel_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("kind") != "collective"
        or contract.get("collective_op") not in SUPPORTED_COLLECTIVE_OPS
    ):
        return _collective_revert_result(
            "unsupported_collective_contract",
            "collective Forge supports " + ", ".join(sorted(SUPPORTED_COLLECTIVE_OPS)),
        )
    try:
        _validate_collective_candidate(candidate)
    except ValueError as exc:
        return _collective_revert_result(
            "invalid_collective_candidate",
            str(exc),
        )

    source_file = candidate["source_file"].strip()
    source_function = candidate["source_function"].strip()
    # The anchor alone hides the op's sibling sources, so a fused variant of the same collective stays uneditable.
    raw_kernel_sources = candidate.get("kernel_sources") or []
    if isinstance(raw_kernel_sources, str):
        raw_kernel_sources = [raw_kernel_sources]
    collective_sources = list(
        dict.fromkeys(str(path).strip() for path in (source_file, *raw_kernel_sources) if str(path or "").strip())
    )
    kernel_repo_raw = candidate.get("kernel_repo")
    if kernel_repo_raw is not None and not isinstance(kernel_repo_raw, str):
        return _collective_revert_result(
            "invalid_collective_candidate",
            "collective candidate kernel_repo must be a string",
        )
    kernel_repo = (kernel_repo_raw or "").strip() or _find_repo_root_for_source(source_file)
    if not kernel_repo:
        return _collective_revert_result(
            "kernel_repo_missing",
            f"cannot resolve a repo root for {source_file!r}",
        )

    tp_raw = payload.get("tp")
    if tp_raw in (None, ""):
        tp_raw = getattr(state, "tp", 0) or os.environ.get("TP")
    try:
        if isinstance(tp_raw, bool) or isinstance(tp_raw, float):
            raise ValueError
        tp = int(tp_raw)
    except (TypeError, ValueError):
        tp = 0
    if tp <= 1:
        return _collective_revert_result(
            "invalid_collective_world_size",
            f"collective TP must be greater than one, got {tp_raw!r}",
        )

    timeout_raw = payload.get("timeout")
    if timeout_raw in (None, ""):
        timeout_raw = os.environ.get("FORGE_COLLECTIVE_TIMEOUT")
    try:
        if isinstance(timeout_raw, bool) or isinstance(timeout_raw, float):
            raise ValueError
        timeout = int(timeout_raw) if timeout_raw not in (None, "") else 0
        max_hours, timeout = _collective_budget(state, payload.get("max_hours"), timeout)
    except (OverflowError, TypeError, ValueError) as exc:
        return _collective_revert_result(
            "invalid_collective_budget",
            str(exc),
        )
    if timeout <= 0:
        return _collective_revert_result(
            "insufficient_collective_budget",
            (
                "remaining session budget cannot fit collective preparation, "
                "KernelForge's one-hour minimum campaign, and finalization"
            ),
            status="skipped",
            analysis_key=collective_analysis_key(state),
        )
    workspace = (
        session_dir
        / "runs"
        / "collective"
        / str(payload.get("task_id") or "kernel_entry_collective")
        / f"attempt-{time.time_ns()}"
    )
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        agent_backend, llm_model = _resolve_forge_agent(payload)
    except (RuntimeError, ValueError) as exc:
        return _collective_revert_result(
            "llm_provider_unconfigured" if isinstance(exc, RuntimeError) else "invalid_agent_backend",
            str(exc),
            analysis_key=collective_analysis_key(state),
        )

    input_payload = {
        "candidate": candidate,
        "source_file": source_file,
        "kernel_repo": kernel_repo,
        "output_dir": str(workspace),
        "tp": tp,
        "timeout": timeout,
        "finalize_grace_sec": _COLLECTIVE_FINALIZE_GRACE_SEC,
        "agent_timeout_sec": payload.get("agent_timeout_sec") or os.environ.get("FORGE_COLLECTIVE_AGENT_TIMEOUT"),
        "gpu_target": str(payload.get("gpu_target") or getattr(state, "gpu_type", "") or ""),
        "max_hours": max_hours,
        "agent_backend": agent_backend,
        "llm_model": llm_model,
        "target_functions": [source_function],
        "source_files": collective_sources,
        "operator_name": source_function,
        "framework": str(getattr(state, "framework", "") or ""),
        "experience_id": workspace.name,
    }
    input_json = workspace / "forge_collective_input.json"
    input_json.write_text(
        json.dumps(input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cmd = ["python3", str(_kernel_agent_tool_path("forge_collective.py")), "--input-json", str(input_json)]

    wrapper_timeout = timeout + _COLLECTIVE_FINALIZE_GRACE_SEC
    try:
        rc, stdout, stderr = await _run_subprocess(
            cmd,
            timeout_sec=wrapper_timeout,
        )
        try:
            result = _parse_forge_collective_sentinel(stdout)
        except ValueError as exc:
            result = _collective_revert_result(
                "invalid_collective_result",
                str(exc),
                returncode=rc,
                stdout_tail=stdout[-2000:],
                stderr_tail=stderr[-2000:],
            )
    except subprocess.TimeoutExpired as exc:
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = _collective_revert_result(
            "subprocess_timeout",
            f"TimeoutExpired after {wrapper_timeout}s: {cmd_repr[:1500]}",
        )

    result.setdefault("backend", "forge")
    result.setdefault("engine", "forge_collective")
    result.setdefault("workspace", str(workspace))
    result.setdefault("kernel_id", candidate.get("kernel_id"))
    result.setdefault("kernel_name", candidate.get("name"))
    result.setdefault("source_file", source_file)
    result.setdefault("kernel_repo", kernel_repo)
    result.setdefault("gpu_pct", candidate.get("gpu_pct"))
    result.setdefault("collective_op", contract["collective_op"])
    result.setdefault("world_size", tp)
    result.setdefault("requires_e2e_validation", False)
    result.setdefault("source", "forge_collective")
    if str(result.get("status") or "") == "skipped" and str(result.get("error_class") or "") in {
        "no_collective_candidate",
        "insufficient_collective_budget",
    }:
        result.setdefault("analysis_key", collective_analysis_key(state))
    return result


def _find_repo_root_for_source(source_file: str) -> str:
    """Nearest ancestor of ``source_file`` containing a ``.git`` directory."""
    if not source_file:
        return ""
    try:
        current = Path(source_file).resolve()
    except OSError:
        return ""
    for parent in current.parents:
        if (parent / ".git").exists():
            return str(parent)
    return ""


async def run_collective_handler(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Run the coordinator-owned collective optimization lane."""
    result = await _run_forge_collective(payload, session_dir=session_dir)
    if not isinstance(result, dict):
        raise TypeError("Collective handler result must be a mapping")
    result.setdefault("requires_e2e_validation", False)
    return result


# A tuner error is a diagnostic pointer, not the diagnosis: the full text lives in the run's own result.json and
# tune.log. 400 characters is enough to carry the argparse line or the aiter marker that says which of the two it was.
_TRACE_TUNER_ERROR_MAXLEN = 400

# Emitted even when null.
_TRACE_TUNER_ALWAYS_KEYS = ("tuner", "best_micro_speedup", "kept")


def _trace_tuner_row(tuner: dict[str, Any]) -> dict[str, Any]:
    """One per-tuner entry for the audit row, keeping why it ended as it did."""
    error = tuner.get("error")
    if isinstance(error, str) and len(error) > _TRACE_TUNER_ERROR_MAXLEN:
        error = error[:_TRACE_TUNER_ERROR_MAXLEN] + "..."
    row = {
        "tuner": tuner.get("tuner") or tuner.get("name"),
        "best_micro_speedup": tuner.get("best_micro_speedup"),
        "kept": tuner.get("kept"),
        "status": tuner.get("status"),
        "elapsed_s": tuner.get("elapsed_s"),
        "error_class": tuner.get("error_class"),
        "error": error,
    }
    # A clean run stays as compact as before: everything added here is dropped when it is null, so a successful row
    # gains only status and elapsed_s.
    return {k: v for k, v in row.items() if k in _TRACE_TUNER_ALWAYS_KEYS or v is not None}


def _trace_gemm_tuning_run(result: Any, *, session_dir: Path) -> None:
    """Append one ``gemm_tuning.jsonl`` audit row for a GEMM-tuning run."""
    if not isinstance(result, dict):
        return
    from datetime import datetime, timezone

    from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path

    engine = str(result.get("engine") or result.get("backend") or "").strip().lower() or "unknown"
    tuners: list[dict[str, Any]] = [
        _trace_tuner_row(t) for t in (result.get("tuners_run") or []) if isinstance(t, dict)
    ]
    # The envelope reported no error class even when a tuner had named one, so a crashed run and a barren one looked
    # alike at the top level too.
    error_class = result.get("error_class") or next((t["error_class"] for t in tuners if t.get("error_class")), None)
    row = {
        "kind": "gemm_tuning",
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "engine": engine,
        "backend": result.get("backend"),
        "status": result.get("status"),
        "decision": result.get("decision"),
        "micro_decision": result.get("micro_decision"),
        "best_speedup": result.get("best_speedup"),
        "precision": result.get("precision"),
        "framework": result.get("framework"),
        "gpu_type": result.get("gpu_type"),
        "tuned_file": result.get("tuned_file"),
        "workspace": result.get("workspace"),
        "requires_e2e_validation": result.get("requires_e2e_validation"),
        "tuners_run": tuners,
        # A crash stays in the audit even when a sibling tuner won and the run was kept -- otherwise a KEEP row hid
        # the failures behind it.
        "failed_tuners": result.get("failed_tuners") or None,
        "error_class": error_class,
    }
    row = {k: v for k, v in row.items() if v is not None}
    try:
        append_jsonl(gemm_tuning_steps_path(session_dir), row, make_parents=True, sort_keys=True)
    except OSError:
        log.debug("full-trace: gemm_tuning audit append failed", exc_info=True)


def _build_trace_analyze_cmd(
    payload: dict,
    *,
    session_dir: Path,
    state: Any,
    workspace_path: str,
    trace_input: Any,
    tracelens_root: "Path | None",
    is_bypass: bool,
    scriptable: bool,
    workload: dict,
    model_name: str,
    framework: str,
    target_platform: str,
    analysis_mode: str,
) -> "tuple[list[str], str]":
    """Assemble the trace-analysis tool argv (TraceLens or bypass); returns ``(cmd, steady_state_mode)`` so the caller can record discovery provenance."""
    # Both tools share the CLI surface below except ``--tracelens-root``.
    tool_name = "bypass_trace_analysis.py" if is_bypass else "tracelens_analysis.py"
    cmd = [
        "python3",
        str(_kernel_agent_tool_path(tool_name)),
        "--trace-input",
        str(trace_input),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--workspace-path",
        workspace_path,
    ]
    if not is_bypass:
        # Pass the resolved root explicitly so the tool never relies on inherited env.
        cmd += ["--tracelens-root", str(tracelens_root)]
    elif str(getattr(state, "benchmark_mode", "") or "").strip().lower() == "agentx":
        cmd += ["--require-single-rank"]
        try:
            state_tp = int(getattr(state, "tp", 0) or 0)
        except (TypeError, ValueError):
            state_tp = 0
        if state_tp > 0:
            cmd += ["--tensor-parallel-size", str(state_tp)]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if framework:
        cmd += ["--framework", str(framework)]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if analysis_mode:
        cmd += ["--analysis-mode", str(analysis_mode)]

    # Model identity informs source resolution for every framework, not only the diffusion roofline.
    model_path = str(
        payload.get("model_path") or getattr(state, "model_path", "") or os.environ.get("MODEL_PATH") or ""
    ).strip()
    if model_path:
        cmd += ["--model-path", model_path]
    precision = str(
        payload.get("precision") or getattr(state, "precision", "") or workload.get("precision") or ""
    ).strip()
    if precision:
        cmd += ["--precision", precision]
    runtime_config = str(payload.get("runtime_config") or getattr(state, "baseline_config_path", "") or "").strip()
    if runtime_config and not is_bypass:
        cmd += ["--runtime-config", runtime_config]

    if scriptable:
        # --skip-split is TraceLens-only; the bypass backend has its own windowing.
        if not is_bypass:
            cmd += ["--skip-split"]
        # Forward the denoise-step count for per-step roofline timings.
        num_denoise = payload.get("num_denoise_steps") or workload.get("num_inference_steps")
        if num_denoise not in (None, ""):
            try:
                if int(num_denoise) > 0:
                    cmd += ["--num-denoise-steps", str(int(num_denoise))]
            except (TypeError, ValueError):
                pass
    else:
        # Splitter workload hints.
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
        payload.get("capture_folder") or payload.get("graph_capture_path") or payload.get("capture_folder_path")
    )
    if capture_folder:
        cmd += ["--capture-folder", str(capture_folder)]
    # Forward TraceLens splitter steady-state mode via payload or env.
    steady_state_mode = payload.get("steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
    steady_state_mode = str(steady_state_mode).strip()
    if steady_state_mode:
        cmd += ["--steady-state-mode", steady_state_mode]
    # Post-kernel-opt roofline writes a separate report so it never overwrites the baseline kernel_roofline.json.
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    if roofline_output_name:
        cmd += ["--roofline-output-name", roofline_output_name]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    return cmd, steady_state_mode


# TraceLens picks its steady-state window by writing split chunks and selecting one file; the TraceLens-free reader
# picks a window in memory and never writes chunks.
_STEADY_SOURCE_SPLIT_CHUNK = "split_chunk"
_STEADY_SOURCE_READER_WINDOW = "in_reader_window"


def _analysis_steady_state(
    result: dict[str, Any],
    *,
    requested_mode: str,
    tool: str,
) -> dict[str, Any]:
    """Normalize the steady-state window across analysis tools."""
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    scope = str(result.get("aggregation_scope") or run_meta.get("aggregation_scope") or "")
    if tool == "bypass":
        selected = result.get("steady_window") or {}
        fell_back = bool(result.get("estimated")) or (bool(scope) and scope != "steady_state")
        source = _STEADY_SOURCE_READER_WINDOW
    else:
        selection = run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {}
        selected = selection or {}
        fell_back = bool(selection.get("fell_back_to_full_trace"))
        source = _STEADY_SOURCE_SPLIT_CHUNK
    return {
        "requested_mode": str(requested_mode or ""),
        "source": source,
        "selected": selected if isinstance(selected, dict) else {"value": selected},
        "fell_back_to_full_trace": fell_back,
        "aggregation_scope": scope,
    }


def _build_analysis_meta(
    result: dict[str, Any],
    *,
    route: str,
    tool: str,
    requested_mode: str,
    trace_input: str,
    duration_sec: float,
) -> dict[str, Any]:
    """Assemble the per-run analysis metadata the roofline timeline event carries."""
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    steps = run_meta.get("steps")
    return {
        "route": str(route or ""),
        "tool": str(tool or ""),
        "steady_state_mode": str(requested_mode or ""),
        "trace_input": str(trace_input or ""),
        "duration_sec": duration_sec,
        "steady_state": _analysis_steady_state(result, requested_mode=requested_mode, tool=tool),
        "preflight": run_meta.get("preflight") if isinstance(run_meta.get("preflight"), dict) else {},
        "split": run_meta.get("split") if isinstance(run_meta.get("split"), dict) else {},
        "selection": run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {},
        "steps": [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else [],
        "route_ext": run_meta.get("route_ext") if isinstance(run_meta.get("route_ext"), dict) else {},
    }


async def trace_analyze_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir."""
    trace_input = payload.get("trace_input") or payload.get("trace_dir")
    if not trace_input:
        return {"status": "failed", "error": "missing 'trace_input' in payload"}
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}
    # Backfill workload context from SharedState when Orchestration omits it.
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state_framework = str(state.framework or "").strip()
    payload_framework = str(payload.get("framework") or "").strip()
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    # Payload metadata remains authoritative for ordinary serving frameworks.
    framework = payload_framework or state_framework
    framework_warnings: list[dict[str, Any]] = []
    if payload_framework and is_scriptable(state_framework) and not is_scriptable(payload_framework):
        framework = state_framework
        framework_warnings.append(
            {
                "code": "stale_framework_overridden",
                "severity": "warning",
                "message": (
                    f"overrode non-scriptable payload framework {payload_framework!r} "
                    f"with scriptable session framework {state_framework!r} "
                    "to preserve the raw trace"
                ),
                "payload_framework": payload_framework,
                "session_framework": state_framework,
            }
        )
        log.warning(
            "trace_analyze: overriding payload framework %r with session "
            "scriptable framework %r to preserve the raw trace",
            payload_framework,
            state_framework,
        )
    target_platform = (payload.get("target_platform") or state.gpu_type or "").strip()
    model_name = (payload.get("model_name") or state.model_name or state.model_path or "").strip()
    analysis_mode = (payload.get("analysis_mode") or "").strip()
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    # Analysis route: default ``agent`` (TraceLens); ``bypass`` (TraceLens-free) is the explicit route via payload
    # ``analysis_route`` / ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``.
    raw_route = payload.get("analysis_route")
    route_text = "" if raw_route is None else str(raw_route).strip()
    if not route_text:
        route_text = os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "").strip()
    explicit_route = route_text.lower()
    # An explicit unknown route is a configuration error.
    if explicit_route and explicit_route not in _VALID_ANALYSIS_ROUTES:
        valid_routes = sorted(_VALID_ANALYSIS_ROUTES)
        message = (
            f"unknown analysis_route {explicit_route!r} (expected one of {valid_routes}); "
            "refusing to fall back to 'agent' because that may start an LLM session. "
            "Use 'bypass' for no-LLM trace analysis."
        )
        log.error("trace_analyze: %s", message)
        return {
            "status": "failed",
            "error_class": "invalid_analysis_route",
            "error": message,
            "requested_route": explicit_route,
            "valid_routes": valid_routes,
        }
    analysis_route = explicit_route or "agent"
    is_bypass = analysis_route == "bypass"
    # Resolve TraceLens root independently of inherited env, self-healing a vanished checkout before validation.
    tracelens_root: Path | None = None
    if not is_bypass:
        tracelens_root = _resolve_tracelens_root()
        # Self-heal when the checkout is missing or incomplete (no .git).
        if not (tracelens_root / ".git").exists():
            _maybe_selfheal_tracelens_root(tracelens_root, log=log)
        tl_err = _tracelens_root_error(tracelens_root)
        if tl_err:
            return {"status": "failed", "error_class": "tracelens_root_missing", "error": tl_err}

    # Pass the session root so artefacts settle under ``<session_dir>/kernel-agent/runs/...``.
    workspace_path = payload.get("workspace_path") or str(session_dir)
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    # Scriptable frameworks (xDiT) have no decode steady-state window, so feed the raw trace and drop the --split-*
    # hints.
    scriptable = is_scriptable(framework)

    # Load materialized baseline workload metadata once.
    metadata = _load_materialized_workload_metadata(state.baseline_config_path)
    workload = metadata.get("runtime_args", {}).get("workload", {}) if isinstance(metadata, dict) else {}

    cmd, steady_state_mode = _build_trace_analyze_cmd(
        payload,
        session_dir=session_dir,
        state=state,
        workspace_path=workspace_path,
        trace_input=trace_input,
        tracelens_root=tracelens_root,
        is_bypass=is_bypass,
        scriptable=scriptable,
        workload=workload,
        model_name=model_name,
        framework=framework,
        target_platform=target_platform,
        analysis_mode=analysis_mode,
    )
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
        }
    _disc_duration_sec = round(time.monotonic() - _disc_started, 3)
    artifacts = result.get("artifact_paths") if isinstance(result, dict) else None
    if isinstance(artifacts, dict) and artifacts.get("kernel_candidates"):
        result["candidates_path"] = artifacts["kernel_candidates"]
    # Surface analysis.md path at the handler boundary for the Coordinator.
    if isinstance(result, dict):
        report_path = result.get("trace_report_path")
        if not report_path and isinstance(artifacts, dict):
            report_path = artifacts.get("trace_report_path")
        if report_path:
            result["trace_report_path"] = str(report_path)
            _enrich_candidate_trace_report(
                result.get("hot_kernels"),
                str(report_path),
            )
        # Surface the reusable-vs-skipped audit sidecar.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
        if isinstance(artifacts, dict) and artifacts.get("kernel_roofline"):
            result["kernel_roofline_path"] = str(artifacts["kernel_roofline"])

        # A failed TraceLens run is a hard failure, not "empty candidates".
        if result.get("status") == "failed" and "trace_split_no_steady_state" not in str(result.get("error") or ""):
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

        # Prepend handler validation warnings so they reach the LLM.
        result["trace_health_warnings"] = framework_warnings + list(result.get("trace_health_warnings") or [])

        _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
        candidates_path = result.get("candidates_path")
        if isinstance(candidates_path, str):
            _enrich_candidates_artifact(
                candidates_path,
                metadata,
                trace_report_path=str(report_path or ""),
            )

        # Route and tool are one-to-one after the no-LLM TraceLens route was removed: agent runs TraceLens, while
        # bypass runs its standalone reader.
        _disc_route = analysis_route
        _disc_tool = "bypass" if is_bypass else "tracelens"
        # Surfaced for the caller's SBD V6 roofline event, which records the run as it happens rather than re-deriving
        # it at export time.
        result["analysis_meta"] = _build_analysis_meta(
            result,
            route=_disc_route,
            tool=_disc_tool,
            requested_mode=steady_state_mode,
            trace_input=str(trace_input),
            duration_sec=_disc_duration_sec,
        )

        # Record hot-kernel discovery provenance (best-effort).
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            _hot = result.get("hot_kernels_top15") or result.get("hot_kernels") or []
            instrument.record_kernel_discovery(
                session_dir,
                source=_disc_tool,
                status=str(result.get("status") or ""),
                hot_kernels=_hot if isinstance(_hot, list) else [],
                scan={
                    "splitter_mode": steady_state_mode,
                    "trace_dir": str(trace_input),
                    "candidates_path": str(result.get("candidates_path") or ""),
                    "trace_report_path": str(result.get("trace_report_path") or ""),
                    "analysis_route": _disc_route,
                },
                duration_sec=_disc_duration_sec,
                error=(str(result.get("error") or "") or None if str(result.get("status") or "") == "failed" else None),
            )
        except Exception as exc:  # noqa: BLE001
            trace_recording_skipped(
                "kernel_discovery",
                reason="caller raised before the recorder",
                error=exc,
            )
    return result


#: Rewrite targets forge can actually execute in one ``--auto`` call. Running
#: several needs a per-target base commit and scratch path, so its CLI refuses
#: more than one; asking for what the budget funds would fail the whole lane.
_REWRITE_EXECUTABLE_TARGETS = 1


def _summarize_dropped_patches(dropped: Any) -> dict[str, int]:
    """Count dropped patch entries by reason, logging each one as it is counted."""
    counts: dict[str, int] = {}
    for entry in dropped or ():
        reason = str(getattr(entry, "reason", "") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
        log.warning(
            "nomination landing: refused patch kernel=%s reason=%s",
            str(getattr(entry, "kernel_name", "") or "<unnamed>"),
            reason,
        )
    return counts


def _raw_kernel_backend_order(payload: dict | None = None) -> list[str]:
    """Return the effective kernel backend order."""
    if forge_explicitly_enabled():
        return ["forge"]
    return list(_DEFAULT_KERNEL_PHASE_BACKEND_ORDER)


def geak_selected(payload: dict | None = None) -> bool:
    """Whether ``geak`` (the whole-pipeline e2e delegate) is in the kernel backend order."""
    return "geak" in _raw_kernel_backend_order(payload)


def _shape_tool_result(rc: int, stdout: str, stderr: str) -> HandlerResult:
    """Wrap a kernel-agent tool's exit + stdout into our schema (prefer the tool's own JSON, synthesize only on parse failure)."""
    parsed = _parse_tool_stdout(stdout)
    if parsed and set(parsed) == {"raw_stdout_tail"}:
        # Unparseable output is not a result.
        return {
            "status": "failed",
            "error_class": "tool_output_unparseable",
            "error": ("tool exited rc=%d but its stdout held no JSON object" % rc),
            "returncode": rc,
            "raw_stdout_tail": parsed["raw_stdout_tail"],
            "stderr_tail": stderr[-2000:] if stderr.strip() else "",
        }
    if parsed:
        # Trust the tool's own status; else infer from rc.
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
    """Parse a tool's stdout into a dict, surviving non-JSON noise."""
    text = stdout.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    # Fallback: scan for the last JSON object on its own line.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # Last: a pretty-printed object opening at the start of a line.
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"^\{", text, re.MULTILINE)]
    for start in reversed(starts):
        try:
            obj, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {"raw_stdout_tail": text[-2000:]}


def _sweep_integrate_aiter_locks(*, reason: str) -> dict[str, Any]:
    """Best-effort orphaned-lock sweep immediately before an integrate boot."""
    from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

    try:
        stats = sweep_stale_aiter_locks_if_dead()
    except Exception as exc:  # noqa: BLE001 - cache hygiene must not hide benchmark results
        log.warning("integrate_handler: aiter lock sweep failed before %s: %r", reason, exc)
        return {"errors": 1, "exception": repr(exc)}
    if stats.get("skipped_live"):
        log.info(
            "integrate_handler: aiter lock sweep skipped before %s; a compiler is alive",
            reason,
        )
    elif stats.get("deleted"):
        log.warning(
            "integrate_handler: reaped %d orphaned aiter lock(s) across %s before %s",
            stats.get("deleted"),
            stats.get("dirs") or [stats.get("dir")],
            reason,
        )
    return stats


async def _run_integrate_rebaseline_with_lock_retry(
    executor: Any,
    ctx: Any,
    *,
    workspace: Path,
    reason: str,
) -> dict[str, Any]:
    """Run one integrate baseline and retry once after a confirmed baton stall."""
    from ..actions.executors._aiter_jit import find_aiter_baton_wait

    prelaunch_sweep = _sweep_integrate_aiter_locks(reason=reason)
    first_started_unix = time.time()
    result = await executor(ctx)
    if not isinstance(result, dict) or result.get("status") == "succeeded":
        return result

    evidence = find_aiter_baton_wait(
        workspace,
        since_unix=first_started_unix - 1.0,
    )
    if evidence is None:
        return result

    cleanup = _sweep_integrate_aiter_locks(reason=f"{reason} stale-lock retry")
    result["error_class"] = "stale_jit_lock"
    result["stale_jit_lock"] = {
        "evidence": evidence,
        "prelaunch_sweep": prelaunch_sweep,
        "post_failure_sweep": cleanup,
        "retry_attempted": False,
    }

    # A live compiler may legitimately own the observed lock.
    cleanup_safe = not cleanup.get("skipped_live") and not cleanup.get("errors")
    lock_removed = bool(cleanup.get("deleted")) or (cleanup.get("scanned", 0) == 0 and not cleanup.get("skipped_fresh"))
    if not (cleanup_safe and lock_removed):
        return result

    log.warning(
        "integrate_handler: classified %s as stale_jit_lock; retrying once after cleanup",
        reason,
    )
    retry_started_unix = time.time()
    retry_result = await executor(ctx)
    if not isinstance(retry_result, dict):
        return retry_result
    retry_evidence = (
        find_aiter_baton_wait(
            workspace,
            since_unix=retry_started_unix - 1.0,
        )
        if retry_result.get("status") != "succeeded"
        else None
    )
    retry_result["stale_jit_lock_retry"] = {
        "evidence": evidence,
        "cleanup": cleanup,
        "retry_attempted": True,
        "retry_succeeded": retry_result.get("status") == "succeeded",
    }
    if retry_evidence is not None:
        retry_result["error_class"] = "stale_jit_lock"
        retry_result["stale_jit_lock_retry"]["retry_evidence"] = retry_evidence
    return retry_result


def _eval_generation_budget() -> int:
    """Completion tokens the eval harness reserves per sample."""
    raw = (os.environ.get("HYPERLOOM_EVAL_MAX_TOKENS") or "").strip()
    if not raw:
        return _EVAL_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _EVAL_DEFAULT_MAX_TOKENS
    return value if value >= 0 else _EVAL_DEFAULT_MAX_TOKENS


def _grade_integrate_accuracy(
    bench_result: dict[str, Any],
    *,
    session_dir: Path,
    workspace: Path,
    strict: bool = False,
    server_args: str = "",
) -> dict[str, Any]:
    """Grade a kernel re-baseline's accuracy against the session baseline."""
    from ..actions.executors._accuracy_gate import (
        accuracy_keep_block,
        accuracy_passed,
        parse_eval_results,
        require_kernel_accuracy_default,
        resolve_served_context,
        served_context_hosts_eval,
    )

    baseline_accuracy = 0.0
    try:
        from ..state.shared_state import SharedState

        baseline_accuracy = float(SharedState.load_or_init(session_dir).baseline_accuracy or 0.0)
    except Exception:  # noqa: BLE001 - an unresolvable baseline degrades, never raises
        log.debug("integrate_handler: could not resolve baseline_accuracy", exc_info=True)

    measured = bench_result.get("accuracy")
    new_accuracy = float(measured) if isinstance(measured, (int, float)) else None
    task = str(bench_result.get("accuracy_task") or "")
    metric = str(bench_result.get("accuracy_metric") or "")
    source_file = str(bench_result.get("accuracy_source") or "")
    if new_accuracy is None:
        try:
            eval_out = parse_eval_results(workspace, framework=os.environ.get("FRAMEWORK") or None)
            parsed = eval_out.get("accuracy")
            if isinstance(parsed, (int, float)):
                new_accuracy = float(parsed)
                task = str(eval_out.get("task") or "")
                metric = str(eval_out.get("metric") or "")
                source_file = str(eval_out.get("source_file") or "")
        except Exception:  # noqa: BLE001 - a failed parse degrades to "no verdict"
            log.debug("integrate_handler: accuracy re-parse failed", exc_info=True)

    accuracy_pass: bool | None = None
    if new_accuracy is not None and baseline_accuracy > 0:
        accuracy_pass = accuracy_passed(baseline_accuracy, new_accuracy)

    blocked, reason, degraded = accuracy_keep_block(
        accuracy_pass,
        required=True if strict else require_kernel_accuracy_default(),
        baseline_accuracy=baseline_accuracy,
    )
    if strict and degraded:
        blocked = True
        reason = "accuracy gate produced no eval result and this artifact has no other end-to-end correctness evidence"
    # A verdict can be missing because the eval broke, or because the serving configuration cannot answer an eval
    # request at all.
    infeasible = False
    if accuracy_pass is None:
        fits, why = served_context_hosts_eval(
            served_max_model_len=resolve_served_context(
                server_args=server_args,
                env_max_model_len=os.environ.get("MAX_MODEL_LEN", 0),
            ),
            eval_max_tokens=_eval_generation_budget(),
        )
        if not fits:
            infeasible = True
            reason = why
            log.warning(
                "integrate_handler: the accuracy gate cannot run under this "
                "serving configuration, so no kernel can clear it until the "
                "configuration changes: %s",
                why,
            )
    log.info(
        "integrate_handler: accuracy gate pass=%s blocked=%s degraded=%s new=%s baseline=%.4f source=%s",
        accuracy_pass,
        blocked,
        degraded,
        "n/a" if new_accuracy is None else f"{new_accuracy:.4f}",
        baseline_accuracy,
        source_file or "none",
    )
    return {
        "blocked": blocked,
        "accuracy_pass": accuracy_pass,
        "reason": reason,
        "degraded": degraded,
        "infeasible": infeasible,
        "accuracy": new_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "task": task,
        "metric": metric,
        "source_file": source_file,
    }


def _agentx_rebaseline_timeout(resolved_sec: int, *, shared_state: Any = None) -> int:
    """Raise a re-baseline timeout to what an AgentX round needs."""
    from ..actions.executors._workload_envs import agentx_active

    if not agentx_active(shared_state):
        return resolved_sec
    from ..actions.executors.baseline import agentx_baseline_timeout_sec

    agentx_sec = agentx_baseline_timeout_sec()
    if agentx_sec <= resolved_sec:
        return resolved_sec
    log.warning(
        "integrate_handler: raising re-baseline timeout %ds -> %ds "
        "(AgentX: AGENTX_DURATION + overhead; a synthetic-sized param cannot "
        "cover a canonical agentic warmup and kills the round mid-warmup)",
        resolved_sec,
        agentx_sec,
    )
    return agentx_sec


def _cold_start_rebaseline_timeout(resolved_sec: int) -> int:
    """Raise a re-baseline timeout to the cold-start cap when the JIT cache is empty."""
    from ..actions.executors._aiter_jit import (
        BASELINE_COLD_START_TIMEOUT_SEC,
        probe_aiter_jit_cache,
    )

    cache = probe_aiter_jit_cache()
    if cache.get("probe_status") != "found" or not cache.get("is_cold"):
        return resolved_sec
    cold_cap = int(
        os.environ.get(
            "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
            BASELINE_COLD_START_TIMEOUT_SEC,
        )
    )
    if cold_cap <= resolved_sec:
        return resolved_sec
    log.warning(
        "integrate_handler: aiter JIT cache is cold (%s kernels); raising "
        "re-baseline timeout %ds -> %ds for the recompile the patch forces",
        cache.get("kernel_count"),
        resolved_sec,
        cold_cap,
    )
    return cold_cap


def _integrate_rebaseline_timeout_sec(
    payload: dict,
    *,
    default_timeout_sec: int,
) -> int:
    """Resolve the E2E timeout from explicit input or benchmark contract."""
    explicit = payload.get("timeout_sec")
    if explicit is not None:
        try:
            value = int(explicit)
            if value > 0:
                return value
        except (TypeError, ValueError):
            log.debug(
                "integrate_handler: invalid timeout_sec; trying fallback timeout sources",
                exc_info=True,
            )
    if "budget_minutes" in payload:
        try:
            value = int(float(payload["budget_minutes"]) * 60)
            if value > 0:
                return value
        except (TypeError, ValueError):
            log.debug(
                "integrate_handler: invalid budget_minutes; trying fallback timeout sources",
                exc_info=True,
            )
    config_path = str(payload.get("config_path") or "")
    if config_path and Path(config_path).is_file():
        try:
            import yaml  # type: ignore[import-untyped]

            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            benchmark = config.get("benchmark")
            if isinstance(benchmark, dict):
                value = int(benchmark.get("timeout_seconds") or 0)
                if value > 0:
                    return value
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            log.debug(
                "integrate_handler: invalid benchmark timeout config; using executor default",
                exc_info=True,
            )
    return max(1, int(default_timeout_sec))


async def integrate_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision."""
    from ..actions.executors.baseline import SBD_INNER_STEP_PARAM, BaselineExecutor
    from ..actions.executors.benchmark_result import is_valid_measurement
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare {kernel_id} payload isn't failed
    # with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)

    if payload.get("_vendor_playbook_deploy_blocked"):
        # A vendor-playbook KEEP (e.g. mori dispatch/combine launch-config tuning) has no deployable artifact:
        # best_artifact_path is a copy of a KernelForge task-bundle config file, not a rewrite of the real installed
        # operator, and apply_kernel_patch's legacy full-file replace would happily overwrite the real site-packages
        # module with it (PR #1191 review finding #1).
        return {
            "status": "failed",
            "error_class": "vendor_playbook_not_deployable",
            "error": (
                "integrate refused: kernel_id="
                f"{payload.get('kernel_id')!r} is a vendor-playbook result "
                "(closed-source operator launch-config tuning); it has no "
                "deployable artifact and must not be applied as a source patch"
            ),
            "decision": "NEEDS_REVIEW",
            "kernel_id": payload.get("kernel_id"),
        }

    base_tput = float(payload.get("base_tput", 0.0))
    if base_tput <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }

    # Env-only is a property of the request, not of who sent it: a payload that carries a runtime bundle and names no
    # artifact has nothing to apply, so it is graded on the bundle alone.
    _has_artifact = bool(str(payload.get("patch_path") or "").strip()) or bool(
        str(payload.get("target_file") or payload.get("source_file") or "").strip()
    )
    env_only_validation = not _has_artifact and (
        bool(payload.get("extra_envs")) or bool(str(payload.get("extra_server_args") or "").strip())
    )
    if not env_only_validation:
        payload, missing_inputs = _resolve_integrate_payload(
            payload,
            session_dir=session_dir,
        )
        if missing_inputs is not None:
            return missing_inputs

    patch_path = payload.get("patch_path")
    kernel_id = payload.get("kernel_id")
    preapplied = payload.get("preapplied_apply_result")
    if isinstance(preapplied, dict) and preapplied.get("status") == "ok":
        manifest_path = Path(str(preapplied.get("manifest_path") or ""))
        patches_root = (Path(session_dir) / "patches").resolve()
        try:
            trusted_preapplied = manifest_path.is_file() and manifest_path.resolve().is_relative_to(patches_root)
        except OSError:
            trusted_preapplied = False
        apply_result = (
            dict(preapplied)
            if trusted_preapplied
            else {
                "status": "failed",
                "error_class": "untrusted_preapplied_manifest",
                "error": f"invalid pre-applied manifest: {manifest_path}",
            }
        )
    else:
        apply_result = _maybe_apply_kernel_patch(
            payload,
            session_dir=session_dir,
            kernel_id=kernel_id,
        )
    _checkpoint_collective_apply(
        str(payload.get("apply_checkpoint_path") or ""),
        apply_result,
    )
    if apply_result.get("status") == "skipped" and env_only_validation:
        apply_result = {
            "status": "ok",
            "reason": "env_only_validation",
            "kernel_id": kernel_id,
        }
    log.info("integrate_handler: apply_result=%s", apply_result)
    if apply_result.get("status") == "failed":
        # Apply crash: the patch was never measured.
        return {
            "status": "failed",
            "error_class": "apply_failed",
            "error": "kernel patch apply failed",
            "decision": "REVERT",
            "apply_result": apply_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": payload.get("target_file") or payload.get("source_file"),
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
    extra_args = str(payload.get("extra_server_args") or "").strip()
    # VRAM barrier (HL_HONEST_E2E umbrella, default ON; opt out with HL_HONEST_E2E=0 or HL_INTEGRATE_VRAM_GUARD=0):
    # cap re-baseline util on vLLM so the integrate server cannot OOM on a tighter node.
    extra_args = _vram_guarded_server_args(extra_args)

    # Wrap BaselineExecutor in a Task/RunnerContext.
    from hyperloom.inference_optimizer.session.session_paths import fs_safe_id, unique_runs_dir

    # A fusion sibling's kernel_id is its operator name (``llm:<recipe>``), which only reaches this handler now that
    # fusion lands through the generic queue.
    fake_task_id = f"integrate-{fs_safe_id(kernel_id)}"
    workspace = unique_runs_dir(session_dir, "integrate", fake_task_id)
    baseline_executor = BaselineExecutor(session_dir=session_dir)
    from ..state.shared_state import SharedState

    # Read-only, and only to learn whether this session is AgentX.
    try:
        _state_for_mode = SharedState.load_or_init(session_dir)
    except Exception as exc:  # noqa: BLE001 - advisory read, never fatal
        log.warning(
            "integrate: could not read session state to detect the benchmark mode "
            "(%s: %s); falling back to the HYPERLOOM_AGENTX env signal. The applied "
            "patch is unaffected.",
            type(exc).__name__,
            exc,
        )
        _state_for_mode = None

    rebaseline_timeout_sec = _agentx_rebaseline_timeout(
        _cold_start_rebaseline_timeout(
            _integrate_rebaseline_timeout_sec(
                payload,
                default_timeout_sec=baseline_executor.default_timeout_sec,
            )
        ),
        shared_state=_state_for_mode,
    )
    fake_task = Task(
        task_id=fake_task_id,
        kind="baseline",
        state="running",
        params={
            "config_path": payload.get("config_path"),
            "output_dir": str(workspace),
            "timeout_sec": rebaseline_timeout_sec,
            "extra_server_args": extra_args,
            "extra_envs": dict(payload.get("extra_envs") or {}),
            # The only artifact that patches FlyDSL sources, so the only run that needs the JIT cache key widened.
            "flydsl_source_dirs": (str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND),
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": base_tput * (1.0 + keep_threshold_pct / 100.0),
            "accuracy_timeout_sec": rebaseline_timeout_sec,
            # Synthetic kind="baseline": candidate A/B validation against the already-anchored reference.
            "quality_ref_exempt": True,
            # A sub-step of the KERNEL phase's own event, not a dispatched measurement, so it leaves no baseline
            # event.
            SBD_INNER_STEP_PARAM: True,
        },
        idempotency_key=f"{fake_task_id}-rebaseline",
    )
    ctx = RunnerContext(task=fake_task, lease=None)

    # aiter cpp_itfs kernels recompile at runtime and its cache hashes params not source, so set AITER_REBUILD=1 for
    # the re-baseline server to force a rebuild of the patched kernel.
    cpp_itfs_backup = apply_result.get("cpp_itfs_cache_backup") or {}
    force_aiter_rebuild = bool(cpp_itfs_backup.get("is_cpp_itfs"))
    _prev_aiter_rebuild = os.environ.get("AITER_REBUILD")
    if force_aiter_rebuild:
        os.environ["AITER_REBUILD"] = "1"

    def _restore_aiter_rebuild_env() -> None:
        """Restore the ``AITER_REBUILD`` env var to its prior value."""
        if not force_aiter_rebuild:
            return
        if _prev_aiter_rebuild is None:
            os.environ.pop("AITER_REBUILD", None)
        else:
            os.environ["AITER_REBUILD"] = _prev_aiter_rebuild

    # Multi-node: force a FULL sglang restart so it re-imports the patched modules (a resume would measure the
    # pre-patch process). mn_round_restarted stops a double restart; force_full_restart scopes the resume override
    # here.
    from ..actions.executors._multi_node_env import is_multi_node

    # This must run even when the regular JIT cache is warm: cpp_itfs attention uses the separate AITER_ROOT_DIR/build
    # tree and can carry a stale baton from a timed-out Forge driver.
    _sweep_integrate_aiter_locks(reason="integrate server startup")

    if is_multi_node():
        from ..actions.executors._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        try:
            await restart_server_for_round(
                extra_server_args=extra_args,
                framework=os.environ.get("FRAMEWORK") or None,
                model_path=(str(payload.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None),
                tp=int(os.environ.get("TP") or 0) or None,
                ep=int(os.environ.get("EP") or 0) or None,
                force_full_restart=True,
            )
            ctx.extra = {**(getattr(ctx, "extra", None) or {}), "mn_round_restarted": True}
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
        bench_result = await _run_integrate_rebaseline_with_lock_retry(
            baseline_executor,
            ctx,
            workspace=workspace,
            reason=f"integrate {kernel_id or 'anonymous'} rebaseline",
        )
    except Exception as exc:  # noqa: BLE001
        revert_result = _maybe_revert_kernel_patch(apply_result)
        return {
            "status": "failed",
            "error_class": "rebaseline_exception",
            "error": repr(exc),
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": payload.get("target_file") or payload.get("source_file"),
            "apply_result": apply_result,
            "revert_result": revert_result,
        }
    finally:
        # Restore AITER_REBUILD on every path so the override never leaks past this integrate.
        _restore_aiter_rebuild_env()

    if not is_valid_measurement(bench_result):
        revert_result = _maybe_revert_kernel_patch(apply_result)
        # The re-baseline produced no usable measurement, so the patch was never fairly scored.
        rebaseline_error_class = (
            str((bench_result or {}).get("error_class") or "").strip() if isinstance(bench_result, dict) else ""
        ) or "bench_exception"
        stopped = stopped_by_the_run_class(rebaseline_error_class)
        if stopped is not None:
            # Nothing was measured, so the patch has no verdict to answer for.
            return {
                "status": "failed",
                "error_class": stopped.error_class,
                "error": stopped.interrupted,
                "decision": "NEEDS_REVIEW",
                "rebaseline_detail": bench_result,
                "kernel_id": kernel_id,
                "patch_path": patch_path,
                "target_file": payload.get("target_file") or payload.get("source_file"),
                "apply_result": apply_result,
                "revert_result": revert_result,
            }
        return {
            "status": "failed",
            "error_class": rebaseline_error_class,
            "error": "re-baseline did not succeed",
            "decision": "REVERT",
            "rebaseline_detail": bench_result,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": payload.get("target_file") or payload.get("source_file"),
            "apply_result": apply_result,
            "revert_result": revert_result,
        }

    # Don't score a stale binary: for cpp_itfs targets the served kernel is runtime-compiled, so a reused
    # params-hashed lib.so would measure the PRE-patch kernel.
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
    from hyperloom.common.gain_math import gain_pct_or_zero, incremental_gain_pct

    gain_pct = gain_pct_or_zero(new_tput, base_tput)
    stack_positive_keep = False
    stack_incremental_gain_pct: float | None = None
    try:
        from ..state.shared_state import SharedState

        state = SharedState.load_or_init(session_dir)
        current_best = state.current_best or {}
        current_best_tput = float(current_best.get("tput") or 0.0)
        if current_best_tput > 0:
            stack_incremental_gain_pct = incremental_gain_pct(new_tput, current_best_tput)
        stack_positive_keep = (
            bool(state.optimization_stack)
            and str(current_best.get("action") or "") == "integrate"
            and current_best_tput > 0
            and stack_incremental_gain_pct >= STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
        )
    except Exception:  # noqa: BLE001 - fall back to the original threshold
        stack_positive_keep = False
    decision = (
        "KEEP"
        if (gain_pct > keep_threshold_pct or stack_positive_keep)
        else ("REVERT" if gain_pct < -keep_threshold_pct else "NEEDS_REVIEW")
    )

    # Accuracy gate: a kernel patch only KEEPs if it also holds accuracy.
    applyback_pending = (
        str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
        and str(payload.get("integration_validation_status") or "") != "passed"
    )
    accuracy_gate: dict[str, Any] | None = None
    if decision == "KEEP":
        accuracy_gate = _grade_integrate_accuracy(
            bench_result,
            session_dir=session_dir,
            workspace=workspace,
            strict=applyback_pending,
            server_args=extra_args,
        )
        if accuracy_gate["blocked"]:
            if accuracy_gate.get("infeasible"):
                # The gate cannot run under this configuration, so this round measured nothing about the patch.
                from ..actions.executors._accuracy_gate import (
                    EVAL_KIND_CONTEXT_TOO_SMALL,
                )

                revert_result = _maybe_revert_kernel_patch(apply_result)
                return {
                    "status": "failed",
                    "error_class": EVAL_KIND_CONTEXT_TOO_SMALL,
                    "error": accuracy_gate["reason"],
                    "decision": "NEEDS_REVIEW",
                    "gain_pct": gain_pct,
                    "accuracy_gate": accuracy_gate,
                    "revert_result": revert_result,
                }
            # A measured regression is hard negative evidence -> REVERT.
            decision = "REVERT" if accuracy_gate["accuracy_pass"] is False else "NEEDS_REVIEW"

    # import-grep source confirmation (HL_HONEST_E2E umbrella, default ON; opt out with HL_HONEST_E2E=0 or
    # HL_CONFIRM_SOURCE_IMPORTED=0).
    source_import_confirmed: bool | None = None
    source_import_evidence: dict[str, bool | None] = {}
    source_not_imported_downgrade = False
    if _honest_flag("HL_CONFIRM_SOURCE_IMPORTED"):
        # Grade the whole write set; the single target is the fallback for a patch whose bundle declared none.
        _written = [str(path) for path in (payload.get("patch_write_paths") or []) if str(path or "").strip()]
        if not _written:
            _written = [str(payload.get("target_file") or payload.get("source_file") or "")]
        source_import_confirmed, source_import_evidence = _confirm_sources_imported(
            _written,
            bench_result.get("workspace"),
        )
        if (
            decision == "KEEP"
            and source_import_confirmed is False
            and _honest_flag("HL_CONFIRM_SOURCE_IMPORTED_STRICT")
        ):
            decision = "NEEDS_REVIEW"
            source_not_imported_downgrade = True

    revert_result = (
        {"status": "skipped", "reason": "KEEP decision"}
        if decision == "KEEP"
        else _maybe_revert_kernel_patch(apply_result)
    )
    defer_patch_finalize = bool(payload.get("defer_patch_finalize", False))
    finalize_result = (
        {"status": "skipped", "reason": "deferred to caller durability checkpoint"}
        if decision == "KEEP" and defer_patch_finalize
        else (
            _maybe_finalize_kernel_patch(apply_result)
            if decision == "KEEP"
            else {"status": "skipped", "reason": "non-KEEP decision"}
        )
    )
    revert_required = decision != "KEEP" and bool(apply_result.get("manifest_path"))
    top_status, patch_cleanup_status, patch_cleanup_action = _cleanup_verdict(
        decision=decision,
        revert_result=revert_result,
        finalize_result=finalize_result,
        revert_required=revert_required,
    )

    result: dict[str, Any] = {
        "status": top_status,
        "decision": decision,
        "patch_cleanup_status": patch_cleanup_status,
        "patch_cleanup_action": patch_cleanup_action,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "base_tput": base_tput,
        "new_tput": new_tput,
        "gain_pct": gain_pct,
        "report_path": bench_result.get("report_path"),
        "workspace": bench_result.get("workspace"),
        "extra_server_args": extra_args,
        "extra_envs": dict(payload.get("extra_envs") or {}),
        "apply_result": apply_result,
        "revert_result": revert_result,
        "finalize_result": finalize_result,
        "rebuild_check": rebuild_check,
        "task_group_key": str(payload.get("task_group_key") or ""),
        "identity_route": str(payload.get("identity_route") or ""),
        "integration_id": str(payload.get("integration_id") or ""),
    }
    # Fusion provenance rides through so the KEEP writeback lifts the stack row as ``fusion`` (not ``integrate``) and
    # sets ``last_fusion_integrate`` -- both are read by the idempotency short-circuit and the remote-recipe fusion
    # export.
    if str(payload.get("source") or "") == "forge_fusion":
        result["source"] = "forge_fusion"
        result["action_label"] = str(payload.get("action_label") or "fusion")
    if top_status == "failed":
        result["error_class"] = "patch_revert_incomplete"
        result["error"] = str(revert_result.get("error") or "Kernel patch revert did not complete")
    if stack_positive_keep and gain_pct <= keep_threshold_pct:
        result["decision_reason"] = "stack_positive_increment"
        result["stack_incremental_gain_pct"] = stack_incremental_gain_pct
        result["stack_incremental_keep_threshold_pct"] = STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
    if source_import_confirmed is not None:
        result["source_import_confirmed"] = source_import_confirmed
    if len(source_import_evidence) > 1:
        result["source_import_evidence"] = source_import_evidence
    if source_not_imported_downgrade:
        result["decision_reason"] = "source_not_confirmed_imported"
    # Recorded last so a blocking accuracy verdict owns ``decision_reason``: it is the reason this candidate lost its
    # KEEP, outranking the throughput-side annotations above.
    if accuracy_gate is not None:
        result["accuracy"] = accuracy_gate["accuracy"]
        result["baseline_accuracy"] = accuracy_gate["baseline_accuracy"]
        result["accuracy_pass"] = accuracy_gate["accuracy_pass"]
        result["accuracy_gate"] = accuracy_gate
        if accuracy_gate["blocked"]:
            result["decision_reason"] = (
                "accuracy_regression" if accuracy_gate["accuracy_pass"] is False else "accuracy_evidence_missing"
            )
    if applyback_pending:
        result["artifact_kind"] = _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
        # Only a KEEP settles the outstanding verdict.
        if decision == "KEEP":
            result["integration_validation_status"] = "passed"
            result["validation_tier"] = _INTEGRATE_ACCURACY_VALIDATION_TIER
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_kernel_e2e(
            session_dir,
            kernel_id=str(kernel_id or ""),
            integrated=decision == "KEEP",
            e2e_gain_pct=gain_pct,
            validated=True if decision == "KEEP" else False,
            decision=decision,
            patch_path=str(patch_path or "") or None,
            target_file=str(payload.get("target_file") or payload.get("source_file") or "") or None,
            extra_server_args=extra_args,
            result=result,
            validation_tier=str(result.get("validation_tier") or "integrate_e2e"),
        )
    except Exception:  # noqa: BLE001
        log.debug("kernel integrate v4 result recording failed", exc_info=True)
    return result


# Kernel-agent programmatic dispatch table.
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "trace_analyze": trace_analyze_handler,
    "run_gemm_tuning": run_gemm_tuning_handler,
    # No run_fusion entry: KernelPhase awaits run_fusion_handler directly.
    "run_collective": run_collective_handler,
    "integrate": integrate_handler,
    "apply_patch": integrate_handler,  # alias — same flow
}


def has_handler(kind: str) -> bool:
    """Report whether a programmatic handler is registered for a request kind."""
    return kind in KERNEL_REQUEST_HANDLERS


def get_handler(kind: str) -> HandlerFn | None:
    """Look up the programmatic handler registered for a request kind."""
    return KERNEL_REQUEST_HANDLERS.get(kind)


__all__ = [
    "KERNEL_REQUEST_HANDLERS",
    "get_handler",
    "has_handler",
    "integrate_handler",
    "run_gemm_tuning_handler",
    "trace_analyze_handler",
]
