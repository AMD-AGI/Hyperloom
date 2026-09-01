# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator-side programmatic handlers for kernel REQUEST kinds.

Handler signature::

    async def handler(payload: dict, *, session_dir: Path) -> dict:

Dispatch table is exposed via :data:`KERNEL_REQUEST_HANDLERS` for test monkey-patching.
"""

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
from hyperloom.common.kernel_shape_contract import (
    ALLOWED_SHAPE_PROVENANCE as _ALLOWED_SHAPE_PROVENANCE,
)
from hyperloom.orchestrator.roles.agent_role import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)

from ..actions.stop_attribution import stopped_by_the_run_class
from .patch_lifecycle import cleanup_verdict as _cleanup_verdict
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from ..trace.task_progress import heartbeat_while_output_flows
from ..trace.parse_usage import (
    parse_forge_steps,
    parse_forge_usage,
    reasoning_output_tokens,
)

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
    record_kernel_opt as record_kernel_opt,
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
    unattempted_skip_reason as unattempted_skip_reason,
    SUPPORTED_COLLECTIVE_OPS as SUPPORTED_COLLECTIVE_OPS,
)
from ..state.kernel_decision_settings import (
    effective_hot_kernel_gpu_pct,
    effective_hot_kernel_min_gpu_pct,
)


log = logging.getLogger(__name__)

# Recognized trace-analysis routes; an unknown value falls back to ``agent``.
_VALID_ANALYSIS_ROUTES = frozenset({"bypass", "deterministic", "agent"})
STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT = 1.0
# A patch whose correctness was only established against a reference kernel;
# serving accuracy is what settles it.
_FRAMEWORK_APPLYBACK_ARTIFACT_KIND = "framework_applyback"
_INTEGRATE_ACCURACY_VALIDATION_TIER = "integrate_e2e_accuracy"

# Mirrors the completion ceiling the inferencex eval shim installs; kept in sync
# so the feasibility check reasons about the budget the eval will really ask for.
_EVAL_DEFAULT_MAX_TOKENS = 4096


def _vram_guarded_server_args(extra_args: str) -> str:
    """Optionally cap ``--gpu-memory-utilization`` for the integrate re-baseline.

    When ``HL_INTEGRATE_VRAM_GUARD`` is on and the caller has not already pinned
    ``--gpu-memory-utilization``, append a conservative cap
    (``HL_INTEGRATE_VRAM_UTIL_CAP``, default 0.90) so a re-baseline server cannot
    OOM. A strict no-op when the flag is off or a util is already specified.

    Args:
        extra_args: The resolved ``extra_server_args`` string for the server.

    Returns:
        str: ``extra_args`` unchanged, or with a util cap appended.
    """
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
    """Best-effort confirm the patched source was actually imported/compiled.

    Greps the re-baseline server log for evidence the patched module's basename
    was imported/loaded/compiled, so a measured E2E delta is attributed to code
    the workload really ran. Returns a tri-state:

    * ``True``  — the module basename appears in import/load/compile context.
    * ``False`` — the server log is readable and the basename never appears
      anywhere (positive evidence the patched file was not exercised).
    * ``None``  — unknown (no source_file, no readable log) — never penalized.

    Args:
        source_file: Resolved path of the patched kernel source.
        workspace: Re-baseline workspace dir (holds ``server.log``).

    Returns:
        bool | None: Tri-state confirmation as described above.
    """
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
    """Confirm every file a patch wrote was exercised by the served process.

    A patch can span several files -- a new module, the dispatcher that routes
    to it, the original source it replaces -- so each is graded on its own with
    :func:`_confirm_source_imported` and the verdicts are combined:

    * ``True``  — every file shows import evidence.
    * ``False`` — no file appears in the log at all, which is the unambiguous
      "the served process never ran any of this" case.
    * ``None``  — anything mixed. A module can be imported lazily or folded
      into another, so partial evidence is recorded for audit rather than held
      against the patch.

    Args:
        source_files: Paths the patch wrote; duplicates and blanks are ignored.
        workspace: Re-baseline workspace dir (holds ``server.log``).

    Returns:
        tuple[bool | None, dict[str, bool | None]]: The aggregate tri-state and
            the per-file verdicts kept for audit.
    """
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
    """Read the kernel-agent install root from the environment at call time.

    Resolved lazily on every call so a late ``os.environ`` injection by the CLI
    preflight still wins.

    Returns:
        Path | None: The kernel-agent root as a :class:`~pathlib.Path`, or
            ``None`` when ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset or empty.
    """
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
    """Framework install roots for the runtime-generated kernel classifier.

    Emits a lower-case variant per root because that classifier matches against
    a lower-cased source path. Path containment uses
    :func:`~hyperloom.orchestrator.framework.paths.resolved_within` instead.

    Returns:
        The de-duplicated framework install roots (each with a lower-case
        variant), including FlyDSL checkout roots.
    """
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
# forge is the only per-kernel backend. The default phase-level backend is the
# whole-pipeline GEAK delegate (``geak``); per-kernel selection is opt-in via
# KERNEL_OPT_BACKEND_ORDER=forge.
_DEFAULT_KERNEL_PHASE_BACKEND_ORDER = ("geak",)
# Soft cap on concurrent kernel-backend coroutines (pin with KERNEL_OPT_MAX_PARALLEL).
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
# forge-loop holds back a finalize reserve of half this window, so the figure
# here buys only half as much search as it reads. At 60 a campaign completed one
# iteration -- planning alone took 16 of its 30 usable minutes -- and terminated
# on budget_exhausted with nothing kept, which reads as "the kernel cannot be
# optimized" rather than "the kernel was tried once". 90 leaves ~45 usable
# minutes, enough for a second iteration to act on what the first measured.
_DEFAULT_BACKEND_BUDGET_MINUTES = 90.0
# Minimum wall-clock a fallback backend needs; below this the ladder stops.
_KERNEL_LADDER_MIN_BACKEND_SEC = 180
# Outer subprocess cap for the whole GEMM-tuning run (all shapes/tuners); sized
# for large models with many GEMM shapes. Independent of the session --max-hours
# budget; override via HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC (or payload timeout_sec).
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 5 * 60 * 60
_FORGE_FUSION_WRAPPER_TIMEOUT_GRACE_SEC = 30


def _visible_gpu_count() -> int | None:
    """Visible GPU count via ``torch.cuda.device_count()``.

    Returns ``None`` when torch can't tell us (missing / driver-init
    failure) so callers can distinguish "no GPUs" (``0``) from "unknown"
    and pick the right fallback. Works for both ROCm and CUDA backends.

    Returns:
        The visible GPU count, or ``None`` when torch is unavailable or
        driver init fails.
    """
    try:
        import torch  # local import: torch driver init is expensive

        return int(torch.cuda.device_count() or 0)
    except Exception:  # noqa: BLE001 -- torch missing / driver init failure
        return None


def _per_task_gpus() -> int:
    """GPUs reserved per kernel-opt attempt (``$KERNEL_AGENT_NUM_GPUS``).

    Floors at 1 so a missing / invalid env never zero-divides or stalls
    the batch fanout.

    Returns:
        The per-attempt GPU reservation, always ``>= 1``.
    """
    try:
        per_task = int(os.environ.get("KERNEL_AGENT_NUM_GPUS", "0") or 0)
    except (TypeError, ValueError):
        per_task = 0
    return per_task if per_task > 0 else 1


@functools.lru_cache(maxsize=1)
def _default_kernel_batch_parallel() -> int:
    """Adaptive batch fanout: ``min(cap, visible_gpus // per_task_gpus)``.

    Uses ``torch.cuda.device_count()`` for the visible-GPU count and
    ``$KERNEL_AGENT_NUM_GPUS`` for the per-attempt reservation, falling back to
    ``_DEFAULT_KERNEL_BATCH_PARALLEL`` when torch can't tell us. Operators can
    pin via ``KERNEL_OPT_MAX_PARALLEL``.

    Cached (driver query); tests that monkeypatch torch / env must call
    ``cache_clear()`` (the conftest autouse fixture handles this).

    Returns:
        int: The adaptive maximum number of concurrent sibling kernel
        attempts, ``min(cap, visible_gpus // per_task_gpus)``.
    """
    n_gpus = _visible_gpu_count()
    if not n_gpus or n_gpus <= 0:
        return _DEFAULT_KERNEL_BATCH_PARALLEL
    return max(1, min(_DEFAULT_KERNEL_BATCH_PARALLEL, n_gpus // _per_task_gpus()))


def _should_parallelize_backends(payload: dict, num_candidates: int) -> bool:
    """Decide whether to run backend ladders in parallel per kernel.

    With the ladder converged to a single forge backend there is no second
    ladder to race, so the auto-derived default is always sequential. Operators
    / tests can still force the flag via payload ``parallel_backends`` or env
    ``KERNEL_OPT_PARALLEL_BACKENDS`` (truthy ``1/true/yes/on`` enables).

    Args:
        payload: Request payload; ``parallel_backends`` may force the choice.
        num_candidates: Number of kernel candidates in this request.

    Returns:
        ``True`` only when explicitly forced on, else ``False``.
    """
    override = payload.get("parallel_backends")
    if override is None:
        raw_env = os.environ.get("KERNEL_OPT_PARALLEL_BACKENDS")
        if raw_env is not None and raw_env.strip() != "":
            override = raw_env
    if override is not None:
        return str(override).strip().lower() in {"1", "true", "yes", "on"}
    return False


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
    """Validate that the kernel-agent install root is configured and present.

    Returns:
        str | None: A human-readable error message when the root env var is
            unset or points at a missing directory, or ``None`` when the root
            exists and is usable.
    """
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
    """Resolve the TraceLens checkout, independent of inherited env.

    Falls back to the install-script-derived pod-local path so trace analysis
    works even when the coordinator process did not source kernel-agent.env.sh.

    Returns:
        Path: The resolved TraceLens root (may not exist yet; callers validate).
    """
    from hyperloom.inference_optimizer.session import paths

    return paths.tracelens_root()


def _tracelens_root_error(root: Path) -> str | None:
    """Validate that the resolved TraceLens root is a usable git checkout.

    A directory that exists but lacks ``.git`` is not usable and must be reported
    so a non-default override fails fast and a default path is self-healed.

    Returns:
        str | None: A human-readable error when the checkout is missing or
            incomplete, or ``None`` when it is a usable git checkout.
    """
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
    """Rebuild the pod-local TraceLens checkout if it vanished mid-run.

    Only the installer-managed default path is healed; an explicit
    ``TRACELENS_ROOT`` override must fail fast when missing. Best-effort: any
    failure is swallowed so the caller's validation produces the error.
    """
    from hyperloom.inference_optimizer.session import paths

    # The installer-managed checkout is <deps_cache_root>/TraceLens or the
    # per-revision <deps_cache_root>/TraceLens@<sha>; both are healable. An
    # explicit override elsewhere must fail fast (never auto-clone).
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
    """Resolve the absolute path to a kernel-agent shell tool.

    Args:
        tool_name (str): File name of the tool under ``<root>/tools/`` (for
            example ``tracelens_analysis.py``).

    Returns:
        Path: The resolved path to the requested tool.

    Raises:
        RuntimeError: If the kernel-agent root is unset/missing, or the named
            tool does not exist under ``<root>/tools/``.
    """
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
    """Detect torch.compile/Inductor/Triton runtime-generated kernels.

    Such kernels are regenerated each run, so patching them would not yield a
    reusable optimization. A name matching a compile-generated marker is only
    treated as runtime-generated when its source path is *not* under a known
    reusable framework root.

    Args:
        name (str): Kernel name (e.g. ``triton_poi_fused_...``).
        source_file (str): Resolved source path for the kernel.

    Returns:
        bool: ``True`` if the kernel appears runtime-generated and therefore
            non-reusable, ``False`` otherwise.
    """
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        return not any(root in lower_file for root in _reusable_source_roots())
    return False


def _load_candidate_metadata(payload: dict) -> dict[str, Any]:
    """Find candidate metadata for the requested ``kernel_id`` if available.

    Prefers an inline ``payload['candidate']`` dict; otherwise reads the
    ``candidates_path`` JSON artifact and looks up the matching entry in its
    ``hot_kernels`` list by ``kernel_id``.

    Args:
        payload (dict): Request payload, expected to carry either a
            ``candidate`` dict or both ``candidates_path`` and ``kernel_id``.

    Returns:
        dict[str, Any]: The candidate metadata dict, or an empty dict when no
            match is found or the artifact cannot be read/parsed.
    """
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
    """Best-effort coercion of a string runtime value to ``int`` or ``float``.

    Integer-looking strings become ``int``; strings containing ``.`` that
    parse as a float become ``float``. Anything else (including unparseable
    strings and non-string inputs) is returned unchanged.

    Args:
        value (Any): The raw value to coerce.

    Returns:
        Any: The coerced numeric value, or the original value when no safe
            numeric coercion applies.
    """
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
    """Decide whether an env var may be forwarded as candidate metadata.

    Rejects anything that looks sensitive (keys, tokens, secrets, passwords,
    credentials); otherwise allows the key if it is in the explicit allowlist
    or starts with a known safe prefix (e.g. ``SGLANG_``, ``VLLM_``).

    Args:
        key (str): Environment variable name to test.

    Returns:
        bool: ``True`` if the env var is safe to surface, ``False`` otherwise.
    """
    upper = key.upper()
    if any(part in upper for part in _SENSITIVE_ENV_PARTS):
        return False
    return key in _CANDIDATE_ENV_KEYS or any(key.startswith(prefix) for prefix in _CANDIDATE_ENV_PREFIXES)


def _split_server_args(raw: str) -> list[str]:
    """Tokenize a raw server-args string into an argv list.

    Args:
        raw (str): Raw shell-style server argument string.

    Returns:
        list[str]: The parsed argv tokens, or an empty list when ``raw`` is
            falsy or cannot be parsed (a warning is logged on parse failure).
    """
    try:
        return shlex.split(raw) if raw else []
    except ValueError:
        log.warning("failed to parse materialized server args; preserving raw string")
        return []


def _load_materialized_workload_metadata(config_path: str) -> dict[str, Any]:
    """Extract runtime workload context from a materialized Magpie YAML config.

    Reads the config's ``benchmark`` block and derives the per-framework
    server-args env name, the allowed candidate env vars, and a normalized
    ``runtime_args`` view (framework, model, precision, server args, and the
    coerced workload knobs such as ``tp`` / ``conc`` / ``isl`` / ``osl``).

    Args:
        config_path (str): Path to the materialized workload YAML config.

    Returns:
        dict[str, Any]: A dict with ``env_vars`` and ``runtime_args`` keys, or
            an empty dict when the path is missing/unreadable. Empty/``None``
            ``runtime_args`` entries are dropped.
    """
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
    """Backfill runtime env/args metadata onto each candidate kernel in place.

    For every dict candidate, sets default ``env_vars`` and ``runtime_args``
    entries from ``metadata`` without overwriting values the candidate already
    carries (uses ``setdefault`` semantics).

    Args:
        candidates (Any): Expected to be a list of candidate dicts; ignored if
            not a list.
        metadata (dict[str, Any]): Metadata with ``env_vars`` / ``runtime_args``
            sub-dicts as produced by
            :func:`_load_materialized_workload_metadata`.

    Returns:
        None: The ``candidates`` list is mutated in place.
    """
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
    """Stamp the TraceLens report path onto each candidate kernel in place.

    Args:
        candidates (Any): Expected to be a list of candidate dicts; ignored if
            not a list.
        report_path (str): Path to the TraceLens ``analysis.md`` report; ignored
            if empty.

    Returns:
        None: Each dict candidate gains a default ``trace_report_path`` entry.
    """
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
    """Rewrite the on-disk candidates artifact with enriched metadata.

    Loads the ``candidates_path`` JSON, enriches its ``hot_kernels`` and
    ``hot_kernels_top15`` lists with runtime metadata and (optionally) the
    TraceLens report path, then writes the artifact back out (pretty-printed,
    key-sorted). No-op when the path is missing or unreadable.

    Args:
        candidates_path (str): Path to the candidates JSON artifact to update.
        metadata (dict[str, Any]): Runtime metadata to merge into each kernel.
        trace_report_path (str): Optional TraceLens report path to record at
            both the top level and on each kernel entry.

    Returns:
        None: The artifact file is rewritten in place when changes apply.
    """
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


def _validate_reusable_native_kernel(payload: dict) -> HandlerResult | None:
    """Reject compile-generated or otherwise non-reusable kernel targets.

    Validates the requested kernel before optimization: it must not be marked
    ``reusable_native_kernel=False``, must have a resolved ``source_file``,
    must not be runtime-generated, and that source must live under a known
    reusable framework root. On success, defaults ``payload['source_file']``
    to the resolved source.

    Args:
        payload (dict): Request payload describing the target kernel (carries
            ``kernel_id`` and optionally ``candidate`` / ``source_file``).

    Returns:
        HandlerResult | None: A structured ``status="failed"`` result (with an
            ``error_class`` such as ``non_reusable_kernel`` or
            ``runtime_generated_kernel``) when the kernel is rejected, or
            ``None`` when the kernel passes validation.
    """
    candidate = _load_candidate_metadata(payload)
    kernel_id = str(payload.get("kernel_id") or "")
    name = str(candidate.get("name") or payload.get("kernel_name") or kernel_id)
    source_file = str(payload.get("source_file") or candidate.get("source_file") or "")
    reusable = candidate.get("reusable_native_kernel")
    if reusable is False:
        return {
            "status": "failed",
            "error_class": "non_reusable_kernel",
            "error": "kernel-opt only accepts reusable native kernel sources",
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
            "reason": candidate.get("optimization_notes") or "candidate marked reusable_native_kernel=false",
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
                "refusing to optimize torch.compile/Inductor runtime-generated kernel; result would not be reusable"
            ),
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
        }
    from ..framework.paths import (
        resolve_patch_target_roots,
        resolved_within,
        source_file_candidates,
    )

    if not any(
        resolved_within(candidate, root)
        for candidate in source_file_candidates(source_file)
        for root in resolve_patch_target_roots()
    ):
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


def _validate_kernel_shape_and_paths(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult | None:
    """Reject a kernel-opt dispatch with untrusted shape provenance or a missing source/workspace path.

    A shapeless candidate is NOT rejected. A graph-launched kernel records no
    CPU-side parent, so the profiler strips its argument dims; refusing to
    dispatch those means the hottest kernels of a captured model can never be
    optimized. The invocation spec reports the absent operands under its
    ``missing`` list and the backend's driver preparation recovers them from the
    kernel source, the tests it names and the deployment context the spec
    carries. Provenance is still checked, but only for a shape that is actually
    there: on a shapeless row the marker names why the dims are absent
    (``unresolved``), and reading that as an untrusted operand dim would close
    the same door from the other side.

    Args:
        payload: Kernel-opt dispatch payload to validate.
        session_dir: Session directory used as the default workspace path.

    Returns:
        A failure ``HandlerResult`` describing the rejection, or ``None`` when
        the dispatch is valid.
    """
    # ``dry_run`` exercises the plumbing without a backend.
    if bool(payload.get("dry_run")):
        return None
    candidate = _load_candidate_metadata(payload)
    kernel_id = str(payload.get("kernel_id") or "")
    name = str(candidate.get("name") or payload.get("kernel_name") or kernel_id)

    # Absent dims are dispatchable; malformed ones are not. A dict or a bare
    # string here is truthy, so a bare truthiness test admitted it and the type
    # error surfaced deep in driver preparation -- against this kernel's retry
    # quota, reported as an optimization failure. Nor is it right to read one as
    # shapeless: an empty ``shapes`` is evidence the trace could not record,
    # which the backend recovers from source, while a malformed one is a
    # producer that broke, and reading it as absent hides that indefinitely.
    raw_shapes = candidate.get("shapes")
    if raw_shapes and not isinstance(raw_shapes, list):
        return {
            "status": "failed",
            "error_class": "malformed_kernel_shapes",
            "error": f"shapes must be a list of operand dims, got {type(raw_shapes).__name__}",
            "kernel_id": kernel_id,
            "kernel_name": name,
        }
    has_shapes = bool(raw_shapes)
    provenance = str(candidate.get("shape_provenance") or payload.get("shape_provenance") or "").strip()
    if has_shapes and provenance and provenance not in _ALLOWED_SHAPE_PROVENANCE:
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

    source_file = str(payload.get("source_file") or candidate.get("source_file") or "").strip()
    if source_file and not Path(source_file).exists():
        return {
            "status": "failed",
            "error_class": "missing_source_path",
            "error": f"kernel source path does not exist: {source_file}",
            "kernel_id": kernel_id,
            "kernel_name": name,
            "source_file": source_file,
        }
    workspace_path = str(payload.get("workspace_path") or session_dir or "").strip()
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
    """Lazily import and cache the kernel-agent ``apply_kernel_patch.py`` module.

    Loaded by file path via :mod:`importlib.util` and memoized in the module
    global ``_APPLY_TOOL_MODULE`` so subsequent calls reuse the same module.

    Returns:
        Any: The imported ``apply_kernel_patch`` module object.

    Raises:
        RuntimeError: If the kernel-agent root/tool path cannot be resolved.
        ImportError: If the module cannot be loaded from its resolved path.
    """
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
    """Normalize compiled-artifact paths from a payload into a list of strings.

    Accepts either ``artifact_paths`` or ``compiled_artifact_paths``; a single
    string is wrapped into a one-element list and falsy entries are dropped.

    Args:
        payload (dict): Request payload that may carry artifact path(s).

    Returns:
        list[str]: The collected artifact paths (possibly empty).
    """
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
    """Apply a kernel patch via the kernel-agent ``apply_kernel_patch`` tool.

    Resolves a backup root under the session's patches dir when none is given,
    then delegates to the tool with rebuild / dry-run / target options pulled
    from the payload.

    Args:
        payload (dict): Request payload carrying ``patch_path`` plus
            ``target_file`` / ``source_file`` and optional apply/rebuild flags.
        session_dir (Path): Session directory used to derive the backup root.
        kernel_id (str | None): Kernel identifier for backup namespacing;
            falls back to ``payload['kernel_id']`` or ``"anon"``.

    Returns:
        HandlerResult: A ``status="skipped"`` result when required inputs are
            missing, otherwise the tool's apply result dict.
    """
    patch_path = str(payload.get("patch_path") or "").strip()
    target_file = str(payload.get("target_file") or payload.get("source_file") or "").strip()
    if not patch_path or not target_file:
        return {
            "status": "skipped",
            "reason": "missing patch_path or target_file/source_file",
        }
    from hyperloom.inference_optimizer.session.session_paths import patches_dir

    kid = str(kernel_id or payload.get("kernel_id") or "")
    backup_root = payload.get("backup_root") or (patches_dir(session_dir, kid or "anon") / "backup")
    tool = _load_apply_tool()
    # Snapshot mode: a snapshot dir of byte-exact final files lands atomically.
    snapshot_dir = str(payload.get("snapshot_dir") or "").strip() or None
    repo_root = str(payload.get("kernel_repo") or payload.get("repo") or "").strip() or None
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
    """Materialize final file contents for apply_kernel_patch snapshot mode.

    Applies a ``forge-fusion`` unified diff to a minimal throwaway mirror of the
    touched files and returns that mirror path (snapshot mode treats the diff as
    a manifest with final bytes under ``snapshot_dir``).
    """
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

    # Paths the patch CREATES: these must be produced by ``git apply``, never
    # pre-seeded with a base, or apply fails "already exists". Everything else
    # is a modify whose base we must supply. ``is_new`` comes from
    # ``parse_patch_manifest`` (single source of truth for both the path
    # normalization and the create/modify disposition), which avoids a second,
    # drift-prone parse of the raw patch text.
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
            # ``git show HEAD:`` failed and this is a MODIFY (not a create):
            # non-git repo_root (e.g. vLLM/sglang under site-packages/
            # dist-packages) or an untracked-but-present file. Fall back to the
            # on-disk source. forge-fusion (PR #75) emits the patch for these
            # non-git frameworks; without this fallback the snapshot lacks the
            # base file and ``git apply`` fails "<path>: No such file or
            # directory". New files are intentionally left for ``git apply`` to
            # create.
            src = root / rel
            if not src.is_file():
                # Neither git HEAD nor the on-disk layout has the base. Surface
                # a precise error here instead of the opaque ``git apply`` "No
                # such file or directory" that would otherwise follow.
                raise FileNotFoundError(
                    f"patch base missing for {rel.as_posix()}: not in git HEAD and not on disk under {root}"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    # ``git apply <path>`` rejects an otherwise valid final hunk when the patch
    # artifact lacks a trailing newline (observed in legacy KB records). Feed a
    # normalized in-memory copy so materialization is tolerant without mutating
    # the content-addressed downloaded artifact.
    normalized_patch_text = patch_text if patch_text.endswith(("\n", "\r")) else f"{patch_text}\n"
    proc = subprocess.run(
        ["git", "apply", "--unsafe-paths", "-"],
        cwd=snap,
        input=normalized_patch_text,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"could not materialize patch snapshot: {msg[:500]}")

    for desc in descriptors:
        if desc.get("op") == "write" and not (snap / str(desc["path"])).is_file():
            raise RuntimeError(f"snapshot missing final content for {desc['path']}")
    return str(snap)


def _maybe_revert_kernel_patch(apply_result: HandlerResult) -> HandlerResult:
    """Revert a kernel patch using its apply manifest.

    A manifest is enough; the apply's ``status`` is not required, so a partial
    apply reverts the files it managed to touch. Gating on ``status == "ok"``
    used to leave exactly those applied.

    Args:
        apply_result: Apply metadata carrying ``manifest_path``.

    Returns:
        The revert result, or an explicit failure result.
    """
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
    """Look up a kernel's source file from the last trace-analyze result.

    Searches ``state.last_trace_analyze`` (preferring ``hot_kernels_top15``,
    falling back to ``hot_kernels``) for the entry matching ``kernel_id``.

    Args:
        state (Any): SharedState snapshot exposing ``last_trace_analyze``.
        kernel_id (str): Kernel identifier to match.

    Returns:
        str: The matching candidate's ``source_file``, or an empty string when
            no match is found.
    """
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
    payload: dict,
    *,
    session_dir: Path,
) -> dict:
    """Pull ``base_tput`` / ``config_path`` / ``extra_server_args`` defaults from SharedState.

    Runs before the ``base_tput > 0`` hard-check in ``integrate_handler`` for
    bare ``{"kernel_id": ...}`` payloads. Always returns a shallow copy; never
    raises on a missing snapshot.

    Args:
        payload: The integrate request payload.
        session_dir: Session directory to load SharedState from.

    Returns:
        A shallow copy of ``payload`` with defaults filled from state.
    """
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
        pending_record = next(
            (
                record
                for record in pending_records
                if str(record.get("kernel_id") or "") == requested_kernel_id
                and (not requested_task_key or str(record.get("task_group_key") or "") == requested_task_key)
            ),
            None,
        )
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

    current_best = getattr(state, "current_best", None) or {}

    if float(resolved.get("base_tput", 0.0) or 0.0) <= 0:
        # ``extra_server_args`` below is filled from current_best, so the
        # candidate must be graded against that recipe too.
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
            # The candidate stacks onto current_best. Candidate-specific
            # overrides win, but omitting an env must not silently drop the
            # accepted recipe during E2E validation.
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
        # Defense-in-depth mirror of _queue_kernel_keep()'s refusal to queue
        # a vendor-playbook KEEP for auto-integration (PR #1191 review
        # finding #1): this also catches an LLM-initiated integrate request
        # that names the kernel_id directly, bypassing the pending-queue
        # lookup above via _resolve_kernel_patch_identity()'s
        # last_kernel_opt.best_artifact_path backfill.
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
    """Backfill artifact provenance for an integrate resolved from a ledger entry.

    These two fields arm the strict accuracy gate. A KEEP the ``source_file`` dedup
    drops from the pending queue resolves through a fallback instead, and without
    them a reference-only apply-back reads as an ordinary kernel patch.
    """
    if not resolved.get("artifact_kind") and isinstance(framework_applyback, dict):
        kind = str(framework_applyback.get("artifact_kind") or "")
        if kind:
            resolved["artifact_kind"] = kind
    if not resolved.get("integration_validation_status"):
        status = str(integration_validation_status or "")
        if status:
            resolved["integration_validation_status"] = status


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id (artifact in ``last_kernel_opt``, source in ``last_trace_analyze``).

    Args:
        payload: The integrate request payload.
        session_dir: Session directory to load SharedState from.

    Returns:
        A tuple of ``(resolved_payload, error_result)`` where ``error_result``
        is a failure ``HandlerResult`` when required inputs are missing, else
        ``None``.
    """
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
        # Provenance travels with the artifact so the serving verdict can tell a
        # reference-only apply-back from one already proven in place.
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
        # Snapshot deploy: prefer the original patch + snapshot dir so the whole
        # multi-file patch lands atomically.
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

    # Multi-KEEP queue fallback: pull patch_path/source_file from the per-kernel
    # ledger for KEEPs other than the strongest pending one.
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
    """Name the tool a command runs, for the progress note.

    Args:
        cmd (list[str]): The command and arguments.

    Returns:
        str: The first ``.py`` argument's stem, else the executable's name.
    """
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
    """Run a bounded subprocess without blocking the reactor.

    Args:
        cmd: The command and arguments to run.
        timeout_sec: Per-run timeout in seconds.

    Returns:
        A tuple of ``(returncode, stdout, stderr)``.
    """
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(float(timeout_sec))
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be finite and positive")

    def _run(on_output: Callable[[], None]) -> tuple[int, str, str]:
        """Run the command synchronously in a worker thread.

        Copies the environment, injects the Ray GCS address in multi-node mode,
        and prepends the venv ``bin`` to ``PATH``. Launches the child in its own
        POSIX session and, on timeout, reaps the whole process group so a hung
        grandchild dies with the wrapper. Mirrors ``subprocess.run``: captures
        stdout/stderr and re-raises ``TimeoutExpired``.

        Args:
            on_output: Liveness callback invoked per line the child emits.

        Returns:
            tuple[int, str, str]: ``(returncode, stdout, stderr)``.

        Raises:
            subprocess.TimeoutExpired: When the command exceeds ``timeout_sec``.
        """
        env = os.environ.copy()
        from ..actions.executors._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
            infera_ssh_env_from_state,
        )
        from ..actions.executors._subprocess_kill import run_with_session_kill

        if is_multi_node():
            # Infera backend: route GEAK GPU work to a pod over SSH (no Ray).
            # infera_ssh_env_from_state() returns {} for RayJob/single-node, so
            # the RAY_ADDRESS path below is unchanged for those.
            ssh_env = infera_ssh_env_from_state()
            if ssh_env:
                env.update(ssh_env)
            addr = "" if ssh_env else ray_gcs_address_from_state()
            if addr:
                env.setdefault("RAY_ADDRESS", addr)
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        # The heartbeat around this call is only as honest as the child's
        # flushing: block-buffered on a pipe, it looks dead between flushes.
        # ``setdefault`` so an operator who set this deliberately still wins.
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
    """Normalize a precision label to a trimmed lower-case string.

    Args:
        value (Any): Raw precision value (e.g. ``"FP8"``, ``None``).

    Returns:
        str: The lower-cased, whitespace-stripped precision, or an empty
            string for falsy input.
    """
    return str(value or "").strip().lower()


def _gemm_tuning_timeout_sec(payload: dict) -> int:
    """Resolve the GEMM-tuning subprocess timeout in seconds.

    Reads ``payload['timeout_sec']`` then the
    ``HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC`` env var, falling back to the module
    default; the result is floored at 60 seconds.

    Args:
        payload (dict): Request payload that may carry ``timeout_sec``.

    Returns:
        int: The resolved timeout in seconds (>= 60).
    """
    raw = payload.get("timeout_sec") or os.environ.get(
        "HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC",
        "",
    )
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = _DEFAULT_GEMM_TUNING_TIMEOUT_SEC
    return max(60, value)


def _forge_fusion_timeout_sec(payload: dict) -> int:
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
        value = 7200
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
    """Resolve the workspace directory for a GEMM-tuning run.

    Honors an explicit ``payload['workspace_path']``; otherwise builds a path
    under ``<session_dir>/runs/gemm_tuning/`` keyed by ``task_id`` /
    ``request_id`` (or a timestamped fallback).

    Args:
        payload (dict): Request payload that may carry ``workspace_path``,
            ``task_id`` or ``request_id``.
        session_dir (Path): Session directory used to build the default path.

    Returns:
        Path: The resolved (not yet created) workspace directory.
    """
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
    """Create an isolated benchmark wrapper for GEAK GEMM tuning (distinct port + no global ``pgrep sglang`` cleanup, so it can't kill the main optimizer's server).

    Args:
        workspace: Directory to write the benchmark script into.
        model_path: Path to the model under test.
        framework: Serving framework (e.g. ``sglang``).
        gpu_type: GPU type used to select the benchmark runner.
        tp: Tensor-parallel degree.
        conc: Concurrency.
        isl: Input sequence length.
        osl: Output sequence length.

    Returns:
        The path to the written, executable benchmark script.
    """
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
    """Read forge's on-disk ``result.json`` from the tuning workspace.

    forge always writes the full report (including ``tuners_skipped``) to
    ``<output_dir>/result.json``, even when the stdout sentinel omits some
    fields. Returns ``{}`` when missing or unparseable.
    """
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
    """Check exactly what ``_build_cmd`` will run, in the interpreter it runs in.

    The tuner is a subpackage of the ``kernelforge`` that ships in this
    distribution, invoked as ``sys.executable -m kernelforge.cli gemm-tune run``.
    Vendoring forge in-tree removes the cross-checkout failures this probe was
    built for, but not the reason it is a subprocess: ``find_spec`` proves the
    module is importable and says nothing about whether ``gemm-tune`` is
    registered on the CLI, which is the thing ``_build_cmd`` actually needs. So
    ask the subcommand itself -- in a subprocess, so a heavy CLI import cannot
    land in the orchestrator's own process. ``--help`` exits 0 only if
    ``kernelforge.cli`` imported and ``gemm-tune`` is registered on it.
    """
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
    """Resolve the actual runtime precision and quant_type for forge tuning.

    Priority:
    1. Explicit payload override
    2. --quantization from current_best server args (actual runtime)
    3. state.precision (session-level, may be stale)
    4. Default: bf16

    Returns (precision, quant_type) tuple.
    """
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
        # Hand forge the fp8 GEMM path the model runs: explicit per-token env wins,
        # else the checkpoint's static format, else "auto".
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


def _resolve_forge_server_log(state, session_dir: Path) -> str:
    """Find the server log matching the current runtime configuration.

    Priority: current_best workspace (matches the resolved server args)
    → baseline workspace → most recent server.log under runs/.

    The server log is written by the benchmark server at startup and lives in
    the warmup_round benchmark directory (where the server process was first
    launched). When ``current_best.workspace`` points to the measure_round
    benchmark directory (one level sibling), the log is not there — so we also
    check sibling ``warmup_round/`` dirs and walk up to the parent run
    directory.
    """

    def _find_server_log_near(workspace_str: str) -> str | None:
        if not workspace_str:
            return None
        ws = Path(workspace_str)
        # Direct hit (server started in this exact dir).
        if (ws / "server.log").is_file():
            return str(ws / "server.log")
        # Sibling warmup_round — benchmark dirs sit under
        # {run_hash}/{warmup_round|measure_round}/{benchmark_dir}/
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
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
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

    # Fallback: check known run subdirs with sufficient depth for the nested
    # layout ({phase}/{hash}/{warmup_round|measure_round}/{bench_dir}/server.log).
    runs_dir = session_dir / "runs"
    if runs_dir.is_dir():
        best: Path | None = None
        best_mtime: float = 0.0
        for sub in ("baseline", "explore", "gemm_tuning", "roofline"):
            sub_dir = runs_dir / sub
            if not sub_dir.is_dir():
                continue
            for log in sub_dir.glob("**/server.log"):
                try:
                    mt = log.stat().st_mtime
                except OSError:
                    continue
                if mt > best_mtime:
                    best_mtime = mt
                    best = log
        if best is not None:
            return str(best)

    return ""


def _is_forge_compatible_shapes_json(path: Path) -> bool:
    """Validate that a shapes JSON file matches forge's expected format.

    Forge expects: [{"M": int, "N": int, "K": int}, ...]
    or {"shapes": [{"M": int, "N": int, "K": int}, ...]}
    """
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
    """Find TraceLens shapes JSON if available and in forge-compatible format.

    Forge dense tuners expect: [{"M": int, "N": int, "K": int}, ...]
    Only passes files that match this schema; incompatible formats are
    silently skipped so forge falls back to config.json shape derivation.

    ``precision`` scopes the traced shapes to the dtype the tuner will actually
    tune (a trace carries every GEMM dtype the model runs, e.g. FP8 projections
    alongside BF16 router heads). Empty means "no dtype scoping".

    When scoping is requested the candidate extraction runs first, because it is
    the only source whose dtype can be checked: a pre-rendered shapes artifact is
    a bare ``[{M,N,K}]`` list carrying no dtype or provenance, so an artifact
    recorded for another dtype would otherwise be handed to the tuner ahead of
    correctly-scoped shapes. Artifacts stay the fallback for the unscoped case
    and for when the trace yields nothing for this precision.
    """
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

    # Extract the GEMM shapes observed by the latest TraceLens analysis. Older
    # traces can describe a backend that is no longer active.
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
    """Fold a precision name or traced dtype token onto one canonical family.

    Both sides of the comparison spell the same dtype many ways: a tuning
    precision arrives as ``fp8`` / ``mxfp4``, while TraceLens renders whatever
    the framework reported -- ``fp8_e4m3``, ``e4m3fnuz``, ``fp4x2``, and
    ``_TRACE_DTYPE_SUFFIX`` in this repo emits ``f16`` for float16. Matching the
    raw strings drops shapes that do belong to the tuned precision, so both are
    folded onto a family first.

    Returns "" for anything unrecognised, which callers treat as "do not scope".
    """
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
    """Extract M,N,K from kernel_candidates.json hot_kernels input_shapes.

    Derives the GEMM dimensions actually observed during serving and writes a
    forge-compatible shapes JSON beside the candidates file, returning its path.

    ``precision`` scopes the result to one traced dtype. A trace records every
    GEMM the model runs, so an FP8 tuner would otherwise also receive the BF16
    router/head shapes; those rows are never looked up at serve time and they
    displace real FP8 shapes in the call-count ordering below. Empty keeps every
    dtype (historical behaviour).
    """
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

    # Tolerate whitespace after the comma ("(1024, 5120)") and any leading token
    # before the tuple; TraceLens formats vary. .search() rather than .match() so
    # a leading dtype/name does not defeat it.
    dim_pattern = _re.compile(r"\((\d+)\s*,\s*(\d+)\)")
    # TraceLens renders the dtype right after the dims: "(64,3072) fp8".
    # Dots are allowed so a fully-qualified spelling ("torch.float8_e4m3fn") is
    # captured whole rather than truncated at "torch".
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
        # B is stored either (N,K) or (K,N); pick the orientation whose
        # contracted dim matches K, else keep the legacy first-dim reading.
        N = b0 if b1 == K else (b1 if b0 == K else b0)
        # ``N == 1`` is a matrix-vector head (e.g. a scalar projection), not a
        # tunable GEMM tile; it would otherwise sort first on call count and
        # burn a tuning slot.
        return (M, N, K) if min(M, K) > 0 and N > 1 else None

    # ``weight`` is the observed call count: decode GEMMs are invoked far more
    # often than prefill ones, so ordering by it puts the throughput-dominant
    # shapes first and they still get tuned when the tuner runs out of budget.
    # The same (M,N,K) can be reported by several kernels; keep the largest
    # count, otherwise a rare first sighting would outrank the hot one.
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

        # Legacy format: one entry carries every tensor, "<br>"-joined. The tag
        # is spelled several ways across TraceLens versions (<br>, <br/>, <BR/>).
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


# Map the resolved (precision, quant_type) to the aiter untuned-GEMM CSV the
# specialist phase records; fp8 "auto" resolves to blockscale (forge default).
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
    """Return True when ``path`` is a CSV carrying at least one data row.

    The aiter recorder leaves header-only or empty files for quant types the
    server never exercised; those must not be passed to forge as a real shape
    source.
    """
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
    """Return the distinct integer ``K`` (contraction-dim) values in a CSV.

    The aiter recorder writes a header containing ``M,N,K`` (optionally with
    extra columns such as ``q_dtype_w``). ``K`` is the GEMM contraction dim,
    which for a transformer layer equals its input dim (``hidden_size`` for
    QKV/gate-up/o projections, ``intermediate_size`` for the down projection).
    """
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
    # ``model_path`` may be an HF repo id; resolve to the local weights dir
    # (shared resolver) so the config read works for repo-id launches.
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
    """Pick the fp8 dense tuner quant_type from the checkpoint's static format.

    forge accepts an explicit ``quant_type``; rather than letting it fall back to
    its internal blockscale default, hand it the path the model actually runs:

    - ``blockscale_bpreshuffle`` when the checkpoint uses block-quantized
      weights AND the target GPU is gfx950 (MI355X) AND framework is sglang --
      sglang/aiter automatically upgrades blockscale to the bpreshuffle kernel
      on CDNA4. vLLM does NOT use this path (it reads
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE).
    - ``blockscale`` when the checkpoint uses block-quantized weights on gfx942,
      on vllm, or when GPU type is unknown.
    - ``per_token`` for a plain fp16/bf16 checkpoint served under dynamic
      ``--quantization fp8`` (the a8w8 per-token path).
    - ``auto`` when ``config.json`` cannot be read, so forge sniffs the
      ``kernel_signature_log`` itself (preserves the legacy behaviour and keeps
      the no-readable-config case unchanged).
    """
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
    """Return True when an untuned CSV plausibly belongs to ``model_path``.

    A real per-model dense untuned CSV always contains GEMMs whose ``K`` equals
    the model ``hidden_size``. When ``hidden_size`` is known and absent from the
    CSV's ``K`` column, the CSV was recorded for a different model and is
    rejected so forge derives shapes from the model config instead.

    Returns True when validation is not possible (``hidden_size`` unreadable or
    the CSV exposes no ``K`` column) to avoid false rejections.
    """
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


def _resolve_forge_untuned_csv(session_dir: Path, precision: str, quant_type: str, model_path: str = "") -> str:
    """Find an aiter untuned-GEMM CSV in a specialist worktree.

    Specialist runs may materialize or modify these files under
    ``runs/specialist/<hash>/worktree/aiter/configs/*_untuned_gemm.csv``; this
    resolver picks the newest non-empty CSV matching the resolved quant type.
    Because an unchanged checkout can also contain static upstream rows, this is
    a fallback behind explicit benchmark input and the latest runtime profile.

    When ``model_path`` is given, candidate CSVs whose GEMM shapes do not match
    the model are rejected so forge derives per-model shapes from ``config.json``.
    Returns the CSV path, or "" when none is available.
    """
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
    """Safe ``Path.is_file()`` that never raises on an over-long pathname.

    A caller may hand us inline JSON content instead of a path; ``is_file()``
    raises ``OSError(ENAMETOOLONG)`` on such input. Treat any OSError as
    "not a file".
    """
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _normalize_tokens(value: Any) -> str:
    """Return a clean comma-separated token string for forge's ``--tokens``.

    forge parses ``--tokens`` as ``int(t) for t in value.split(",")``, so accept
    lists and bracketed strings and emit a bare comma-separated list.
    """
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
    """Return a usable shapes-JSON *file path*, materializing inline content.

    Callers sometimes pass GEMM shapes as inline JSON in ``shapes_json`` instead
    of a file path; forge treats it strictly as a path. Normalize here:

    - existing file path -> returned unchanged
    - list/dict, or a string that parses as JSON -> written to
      ``<workspace>/forge_shapes.json`` and that path returned
    - anything else (empty / unparseable / non-existent path) -> ""
    """
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


# Forge tuner families whose deliverable is an aiter tuned-GEMM CSV, i.e. the
# ones whose rows are resolved through aiter's padded (M, N, K) lookup.
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
    """Re-key profiled GEMM shapes onto the M values aiter actually looks up.

    Captured shapes carry the raw runtime M, which for prefill is the
    data-dependent scheduled-token count and so never recurs between runs. aiter
    resolves a tuned row by trying the raw M and then two padded M variants, so a
    CSV keyed on raw M is unreachable and the tuner's micro win never reaches the
    server. Padding the shapes first makes each tuned row serve the whole bucket
    that pads onto it.

    Returns the shapes-JSON path to hand forge plus an alignment report, or the
    input path and ``None`` when alignment does not apply.
    """
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
    """Return whether aiter's CK MoE codegen has a kernel family for this pair.

    ``get_gemm1_kernels_list`` / ``get_gemm2_kernels_list`` pick a family from the
    activation/weight widths and raise ``Unsupported data type combination`` for
    anything else. Notably a BF16 activation against FP4 weights -- which the
    serving path runs happily -- matches no family, so handing it to the tuner
    trades a silent no-op for a hard error.
    """
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
    """Return the distinct MoE problems a server log shows aiter dispatching.

    Deduplicated on everything but the token count, preserving first-seen order.
    One model routinely yields several problems -- the same checkpoint dispatches
    both a BF16-activation and an FP8-activation variant, and the EP path appends
    a masked fake-expert slot so ``expert``/``topk`` arrive one higher than the
    model config states. Neither is derivable from the config, which is why the
    log is the authoritative source for what to tune.
    """
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
    """Return whether aiter's CK MoE tuner can tune anything the server dispatched.

    The tuner builds its kernel candidates from the activation/weight dtype pair
    and rejects some combinations the serving path happily runs. Measured on
    gpt-oss-120b at TP=1, a BF16-activation / FP4-weight MoE (the
    ``AITER_MXFP4_BF16`` backend) benchmarks fine but fails candidate generation
    with ``Unsupported data type combination: b16, fp4x2``, so routing it to
    ``fmoe_ck`` would only trade silent no-op for a hard tuner error.

    A single checkpoint can dispatch several dtype pairs at once, so this asks
    whether *any* of them is tunable; per-problem filtering happens where the
    tuning input is written.
    """
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
    """Positive token counts to sweep, keyed off whatever the caller sends.

    Accepts forge's comma-separated string (what :func:`_normalize_tokens`
    produces) or a sequence. Unparseable and non-positive entries are dropped
    rather than raising -- one bad entry is not worth the run -- and ``[1]`` is
    the floor so there is always a token to key on.
    """
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
    """Turn the MoE problems observed in ``server_log`` into a tuning input CSV.

    Returns ``(csv_path, report)``; ``csv_path`` is "" when nothing tunable was
    observed. Writing the observed tuple verbatim is the whole point: the
    quantisation pair, the per-partition ``inter_dim`` and the EP-inflated
    expert/topk counts are all properties of what the serving framework chose,
    and every attempt to re-derive them from the model config is a guess that has
    already produced tables no runtime lookup could reach.

    Problems whose dtype pair aiter's codegen rejects are dropped rather than
    passed through, because one unsupported row aborts the whole tuner run.
    """
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
        # A full disk or a read-only workspace must cost the MoE tuner its input,
        # not the whole tuning run: the dense tuners take their shapes from
        # elsewhere and can still produce something useful.
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
    """Return which aiter kernel families a server log shows in use.

    Routing is driven by the log rather than by precision alone because only the
    log says which backend the model actually got: the same checkpoint runs on
    aiter or on the native path depending on the recipe's env.
    """
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
    """Route vLLM runs served by aiter to Forge's AITER tuner family.

    Forge's vLLM branch only offers ``vllm_moe_triton`` and
    ``vllm_dense_tunableop``, which target kernels an aiter-served model never
    executes. Its sglang branch carries the tuners that do write the tables aiter
    reads, and the router already accepts ``vllm-aiter`` as an alias for it.
    """
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

    # Dense BF16 routing is for dense checkpoints; a MoE model's dense side
    # rides along with its MoE routing instead.
    flags["aiter_bf16_dense"] = "bf16_dense" in evidence and not is_moe

    if "fused_moe" in evidence and is_moe and _aiter_ck_moe_tuner_supports(server_log):
        # Only route MoE when aiter's CK fused-MoE can actually serve this
        # checkpoint at this TP -- otherwise the tuner has no reachable target.
        from hyperloom.inference_optimizer.cli.model_gate import (
            model_supports_aiter_ck_fused_moe,
        )

        flags["aiter_fused_moe"] = model_supports_aiter_ck_fused_moe(model_path, tp)

    _warn_if_moe_routing_is_coarser_than_the_log(server_log, flags)
    return flags


def _warn_if_moe_routing_is_coarser_than_the_log(server_log: str, flags: dict[str, bool]) -> None:
    """Say so when one log shows both MoE backends and routing picks one.

    The decision above is a substring scan: seeing an aiter fused-MoE marker
    anywhere routes the whole run to the aiter tuner family, and
    ``vllm_moe_triton`` then never runs. A run can dispatch both -- aiter CK over
    part of the token range and vLLM's Triton path over the rest -- and forge's
    own parser records exactly that as ``impl="mixed"``. Whichever way the single
    flag falls, the range served by the other backend is left untuned.

    Reported rather than acted on here: changing this routing changes which
    tuners run for every aiter-served vLLM model, which is a bigger step than
    the tuner-side addition that already covers the CK half. Forge adds
    ``fmoe_ck`` from the same evidence, so the gap this warns about is the
    Triton half.
    """
    if not flags.get("aiter_fused_moe"):
        return
    try:
        from kernelforge.gemm_tune.evidence import parse_log_file
    except ImportError:
        # Same reasoning as apply_verification._parse: kernelforge is in this
        # wheel, so a miss is a broken install, and a bare return makes the
        # missing routing warning indistinguishable from a clean run.
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
        # Shared classifier, so a layout the kernel-agent routes demote is also
        # kept out of the shape harvest here; the exact-``capture_traces`` test
        # this replaced missed ``graph_capture_profile/``.
        return _shared_is_capture_fragment(path, trace_input if trace_input.is_dir() else trace_input.parent)

    shapes: set[tuple[int, int, int]] = set()
    # ``Path("")`` normalizes to ``Path(".")``, which would otherwise walk the
    # whole process CWD and harvest shapes from unrelated traces.
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
        task_params.update(
            {
                "config_path": config_path,
                "timeout_sec": timeout_sec,
                "disable_run_eval": True,
                "baseline_double_run": False,
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
    """Deterministic GEMM tuning via the ``kernelforge gemm-tune`` CLI.

    Supports bf16/fp8/fp4 + sglang/vllm. Only micro-benchmarks;
    returns recommended_env for Hyperloom E2E validation.

    ``model_path`` accepts either a local directory or a Hugging Face repo ID.
    Forge receives a validated local directory, while result provenance and
    durable artifact names retain the original logical model identifier.
    Missing inputs return ``model_path_missing``; inputs that cannot resolve to
    a local directory return ``model_path_unavailable`` as a ``skipped`` result,
    because forge never ran and so has no verdict to report.
    """
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    # Importing kernelforge.cli is deliberately isolated in a subprocess, but
    # that subprocess may still take until the bounded timeout to fail. Keep the
    # synchronous probe off the orchestrator reactor.
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

    # Bootstrap already walked HL_MODEL_BASE and the hub cache to decide what to
    # serve; probing only the hub cache here would reject a repo id that the
    # running server resolved fine.
    resolved_model_dir = resolve_local_model_dir(resolve_serving_model_path(raw_model_path) or raw_model_path)
    if resolved_model_dir is None:
        # Forge needs the config on disk to derive shapes, so it cannot run --
        # but not running one tuning backend is a skip, not a session failure.
        # Reporting it as failed spends a REVERT verdict on an experiment that
        # never started, which is the misattribution this change set removes.
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
        kernel_sig_log = _resolve_forge_server_log(state, session_dir)

    # Explicit operator/benchmark input wins. Automatic SGLang priority is:
    # latest TraceLens runtime profile, specialist-worktree CSV fallback, then
    # Forge's config-derived fallback. A specialist checkout is not sufficient
    # evidence that its static CSV came from the active benchmark. vLLM instead
    # requires native TunableOp rows or a workload-matched block-FP8 profile.
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

    # MoE shapes come from the runtime, never from inference. The dispatch tuple
    # in the server log states the quantisation pair, the per-partition inter_dim
    # and the EP-inflated expert/topk counts; none of the three is recoverable
    # from the model config, and guessing them is what produced tuned tables no
    # runtime lookup could reach.
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
        # Decode steps replay inside a CUDA Graph and therefore emit no Kineto
        # *op* events, so every profile-derived block-FP8 shape set structurally
        # carries prefill M only -- measured on a real capture, the decode-only
        # trace split yields zero block-FP8 events while the prefill splits yield
        # M=2095. Tuning that alone optimizes an operating point the workload
        # barely uses. TraceLens candidates are built from the device kernel
        # timeline, which does see through the graph and carries the decode M
        # that dominates throughput, so prefer them. ``require_fresh_profile``
        # keeps the vLLM rule that shapes must be workload-matched, and
        # ``precision`` keeps BF16 heads out of an FP8 tuner's input.
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
        # Keyed on the routed framework: a run handed to the AITER tuner family
        # has no use for a TunableOp recording pass, and paying for one costs a
        # full extra server boot.
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
            # Forge dense tuners prefer untuned_csv over shapes_json. A fresh
            # profile capture is workload-matched and must supersede any stale
            # specialist CSV resolved before the capture pass.
            untuned_csv = ""

    timeout = _gemm_tuning_timeout_sec(payload)
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
        "timeout": timeout,
        # Bounds the whole session across all tuners.
        "global_timeout": timeout,
        "skip_gpu_check": True,
        "tokens": tokens,
        "untuned_csv": untuned_csv,
        "moe_untuned_csv": moe_untuned_csv,
        "shapes_json": shapes_json,
        "tunableop_input": tunableop_input,
        "kernel_signature_log": kernel_sig_log,
        "tuner": str(payload.get("tuner") or ""),
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
        # Kept even when nothing was tunable: "no MoE problem was observed" and
        # "the observed pair is one aiter cannot tune" lead to different actions,
        # and neither is visible from the tuner's own status.
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

    # Surface why forge skipped: merge per-tuner skip reasons from the on-disk
    # result.json and derive a top-level skip_reason.
    if not result.get("tuners_skipped"):
        disk_skipped = _read_forge_result_json(workspace).get("tuners_skipped")
        if disk_skipped:
            result["tuners_skipped"] = disk_skipped
    if not result.get("skip_reason"):
        reason = _derive_gemm_skip_reason(result.get("tuners_skipped"))
        if reason:
            result["skip_reason"] = reason

    # The breakdown and the stack read the envelope, not the jsonl audit row, so
    # a tuner's own error class has to surface here too. Lifted before the bridge
    # so a specific class outranks the generic wording. ``tuners_run`` is forge's
    # JSON and may be any shape; this is bookkeeping and must not raise.
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

    # Bridge forge schema → coordinator schema: a "candidate" micro_decision with
    # recommended_env becomes decision="KEEP" + extra_envs.
    micro = str(result.get("micro_decision") or "").strip().lower()
    if micro == "candidate" and result.get("recommended_env"):
        result.setdefault("decision", "KEEP")
        # Make the tuned CSV durable + recipe-portable (mirrors integrate_patch's
        # source-layer snapshot): copy it into the serving aiter config dir,
        # repoint the env there, and snapshot it so the KEEP survives with the
        # recipe instead of referencing the ephemeral tuner-workspace path.
        # Keep the logical ID here: a resolved HF snapshot basename is a commit
        # hash, which would make durable artifact names unstable across revisions.
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
    """Make forge GEMM tuned CSVs durable + recipe-portable.

    The forge KEEP references tuned CSVs by their ephemeral tuner-workspace paths,
    so a recipe replayed after the workspace is gone (or on another box) loses the
    tuning and aiter falls back to its default config. Mirror integrate_patch's
    durability: copy each CSV into the serving aiter config tree, repoint the env
    there, and snapshot the realized files via :func:`snapshot_source_layer` so
    they travel with the recipe.

    The copy lands one level below ``configs/model_configs/`` on purpose. aiter
    merges every ``model_configs/*{table}*.csv`` it can glob whenever the env var
    is unset, and that glob is not recursive. Writing directly into that
    directory would hand the table to every later server start -- including after
    E2E rejected the candidate, and including servers for other models, since the
    scan does not discriminate by model. Replay does not need the scan: it
    restores the env var explicitly (see ``prelude._warm_kernel_extra_envs``) and
    defers a GEMM column that has no env at all.

    The snapshot lands under ``<session_dir>/optimization_stack/src/`` (the same
    durable, run-cleanup-surviving location integrate_patch uses) -- NOT under the
    ephemeral ``runs/gemm_tuning`` workspace, which would be cleaned away and
    defeat the cross-environment recipe-portability this exists for.

    Best-effort: on any error the env is returned unchanged (never breaks the KEEP).
    Returns ``(extra_envs, source_snapshot_dir)``.
    """
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

    # Step 1 -- commit durable copies + env repoints. This is what makes the
    # KEEP survive: each CSV lands in aiter's config dir and the env points
    # there instead of the ephemeral tuner workspace.
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

    # Step 2 -- recipe-portability snapshot. Separate best-effort concern: a
    # snapshot failure must NOT discard the copy + repoint committed above.
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
    """Legacy GEAK GEMM tuning wrapper.

    Hyperloom does not decide precision/framework applicability here; it passes
    the workload metadata through and lets GEAK decide.
    """
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
    """Run GEMM tuning via GEAK, or forge only when explicitly enabled.

    Backend selection:
    1. Exact ``KERNEL_OPT_BACKEND_ORDER=forge`` -> forge.
    2. Everything else -> GEAK.

    Args:
        payload: The GEMM-tuning request payload.
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` describing the tuning outcome.
    """
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
    """Check that KernelForge's fusion pipeline is importable.

    Probes the subpackage rather than ``kernelforge``: an installation
    predating the fusion absorption would satisfy the parent import and only
    fail once the subprocess rejected ``forge-fuse``. PATH is not consulted
    because the tool is invoked through ``sys.executable -m``.
    """
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
    """Reuse the PRELUDE/roofline decode trace for fusion discovery.

    forge-fusion's discover stage needs a CUDA-graph-disabled decode kineto trace,
    already captured in PRELUDE (``state.last_profile_trace``); reuse it instead of
    re-profiling. Explicit ``payload['trace_path']`` wins.
    """

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
    """Resolve the Forge agent backend and model as one decision.

    Shared by forge-fusion and forge-collective (rewrite uses the same model
    ladder via :func:`llm_config.resolve_forge_llm_model`). The canonical
    provider-shape predicates decide the default backend: OpenAI-only uses
    Codex, while Anthropic-only and dual-configured deployments use Claude, the
    established default for this agentic role. A valid explicit
    ``agent_backend`` or ``llm_model`` in the request wins. With no configured
    provider, the request fails instead of silently spawning an unauthenticated
    Claude process.

    Model id precedence (after the backend is chosen) is owned by
    :func:`llm_config.resolve_forge_llm_model`.

    Args:
        payload: Kernel request payload.
        env: Provider environment to inspect; defaults to ``os.environ``.

    Returns:
        The canonical ``(agent_backend, llm_model)`` pair.

    Raises:
        RuntimeError: If neither provider side is configured.
        ValueError: If ``agent_backend`` is not ``"claude"`` or ``"codex"``.
    """
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
    """Resolve and validate the sandbox policy recorded for forge-fusion.

    Codex delegates both defaults and validation to the canonical Hyperloom
    resolver, including its double opt-in for ``bypass``. Claude records
    ``workspace-write`` as the stable audit default; an explicit override is
    validated by that same resolver so both backends share one policy vocabulary
    and unsafe bypass cannot reach the subprocess.

    Args:
        payload: Kernel request payload.
        agent_backend: The already-resolved ``"claude"`` or ``"codex"`` backend.
        env: Environment overlay for the canonical resolver; defaults to the
            process environment.

    Returns:
        A validated KernelForge sandbox mode.

    Raises:
        CodexSessionUnavailableError: If the mode is unknown or bypass lacks
            either operator confirmation.
    """
    explicit = str(payload.get("agent_sandbox_mode") or "").strip()
    if agent_backend == "claude" and not explicit:
        return codex_session.DEFAULT_CODEX_SANDBOX_MODE
    return codex_session.resolve_codex_sandbox_mode(
        sandbox_mode=explicit,
        env=dict(env) if env is not None else None,
    )


async def _run_forge_fusion(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Autonomous kernel fusion via the forge-fusion CLI.

    Builds an input-json with one provider-compatible agent backend, model, and
    validated sandbox policy, shells out to the ``forge_fusion.py`` wrapper, and
    parses the result sentinel. A KEPT fusion carries a source patch + env flags
    and ``requires_e2e_validation`` so the integrate gate confirms the
    end-to-end gain. Reuses the PRELUDE decode trace (no re-profiling).
    """
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
    timeout = _forge_fusion_timeout_sec(payload)

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
        "fuse_all_confirmed": bool(payload.get("fuse_all_confirmed", True)),
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
    """Run autonomous kernel fusion via forge-fusion (serving-validated).

    Registered as the ``run_fusion`` kernel request. Authors serving-safe fused
    kernels and returns a source patch + env flags for the integrate gate.
    """
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
    """Read a forge-loop budget bound, falling back when KernelForge is off the path.

    A local copy of the number drifts the moment upstream changes it, and the
    lane then plans against a budget the campaign will not honour.

    The fallback is logged rather than taken silently. KernelForge now ships in
    this distribution, so a failed import means a renamed module or a broken
    install, not an optional dependency -- and the symptom otherwise is a
    campaign quietly planned against the wrong wall-clock budget, which no run
    ever reports.
    """
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
        # Unbounded session: no budget to divide, so the reserve and
        # minimum-campaign contracts below cannot apply.
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
    # Truncate to two decimals so the hours we hand forge-loop never round up
    # past the budget they were derived from.
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
    # The anchor alone hides the op's sibling sources, so a fused variant of the
    # same collective stays uneditable. Keep the anchor first, then whatever the
    # candidate declares as the op's source set.
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
    """Run the coordinator-owned collective optimization lane.

    The attempt identity is deliberately left unset: the KERNEL phase derives it
    from the result's content so a replayed or salvaged campaign deduplicates
    against its earlier record. Stamping a wall-clock id here would make every
    replay look like a new campaign.
    """
    result = await _run_forge_collective(payload, session_dir=session_dir)
    if not isinstance(result, dict):
        raise TypeError("Collective handler result must be a mapping")
    result.setdefault("requires_e2e_validation", False)
    return result


# A tuner error is a diagnostic pointer, not the diagnosis: the full text lives
# in the run's own result.json and tune.log. 400 characters is enough to carry
# the argparse line or the aiter marker that says which of the two it was.
_TRACE_TUNER_ERROR_MAXLEN = 400

# Emitted even when null. ``kept`` is null on every row observed so far, and an
# absent key would be indistinguishable from ``false``.
_TRACE_TUNER_ALWAYS_KEYS = ("tuner", "best_micro_speedup", "kept")


def _trace_tuner_row(tuner: dict[str, Any]) -> dict[str, Any]:
    """One per-tuner entry for the audit row, keeping why it ended as it did.

    The row used to carry only ``tuner``/``best_micro_speedup``/``kept``, which
    cannot separate a tuner that crashed from one that ran and found nothing --
    the single question the audit trail exists to answer. Across one campaign 38
    of 337 tuner runs ended ``failed`` or ``empty_output`` and the trace showed
    none of them; one of those was 82 runs rejected by argparse in 11 seconds
    and recorded as a clean ``no_improvement`` (#1211), which stayed invisible
    for three weeks because this row had nowhere to put it.
    """
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
    # A clean run stays as compact as before: everything added here is dropped
    # when it is null, so a successful row gains only status and elapsed_s.
    return {k: v for k, v in row.items() if k in _TRACE_TUNER_ALWAYS_KEYS or v is not None}


def _trace_gemm_tuning_run(result: Any, *, session_dir: Path) -> None:
    """Append one ``gemm_tuning.jsonl`` audit row for a GEMM-tuning run.

    Distils the run result into a compact source-attribution row (engine,
    decision, speedup, per-tuner summary) appended to
    ``reports/trace/gemm_tuning.jsonl``. Best-effort; any failure is swallowed.

    Args:
        result: The GEMM-tuning handler result envelope.
        session_dir: Session directory the audit row is appended under.
    """
    if not isinstance(result, dict):
        return
    from datetime import datetime, timezone

    from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path

    engine = str(result.get("engine") or result.get("backend") or "").strip().lower() or "unknown"
    tuners: list[dict[str, Any]] = [
        _trace_tuner_row(t) for t in (result.get("tuners_run") or []) if isinstance(t, dict)
    ]
    # The envelope reported no error class even when a tuner had named one, so a
    # crashed run and a barren one looked alike at the top level too. Take the
    # first one a tuner supplied rather than leaving the field null.
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
    analysis_route: str,
) -> "tuple[list[str], str]":
    """Assemble the trace-analysis tool argv (TraceLens or bypass); returns
    ``(cmd, steady_state_mode)`` so the caller can record discovery provenance."""
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
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if framework:
        cmd += ["--framework", str(framework)]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if analysis_mode:
        cmd += ["--analysis-mode", str(analysis_mode)]

    # Model identity informs source resolution for every framework, not only the
    # diffusion roofline. Keep the standard payload > state > environment
    # precedence so ordinary sglang/vLLM production requests carry config.json
    # selectors into the bounded model context.
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
        # Priority: payload override > baseline workload metadata.
        num_denoise = payload.get("num_denoise_steps") or workload.get("num_inference_steps")
        if num_denoise not in (None, ""):
            try:
                if int(num_denoise) > 0:
                    cmd += ["--num-denoise-steps", str(int(num_denoise))]
            except (TypeError, ValueError):
                pass
    else:
        # Splitter workload hints. Priority: payload override > baseline metadata
        # > drop the flag.
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
    # Forward the analysis route (bypass takes no such flag).
    if analysis_route in ("deterministic", "agent"):
        cmd += ["--analysis-route", analysis_route]
    # Post-kernel-opt roofline writes a separate report so it never overwrites
    # the baseline kernel_roofline.json.
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    if roofline_output_name:
        cmd += ["--roofline-output-name", roofline_output_name]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    return cmd, steady_state_mode


# TraceLens picks its steady-state window by writing split chunks and selecting
# one file; the TraceLens-free reader picks a window in memory and never writes
# chunks. Both answer "is the window this analysis rests on trustworthy", so the
# event normalizes them onto one shape and keeps the raw form under ``selected``.
_STEADY_SOURCE_SPLIT_CHUNK = "split_chunk"
_STEADY_SOURCE_READER_WINDOW = "in_reader_window"


def _analysis_steady_state(
    result: dict[str, Any],
    *,
    requested_mode: str,
    tool: str,
) -> dict[str, Any]:
    """Normalize the steady-state window across analysis tools.

    Args:
        result: The analysis tool's result dict.
        requested_mode: The steady-state mode asked of the tool.
        tool: ``tracelens`` or ``bypass``.

    Returns:
        A dict naming the requested mode, how the window was picked, the raw
        selection, whether the tool fell back to the full trace, and the
        aggregation scope the shares are anchored to.
    """
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
    """Assemble the per-run analysis metadata the roofline timeline event carries.

    The three execution paths (LLM TraceLens, no-LLM TraceLens, TraceLens-free
    reader) share this envelope. ``route`` and ``tool`` are both kept because
    they are independent: the no-LLM TraceLens route reports ``deterministic`` /
    ``tracelens``, while the reader reports ``bypass`` / ``bypass``, and
    collapsing them would erase the distinction between "TraceLens ran without
    an LLM" and "TraceLens never ran". Tool-specific analysis output lands under
    ``route_ext`` rather than widening the shared envelope.

    Args:
        result: The analysis tool's result dict.
        route: The requested analysis route (``agent`` / ``deterministic`` / ``bypass``).
        tool: The tool that actually ran (``tracelens`` / ``bypass``).
        requested_mode: The steady-state mode asked of the tool.
        trace_input: The trace the run analyzed.
        duration_sec: Wall-clock seconds the subprocess took.

    Returns:
        The analysis metadata dict.
    """
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
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir.

    The explicit payload framework normally takes precedence over the persisted
    session value.  A scriptable session overrides a conflicting non-scriptable
    payload framework so a diffusion trace is not sent through the LLM
    prefill/decode splitter.

    Args:
        payload (dict): Request payload (see ``Required payload`` /
            ``Optional payload`` below for the recognized keys).
        session_dir (Path): Session root used for resolving inputs and writing
            the analysis outputs.

    Required payload:
        trace_input: path to a torch_trace dir or single .trace.json.gz file.

    Returns the tool's result dict with ``status``, surfaced artifact paths, and
    ``trace_health_warnings``; on failure, ``returncode`` / ``error`` and empty ``hot_kernels``.
    """
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
    # The exception is a scriptable session receiving a stale non-scriptable
    # default (commonly ``sglang``): that would make xDiT follow the LLM trace
    # splitter, which discards its raw diffusion GPU kernels.
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

    # Analysis route: default ``agent`` (TraceLens); ``bypass`` (TraceLens-free)
    # and ``deterministic`` (no-LLM TraceLens) are explicit routes via payload
    # ``analysis_route`` / ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``. Coerce to str.
    explicit_route = (
        str(payload.get("analysis_route") or os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "")).strip().lower()
    )
    # Reject an unknown route: warn and fall back to the default ``agent`` route.
    route_health_warnings: list[dict[str, Any]] = []
    if explicit_route and explicit_route not in _VALID_ANALYSIS_ROUTES:
        log.warning(
            "trace_analyze: unknown analysis_route %r (expected one of %s); falling back to the default 'agent' route",
            explicit_route,
            sorted(_VALID_ANALYSIS_ROUTES),
        )
        route_health_warnings.append(
            {
                "code": "invalid_analysis_route",
                "severity": "warning",
                "message": (
                    f"unknown analysis_route {explicit_route!r} (expected one of "
                    f"{sorted(_VALID_ANALYSIS_ROUTES)}); fell back to the default 'agent' route."
                ),
                "requested_route": explicit_route,
            }
        )
        explicit_route = ""
    analysis_route = explicit_route or "agent"
    is_bypass = analysis_route == "bypass"
    # Resolve TraceLens root independently of inherited env, self-healing a
    # vanished checkout before validation. Skipped on bypass.
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

    # Scriptable frameworks (xDiT) have no decode steady-state window, so feed the
    # raw trace and drop the --split-* hints.
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
        analysis_route=analysis_route,
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
        result["trace_health_warnings"] = (
            framework_warnings + route_health_warnings + list(result.get("trace_health_warnings") or [])
        )

        _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
        candidates_path = result.get("candidates_path")
        if isinstance(candidates_path, str):
            _enrich_candidates_artifact(
                candidates_path,
                metadata,
                trace_report_path=str(report_path or ""),
            )

        # Discovery source = the route that ran; deterministic maps to
        # ``bypass``, the TraceLens LLM route to ``tracelens``.
        _orch_mode = str(result.get("orchestrator_mode") or "").strip().lower()
        _independent_bypass = _orch_mode == "bypass" or is_bypass
        _is_bypass = _independent_bypass or _orch_mode == "deterministic" or analysis_route == "deterministic"
        _disc_source = "bypass" if _is_bypass else "tracelens"
        _disc_tool = "bypass" if _independent_bypass else "tracelens"
        # Effective route: the reader is only reached via the bypass route, and a
        # TraceLens run without an LLM stays ``deterministic``.
        _disc_route = "bypass" if _independent_bypass else analysis_route
        # Surfaced for the caller's SBD V6 roofline event, which records the run
        # as it happens rather than re-deriving it at export time.
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
                source=_disc_source,
                tool=_disc_tool,
                status=str(result.get("status") or ""),
                hot_kernels=_hot if isinstance(_hot, list) else [],
                scan={
                    "splitter_mode": steady_state_mode,
                    "trace_dir": str(trace_input),
                    "candidates_path": str(result.get("candidates_path") or ""),
                    "trace_report_path": str(result.get("trace_report_path") or ""),
                    "analysis_route": _disc_source,
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


def _exists_with_retry(
    path: str | Path,
    *,
    attempts: int = 5,
    delay_sec: float = 0.5,
) -> bool:
    """Check ``path`` existence, retrying briefly to absorb storage latency.

    On shared/network filesystems a just-written file can take a moment to become
    visible, so retry a few times with a short pause before giving up.

    Args:
        path: Filesystem path to check.
        attempts: Total number of existence checks to perform (>= 1).
        delay_sec: Seconds to sleep between checks.

    Returns:
        ``True`` as soon as the path is visible, else ``False`` after all
        attempts are exhausted.
    """
    target = Path(path)
    for attempt in range(max(1, attempts)):
        if target.exists():
            return True
        if attempt < attempts - 1:
            time.sleep(delay_sec)
    return False


def _validate_trace_analyze_inputs(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult | None:
    """Confirm the run_optimization payload references a valid trace_analyze.

    Args:
        payload: The run_optimization request payload.
        session_dir: Session directory to load SharedState from.

    Returns:
        A failure ``HandlerResult`` when no valid trace_analyze is referenced,
        else ``None``.
    """
    candidates_path = str(payload.get("candidates_path") or "").strip()
    if candidates_path and not _exists_with_retry(candidates_path):
        return {
            "status": "failed",
            "error_class": "missing_candidates_artifact",
            "error": (
                "run_optimization requires a candidates_path that exists on disk; re-run trace_analyze to regenerate it"
            ),
            "candidates_path": candidates_path,
        }
    if candidates_path:
        return None
    if payload.get("dry_run") or payload.get("source_file") or isinstance(payload.get("candidate"), dict):
        return None
    try:
        from ..state.shared_state import SharedState

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

    Args:
        payload: The run_optimization request payload.
        session_dir: Session directory for workspace and state.
        record_partial: Optional callback streaming each batch sub-result into
            SharedState before the gather wait-all returns.

    Returns:
        A ``HandlerResult`` describing the optimization outcome.
    """
    data_guard = _validate_trace_analyze_inputs(payload, session_dir=session_dir)
    if data_guard is not None:
        return data_guard
    if payload.get("_single_kernel"):
        return await _run_optimization_single(payload, session_dir=session_dir)
    dispatch_skips: dict[str, str] = {}
    candidates = _batch_kernel_candidates(
        payload,
        session_dir=session_dir,
        skipped_out=dispatch_skips,
    )
    if len(candidates) <= 1:
        single_payload = dict(payload)
        kernel_id_pinned = False
        if candidates:
            requested_kernel_id = single_payload.get("kernel_id")
            reconciled_id, kernel_id_pinned = _reconcile_kernel_id_for_single_batch(
                requested_kernel_id,
                candidates,
            )
            single_payload["kernel_id"] = reconciled_id
            # Preserve the selected candidate itself: it may carry a task_group
            # that does not exist on the raw hot-kernel row reloaded by the
            # kernel_optimization subprocess.
            single_payload["candidate"] = candidates[0]
            # Assigned, not defaulted. Reconciliation can name a different
            # kernel than the payload asked for, and this is the same object's
            # path: a ``setdefault`` kept the requested kernel's source beside
            # the selected kernel's candidate, so the backend rewrote one
            # kernel's file while the ledger charged the attempt to the other.
            selected_source = str(candidates[0].get("source_file") or "")
            if selected_source:
                single_payload["source_file"] = selected_source
            else:
                single_payload.setdefault("source_file", "")
        else:
            # No routable candidate: canonicalize an aliased id against the full set.
            all_candidates = _all_kernel_candidates(payload)
            canon = _resolve_candidate_id(
                single_payload.get("kernel_id"),
                all_candidates,
            )
            if canon:
                single_payload["kernel_id"] = canon
                # The filter dropped this kernel for a reason it already knows.
                # When that reason means "never dispatched", say so instead of
                # falling through to the validation guards: a failure recorded
                # here spends the source's retry quota on a decision no backend
                # made, and the report then explains a technical failure that
                # never happened.
                skip_reason = dispatch_skips.get(canon, "")
                if unattempted_skip_reason(skip_reason):
                    return {
                        "status": "skipped",
                        "reason": skip_reason,
                        "kernel_id": canon,
                        "kernels_considered": len(all_candidates),
                        "message": (f"kernel {canon} was not dispatched: {skip_reason}"),
                    }
                # Otherwise the guards below decide, and they need the candidate
                # to report against. Without it the attempt ledger files this
                # kernel under an empty source and splits its identity from the
                # one a later dispatch would use.
                named = next(
                    (
                        row
                        for row in all_candidates
                        if isinstance(row, dict) and str(row.get("kernel_id") or "") == canon
                    ),
                    None,
                )
                if named is not None:
                    single_payload.setdefault("candidate", named)
                    if named.get("source_file"):
                        single_payload.setdefault("source_file", named["source_file"])
            elif not _names_specific_kernel(single_payload):
                # Empty eligible queue and no specific target (e.g. the post-GEMM
                # auto pass): finish cleanly as "skipped", not a failure.
                return {
                    "status": "skipped",
                    "reason": "no_eligible_kernels",
                    "kernels_considered": len(_all_kernel_candidates(payload)),
                    "message": (
                        "no eligible kernels to optimize (all candidates already "
                        "tried/rejected, below the size cutoff, or not reusable)"
                    ),
                }
        single_payload["_single_kernel"] = True
        result = await _run_optimization_single(
            single_payload,
            session_dir=session_dir,
        )
        if kernel_id_pinned and isinstance(result, dict):
            result["kernel_id_pinned"] = True
            if requested_kernel_id is not None:
                result["requested_kernel_id"] = str(requested_kernel_id)
        return _stamp_task_group_result(
            result,
            candidates[0] if candidates else None,
            fallback_kernel_id=str(single_payload.get("kernel_id") or ""),
        )
    return await _run_optimization_batch(
        payload,
        candidates,
        session_dir=session_dir,
        record_partial=record_partial,
    )


def _optimization_budget_minutes(payload: dict) -> float:
    """Wall-clock budget mirrored by the kernel_optimization.py wrapper.

    Priority: env ``KERNEL_OPT_BACKEND_BUDGET_MIN`` > payload ``budget_minutes``
    > :data:`_DEFAULT_BACKEND_BUDGET_MINUTES`. The env wins because the payload
    value is LLM-authored from a prompt template, so an operator raising the
    budget must not be silently overridden by it. The rewrite-route floor is
    applied in all cases: an operator who tunes the budget down for an unrelated
    reason still cannot silently disable the route they opted into.

    Args:
        payload (dict): Request payload carrying an optional ``budget_minutes``.

    Returns:
        float: The wall-clock budget in minutes for this optimization.
    """
    floor = _rewrite_route_budget_floor_minutes()
    raw = os.environ.get("KERNEL_OPT_BACKEND_BUDGET_MIN", "").strip()
    if raw:
        try:
            forced = float(raw)
        except ValueError:
            forced = 0.0
        if forced > 0:
            return max(forced, floor)
    budget = float(payload.get("budget_minutes", _DEFAULT_BACKEND_BUDGET_MINUTES))
    return max(budget, floor)


def _rewrite_route_budget_floor_minutes() -> float:
    """Minutes the FlyDSL rewrite route needs, or 0 when it is not opted into.

    Below its own minimum the route declines every candidate as
    ``budget_insufficient`` and falls back to forge-loop without saying so, so a
    budget tuned down for an unrelated reason would switch the route off
    silently. Raising the floor only for a run that opted in leaves every other
    run's budget untouched.

    Returns:
        float: The floor in minutes, or ``0.0`` when the route is disabled.
    """
    from hyperloom.agents.kernel.tools.backends._flydsl_rewrite import (
        MIN_BUDGET_SEC,
        rewrite_enabled,
    )

    return MIN_BUDGET_SEC / 60.0 if rewrite_enabled() else 0.0


def _optimization_wrapper_timeout_sec(payload: dict) -> int:
    """Compute the subprocess timeout for the kernel_optimization.py wrapper.

    Converts the optimization budget to seconds and adds a 180s grace window
    so the wrapper can salvage partial artifacts before being killed.

    Args:
        payload (dict): Request payload used to derive the optimization budget.

    Returns:
        int: The subprocess timeout in seconds.
    """
    return int(_optimization_budget_minutes(payload) * 60) + 180


def _raw_kernel_backend_order(payload: dict | None = None) -> list[str]:
    """Return the effective kernel backend order.

    Forge is deliberately not request-selectable.  The only supported forge
    opt-in is exactly ``KERNEL_OPT_BACKEND_ORDER=forge``; every other value,
    missing value, legacy alias, or payload override stays on the GEAK
    whole-phase backend.
    """
    if forge_explicitly_enabled():
        return ["forge"]
    return list(_DEFAULT_KERNEL_PHASE_BACKEND_ORDER)


def geak_selected(payload: dict | None = None) -> bool:
    """Whether ``geak`` (the whole-pipeline e2e delegate) is in the kernel backend order.

    ``geak`` is not a per-kernel backend: when it appears in the order it
    means "delegate the whole KERNEL_AGENT phase to the GEAK e2e optimizer".
    It therefore *owns* the phase whenever present (any other backends in the
    order are ignored for the kernel phase), so an order of just ``geak``
    runs only the GEAK e2e optimizer. ``forge`` is the per-kernel backend.

    Args:
        payload: Optional request payload that may carry ``backend_order``.

    Returns:
        bool: ``True`` when ``geak`` is in the resolved order.
    """
    return "geak" in _raw_kernel_backend_order(payload)


def _kernel_ladder_budget_sec(payload: dict) -> int:
    """Total wall-clock budget for one kernel's whole backend ladder.

    Bounds the whole ladder so a fallback only runs within the time left and an
    exhausted budget exits cleanly, keeping the ladder from overshooting the
    KERNEL-phase budget cap.

    Priority: payload ``kernel_budget_min`` > env
    ``KERNEL_OPT_KERNEL_BUDGET_MIN`` > the single-backend budget from
    :func:`_optimization_budget_minutes`. A +180s grace mirrors the per-subprocess
    wrapper so the first backend is never capped below its own timeout.

    Args:
        payload (dict): Request payload carrying optional ``kernel_budget_min``.

    Returns:
        int: The per-kernel ladder budget in seconds.
    """
    minutes = (
        payload.get("kernel_budget_min")
        or os.environ.get("KERNEL_OPT_KERNEL_BUDGET_MIN")
        or _optimization_budget_minutes(payload)
    )
    return int(float(minutes) * 60) + 180


def _backend_order(payload: dict) -> list[str]:
    """Resolve the per-kernel backend ladder.

    The per-kernel ladder is disabled unless forge is explicitly opted in via
    ``KERNEL_OPT_BACKEND_ORDER=forge``.  GEAK owns the default whole KERNEL
    phase, so request payloads and legacy aliases cannot select forge.
    """
    order = _raw_kernel_backend_order(payload)
    return ["forge"] if order == ["forge"] else []


def _in_flight_kernel_ids(session_dir: Path) -> set[str]:
    """Scan the kernel-agent run dir for ``state=running`` status files, so :func:`_batch_kernel_candidates` skips kernels still in flight from a prior batch.

    Args:
        session_dir: Session directory whose kernel-agent run dir is scanned.

    Returns:
        The set of kernel ids currently in flight.
    """
    from hyperloom.inference_optimizer.session.session_paths import kernel_agent_runs_dir

    in_flight: set[str] = set()
    sid = session_dir.name
    status_dir = kernel_agent_runs_dir(session_dir, sid) / "status" / "kernel_optimization"
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
    """Fold hallucinated ``kn``/``rn`` prefixes onto the real ``k`` numbering (mirrors ``kernel_optimization._normalize_kernel_id``).

    Args:
        value: The raw kernel id to normalize.

    Returns:
        The normalized kernel id.
    """
    s = str(value or "").strip().lower()
    for prefix in ("kn", "rn"):
        if s.startswith(prefix) and s[len(prefix) :].isdigit():
            return "k" + s[len(prefix) :]
    return s


def _reconcile_kernel_id(
    requested: Any,
    candidates: list[dict[str, Any]],
) -> str:
    """Resolve the LLM kernel_id to a real candidate id (exact kernel_id/name, then normalized; only a missing id falls back to the first candidate).

    Args:
        requested: The (possibly hallucinated) kernel id from the LLM.
        candidates: The real candidate dicts to reconcile against.

    Returns:
        The reconciled candidate id (the first candidate's id when
        ``requested`` is empty; ``requested`` unchanged when no match).
    """
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
            req,
            [str(c.get("kernel_id") or "") for c in candidates],
        )
        return req
    fallback = str(candidates[0].get("kernel_id") or "")
    return fallback


def _reconcile_kernel_id_for_single_batch(
    requested: Any,
    candidates: list[dict[str, Any]],
) -> tuple[str, bool]:
    """Reconcile ``requested`` and pin to the sole batch row when ids diverge.

    When the batch filter leaves exactly one candidate, the LLM may still
    request a different ``kernel_id``; keep candidate metadata and
    ``kernel_id`` aligned so predispatch validation uses the same row.

    Returns:
        ``(kernel_id, kernel_id_pinned)`` where ``kernel_id_pinned`` is True
        when the requested id was overridden to match the sole batch row.
    """
    if not candidates:
        return str(requested or ""), False
    reconciled = _reconcile_kernel_id(requested, candidates)
    valid = {str(c.get("kernel_id") or "") for c in candidates if c.get("kernel_id")}
    if reconciled in valid:
        return reconciled, False
    pinned = str(candidates[0].get("kernel_id") or "")
    if pinned and reconciled != pinned:
        log.warning(
            "kernel_id %r pinned to sole batch candidate %r",
            reconciled,
            pinned,
        )
        return pinned, True
    return pinned or reconciled, False


def _resolve_candidate_id(
    requested: Any,
    candidates: list[dict[str, Any]],
) -> str:
    """Return the canonical ``k00x`` id for ``requested`` or ``""`` (like ``find_candidate`` but with no first-candidate fallback; a pure hallucination returns ``""``).

    Args:
        requested: The (possibly hallucinated) kernel id to canonicalize.
        candidates: The real candidate dicts to match against.

    Returns:
        The canonical candidate id, or ``""`` when no match is found.
    """
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


def _names_specific_kernel(payload: dict) -> bool:
    """Return ``True`` when the payload targets one specific kernel/source.

    A specific target is an explicit ``kernel_id``, a ``source_file`` to
    optimize, or an inline ``candidate`` dict. The post-GEMM auto pass dispatches
    a batch with none of these, which is the empty-work-queue case that should be
    skipped cleanly rather than routed into the single-kernel path.

    Args:
        payload: The run_optimization request payload.

    Returns:
        ``True`` if the request names a specific kernel/source, else ``False``.
    """
    if str(payload.get("kernel_id") or "").strip():
        return True
    if str(payload.get("source_file") or "").strip():
        return True
    if isinstance(payload.get("candidate"), dict):
        return True
    return False


def _all_kernel_candidates(payload: dict) -> list[dict[str, Any]]:
    """Load every unique candidate (``hot_kernels`` ∪ ``skipped_kernels``) so id canonicalization resolves even when hot_kernels is empty.

    Under the P0 contract ``hot_kernels`` is the FULL ranked hotspot set and
    ``skipped_kernels`` is its non-routable subset, so the two on-disk lists
    OVERLAP. Candidates are therefore de-duplicated by kernel identity
    (``kernel_id`` then ``name``), keeping the first (``hot_kernels``) copy, so
    ``kernels_considered`` counts each hotspot once instead of double-counting
    every non-routable kernel. Rows carrying neither id nor name cannot be
    identified and are always kept (never silently dropped).

    Args:
        payload: Request payload carrying ``candidates_path``.

    Returns:
        Every unique candidate dict from the artifact, or an empty list when the
        artifact is missing or unreadable.
    """
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
    seen: set[str] = set()
    for key in ("hot_kernels", "kernel_candidates", "skipped_kernels"):
        value = data.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            ident = str(item.get("kernel_id") or item.get("name") or "")
            if ident:
                if ident in seen:
                    continue
                seen.add(ident)
            out.append(item)
    return out


# Default: one backend-ladder dispatch per kernel/source unless an infra
# failure still has retry budget (see ``_kernel_dispatch_attempt_cap``).
_DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS = 1


def _kernel_dispatch_attempt_cap(entry: dict[str, Any], *, max_failures: int) -> int:
    """Return the batch-eligibility attempt cap for one kernel attempt record.

    Non-infra attempts (PARTIAL, legacy resume rows, etc.) keep the
    single-dispatch rule. Only a retryable backend infra failure widens the
    cap to ``max_failures`` so dispatch, ``record_kernel_opt``, and
    ``kernel_work_pending`` agree on the same budget.
    """
    if not isinstance(entry, dict):
        return _DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS
    try:
        failure_count = int(entry.get("failure_count") or 0)
    except (TypeError, ValueError):
        failure_count = 0
    if failure_count <= 0:
        return _DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS
    last_decision = str(entry.get("last_decision") or "").strip()
    last_status = str(entry.get("last_status") or "").lower()
    rejected_reason = str(entry.get("rejected_reason") or "").strip()
    is_retryable_infra = last_decision == "" and last_status in {"failed", "error", "timeout"} and not rejected_reason
    if failure_count < max_failures and is_retryable_infra:
        return max_failures
    # High-impact infra-retry (HL_HONEST_E2E umbrella, default ON; opt out with
    # HL_HONEST_E2E=0 or HL_INFRA_RETRY_HIGH_IMPACT=0): a high-GPU%-share kernel
    # that keeps infra-failing gets extra attempts before retirement.
    if is_retryable_infra and _honest_flag("HL_INFRA_RETRY_HIGH_IMPACT"):
        try:
            impact_pct = float(entry.get("last_gpu_pct") or 0.0)
        except (TypeError, ValueError):
            impact_pct = 0.0
        try:
            min_gpu = float(os.environ.get("HL_INFRA_RETRY_MIN_GPU_PCT", "5.0") or 5.0)
        except ValueError:
            min_gpu = 5.0
        try:
            infra_max = int(os.environ.get("HL_INFRA_RETRY_MAX", "4") or 4)
        except ValueError:
            infra_max = 4
        if impact_pct >= min_gpu and failure_count < infra_max:
            return infra_max
    return _DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS


def _batch_kernel_candidates(
    payload: dict,
    *,
    session_dir: Path | None = None,
    skipped_out: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Select the reusable native kernels to dispatch for a batch run.

    Reads the ``candidates_path`` artifact and builds the dispatch list,
    collapsing kernels that share a source function into single ``task_group``
    dispatches and falling back to a legacy per-kernel pass for ungrouped
    kernels. Applies the "live" filters (not rejected, not in-flight, under the
    per-source attempt cap) and the minimum GPU-percentage gate. When
    ``session_dir`` is omitted, the SharedState-derived filters degrade to
    empty sets.

    Args:
        payload (dict): Request payload carrying ``candidates_path``.
        session_dir (Path | None): Session directory used to load SharedState
            for rejection / attempt / in-flight filters; optional for legacy
            and dry-run paths.

    Returns:
        list[dict[str, Any]]: The selected candidate dicts (each a shallow copy
            carrying its ``task_group`` when grouped), or an empty list when
            the artifact is missing/unreadable or nothing is eligible.
    """

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
    # Drop geometry-only kernels (bypass path tags shape_dispatchable=False) up
    # front so both the grouped and legacy passes agree: they resolve a source
    # yet fail the kernel-opt gate on untrusted shape provenance. Absent field
    # (TraceLens path) stays dispatchable to avoid regressing it.
    kernels = [
        k
        for k in kernels
        if not (isinstance(k, dict) and (k.get("shape_dispatchable") is False or is_collective_candidate(k)))
    ]
    reusable_ids = data.get("reusable_native_kernel_ids") or []
    reusable_id_set = {str(item) for item in reusable_ids if item}

    # Build the "live" exclusion sets up front (empty without session_dir).
    rejected_kernel_ids: set[str] = set()
    attempts_by_task: dict[str, dict] = {}
    attempts_by_kid: dict[str, dict] = {}
    in_flight: set[str] = set()
    from ..state.shared_state import (
        resolve_hot_kernel_min_gpu_pct,
        resolve_kernel_opt_max_failures,
    )

    max_failures = resolve_kernel_opt_max_failures()
    min_gpu_pct = resolve_hot_kernel_min_gpu_pct()
    if session_dir is not None:
        try:
            from ..state.shared_state import SharedState

            state = SharedState.load_or_init(session_dir)
            rejected_kernel_ids = set(state.rejected_kernel_ids or [])
            attempts_by_task = dict(state.kernel_opt_task_attempts or {})
            attempts_by_kid = index_attempts_by_kernel_id(attempts_by_task)
            in_flight = _in_flight_kernel_ids(session_dir)
        except Exception:
            log.exception(
                "_batch_kernel_candidates: failed to load SharedState from %s; PR-C filters disabled this dispatch",
                session_dir,
            )

    def _entry_allows_dispatch(
        entry: dict[str, Any],
        current_source: str,
    ) -> bool:
        """Apply retry and terminal-state rules to one persisted task ledger."""
        if str(entry.get("last_decision") or "").upper() in {"KEEP", "REVERT"}:
            return False
        attempt_cap = _kernel_dispatch_attempt_cap(entry, max_failures=max_failures)
        if current_source:
            per_source = entry.get("attempts_per_source")
            if isinstance(per_source, dict):
                src_attempts = int(per_source.get(current_source, 0))
                return src_attempts < attempt_cap
        return int(entry.get("attempts", 0)) < attempt_cap

    def _liveness_reason(
        kid: str,
        current_source: str = "",
        current_task_group_key: str = "",
    ) -> str:
        """Return ``""`` when the kernel is batch-eligible, else why it is not.

        The distinction the bare boolean could not draw: a kernel held back
        because a sibling dispatch is in flight has had no backend look at it,
        while one held back for a rejection or an exhausted attempt cap has. A
        single ``not_live`` reason conflated the two, so the "a gate rejection is
        not a failed attempt" exemption could not cover the first without also
        covering the last two -- and covering those would retire a kernel that
        really did spend its attempts.

        Two task-group escapes relax the rejection check: a ledger entry
        recorded under a *different* ``task_group_key`` leaves the kernel live
        even when rejected, and a rejected kernel with no recorded group key is
        still live when a ``current_task_group_key`` is supplied.

        Args:
            kid: The kernel id to test.
            current_source: The current source file for per-source counting.
            current_task_group_key: The task-group key of the pending dispatch;
                empty when the caller has no group identity.

        Returns:
            ``""``, ``"not_live_in_flight"``, ``"not_live_rejected"`` or
            ``"not_live_attempts_exhausted"``.
        """
        if kid in in_flight:
            return "not_live_in_flight"
        entry = attempts_by_kid.get(kid) or {}
        recorded_group_key = str(entry.get("task_group_key") or "")
        if current_task_group_key and recorded_group_key and recorded_group_key != current_task_group_key:
            return ""
        recorded_source = str(entry.get("last_source_file") or "")
        same_source = not current_source or not recorded_source or recorded_source == current_source
        if kid in rejected_kernel_ids and same_source and not (current_task_group_key and not recorded_group_key):
            return "not_live_rejected"
        if not _entry_allows_dispatch(entry, current_source):
            return "not_live_attempts_exhausted"
        return ""

    def _is_live(
        kid: str,
        current_source: str = "",
        current_task_group_key: str = "",
    ) -> bool:
        """Whether ``kid`` is batch-eligible; see :func:`_liveness_reason`."""
        return not _liveness_reason(kid, current_source, current_task_group_key)

    # Collapse kernels sharing a source function into one dispatch via
    # ``task_groups[]``; ungrouped kernels fall through below.
    task_groups = data.get("task_groups") or []
    if not isinstance(task_groups, list):
        task_groups = []
    kernel_by_id: dict[str, dict[str, Any]] = {
        str(k.get("kernel_id") or ""): k for k in kernels if isinstance(k, dict) and k.get("kernel_id")
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
        group_id = str(group.get("task_group_id") or "")
        group_key = str(group.get("task_group_key") or "")
        group_key_aliases = {
            group_key,
            *[str(alias) for alias in (group.get("legacy_task_group_keys") or []) if str(alias)],
        }
        group_key_aliases.discard("")
        primary = str(group.get("primary_kernel_id") or "")
        if any(member_id in in_flight for member_id in member_ids):
            for member_id in member_ids:
                skipped.setdefault(member_id, "group_in_flight")
            continue

        # Once the merged task has a ledger entry, retry or retire that task as
        # one unit. Never rotate to an untouched sibling shape.
        recorded_ledger = next(
            (
                (ledger_id, entry)
                for ledger_id, entry in attempts_by_task.items()
                if isinstance(entry, dict)
                and (
                    (
                        group_key
                        and (
                            str(entry.get("stable_task_key") or "") == group_key
                            or str(entry.get("task_group_key") or "") == group_key
                            or str(entry.get("stable_task_key") or "") in group_key_aliases
                            or str(entry.get("task_group_key") or "") in group_key_aliases
                        )
                    )
                    or (
                        not group_key
                        and group_id
                        and str(entry.get("task_group_id") or "") == group_id
                        and str(entry.get("current_kernel_id") or "") in member_ids
                    )
                )
            ),
            None,
        )
        if recorded_ledger is not None:
            _recorded_member_id, recorded_entry = recorded_ledger
            recorded_candidate = kernel_by_id.get(primary) or next(
                (
                    kernel_by_id[member_id]
                    for member_id in member_ids
                    if member_id in kernel_by_id
                    and kernel_by_id[member_id].get("reusable_native_kernel") is True
                    and kernel_by_id[member_id].get("source_file")
                ),
                None,
            )
            recorded_live = (
                recorded_candidate is not None
                and recorded_candidate.get("reusable_native_kernel") is True
                and bool(recorded_candidate.get("source_file"))
                and _entry_allows_dispatch(
                    recorded_entry,
                    str(recorded_candidate.get("source_file") or ""),
                )
            )
            if not recorded_live:
                for member_id in member_ids:
                    skipped.setdefault(member_id, "group_task_complete")
                continue
            primary_cand = recorded_candidate
        else:
            primary_cand = None

        # Only reusable_native members survive; fall back to the next live one
        # when the primary is rejected, else skip the group.
        if primary_cand is None:
            primary_cand = kernel_by_id.get(primary)
            primary_live = (
                primary_cand is not None
                and primary_cand.get("reusable_native_kernel") is True
                and bool(primary_cand.get("source_file"))
                and _is_live(
                    primary,
                    str(primary_cand.get("source_file") or ""),
                    group_key,
                )
            )
            if not primary_live:
                primary_cand = next(
                    (
                        kernel_by_id[m]
                        for m in member_ids
                        if m in kernel_by_id
                        and kernel_by_id[m].get("reusable_native_kernel") is True
                        and kernel_by_id[m].get("source_file")
                        and _is_live(
                            m,
                            str(kernel_by_id[m].get("source_file") or ""),
                            group_key,
                        )
                    ),
                    None,
                )
                if primary_cand is None:
                    # Every reusable member exhausted -> nothing to dispatch.
                    for m in member_ids:
                        skipped.setdefault(m, "group_exhausted")
                    continue
        if not primary_cand.get("source_file"):
            continue
        # Vendor-playbook groups (mori's dispatch+combine) are gated on the
        # sum of the group's members, not the picked member's own share, and
        # may pin a per-playbook floor -- see
        # effective_hot_kernel_gpu_pct's docstring. No-op for ordinary
        # task_group members, which carry neither field.
        gate_floor = effective_hot_kernel_min_gpu_pct(primary_cand, min_gpu_pct)
        if effective_hot_kernel_gpu_pct(primary_cand) < gate_floor:
            for m in member_ids:
                skipped.setdefault(m, f"below_min_gpu_pct={gate_floor}")
            continue
        # Shallow copy + attach group so the subprocess sees the task_group.
        item = dict(primary_cand)
        item["task_group"] = group
        selected.append(item)

    # Legacy per-kernel pass for ungrouped reusable kernels. Collect eligible rows
    # first; the min_gpu_pct gate is applied below so op-fanout de-dup can sum
    # sibling GPU% before gating.
    legacy_eligible: list[tuple[str, dict[str, Any], float]] = []
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
        liveness = _liveness_reason(kernel_id, str(item.get("source_file") or ""))
        if liveness:
            skipped[kernel_id] = liveness
            continue
        try:
            row_pct = float(item.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            row_pct = 0.0
        legacy_eligible.append((kernel_id, item, row_pct))

    if _honest_flag("HL_KERNEL_OPFANOUT_DEDUP"):
        # Op-fanout de-dup (flag-gated): collapse same-source rows into the
        # highest-GPU% representative carrying the siblings' summed GPU%.
        by_source: dict[str, list[tuple[str, dict[str, Any], float]]] = {}
        order: list[str] = []
        for kid, item, row_pct in legacy_eligible:
            src = str(item.get("source_file") or "")
            if src not in by_source:
                by_source[src] = []
                order.append(src)
            by_source[src].append((kid, item, row_pct))
        deduped: list[tuple[str, dict[str, Any], float]] = []
        for src in order:
            rows = by_source[src]
            summed_pct = sum(p for _, _, p in rows)
            rep_kid, rep_item, _ = max(rows, key=lambda r: r[2])
            if len(rows) > 1:
                rep_item = dict(rep_item)
                rep_item["gpu_pct"] = summed_pct
                rep_item["opfanout_collapsed_ids"] = [k for k, _, _ in rows]
                for k, _, _ in rows:
                    if k != rep_kid:
                        skipped.setdefault(k, f"opfanout_merged_into={rep_kid}")
            deduped.append((rep_kid, rep_item, summed_pct))
        legacy_eligible = deduped

    for kernel_id, item, row_pct in legacy_eligible:
        # See the task_group gate above: prefer a vendor-playbook group's
        # aggregate share and floor over the row's own gpu_pct when stamped.
        gate_floor = effective_hot_kernel_min_gpu_pct(item, min_gpu_pct)
        if max(row_pct, effective_hot_kernel_gpu_pct(item)) < gate_floor:
            skipped[kernel_id] = f"below_min_gpu_pct={gate_floor}"
            continue
        selected.append(item)

    if skipped:
        log.info(
            "batch candidates filtered: %d selected, skipped=%s",
            len(selected),
            skipped,
        )
    if skipped_out is not None:
        skipped_out.update(skipped)
    return selected


def _kernel_result_rank(result: HandlerResult | None) -> tuple[int, float]:
    """Best-selection key shared by the ladder and the batch handler.

    A KEEP verdict always beats a non-KEEP regardless of micro_speedup
    (GEAK frequently reports a higher micro on a NEEDS_REVIEW that has no
    correctness gate, while a KEEP at a lower micro is a real
    integrate-ready patch); among equals, higher ``micro_speedup`` wins.
    Mirrors the max-key in :func:`_run_optimization_batch` so the ladder,
    the backend ladder and batch mode agree on "best".

    Args:
        result: A kernel-opt attempt result, or ``None``.

    Returns:
        A ``(keep, micro_speedup)`` sort key; KEEP verdicts rank above
        non-KEEP, and higher ``micro_speedup`` breaks ties.
    """
    if not isinstance(result, dict):
        return (0, 0.0)
    proposal = result.get("proposal") or {}
    verification = result.get("verification") or {}
    keep = 1 if (result.get("status") == "ok" and proposal.get("decision") == "KEEP") else 0
    micro = float(verification.get("micro_speedup") or 0.0)
    return (keep, micro)


async def _run_backend_ladder(
    base_payload: dict,
    candidate: dict[str, Any],
    kernel_id: str,
    backends: list[str],
    *,
    session_dir: Path,
    deadline: float | None = None,
) -> tuple[HandlerResult | None, list[dict[str, Any]]]:
    """Run ``backends`` as a sequential break-on-KEEP ladder.

    Returns ``(best, attempts)`` where ``best`` is the strongest result by
    :func:`_kernel_result_rank` and ``attempts`` is the ordered per-backend
    attempt log. Stops at the first KEEP so a clean KEEP short-circuits the
    ladder and later backends only fire when an earlier one misses a KEEP.

    When ``deadline`` (a :func:`time.monotonic` timestamp) is given, the ladder
    enforces the per-kernel budget: each backend's subprocess timeout is capped
    to the time left, and once less than :data:`_KERNEL_LADDER_MIN_BACKEND_SEC`
    remains the ladder stops rather than launching a fallback it cannot finish.
    This keeps a backend that hangs to its hard timeout from letting the
    fallback overshoot the budget.

    Args:
        base_payload: The base request payload shared by every backend.
        candidate: The kernel candidate to optimize.
        kernel_id: The kernel id being optimized.
        backends: The ordered backends to try.
        session_dir: Session directory for workspace and state.
        deadline: Optional ``time.monotonic`` deadline bounding the whole
            ladder; ``None`` disables the per-kernel budget cap.

    Returns:
        A tuple of ``(best, attempts)`` where ``best`` is the strongest
        result by ``_kernel_result_rank`` and ``attempts`` is the ordered
        per-backend attempt log.
    """
    attempts: list[dict[str, Any]] = []
    best: HandlerResult | None = None
    for idx, backend in enumerate(backends):
        timeout_override: int | None = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= _KERNEL_LADDER_MIN_BACKEND_SEC:
                # Not enough budget left for another backend.
                log.info(
                    "kernel %s: per-kernel ladder budget exhausted (%.0fs left); skipping remaining backends %s",
                    kernel_id,
                    remaining,
                    backends[idx:],
                )
                break
            timeout_override = int(remaining)
        child = dict(base_payload)
        child["_single_kernel"] = True
        child["kernel_id"] = kernel_id
        child["backends"] = backend
        child["candidate"] = candidate
        child.setdefault("source_file", candidate.get("source_file"))
        result = await _run_optimization_single(
            child,
            session_dir=session_dir,
            timeout_override_sec=timeout_override,
        )
        attempts.append(
            {
                "backend": backend,
                "status": result.get("status"),
                "kernel_id": result.get("kernel_id"),
                "proposal": result.get("proposal"),
                "verification": result.get("verification"),
                "best_artifact_path": result.get("best_artifact_path"),
                "error": result.get("error"),
            }
        )
        if best is None or _kernel_result_rank(result) > _kernel_result_rank(best):
            best = result
        if _kernel_result_rank(result)[0] == 1:  # KEEP -> stop this ladder
            break
    return best, attempts


def _stamp_task_group_result(
    result: HandlerResult,
    candidate: dict[str, Any] | None,
    *,
    fallback_kernel_id: str = "",
) -> HandlerResult:
    """Attach the dispatch grouping to every single or batched result path.

    Two groupings arrive here, and only one of them has a ``task_group``. An
    op-fanout representative carries ``opfanout_collapsed_ids``: the siblings the
    batch filter merged into it, which no backend will be handed separately. The
    ledger has to record them for the same reason it records a task_group's
    members -- ``untried_hot_reusable_kernels`` resolves a member to whichever
    ledger row covers it, and a sibling that resolves to none owes an attempt
    that can never happen.
    """
    if not isinstance(result, dict) or not isinstance(candidate, dict):
        return result

    collapsed = [str(item) for item in (candidate.get("opfanout_collapsed_ids") or []) if str(item)]
    if collapsed:
        result = {**result, "opfanout_collapsed_ids": collapsed}

    task_group = candidate.get("task_group")
    if not isinstance(task_group, dict):
        return result

    stamped = dict(result)
    stamped.setdefault("task_group_id", str(task_group.get("task_group_id") or ""))
    stamped.setdefault("task_group_key", str(task_group.get("task_group_key") or ""))
    stamped.setdefault(
        "identity_route",
        str(task_group.get("identity_route") or ""),
    )
    stamped.setdefault(
        "operator_identity",
        dict(task_group.get("operator_identity") or {}),
    )
    stamped.setdefault(
        "legacy_task_group_keys",
        [str(item) for item in (task_group.get("legacy_task_group_keys") or []) if str(item)],
    )
    stamped.setdefault(
        "task_group_kernel_ids",
        [str(item) for item in (task_group.get("kernel_ids") or []) if str(item)],
    )
    stamped.setdefault(
        "task_group_primary_kernel_id",
        str(task_group.get("primary_kernel_id") or candidate.get("kernel_id") or fallback_kernel_id),
    )
    shape_cases = task_group.get("shape_cases")
    if isinstance(shape_cases, list):
        stamped.setdefault("task_group_shape_case_count", len(shape_cases))
        stamped.setdefault(
            "task_group_shape_case_ids",
            [
                str(case.get("case_id") or "")
                for case in shape_cases
                if isinstance(case, dict) and str(case.get("case_id") or "")
            ],
        )
    return stamped


async def _run_kernel_backend_sequence(
    base_payload: dict,
    candidate: dict[str, Any],
    *,
    session_dir: Path,
    parallel_backends: bool = False,
) -> HandlerResult:
    """Optimize one kernel with the forge backend.

    Args:
        base_payload: The base request payload shared by every backend.
        candidate: The kernel candidate to optimize.
        session_dir: Session directory for workspace and state.
        parallel_backends: Retained for signature compatibility; unused.

    Returns:
        The strongest ``HandlerResult`` produced.
    """
    kernel_id = str(candidate.get("kernel_id") or base_payload.get("kernel_id") or "")
    order = _backend_order(base_payload)

    # Bound the backend to one wall-clock budget so a hang cannot overshoot
    # the KERNEL-phase cap.
    ladder_deadline = time.monotonic() + _kernel_ladder_budget_sec(base_payload)

    best, attempts = await _run_backend_ladder(
        base_payload,
        candidate,
        kernel_id,
        order,
        session_dir=session_dir,
        deadline=ladder_deadline,
    )

    if best is None:
        best = {
            "status": "failed",
            "kernel_id": kernel_id,
            "error": "no backend attempts were run",
        }
    best = dict(best)
    best["backend_fallback_attempts"] = attempts
    best["batch_kernel_id"] = kernel_id
    best = _stamp_task_group_result(
        best,
        candidate,
        fallback_kernel_id=kernel_id,
    )
    # Preserve source_file so the streaming callback can group by file.
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
    """Fan ``run_optimization`` out across reusable native kernels (``record_partial`` streams each sub-attempt into SharedState before gather wait-all unblocks).

    Args:
        payload: The run_optimization request payload.
        candidates: The reusable native kernels to fan out across.
        session_dir: Session directory for workspace and state.
        record_partial: Optional callback streaming each sub-attempt into
            SharedState before the gather wait-all unblocks.

    Returns:
        The strongest ``HandlerResult`` augmented with batch metadata.
    """
    max_parallel = int(
        payload.get("max_parallel") or os.environ.get("KERNEL_OPT_MAX_PARALLEL") or _default_kernel_batch_parallel()
    )
    max_parallel = max(1, max_parallel)
    # Forge edits framework sources in-place; concurrent kernels could race the
    # per-repo lock, so keep the batch serial whenever forge is in the ladder.
    if "forge" in _backend_order(payload):
        max_parallel = 1
    # parallel_backends is off by default; it only matters when explicit
    # per-kernel forge mode is enabled.
    parallel_backends = _should_parallelize_backends(payload, len(candidates))
    # When forced on, halve the GPU budget so pre-Ray backend setup fits.
    if parallel_backends:
        n_gpus = _visible_gpu_count()
        per_task = _per_task_gpus()
        if n_gpus and per_task > 0:
            max_parallel = min(max_parallel, max(1, n_gpus // (2 * per_task)))
    sem = asyncio.Semaphore(max_parallel)

    async def _guarded(candidate: dict[str, Any]) -> HandlerResult:
        """Run one candidate's backend sequence under the concurrency semaphore.

        Acquires the shared ``max_parallel`` semaphore, runs the backend
        sequence for a single candidate, and converts any exception into a
        failed :class:`HandlerResult` so a sub-task error never propagates out
        of ``asyncio.gather`` while sibling tasks are still in flight.

        Args:
            candidate (dict[str, Any]): The kernel candidate descriptor to run
                (expects ``kernel_id`` and ``source_file`` keys when a dict).

        Returns:
            HandlerResult: The backend-sequence result, or a failed result if
                the sub-task raised.
        """
        cand_kid = str(candidate.get("kernel_id") or "") if isinstance(candidate, dict) else ""
        cand_src = str(candidate.get("source_file") or "") if isinstance(candidate, dict) else ""
        async with sem:
            try:
                result = await _run_kernel_backend_sequence(
                    payload,
                    candidate,
                    session_dir=session_dir,
                    parallel_backends=parallel_backends,
                )
            except Exception as exc:  # noqa: BLE001
                # Wrap a sub-task failure as a structured result so gather stays
                # wait-all (a raised exception would unblock mid-batch).
                log.exception(
                    "kernel-opt sub-task crashed for kernel_id=%s; wrapping as failed result so gather wait-all holds",
                    cand_kid or "?",
                )
                result = {
                    "status": "failed",
                    "kernel_id": cand_kid,
                    "source_file": cand_src,
                    "error_class": "subtask_exception",
                    "error": repr(exc),
                }
        # Re-stamp source_file so the same-file conflict guard can detect two
        # KEEPs on one file (defensive; the sequence already preserves it).
        if isinstance(result, dict) and not result.get("source_file") and cand_src:
            result["source_file"] = cand_src
        result = _stamp_task_group_result(
            result,
            candidate,
            fallback_kernel_id=cand_kid,
        )
        if record_partial is not None:
            try:
                record_partial(result)
            except Exception:  # noqa: BLE001
                # Callback failure must not abort the batch; the post-gather
                # record path recovers the lost streaming write.
                log.exception(
                    "record_partial callback failed for kernel_id=%s",
                    (result or {}).get("kernel_id") if isinstance(result, dict) else None,
                )
        return result

    results = await asyncio.gather(*(_guarded(c) for c in candidates))
    best = max(results, key=_kernel_result_rank, default=None)
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
    out["parallel_backends"] = parallel_backends
    out["batch_results"] = results
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_native_kernel_run_result(session_dir, result=out)
    except Exception:  # noqa: BLE001
        log.debug("native kernel v4 batch-result recording failed", exc_info=True)
    return out


def _backends_cli_arg(value: Any) -> str:
    """Normalize a payload ``backends`` field into a bare ``--backends`` value.

    The orchestration payload may carry ``backends`` as a bare string
    (``"forge"``) or a JSON list (``["forge"]``) when an upstream request
    serializes it as an array. A list MUST be comma-joined into bare names,
    never ``str()``-ed into the repr of a list (``"['forge']"``) — the
    kernel-agent's ``parse_backends`` validator correctly rejects that opaque
    token and the dispatch fails with the self-contradictory
    "unsupported backend(s): ['forge']".

    Args:
        value: The raw ``payload["backends"]`` value (str / list / tuple / None).

    Returns:
        A bare, comma-joined backend string (possibly empty).
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(b).strip() for b in value if str(b).strip())
    return str(value).strip()


async def _run_optimization_single(
    payload: dict,
    *,
    session_dir: Path,
    timeout_override_sec: int | None = None,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's kernel_optimization.py on one kernel.

    Required payload: ``kernel_id``. Returns the tool's JSON output verbatim.

    Args:
        payload: The single-kernel request payload (requires ``kernel_id``).
        session_dir: Session directory for workspace and state.

    Returns:
        The kernel_optimization tool's JSON output as a ``HandlerResult``.
    """
    kernel_id = payload.get("kernel_id")
    if not kernel_id:
        return {"status": "failed", "error": "missing 'kernel_id' in payload"}
    # A guard result is recorded as an attempt, and the attempt ledger keys on
    # kernel_id plus source_file. Carrying the source through means a rejection
    # and a later real dispatch share one identity instead of splitting into an
    # empty-source row nothing can reconcile.
    guard_source = str(payload.get("source_file") or (payload.get("candidate") or {}).get("source_file") or "")

    def _with_source(guard: HandlerResult) -> HandlerResult:
        """Stamp the resolved source onto a guard result that omitted it."""
        if guard_source and not guard.get("source_file"):
            guard = {**guard, "source_file": guard_source}
        return guard

    guard = _validate_reusable_native_kernel(payload)
    if guard is not None:
        return _with_source(guard)
    shape_guard = _validate_kernel_shape_and_paths(
        payload,
        session_dir=session_dir,
    )
    if shape_guard is not None:
        return _with_source(shape_guard)
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}

    # Pass the session root so artefacts land under ``<session_dir>/kernel-agent/runs/...``.
    workspace_path = payload.get("workspace_path") or str(session_dir)
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    target_platform = (payload.get("target_platform") or state.gpu_type or "").strip()
    if target_platform:
        os.environ["TARGET_GPU_TYPE"] = target_platform

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("kernel_optimization.py")),
        "--kernel-id",
        str(kernel_id),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--workspace-path",
        workspace_path,
    ]
    candidate_payload = payload.get("candidate")
    if isinstance(candidate_payload, dict):
        task_group = candidate_payload.get("task_group")
        group_id = str(task_group.get("task_group_id") or "") if isinstance(task_group, dict) else ""
        identity = group_id or str(candidate_payload.get("kernel_id") or kernel_id)
        safe_identity = re.sub(r"[^A-Za-z0-9._-]+", "_", identity).strip("._-") or "candidate"
        candidate_json_path = (
            Path(workspace_path)
            / "kernel-agent"
            / "runs"
            / str(payload.get("session_id") or session_dir.name)
            / "inputs"
            / f"candidate_{safe_identity}.json"
        )
        try:
            candidate_json_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_json_path.write_text(
                json.dumps(candidate_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            cmd += ["--candidate-json", str(candidate_json_path)]
        except OSError as exc:
            log.warning(
                "could not persist grouped candidate input %s: %s",
                candidate_json_path,
                exc,
            )
    backends_arg = _backends_cli_arg(payload.get("backends"))
    if backends_arg:
        cmd += ["--backends", backends_arg]
    if payload.get("source_file"):
        cmd += ["--source-file", str(payload["source_file"])]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    extra_args = str(payload.get("extra_server_args") or "").strip()
    if extra_args:
        cmd += ["--extra-sglang-args", extra_args]
    if payload.get("candidates_path"):
        cmd += ["--candidates-path", str(payload["candidates_path"])]
    if payload.get("benchmark_file"):
        cmd += ["--benchmark-file", str(payload["benchmark_file"])]
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
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    # Always pass the resolved budget so the operator env override reaches the
    # tool: reading payload directly here would bypass
    # _optimization_budget_minutes and silently leave forge on its own 60-min
    # default (of which forge-loop reserves half for finalize).
    cmd += ["--budget-minutes", str(_optimization_budget_minutes(payload))]
    # Let the tool handle its own backend timeout and salvage partial artifacts.
    timeout_sec = _optimization_wrapper_timeout_sec(payload)
    if timeout_override_sec is not None:
        # Cap each subprocess to the time left in the per-kernel budget.
        timeout_sec = max(1, min(timeout_sec, int(timeout_override_sec)))

    from ..actions.executors._multi_node_env import is_multi_node

    if is_multi_node():
        from hyperloom.inference_optimizer.multi_node.cli import (
            kill_inference_for_kernel_agent_best_effort,
        )

        await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)

    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_native_kernel_run_start(
            session_dir,
            payload={
                "kernel_id": str(kernel_id),
                "backend": backends_arg,
                "source_file": payload.get("source_file"),
            },
        )
        instrument.record_kernel_dispatch(
            session_dir,
            kernel_id=str(kernel_id),
            dispatched=True,
            backends=[value for value in backends_arg.split(",") if value],
            task_group=(payload.get("candidate") or {}).get("task_group")
            if isinstance(payload.get("candidate"), dict)
            else None,
        )
    except Exception:  # noqa: BLE001
        log.debug("native kernel v4 start recording failed", exc_info=True)

    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # Shape a failed result here instead of letting TimeoutExpired propagate
        # to the batch wrapper (which would drop the real backend attribution).
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
        }
    # Stamp source_file / kernel_id from the payload onto the result so the
    # multi-KEEP integrate queue can group same-file KEEPs (the tool may omit
    # them on timeout/crash).
    if isinstance(result, dict):
        if not result.get("kernel_id") and payload.get("kernel_id"):
            result["kernel_id"] = str(payload["kernel_id"])
        if not result.get("source_file") and payload.get("source_file"):
            result["source_file"] = str(payload["source_file"])
        # Attribute a result with no per-backend attempt ladder to the backend
        # that ran, but only when a single unambiguous backend was dispatched.
        dispatched_backend = backends_arg.lower()
        if (
            dispatched_backend
            and "," not in dispatched_backend
            and not result.get("backend")
            and not result.get("attempts")
        ):
            result["backend"] = dispatched_backend
        result = _stamp_task_group_result(
            result,
            candidate_payload if isinstance(candidate_payload, dict) else None,
            fallback_kernel_id=str(kernel_id),
        )
    # Full-trace: mine each forge attempt's stdout for token usage and append an
    # ``llm_calls.jsonl`` row. Best-effort; no-op without a usage block.
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    # Full-trace: record each forge attempt's key-step timeline as a forge_steps
    # audit. Best-effort; no-op without a step marker.
    _trace_kernel_attempt_steps(result, session_dir=session_dir)
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_kernel_backend_result(
            session_dir,
            result,
            route_strategy="kernel_agent_forge",
        )
        instrument.record_native_kernel_run_result(session_dir, result=result)
    except Exception:  # noqa: BLE001
        log.debug("native kernel v4 result recording failed", exc_info=True)
    return result


def _trace_kernel_attempt_usage(
    result: Any,
    *,
    session_dir: Path,
) -> None:
    """Append ``llm_calls.jsonl`` rows for out-of-process attempts in ``result``.

    Each ``kernel_optimization`` attempt record carries ``backend`` plus
    ``optimized_path`` (the backend's full ``*_stdout.log``). For the
    token-traced backends (:data:`_TOKEN_TRACED_KERNEL_BACKENDS`) we read that
    log and run the matching usage parser (``forge`` →
    :func:`parse_forge_usage`). A row is appended only when a
    usage block is actually recovered — backends that don't emit usage stay a
    silent no-op rather than logging fabricated zeros.

    Best-effort end to end: any read/parse/append failure is logged at debug
    and swallowed so kernel optimization never breaks on a trace write.

    Args:
        result: A kernel_optimization result whose ``attempts`` are mined.
        session_dir: Session directory the ``llm_calls.jsonl`` rows append to.
    """
    if not isinstance(result, dict):
        return
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        return
    kernel_id = str(result.get("kernel_id") or "") or None
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        backend = str(attempt.get("backend") or "").strip().lower()
        if backend not in _TOKEN_TRACED_KERNEL_BACKENDS:
            continue
        log_path = str(attempt.get("optimized_path") or "").strip()
        if not log_path:
            continue
        try:
            stdout_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        try:
            usage = parse_forge_usage(stdout_text)
            if not usage:
                # No failure row is written here, deliberately. This is a
                # post-hoc log scrape of an out-of-process child, and the child
                # prints FORGE_LLM_USAGE only on success — so a missing marker
                # cannot be told apart from "the child's LLM calls failed".
                # The attempt dict carries only business outcomes (improved,
                # best_ms), and synthesizing an LLM error from those is exactly
                # the conflation that makes an error rate untrustworthy.
                # Closing this gap needs the child (GEAK / KernelForge
                # forge_submit) to emit a failure marker of its own.
                continue
            record = LLMCallRecord(
                session_id=session_dir.name,
                component=backend,
                task_id=kernel_id,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                # Read through the shared helper so a reasoning model spends the
                # same way here as it does on the in-process backends' rows.
                reasoning_output_tokens=reasoning_output_tokens(usage),
            )
            append_llm_call(session_dir=session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break optimization
            log.debug(
                "full-trace: kernel attempt usage append failed (backend=%s, log=%s)",
                backend,
                log_path,
                exc_info=True,
            )


def _trace_kernel_attempt_steps(
    result: Any,
    *,
    session_dir: Path,
) -> None:
    """Record each forge attempt's key-step timeline to the forge_steps audit.

    Reads the ``FORGE_STEPS`` marker off each forge attempt's stdout log
    (``optimized_path``) — the per-iteration rationale / validation / bench /
    keep-revert steps plus a run summary — and appends one audit row per step to
    ``reports/trace/forge_steps.jsonl``, keyed by kernel id. The Langfuse emitter
    backfills these as ``forge:iter:<n>`` / ``forge:summary`` spans so a trace
    shows forge's decision process. Best-effort end to end: any read/parse/write
    failure degrades to a debug log and is swallowed.
    """
    if not isinstance(result, dict):
        return
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        return
    from datetime import datetime, timezone
    from hyperloom.inference_optimizer.session.session_paths import forge_steps_path

    kernel_id = str(result.get("kernel_id") or "") or None
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if str(attempt.get("backend") or "").strip().lower() != "forge":
            continue
        log_path = str(attempt.get("optimized_path") or "").strip()
        if not log_path:
            continue
        try:
            stdout_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        payload = parse_forge_steps(stdout_text)
        if not payload:
            continue
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        for step in payload.get("steps") or []:
            if not isinstance(step, dict):
                continue
            rows.append(
                {
                    "kernel_id": kernel_id,
                    "kind": "iteration",
                    "ts": ts,
                    **step,
                }
            )
        summary = payload.get("summary")
        if isinstance(summary, dict):
            rows.append(
                {
                    "kernel_id": kernel_id,
                    "kind": "summary",
                    "ts": ts,
                    **summary,
                }
            )
    if not rows:
        return
    try:
        path = forge_steps_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            append_jsonl(path, row, sort_keys=True)
    except OSError:
        log.debug("full-trace: forge_steps append failed", exc_info=True)


def _shape_tool_result(rc: int, stdout: str, stderr: str) -> HandlerResult:
    """Wrap a kernel-agent tool's exit + stdout into our schema (prefer the tool's own JSON, synthesize only on parse failure).

    Args:
        rc: The tool's process return code.
        stdout: The tool's captured standard output.
        stderr: The tool's captured standard error.

    Returns:
        The tool's own JSON result (status filled from ``rc`` if absent), or a
        synthesized failure result when stdout has no parseable JSON.
    """
    parsed = _parse_tool_stdout(stdout)
    if parsed and set(parsed) == {"raw_stdout_tail"}:
        # Unparseable output is not a result. Inferring ``ok`` from rc==0 here
        # made a tool whose output we could not read indistinguishable from one
        # that succeeded: the roofline executor read status=ok, recorded an
        # empty analysis over the real one, and the leg reported success while
        # twenty minutes of GPU evidence went in the bin.
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
    """Parse a tool's stdout into a dict, surviving non-JSON noise.

    Tries the whole stdout as a JSON object first; if that fails, scans
    backwards for the last line that is a standalone JSON object. As a last
    resort returns the stdout tail under ``raw_stdout_tail``.

    Args:
        stdout (str): Captured standard output from a kernel-agent tool.

    Returns:
        dict[str, Any]: The parsed JSON object, an empty dict for empty input,
            or ``{"raw_stdout_tail": ...}`` when no JSON object is found.
    """
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
    # Last: a pretty-printed object opening at the start of a line. A tool that
    # indents its result spans many lines, so neither whole-text nor per-line
    # parsing sees it, and it is exactly the tools with a lot to say that
    # indent. tracelens_analysis returned a megabyte of hot-kernel analysis this
    # way, interleaved with progress chatter and followed by an import banner;
    # every field of it was dropped and the run still reported ``ok``.
    # ``raw_decode`` stops at the end of the object, so trailing noise is fine.
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

    # A live compiler may legitimately own the observed lock. When liveness is
    # unknown, a fresh skipped lock is also not safe to remove. Retry only after
    # at least one deletion or after confirming the lock disappeared.
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
    """Completion tokens the eval harness reserves per sample.

    Mirrors the clamp installed by the inferencex shim: ``HYPERLOOM_EVAL_MAX_TOKENS``
    when it parses as a positive integer, else the shim's own default. ``0``
    means the operator disabled the clamp, so no budget can be assumed.
    """
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
    """Grade a kernel re-baseline's accuracy against the session baseline.

    The staged re-baseline already ran the serving eval after hot throughput
    passed, so the score is read back rather than re-measured. A measured drop
    beyond ``ACCURACY_THRESHOLD`` blocks the KEEP. A missing verdict blocks only
    when a positive baseline accuracy proves eval works in this environment;
    otherwise the gate degrades to throughput-only so eval-less setups are not
    universally blocked.

    The preferred path is the accuracy attached by BaselineExecutor's staged
    accuracy round. Workspace parsing remains as a compatibility fallback for
    older runs where eval lived in the warmup slot.

    Args:
        bench_result: The re-baseline result dict from ``BaselineExecutor``.
        session_dir: Session directory used to resolve ``baseline_accuracy``.
        workspace: The integrate task workspace holding both round slots.
        strict: Grade an artifact whose correctness has only ever been proven
            against a reference. This serving run is its first and only
            end-to-end evidence, so the operator opt-out does not apply and a
            gate that produced no verdict blocks instead of degrading.

    Returns:
        ``{"blocked": bool, "accuracy_pass": bool | None, "reason": str,
        "degraded": bool, "accuracy": float | None, "baseline_accuracy": float,
        "task": str, "metric": str, "source_file": str}``.
    """
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
    # A verdict can be missing because the eval broke, or because the serving
    # configuration cannot answer an eval request at all. Only the first says
    # anything about the patch. The second reproduces on every retry, so
    # charging it to the patch discards a kernel over a configuration choice.
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


def _cold_start_rebaseline_timeout(resolved_sec: int) -> int:
    """Raise a re-baseline timeout to the cold-start cap when the JIT cache is empty.

    An apply moves the cache aside, so the re-baseline recompiles from scratch;
    the explicit ``timeout_sec`` integrate passes also suppresses the baseline
    executor's own cold-start branch, leaving the warm budget as the only one.
    """
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
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Applies an optimized kernel artifact, re-runs the active Magpie baseline,
    and KEEPs only when measured E2E throughput clears the threshold AND the
    re-baseline's accuracy holds (source + artifacts are backed up first so
    non-KEEP can restore without a rebuild). Accuracy is graded only for a
    candidate that already cleared the throughput bar -- see
    :func:`_grade_integrate_accuracy`.

    Payload: ``base_tput`` must be > 0 at decision time, but is auto-filled from
    SharedState when a baseline has been recorded, so a bare ``{kernel_id}`` (or
    ``{integration_id}``) payload is accepted. Optional: patch_path,
    target_file, snapshot_dir, kernel_repo, config_path, extra_server_args,
    extra_envs, source, task_group_key, keep_threshold_pct (1.0), timeout_sec,
    or budget_minutes. Without an explicit timeout, the benchmark config's
    timeout contract is used. Returns ``{status, decision, base_tput, new_tput,
    gain_pct, kernel_id, patch_path, report_path, workspace}``.

    Args:
        payload: The integrate request payload.
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` with the KEEP/REVERT decision and re-baseline
        metrics (``status``, ``decision``, ``base_tput``, ``new_tput``,
        ``gain_pct``, ``kernel_id``, ``patch_path``, ``report_path``,
        ``workspace``), plus ``accuracy`` / ``baseline_accuracy`` /
        ``accuracy_pass`` / ``accuracy_gate`` when the gate was graded.
    """
    from ..actions.executors.baseline import BaselineExecutor
    from ..actions.executors.benchmark_result import is_valid_measurement
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare
    # {kernel_id} payload isn't failed with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)

    if payload.get("_vendor_playbook_deploy_blocked"):
        # A vendor-playbook KEEP (e.g. mori dispatch/combine launch-config
        # tuning) has no deployable artifact: best_artifact_path is a copy of
        # a KernelForge task-bundle config file, not a rewrite of the real
        # installed operator, and apply_kernel_patch's legacy full-file
        # replace would happily overwrite the real site-packages module with
        # it (PR #1191 review finding #1). Refuse before touching the
        # filesystem rather than letting a config-file copy silently
        # corrupt a live install.
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

    # Env-only is a property of the request, not of who sent it: a payload that
    # carries a runtime bundle and names no artifact has nothing to apply, so it
    # is graded on the bundle alone. Deciding by shape also keeps such a request
    # away from _resolve_integrate_payload, which would otherwise back-fill
    # patch_path/target_file from the last kernel optimization and silently
    # measure an unrelated patch.
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
        # Apply crash: the patch was never measured. Stamp a top-level fault
        # error_class so SharedState routes this through the fault retry budget.
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
    # VRAM barrier (HL_HONEST_E2E umbrella, default ON; opt out with
    # HL_HONEST_E2E=0 or HL_INTEGRATE_VRAM_GUARD=0): cap re-baseline util on
    # vLLM so the integrate server cannot OOM on a tighter node.
    extra_args = _vram_guarded_server_args(extra_args)

    # Wrap BaselineExecutor in a Task/RunnerContext.
    from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir

    fake_task_id = f"integrate-{kernel_id or 'anon'}"
    workspace = unique_runs_dir(session_dir, "integrate", fake_task_id)
    baseline_executor = BaselineExecutor(session_dir=session_dir)
    rebaseline_timeout_sec = _cold_start_rebaseline_timeout(
        _integrate_rebaseline_timeout_sec(
            payload,
            default_timeout_sec=baseline_executor.default_timeout_sec,
        )
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
            # The only artifact that patches FlyDSL sources, so the only run that
            # needs the JIT cache key widened.
            "flydsl_source_dirs": (str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND),
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": base_tput * (1.0 + keep_threshold_pct / 100.0),
            "accuracy_timeout_sec": rebaseline_timeout_sec,
            # Synthetic kind="baseline": candidate A/B validation against the
            # already-anchored reference. It runs eval for the kernel accuracy
            # gate but never establishes a replacement quality reference.
            "quality_ref_exempt": True,
        },
        idempotency_key=f"{fake_task_id}-rebaseline",
    )
    ctx = RunnerContext(task=fake_task, lease=None)

    # aiter cpp_itfs kernels recompile at runtime and its cache hashes params not
    # source, so set AITER_REBUILD=1 for the re-baseline server to force a rebuild
    # of the patched kernel. Scoped to cpp_itfs applies and always restored.
    cpp_itfs_backup = apply_result.get("cpp_itfs_cache_backup") or {}
    force_aiter_rebuild = bool(cpp_itfs_backup.get("is_cpp_itfs"))
    _prev_aiter_rebuild = os.environ.get("AITER_REBUILD")
    if force_aiter_rebuild:
        os.environ["AITER_REBUILD"] = "1"

    def _restore_aiter_rebuild_env() -> None:
        """Restore the ``AITER_REBUILD`` env var to its prior value.

        No-op unless a forced rebuild was applied for this re-baseline.
        """
        if not force_aiter_rebuild:
            return
        if _prev_aiter_rebuild is None:
            os.environ.pop("AITER_REBUILD", None)
        else:
            os.environ["AITER_REBUILD"] = _prev_aiter_rebuild

    # Multi-node: force a FULL sglang restart so it re-imports the patched
    # modules (a resume would measure the pre-patch process). mn_round_restarted
    # stops a double restart; force_full_restart scopes the resume override here.
    from ..actions.executors._multi_node_env import is_multi_node

    # This must run even when the regular JIT cache is warm: cpp_itfs attention
    # uses the separate AITER_ROOT_DIR/build tree and can carry a stale baton
    # from a timed-out Forge driver.
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
        # Restore AITER_REBUILD on every path so the override never leaks past
        # this integrate.
        _restore_aiter_rebuild_env()

    if not is_valid_measurement(bench_result):
        revert_result = _maybe_revert_kernel_patch(apply_result)
        # The re-baseline produced no usable measurement, so the patch was never
        # fairly scored. Surface a top-level fault error_class (the re-baseline's
        # own when present, else bench_exception) so this routes through the fault
        # retry budget rather than being discarded as a genuine REVERT.
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

    # Don't score a stale binary: for cpp_itfs targets the served kernel is
    # runtime-compiled, so a reused params-hashed lib.so would measure the
    # PRE-patch kernel. Assert a fresh lib.so (newer than the invalidation) landed
    # before trusting gain_pct; otherwise flag for review.
    #
    # Single-node only: in multi-node the served cache lives on the serving pod,
    # so AITER_REBUILD=1 on the pod restart is the mechanism and this local check
    # is skipped. verify_cpp_itfs_rebuilt() returns verified=True off the
    # cpp_itfs path, so this gate is a strict no-op there.
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

    # Accuracy gate: a kernel patch only KEEPs if it also holds accuracy. Graded
    # ONLY for a candidate that already cleared the throughput bar, so a
    # regressing patch never spends a verdict on itself, and graded from the
    # re-baseline's own eval output, so the verdict costs no extra GPU time.
    # Placed ahead of the optional source-import pass so a patch that loses
    # accuracy short-circuits before it runs.
    # An apply-back carries only reference correctness, so this run is the sole
    # end-to-end evidence it will ever get.
    # Anything other than a recorded pass still owes the verdict, so an absent or
    # unrecognised status keeps the gate armed rather than disarming it.
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
                # The gate cannot run under this configuration, so this round
                # measured nothing about the patch. Report it as an integration
                # fault: faults carry their own budget and never consume one of
                # the three attempts a patch gets to prove itself.
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
            # A measured regression is hard negative evidence -> REVERT. A
            # missing verdict is only an evidence gap -> NEEDS_REVIEW.
            decision = "REVERT" if accuracy_gate["accuracy_pass"] is False else "NEEDS_REVIEW"

    # import-grep source confirmation (HL_HONEST_E2E umbrella, default ON; opt
    # out with HL_HONEST_E2E=0 or HL_CONFIRM_SOURCE_IMPORTED=0). Advisory:
    # annotate whether the served process imported/compiled the patched source.
    # Only the strict sub-flag enforces it, and only on positive non-import
    # evidence (confirmed is False); an "unknown" (None) never penalizes.
    source_import_confirmed: bool | None = None
    source_import_evidence: dict[str, bool | None] = {}
    source_not_imported_downgrade = False
    if _honest_flag("HL_CONFIRM_SOURCE_IMPORTED"):
        # Grade the whole write set; the single target is the fallback for a
        # patch whose bundle declared none.
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
    # Recorded last so a blocking accuracy verdict owns ``decision_reason``: it
    # is the reason this candidate lost its KEEP, outranking the throughput-side
    # annotations above.
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
        # Only a KEEP settles the outstanding verdict. A non-KEEP is left
        # unstamped: the attempt ledger already distinguishes a rejection from a
        # retryable fault, and this field must not blur the two.
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
    "run_optimization": run_optimization_handler,
    "integrate": integrate_handler,
    "apply_patch": integrate_handler,  # alias — same flow
}


def has_handler(kind: str) -> bool:
    """Report whether a programmatic handler is registered for a request kind.

    Args:
        kind (str): The kernel request ``kind`` to check.

    Returns:
        bool: ``True`` if a handler is registered for ``kind``, else ``False``.
    """
    return kind in KERNEL_REQUEST_HANDLERS


def get_handler(kind: str) -> HandlerFn | None:
    """Look up the programmatic handler registered for a request kind.

    Args:
        kind (str): The kernel request ``kind`` to resolve.

    Returns:
        HandlerFn | None: The registered handler coroutine function, or
            ``None`` when no handler is registered for ``kind``.
    """
    return KERNEL_REQUEST_HANDLERS.get(kind)


__all__ = [
    "KERNEL_REQUEST_HANDLERS",
    "get_handler",
    "has_handler",
    "integrate_handler",
    "run_gemm_tuning_handler",
    "run_optimization_handler",
    "trace_analyze_handler",
]
