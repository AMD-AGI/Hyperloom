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
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .trace.llm_trace import LLMCallRecord, append_llm_call
from .trace.parse_usage import (
    parse_forge_steps,
    parse_forge_usage,
    parse_geak_usage,
    parse_oob_json_usage,
)


log = logging.getLogger(__name__)
_BACKGROUND_ROCPROF_TASKS: set[asyncio.Task[Any]] = set()
STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT = 1.0

# kernel_optimization attempt backends whose stdout log we mine for token
# usage. ``geak`` uses litellm (OpenAI-shape usage); ``oob`` runs ``oob run
# --json`` whose envelope may carry a ``usage`` block; ``forge`` (Kernel-Forge
# autonomous loop) prints a ``FORGE_LLM_USAGE {json}`` marker aggregated from
# its claude-agent-sdk ResultMessages. The other backends (claude/codex/cursor)
# already account their spend via their own paths.
_TOKEN_TRACED_KERNEL_BACKENDS: frozenset[str] = frozenset({"geak", "oob", "forge"})


# Where the kernel-agent shell tools live; read lazily so cli.py's late env injection wins.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"


def _kernel_agent_root_from_env() -> Path | None:
    """Read the kernel-agent install root from the environment at call time.

    Resolved lazily on every call (rather than snapshotted at import) so a
    late ``os.environ`` injection by the CLI preflight still wins.

    Returns:
        Path | None: The kernel-agent root as a :class:`~pathlib.Path`, or
            ``None`` when ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset or empty.
    """
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
    """Framework install roots for patchability checks (from :func:`framework_paths.resolve_patch_target_roots`; emits a lower-case variant per root for case-insensitive matching).

    Returns:
        The de-duplicated framework install roots (each with a lower-case
        variant), including FlyDSL checkout roots.
    """
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
# Default ladder: forge first, then geak. claude/codex/cursor are NOT in the
# default anymore — they only run when explicitly requested via
# KERNEL_OPT_BACKEND_ORDER / KERNEL_OPT_BACKENDS (or payload backend_order).
# They remain in `allowed` (see _backend_order) so env-opt-in still works.
# Cursor is additionally key-gated (dropped when CURSOR_API_KEY is unset).
_DEFAULT_KERNEL_BACKEND_ORDER = ("forge", "geak")
# Soft cap on concurrent kernel-backend coroutines (legacy MI300X 8-GPU fallback; pin with KERNEL_OPT_MAX_PARALLEL).
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
_DEFAULT_OOB_BUDGET_MINUTES = 60.0
# Minimum wall-clock a fallback backend needs to do anything useful (and still
# salvage partial artifacts). When less than this remains in the per-kernel
# ladder budget, the ladder stops instead of launching a backend it cannot
# finish. Mirrors the +180s wrapper grace. See Hyperloom#602.
_KERNEL_LADDER_MIN_BACKEND_SEC = 180
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 3 * 60 * 60


@functools.lru_cache(maxsize=1)
def _default_geak_budget_minutes() -> float:
    """Default per-GEAK-attempt budget tracking ``$GEAK_RUN_MODE`` (quick→70, full→130). PR #301: mirrors kernel-agent tool defaults.

    Returns:
        The default per-attempt GEAK budget in minutes.
    """
    raw = (os.environ.get("GEAK_RUN_MODE") or "").strip().lower()
    return 70.0 if raw == "quick" else 130.0


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
        import torch  # local import: torch driver init can be expensive

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

    The legacy hard-coded 8 assumed a full MI300X / MI355X node. On
    smaller pods it lets the asyncio semaphore admit more concurrent
    sibling attempts than Ray can schedule, so they stack against the
    GPU lock and one fast kernel waits behind a stuck GEAK for many
    minutes. We use ``torch.cuda.device_count()`` for the visible-GPU
    count (works for both ROCm and CUDA backends) and
    ``$KERNEL_AGENT_NUM_GPUS`` for the per-attempt GPU reservation
    (set by the kernel-agent submitter). Falls back to the legacy
    ``_DEFAULT_KERNEL_BATCH_PARALLEL`` when torch can't tell us
    (CI / mocks / pre-driver init). Operators can still pin via
    ``KERNEL_OPT_MAX_PARALLEL``.

    Cached: visible GPU count and ``$KERNEL_AGENT_NUM_GPUS`` are fixed
    at process start; ``torch.cuda.device_count()`` is a driver query
    we don't want to re-issue on every batch dispatch. Tests that
    monkeypatch torch / env must call ``cache_clear()``; the
    ``inference_optimizer/tests/conftest.py`` autouse fixture handles
    this for every test.

    Returns:
        int: The adaptive maximum number of concurrent sibling kernel
        attempts, ``min(cap, visible_gpus // per_task_gpus)``.
    """
    n_gpus = _visible_gpu_count()
    if not n_gpus or n_gpus <= 0:
        return _DEFAULT_KERNEL_BATCH_PARALLEL
    return max(1, min(_DEFAULT_KERNEL_BATCH_PARALLEL, n_gpus // _per_task_gpus()))


def _should_parallelize_backends(payload: dict, num_candidates: int) -> bool:
    """Decide whether to race GEAK against the OOB ladder per kernel.

    Default policy ("GPU-aware"): enable whenever the node can run a single
    kernel's GEAK *and* OOB ladder side-by-side, i.e.
    ``visible_gpus >= 2 * per_task_gpus``. This is intentionally independent
    of ``num_candidates`` -- batch width (how many kernels race at once) is
    throttled separately by :func:`_run_optimization_batch`, which caps
    concurrency to ``visible_gpus // (2 * per_task_gpus)`` so the per-kernel
    before_kernel_opt rocprof (a pre-Ray subprocess NOT bound by the Ray GPU
    lease) never overcommits the GPUs. Below ``2 * per_task`` there isn't
    room for both ladders even for one kernel, so we keep the legacy
    sequential ladder (GEAK first, OOB only as a fallback when GEAK misses a
    KEEP).

    Operators / tests can force the decision via payload
    ``parallel_backends`` or env ``KERNEL_OPT_PARALLEL_BACKENDS``
    (truthy ``1/true/yes/on`` enables, anything else disables).

    Args:
        payload: Request payload; ``parallel_backends`` may force the choice.
        num_candidates: Number of kernel candidates in this request.

    Returns:
        ``True`` to race GEAK against the OOB ladder, else ``False``.
    """
    override = payload.get("parallel_backends")
    if override is None:
        raw_env = os.environ.get("KERNEL_OPT_PARALLEL_BACKENDS")
        if raw_env is not None and raw_env.strip() != "":
            override = raw_env
    if override is not None:
        return str(override).strip().lower() in {"1", "true", "yes", "on"}
    if num_candidates <= 0:
        return False
    n_gpus = _visible_gpu_count()
    if not n_gpus or n_gpus <= 0:
        return False
    return n_gpus >= 2 * _per_task_gpus()


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
            "inference_optimizer/scripts/install.sh and source $KERNEL_AGENT_ENV "
            "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
        )
    if not root.is_dir():
        return f"{_KERNEL_AGENT_ROOT_ENV} does not exist: {root}"
    return None


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
    """Escape hatch (default off) via ``payload['allow_empty_kernel_shape']`` or ``HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE=1``.

    Args:
        payload: Request payload that may carry ``allow_empty_kernel_shape``.

    Returns:
        ``True`` when empty kernel shapes are explicitly permitted.
    """
    if bool(payload.get("allow_empty_kernel_shape")):
        return True
    return str(os.environ.get("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _validate_kernel_shape_and_paths(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult | None:
    """Reject a kernel-opt dispatch with no trace-anchored shape or a missing source/workspace path (would burn budget with no anchor; guides back to ``trace_analyze``).

    Args:
        payload: Kernel-opt dispatch payload to validate.
        session_dir: Session directory used as the default workspace path.

    Returns:
        A failure ``HandlerResult`` describing the rejection, or ``None`` when
        the dispatch is valid.
    """
    # ``dry_run`` exercises the plumbing without a backend, so no GPU budget and fake fixture paths need not exist.
    if bool(payload.get("dry_run")):
        return None
    candidate = _load_candidate_metadata(payload)
    kernel_id = str(payload.get("kernel_id") or "")
    name = str(candidate.get("name") or payload.get("kernel_name") or kernel_id)

    shapes = candidate.get("shapes")
    if not isinstance(shapes, list):
        shapes = []
    provenance = str(candidate.get("shape_provenance") or payload.get("shape_provenance") or "").strip()
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
    from ..session_paths import patches_dir

    kid = str(kernel_id or payload.get("kernel_id") or "")
    backup_root = payload.get("backup_root") or (patches_dir(session_dir, kid or "anon") / "backup")
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
    """Revert a previously applied kernel patch using its apply manifest.

    Args:
        apply_result (HandlerResult): The result returned by
            :func:`_maybe_apply_kernel_patch`; must be ``status="ok"`` with a
            ``manifest_path`` to be revertible.

    Returns:
        HandlerResult: A ``status="skipped"`` result when there is no applied
            manifest, otherwise the tool's revert result dict.
    """
    if apply_result.get("status") != "ok" or not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    tool = _load_apply_tool()
    return tool.revert_kernel_patch(apply_result["manifest_path"])


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
        cb_args = current_best.get("extra_server_args") or current_best.get("extra_sglang_args") or ""
        if cb_args:
            resolved["extra_server_args"] = cb_args

    return resolved


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
    if kernel_id:
        attempt = (state.kernel_opt_attempts or {}).get(kernel_id) or {}
        if not resolved.get("patch_path") and attempt.get("last_artifact_path"):
            resolved["patch_path"] = str(attempt["last_artifact_path"])
        if not resolved.get("source_file") and attempt.get("last_source_file"):
            resolved["source_file"] = str(attempt["last_source_file"])

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


async def _run_subprocess(cmd: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """asyncio-friendly wrapper around blocking subprocess.run (keeps the reactor responsive).

    Args:
        cmd: The command and arguments to run.
        timeout_sec: Per-run timeout in seconds.

    Returns:
        A tuple of ``(returncode, stdout, stderr)``.
    """

    def _run() -> subprocess.CompletedProcess[str]:
        """Run the command synchronously in a worker thread.

        Copies the current environment, injects the Ray GCS address when running
        in multi-node mode, and prepends the venv ``bin`` directory to ``PATH``
        before invoking the command with output capture and the timeout.

        Returns:
            subprocess.CompletedProcess[str]: The completed process with captured
                text stdout/stderr.
        """
        env = os.environ.copy()
        from .action_executors._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
            dynamo_ssh_env_from_state,
        )

        if is_multi_node():
            # Dynamo backend: route GEAK GPU work to a pod over SSH (no Ray).
            # dynamo_ssh_env_from_state() returns {} for RayJob/single-node, so
            # the RAY_ADDRESS path below is unchanged for those.
            ssh_env = dynamo_ssh_env_from_state()
            if ssh_env:
                env.update(ssh_env)
            addr = "" if ssh_env else ray_gcs_address_from_state()
            if addr:
                env.setdefault("RAY_ADDRESS", addr)
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )

    proc = await asyncio.to_thread(_run)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


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
export RUN_EVAL="${{RUN_EVAL:-true}}"
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
    """Resolve GEMM tuning backend: forge or geak.

    Precedence:
    1. payload['gemm_tuning_backend']
    2. GEMM_TUNING_BACKEND env var
    3. Default: 'forge'
    """
    raw = str(
        payload.get("gemm_tuning_backend")
        or os.environ.get("GEMM_TUNING_BACKEND")
        or ""
    ).strip().lower()
    if raw in ("forge", "geak"):
        return raw
    return "forge"


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


def _forge_gemm_tune_available() -> bool:
    """Check if forge-gemm-tune CLI is importable or on PATH."""
    if shutil.which("forge-gemm-tune"):
        return True
    try:
        spec = importlib.util.find_spec("forge_gemm_tune")
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


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

    # Explicit override from payload
    if payload.get("precision"):
        precision = _normalize_precision(payload["precision"])
        quant_type = str(payload.get("quant_type") or "auto").strip()
        return precision, quant_type

    # Resolve from actual server args (baseline yaml + current_best overlay).
    current_best = getattr(state, "current_best", None) or {}
    try:
        server_args = resolve_runtime_workload(state, arm="current_best").server_args
    except Exception:  # noqa: BLE001 - best-effort fallback for partial state/test doubles
        server_args = ""
        if isinstance(current_best, dict):
            server_args = str(current_best.get("extra_server_args") or "")
    # Check all env sources for per-token signal: current_best.extra_envs,
    # reference_envs, and baseline yaml envs.
    extra_envs = dict(current_best.get("extra_envs") or {}) if isinstance(current_best, dict) else {}
    ref_envs = dict(getattr(state, "reference_envs", None) or {})
    per_token_signal = (
        _truthy_env_value(extra_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN"))
        or _truthy_env_value(ref_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN"))
    )

    quantization_arg = _parse_server_arg(server_args, "--quantization").lower()

    if quantization_arg == "fp8":
        precision = "fp8"
        # Only explicit per-token env should route to per_token.
        # Otherwise keep auto so forge can inspect kernel_signature_log
        # for QuantType.per_Token / blockscale detection.
        if per_token_signal:
            quant_type = "per_token"
        else:
            quant_type = "auto"
        return precision, quant_type

    if quantization_arg in ("fp4", "mxfp4"):
        return quantization_arg, "fp4"

    # Fall back to session precision
    precision = _normalize_precision(state.precision)
    if not precision:
        precision = "bf16"
    quant_type = str(payload.get("quant_type") or "auto").strip()
    return precision, quant_type


def _truthy_env_value(value: Any) -> bool:
    """Return True for common env truthy values."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_forge_server_log(state, session_dir: Path) -> str:
    """Find the server log matching the current runtime configuration.

    Priority: current_best workspace (matches the resolved server args)
    → baseline workspace → most recent server.log under runs/.
    """
    # 1. current_best workspace — matches the runtime args we resolved precision from.
    current_best = getattr(state, "current_best", None) or {}
    if isinstance(current_best, dict):
        cb_workspace = str(current_best.get("workspace") or "").strip()
        if cb_workspace:
            log_path = Path(cb_workspace) / "server.log"
            if log_path.is_file():
                return str(log_path)

    # 2. Baseline workspace — the initial server run.
    last_baseline = getattr(state, "last_baseline", None) or {}
    if isinstance(last_baseline, dict):
        bl_workspace = last_baseline.get("workspace") or ""
        if bl_workspace:
            log_path = Path(bl_workspace) / "server.log"
            if log_path.is_file():
                return str(log_path)

    # 3. Fallback: check known run subdirs (bounded, not recursive glob).
    runs_dir = session_dir / "runs"
    if runs_dir.is_dir():
        best: Path | None = None
        best_mtime: float = 0.0
        for sub in ("baseline", "explore", "gemm_tuning"):
            sub_dir = runs_dir / sub
            if not sub_dir.is_dir():
                continue
            for log in sub_dir.glob("*/server.log"):
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
        # Must have M/N/K keys (case-insensitive check)
        keys = {k.upper() for k in sample}
        return {"M", "N", "K"}.issubset(keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _resolve_forge_shapes(state, session_dir: Path) -> str:
    """Find TraceLens shapes JSON if available and in forge-compatible format.

    Forge dense tuners expect: [{"M": int, "N": int, "K": int}, ...]
    Only passes files that match this schema; incompatible formats are
    silently skipped so forge falls back to config.json shape derivation.
    """
    last_trace = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(last_trace, dict):
        return ""

    candidates: list[str] = []

    # Prefer explicit artifact fields when newer TraceLens versions expose them.
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

    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and _is_forge_compatible_shapes_json(p):
            return str(p)

    return ""


async def _run_forge_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Deterministic GEMM tuning via forge-gemm-tune CLI.

    Supports bf16/fp8/fp4 + sglang/vllm. Only micro-benchmarks;
    returns recommended_env for Hyperloom E2E validation.
    """
    from .shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    if not _forge_gemm_tune_available():
        forge_path = os.environ.get("FORGE_GEMM_TUNE_PATH", "")
        return {
            "status": "failed",
            "error_class": "forge_gemm_tune_not_found",
            "error": (
                "forge-gemm-tune CLI not found. Install via "
                "'pip install -e <path>/forge_gemm_tune' or set FORGE_GEMM_TUNE_PATH."
                f" (checked: FORGE_GEMM_TUNE_PATH={forge_path!r})"
            ),
            "backend": "forge",
        }

    # Resolve precision from actual runtime (not just session-level state)
    precision, quant_type = _resolve_forge_precision_and_quant(state, payload)
    framework = str(payload.get("framework") or state.framework or "sglang").strip().lower()

    workspace = _gemm_tuning_workspace(payload, session_dir=session_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    model_path = str(
        payload.get("model_path") or state.model_path or os.environ.get("MODEL_PATH") or ""
    ).strip()
    if not model_path:
        return {"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"}

    tp = int(payload.get("tp") or state.tp or os.environ.get("TP") or 1)
    conc = int(payload.get("conc") or state.conc or os.environ.get("CONC") or 64)
    gpu_type = str(
        payload.get("gpu_type") or state.gpu_type or os.environ.get("GPU_TYPE") or "mi300x"
    ).strip().lower()
    tokens = str(payload.get("tokens") or "").strip()
    # Default mp = all visible GPUs (server is stopped during tuning).
    from .policy import detect_gpu_count

    detected_gpus = detect_gpu_count() or tp
    mp = int(payload.get("mp") or os.environ.get("FORGE_GEMM_TUNE_MP") or detected_gpus)

    # Resolve server log for 1-stage ASM detection
    kernel_sig_log = str(payload.get("kernel_signature_log") or "").strip()
    if not kernel_sig_log:
        kernel_sig_log = _resolve_forge_server_log(state, session_dir)

    # Resolve TraceLens shapes if available
    shapes_json = str(payload.get("shapes_json") or "").strip()
    if not shapes_json:
        shapes_json = _resolve_forge_shapes(state, session_dir)

    cmd = [
        "python3", "-m", "forge_gemm_tune.cli", "run",
        "--model-path", model_path,
        "--framework", framework,
        "--precision", precision,
        "--quant-type", quant_type,
        "--gpu-type", gpu_type,
        "--tp", str(tp),
        "--conc", str(conc),
        "--mp", str(mp),
        "--output-dir", str(workspace),
        "--skip-gpu-check",
    ]
    if tokens:
        cmd.extend(["--tokens", tokens])
    if payload.get("untuned_csv"):
        cmd.extend(["--untuned-csv", str(payload["untuned_csv"])])
    if shapes_json:
        cmd.extend(["--shapes-json", shapes_json])
    if payload.get("tunableop_input"):
        cmd.extend(["--tunableop-input", str(payload["tunableop_input"])])
    if kernel_sig_log:
        cmd.extend(["--kernel-signature-log", kernel_sig_log])

    timeout = _gemm_tuning_timeout_sec(payload)
    cmd.extend(["--timeout", str(timeout)])
    # Global timeout ensures the whole session (all tuners combined) stays
    # within the budget. Forge will skip lower-priority tuners if time runs out,
    # guaranteeing the highest-value tuner (MoE fmoe_ck) always runs first.
    cmd.extend(["--global-timeout", str(timeout)])

    # Thorough mode: exhaustive search when session budget allows (>= 24h)
    # and enough GPUs are available (>= 4) to parallelize the sweep.
    session_max_min = float(getattr(state, "max_minutes", 0) or 0)
    if session_max_min >= 1440 and mp >= 4:
        cmd.append("--thorough")

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout)

    result = _parse_forge_gemm_sentinel(stdout)
    if result is None:
        result = _shape_tool_result(rc, stdout, stderr)

    result.setdefault("backend", "forge")
    result.setdefault("workspace", str(workspace))
    result.setdefault("precision", precision)
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)

    # Bridge forge schema → coordinator-consumable schema:
    # forge returns micro_decision="candidate" with recommended_env;
    # translate to decision="KEEP" + extra_envs for the promote path.
    micro = str(result.get("micro_decision") or "").strip().lower()
    if micro == "candidate" and result.get("recommended_env"):
        result.setdefault("decision", "KEEP")
        result.setdefault("extra_envs", dict(result["recommended_env"]))
        # Derive best_speedup from tuners_run if not already set.
        if "best_speedup" not in result:
            best = 1.0
            for t in result.get("tuners_run") or []:
                if isinstance(t, dict):
                    sp = float(t.get("best_micro_speedup") or 1.0)
                    if sp > best:
                        best = sp
            if best > 1.0:
                result["best_speedup"] = best
        # Flag that E2E validation is still needed (micro-only).
        result.setdefault("requires_e2e_validation", True)
    elif micro in ("no_improvement", "skipped"):
        result.setdefault("decision", "REVERT")
    elif micro == "failed":
        result.setdefault("decision", "REVERT")
        result.setdefault("status", "failed")

    return result


async def _run_geak_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Legacy GEAK FP8 block-scale GEMM tuning (sglang-only)."""
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
    }
    if geak_config:
        input_payload["config"] = geak_config
    if payload.get("dry_run"):
        input_payload["dry_run"] = True
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("gemm_tuning.py")),
        "--input-json",
        str(input_json),
    ]

    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=_gemm_tuning_timeout_sec(payload))
    result = _shape_tool_result(rc, stdout, stderr)
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
    """Run GEMM tuning via forge-gemm-tune (deterministic) or GEAK (legacy).

    Backend selection:
    1. payload['gemm_tuning_backend']
    2. GEMM_TUNING_BACKEND env var
    3. Default: 'forge'

    Args:
        payload: The GEMM-tuning request payload.
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` describing the tuning outcome.
    """
    backend = _resolve_gemm_tuning_backend(payload)
    log.info("run_gemm_tuning: backend=%s", backend)

    if backend == "forge":
        return await _run_forge_gemm_tuning(payload, session_dir=session_dir)
    return await _run_geak_gemm_tuning(payload, session_dir=session_dir)


async def trace_analyze_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir.

    Args:
        payload (dict): Request payload (see ``Required payload`` /
            ``Optional payload`` below for the recognized keys).
        session_dir (Path): Session root used for resolving inputs and writing
            the analysis outputs.

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
    target_platform = (payload.get("target_platform") or state.gpu_type or "").strip()
    model_name = (payload.get("model_name") or state.model_name or state.model_path or "").strip()
    analysis_mode = (payload.get("analysis_mode") or "").strip()
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    # Load materialized baseline workload metadata once: feeds splitter CLI flags (--split-*) so the steady-state window is correct, and enriches hot_kernels downstream.
    metadata = _load_materialized_workload_metadata(state.baseline_config_path)
    workload = metadata.get("runtime_args", {}).get("workload", {}) if isinstance(metadata, dict) else {}

    cmd = [
        "python3",
        str(_kernel_agent_tool_path("tracelens_analysis.py")),
        "--trace-input",
        str(trace_input),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--top-k",
        str(payload.get("top_k", 10)),
        "--workspace-path",
        workspace_path,
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
        payload.get("capture_folder") or payload.get("graph_capture_path") or payload.get("capture_folder_path")
    )
    if capture_folder:
        cmd += ["--capture-folder", str(capture_folder)]
    # Forward TraceLens splitter steady-state mode (mixed/decode_only/prefilldecode) via payload or env, so the coordinator can re-issue after a steady_state_chunk warning.
    steady_state_mode = payload.get("steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
    steady_state_mode = str(steady_state_mode).strip()
    if steady_state_mode:
        cmd += ["--steady-state-mode", steady_state_mode]
    # Forward the analysis route switch (deterministic vs agent). Coerce to str
    # first (mirrors steady_state_mode) so a non-string payload value (e.g. a
    # bool/list emitted by the LLM) cannot raise AttributeError here.
    analysis_route = (
        str(payload.get("analysis_route") or os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "")).strip().lower()
    )
    if analysis_route in ("deterministic", "agent"):
        cmd += ["--analysis-route", analysis_route]
    # ``--roofline-json`` CLI param retired with the ``pmc_roofline`` action; a stale payload key is silently ignored.
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    _disc_duration_sec = round(time.monotonic() - _disc_started, 3)
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
                result.get("hot_kernels"),
                str(report_path),
            )
        # Surface tracelens/summary.json — the per-run audit sidecar of reusable vs skipped kernels.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
        if isinstance(artifacts, dict) and artifacts.get("kernel_roofline"):
            result["kernel_roofline_path"] = str(artifacts["kernel_roofline"])

        # A failed TraceLens run is a hard failure, not "empty candidates"; keep status=failed and attach a structured warning.
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

        # kernel_journey stage 1 (hot-kernel discovery): additive, best-effort.
        # Records the discovery run + its hot-kernel list + tool provenance so
        # the journey can thread discovery -> dispatch -> backends -> e2e.
        try:
            from ..breakdown.recorder import instrument

            _hot = result.get("hot_kernels_top15") or result.get("hot_kernels") or []
            # Discovery source = the route that actually ran. The tool reports
            # the authoritative mode (``orchestrator_mode``); the deterministic
            # (no-LLM) route is surfaced to the dashboard as ``bypass`` while the
            # LLM route stays ``tracelens``. Fall back to the requested
            # ``analysis_route`` when the tool didn't echo a mode (e.g. early
            # failure). Both routes drive the same TraceLens toolchain, so the
            # version provenance (``tool``) stays ``tracelens`` either way.
            _orch_mode = str(result.get("orchestrator_mode") or "").strip().lower()
            _is_bypass = _orch_mode == "deterministic" or analysis_route == "deterministic"
            _disc_source = "bypass" if _is_bypass else "tracelens"
            instrument.record_kernel_discovery(
                session_dir,
                source=_disc_source,
                tool="tracelens",
                status=str(result.get("status") or ""),
                hot_kernels=_hot if isinstance(_hot, list) else [],
                scan={
                    "splitter_mode": steady_state_mode,
                    "trace_dir": str(trace_input),
                    "candidates_path": str(result.get("candidates_path") or ""),
                    "trace_report_path": str(result.get("trace_report_path") or ""),
                    "analysis_route": _disc_source,
                },
                # tracelens version/commit is read from $TRACELENS_ROOT (its
                # own checkout), resolved by the recorder's tool registry; we
                # don't pin it to the kernel-agent root here.
                duration_sec=_disc_duration_sec,
                error=(str(result.get("error") or "") or None if str(result.get("status") or "") == "failed" else None),
            )
        except Exception:  # noqa: BLE001
            pass
    return result


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
    if candidates_path and not Path(candidates_path).exists():
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
    candidates = _batch_kernel_candidates(payload, session_dir=session_dir)
    if len(candidates) <= 1:
        single_payload = dict(payload)
        if candidates:
            # Reconcile the (often hallucinated) LLM kernel_id against the real candidate id so the CLI doesn't KeyError.
            single_payload["kernel_id"] = _reconcile_kernel_id(
                single_payload.get("kernel_id"),
                candidates,
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
        payload,
        candidates,
        session_dir=session_dir,
        record_partial=record_partial,
    )


def _geak_budget_minutes(payload: dict) -> float:
    """Resolve the per-GEAK-attempt budget in minutes.

    Priority: ``payload['geak_budget_min']`` > ``HYPERLOOM_GEAK_BUDGET_MIN``
    env > the mode-derived default from :func:`_default_geak_budget_minutes`.

    Args:
        payload (dict): Request payload that may carry ``geak_budget_min``.

    Returns:
        float: The GEAK budget in minutes.
    """
    return float(
        payload.get("geak_budget_min") or os.environ.get("HYPERLOOM_GEAK_BUDGET_MIN") or _default_geak_budget_minutes()
    )


def _optimization_budget_minutes(payload: dict) -> float:
    """Wall-clock budget mirrored by the kernel_optimization.py wrapper.

    Picks the OOB budget for Claude/Codex/Cursor, the GEAK budget for GEAK,
    and the max of both for empty/multi-backend payloads (which may still run
    GEAK first in the ladder).

    Args:
        payload (dict): Request payload carrying ``backends`` and optional
            ``budget_minutes`` / GEAK budget hints.

    Returns:
        float: The wall-clock budget in minutes for this optimization.
    """
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
    """Compute the subprocess timeout for the kernel_optimization.py wrapper.

    Converts the optimization budget to seconds and adds a 180s grace window
    so the wrapper can salvage partial artifacts before being killed.

    Args:
        payload (dict): Request payload used to derive the optimization budget.

    Returns:
        int: The subprocess timeout in seconds.
    """
    # +180s grace so kernel_optimization.py can salvage partial artifacts.
    return int(_optimization_budget_minutes(payload) * 60) + 180


def _kernel_ladder_budget_sec(payload: dict) -> int:
    """Total wall-clock budget for one kernel's whole backend ladder.

    The ladder runs its backends sequentially (forge -> geak -> oob fallbacks),
    each as a subprocess with its own timeout. Without a shared ceiling a
    backend that hangs to its hard timeout followed by a fallback running its
    full budget could roughly double a kernel's wall clock and overshoot the
    KERNEL-phase budget cap (which is only re-checked between orchestration
    turns, never mid-``run_optimization``). This budget bounds the whole ladder
    so a fallback only runs within the time left and an exhausted budget exits
    the ladder cleanly. See Hyperloom#602.

    Priority: payload ``kernel_budget_min`` > env
    ``KERNEL_OPT_KERNEL_BUDGET_MIN`` > the single-backend wall-clock budget from
    :func:`_optimization_budget_minutes` (the ladder shares roughly one
    backend's budget). A +180s grace mirrors the per-subprocess wrapper so the
    first backend is never capped below its own timeout.

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
    """Resolve the ordered list of optimization backends to try.

    Precedence (highest to lowest):

    1. ``payload['backend_order']`` – explicit per-request override.
    2. ``KERNEL_OPT_BACKEND_ORDER`` env var – comma-separated list.
    3. ``KERNEL_OPT_BACKENDS`` env var – accepted as an alias for
       ``KERNEL_OPT_BACKEND_ORDER``.
    4. The built-in GEAK-first default ladder.

    All backend names are normalized to lowercase before filtering, so
    values like ``"GEAK"`` or ``"Claude"`` are treated the same as their
    lowercase equivalents.  Unknown backends are silently dropped, and
    ``cursor`` is removed from the auto-derived ladder when
    ``CURSOR_API_KEY`` is unset (explicit orders are respected as-is).

    Args:
        payload (dict): Request payload that may carry ``backend_order``.

    Returns:
        list[str]: The filtered, ordered backend names (subset of
            ``{"claude", "codex", "cursor", "geak"}``).
    """
    raw = (
        payload.get("backend_order")
        or os.environ.get("KERNEL_OPT_BACKEND_ORDER")
        or os.environ.get("KERNEL_OPT_BACKENDS")
    )
    if raw:
        order = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
        explicit = True
    else:
        # Ignore legacy payload["backends"]; the default ladder (GEAK first) mirrors ``kernel_optimization.choose_backends`` so single/batch agree.
        order = list(_DEFAULT_KERNEL_BACKEND_ORDER)
        explicit = False
    # `forge` (Kernel-Forge autonomous-loop backend) is first in
    # _DEFAULT_KERNEL_BACKEND_ORDER; keep it in `allowed` so it survives the
    # filter for both the default and any explicit backend_order.
    allowed = {"claude", "codex", "cursor", "geak", "forge"}
    selected = [backend for backend in order if backend in allowed]
    # Drop cursor from the auto-derived ladder when CURSOR_API_KEY is unset (explicit order still wins).
    if not explicit and not os.environ.get("CURSOR_API_KEY", "").strip():
        selected = [b for b in selected if b != "cursor"]
    return selected


def _in_flight_kernel_ids(session_dir: Path) -> set[str]:
    """Scan the kernel-agent run dir for ``state=running`` status files, so :func:`_batch_kernel_candidates` skips kernels still in flight from a prior batch.

    Args:
        session_dir: Session directory whose kernel-agent run dir is scanned.

    Returns:
        The set of kernel ids currently in flight.
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


def _all_kernel_candidates(payload: dict) -> list[dict[str, Any]]:
    """Load every candidate (``hot_kernels`` ∪ ``skipped_kernels``) so id canonicalization resolves even when hot_kernels is empty.

    Args:
        payload: Request payload carrying ``candidates_path``.

    Returns:
        Every candidate dict from the artifact, or an empty list when the
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
    reusable_ids = data.get("reusable_native_kernel_ids") or []
    reusable_id_set = {str(item) for item in reusable_ids if item}

    # Build the "live" exclusion sets up front for both passes (empty without session_dir).
    rejected_kernel_ids: set[str] = set()
    attempts_by_kid: dict[str, dict] = {}
    in_flight: set[str] = set()
    max_attempts = 1
    try:
        max_attempts = max(
            1,
            int(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_ATTEMPTS",
                    "1",
                )
            ),
        )
    except (TypeError, ValueError):
        max_attempts = 1
    # min_gpu_pct must mirror SharedState.untried_hot_reusable_kernels' 3.0 default so the two layers agree and tiny kernels don't eat ladder wall-clock.
    from .shared_state import _DEFAULT_HOT_KERNEL_MIN_GPU_PCT

    try:
        min_gpu_pct = float(
            os.environ.get(
                "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
            )
        )
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
                "_batch_kernel_candidates: failed to load SharedState from %s; PR-C filters disabled this dispatch",
                session_dir,
            )

    def _is_live(kid: str, current_source: str = "") -> bool:
        """A kernel_id is live (batch-eligible) iff NOT rejected, NOT in-flight, and < max_attempts recorded against the CURRENT source_file (PR-K per-source counting).

        Args:
            kid: The kernel id to test.
            current_source: The current source file for per-source counting.

        Returns:
            ``True`` when the kernel is batch-eligible, else ``False``.
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

    # Collapse kernels sharing a source function into one dispatch via ``task_groups[]`` (keyed off ``primary_kernel_id``); unparseable kernels fall through to the legacy per-kernel pass.
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
            len(selected),
            skipped,
        )
    return selected


def _kernel_result_rank(result: HandlerResult | None) -> tuple[int, float]:
    """Best-selection key shared by the ladder and the batch handler.

    A KEEP verdict always beats a non-KEEP regardless of micro_speedup
    (GEAK frequently reports a higher micro on a NEEDS_REVIEW that has no
    correctness gate, while a Claude/Codex KEEP at a lower micro is a real
    integrate-ready patch); among equals, higher ``micro_speedup`` wins.
    Mirrors the max-key in :func:`_run_optimization_batch` so the ladder,
    the GEAK-vs-OOB race, and the batch all agree on "best".

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
    attempt log. Stops at the first KEEP so a clean GEAK KEEP still
    short-circuits *its own* ladder and OOB fallbacks (claude -> codex ->
    cursor) only fire when an earlier backend misses a KEEP.

    When ``deadline`` (a :func:`time.monotonic` timestamp) is given, the ladder
    enforces the per-kernel budget: each backend's subprocess timeout is capped
    to the time left, and once less than :data:`_KERNEL_LADDER_MIN_BACKEND_SEC`
    remains the ladder stops rather than launching a fallback it cannot finish.
    This keeps a backend that hangs to its hard timeout from letting the
    fallback overshoot the budget. See Hyperloom#602.

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
                # Not enough of the per-kernel budget left to run another
                # backend usefully; stop instead of overshooting it. #602
                log.info(
                    "kernel %s: per-kernel ladder budget exhausted (%.0fs left); "
                    "skipping remaining backends %s",
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


async def _run_kernel_backend_sequence(
    base_payload: dict,
    candidate: dict[str, Any],
    *,
    session_dir: Path,
    parallel_backends: bool = False,
) -> HandlerResult:
    """Optimize one kernel across the backend ladder.

    Two modes:

    * **Sequential (default)** -- the legacy ladder. Walk
      ``_backend_order`` (GEAK first), stopping at the first KEEP. OOB
      (claude/codex/cursor) only runs as a fallback when GEAK misses a
      KEEP.
    * **Parallel (``parallel_backends=True``)** -- GPU-rich mode chosen by
      :func:`_should_parallelize_backends` at the batch layer. Race GEAK
      against the OOB ladder concurrently and keep the stronger result by
      :func:`_kernel_result_rank`, so we no longer short-circuit on GEAK's
      first KEEP when there are spare GPUs to let OOB chase a higher
      speedup. Falls back to sequential when GEAK or every OOB backend is
      absent from the ladder (nothing to race).

    Args:
        base_payload: The base request payload shared by every backend.
        candidate: The kernel candidate to optimize.
        session_dir: Session directory for workspace and state.
        parallel_backends: When ``True``, race GEAK against the OOB ladder.

    Returns:
        The strongest ``HandlerResult`` across the backends tried.
    """
    kernel_id = str(candidate.get("kernel_id") or base_payload.get("kernel_id") or "")
    order = _backend_order(base_payload)

    # Bound the whole ladder (all backends for this kernel) to one wall-clock
    # budget so a backend that hangs to its hard timeout cannot let the
    # fallback double the kernel's wall clock and overshoot the KERNEL-phase
    # cap. Each ladder call caps its backends to the time left. See #602.
    ladder_deadline = time.monotonic() + _kernel_ladder_budget_sec(base_payload)

    # Forge edits the live repo in-place (temp branch + per-file restore) and
    # must NOT race with other backends that read/write the same repo. Run it
    # sequentially first; if it KEEPs, short-circuit. Otherwise continue with
    # the geak / oob parallel split on the remaining backends.
    forge_group = [b for b in order if b == "forge"]
    remaining = [b for b in order if b != "forge"]
    forge_best: dict | None = None
    forge_attempts: list = []
    best: dict | None = None
    attempts: list = []
    if forge_group:
        forge_best, forge_attempts = await _run_backend_ladder(
            base_payload,
            candidate,
            kernel_id,
            forge_group,
            session_dir=session_dir,
            deadline=ladder_deadline,
        )
        if forge_best and _kernel_result_rank(forge_best)[0] > 0:
            best = forge_best
            attempts = forge_attempts

    geak_group = [b for b in remaining if b == "geak"]
    oob_group = [b for b in remaining if b != "geak"]

    if best is not None:
        pass
    elif parallel_backends and geak_group and oob_group:
        (geak_best, geak_attempts), (oob_best, oob_attempts) = await asyncio.gather(
            _run_backend_ladder(
                base_payload,
                candidate,
                kernel_id,
                geak_group,
                session_dir=session_dir,
                deadline=ladder_deadline,
            ),
            _run_backend_ladder(
                base_payload,
                candidate,
                kernel_id,
                oob_group,
                session_dir=session_dir,
                deadline=ladder_deadline,
            ),
        )
        attempts = forge_attempts + geak_attempts + oob_attempts
        best = max(
            (r for r in (geak_best, oob_best) if r is not None),
            key=_kernel_result_rank,
            default=None,
        )
    else:
        best, attempts = await _run_backend_ladder(
            base_payload,
            candidate,
            kernel_id,
            remaining,
            session_dir=session_dir,
            deadline=ladder_deadline,
        )
        attempts = forge_attempts + attempts
        if best is None and forge_best is not None:
            best = forge_best

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
    # Forge edits framework sources in-place. The per-repo lock protects one
    # forge run, but if multiple kernels are processed concurrently a second
    # kernel can miss the lock, skip forge, and race another backend against the
    # first kernel's live-tree edits. Keep the whole kernel batch serial whenever
    # forge is in the backend ladder.
    if "forge" in _backend_order(payload):
        max_parallel = 1
    # GPU-rich mode: when the node can fit a kernel's GEAK + OOB ladder
    # side-by-side (see :func:`_should_parallelize_backends`), race them per
    # kernel and keep the stronger result instead of short-circuiting on
    # GEAK's first KEEP.
    parallel_backends = _should_parallelize_backends(payload, len(candidates))
    # Each parallel-backends kernel launches TWO before_kernel_opt rocprof
    # subprocesses (one per ladder) *before* entering Ray, so they are NOT
    # bound by the Ray GPU lease. Cap concurrent kernels to
    # ``visible_gpus // (2 * per_task)`` so those pre-Ray profilers (and the
    # 2 * per_task Ray tasks that follow) stay within the real GPU budget
    # instead of overcommitting it.
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
                # Wrap a sub-task failure as a structured result so gather stays wait-all (a raised exception would unblock mid-batch and collide with running siblings on the GPU).
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
    return out


def _backends_cli_arg(value: Any) -> str:
    """Normalize a payload ``backends`` field into a bare ``--backends`` value.

    The orchestration payload may carry ``backends`` as a bare string
    (``"geak"``), a comma-joined string (``"geak,claude"``), or a JSON list
    (``["geak"]``) when an upstream request serializes it as an array. A list
    MUST be comma-joined into bare names, never ``str()``-ed into the repr of a
    list (``"['geak']"``) — the kernel-agent's ``parse_backends`` validator
    correctly rejects that opaque token and the dispatch fails with the
    self-contradictory "unsupported backend(s): ['geak']". See Hyperloom#601.

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
    guard = _validate_reusable_native_kernel(payload)
    if guard is not None:
        return guard
    shape_guard = _validate_kernel_shape_and_paths(
        payload,
        session_dir=session_dir,
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
    backends_arg = _backends_cli_arg(payload.get("backends"))
    if backends_arg:
        cmd += ["--backends", backends_arg]
    if payload.get("source_file"):
        cmd += ["--source-file", str(payload["source_file"])]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    extra_args = str(payload.get("extra_server_args") or payload.get("extra_sglang_args") or "").strip()
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
    backend = backends_arg.lower()
    if backend == "geak" or not backend:
        cmd += ["--geak-budget-min", str(geak_budget_min)]
    if payload.get("budget_minutes") is not None:
        cmd += ["--budget-minutes", str(payload["budget_minutes"])]
    # Allow the tool to handle its own backend timeout and salvage partial artifacts.
    timeout_sec = _optimization_wrapper_timeout_sec(payload)
    if timeout_override_sec is not None:
        # The backend ladder caps each subprocess to the time left in the
        # per-kernel budget so a fallback never overshoots it. See Hyperloom#602.
        timeout_sec = max(1, min(timeout_sec, int(timeout_override_sec)))

    from .action_executors._multi_node_env import is_multi_node

    if is_multi_node():
        from inference_optimizer.multi_node.cli import (
            kill_inference_for_kernel_agent_best_effort,
        )

        await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)

    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # The kernel-agent subprocess overran the hard outer timeout. Shape a
        # failed result here instead of letting TimeoutExpired propagate to the
        # batch wrapper — that wrapper produces a backend-less result, so the
        # failure was silently bucketed as a GEAK invocation even when a
        # different optimizer (e.g. claude) actually ran. See Hyperloom#602.
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
        }
    # Stamp source_file / kernel_id from the payload onto the result so the multi-KEEP integrate queue can group same-file KEEPs (the tool may omit them on timeout/crash).
    if isinstance(result, dict):
        if not result.get("kernel_id") and payload.get("kernel_id"):
            result["kernel_id"] = str(payload["kernel_id"])
        if not result.get("source_file") and payload.get("source_file"):
            result["source_file"] = str(payload["source_file"])
        # Attribute a result that carries no per-backend attempt ladder
        # (pre-dispatch / infra / timeout) to the backend that actually ran, so
        # downstream recorders never fall back to a silent GEAK default. Only
        # when this run dispatched a single, unambiguous backend. See #602.
        if (
            backend
            and "," not in backend
            and not result.get("backend")
            and not result.get("attempts")
        ):
            result["backend"] = backend
    # Full-trace: mine each geak/oob/forge attempt's stdout log for token usage
    # and append an ``llm_calls.jsonl`` row. Best-effort; a no-op when the
    # backend emits no usage block (claude/codex/cursor account spend elsewhere).
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    # Full-trace: record each forge attempt's key-step timeline (rationale /
    # validation / keep-revert + summary) as a forge_steps audit, backfilled
    # into the trace as forge:* spans. Best-effort; no-op without a step marker.
    _trace_kernel_attempt_steps(result, session_dir=session_dir)
    return result


def _trace_kernel_attempt_usage(
    result: Any,
    *,
    session_dir: Path,
) -> None:
    """Append ``llm_calls.jsonl`` rows for geak/oob attempts in ``result``.

    Each ``kernel_optimization`` attempt record carries ``backend`` plus
    ``optimized_path`` (the backend's full ``*_stdout.log``). For the
    token-traced backends (:data:`_TOKEN_TRACED_KERNEL_BACKENDS`) we read that
    log and run the matching usage parser (``geak`` → :func:`parse_geak_usage`,
    ``oob`` → :func:`parse_oob_json_usage`, ``forge`` →
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
            if backend == "geak":
                usage = parse_geak_usage(stdout_text)
            elif backend == "forge":
                usage = parse_forge_usage(stdout_text)
            else:
                usage = parse_oob_json_usage(stdout_text)
            if not usage:
                continue
            record = LLMCallRecord(
                session_id=session_dir.name,
                component=backend,
                task_id=kernel_id,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
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
    result: Any, *, session_dir: Path,
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
    from ..session_paths import forge_steps_path
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
            rows.append({
                "kernel_id": kernel_id, "kind": "iteration", "ts": ts, **step,
            })
        summary = payload.get("summary")
        if isinstance(summary, dict):
            rows.append({
                "kernel_id": kernel_id, "kind": "summary", "ts": ts, **summary,
            })
    if not rows:
        return
    try:
        path = forge_steps_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
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
def _record_kernel_roofline_sidecar(session_dir: Path) -> None:
    """Transcribe ``reports/kernel_roofline.json`` (written by the external
    kernel-agent tool) into the breakdown recorder as a ``kernel_roofline``
    singleton. Best-effort; never raises.

    Args:
        session_dir: Session directory holding the ``reports/kernel_roofline.json``
            sidecar to transcribe.
    """
    try:
        sidecar_path = Path(session_dir) / "reports" / "kernel_roofline.json"
        if not sidecar_path.is_file():
            return
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            return
        from ..breakdown.recorder import instrument

        instrument.record_singleton_section(
            session_dir,
            "kernel_roofline",
            payload,
            producer="kernel-agent",
        )
    except Exception:  # noqa: BLE001
        pass


def _lookup_kernel_roofline_name(session_dir: Path, kernel_id: str) -> str:
    """Resolve the TraceLens/device kernel name for a roofline sidecar row.

    Args:
        session_dir: Session directory holding the roofline sidecar.
        kernel_id: The kernel id to look up.

    Returns:
        The matched kernel name, or ``""`` when the sidecar/row is absent.
    """
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
    """Best-effort sidecar status update for skipped/failed after-opt rocprof.

    Args:
        session_dir: Session directory holding the roofline sidecar.
        kernel_id: The kernel id whose sidecar row is updated.
        status: The rocprof status to record.
        reason: Optional human-readable reason for the status.
        json_path: Optional path to the rocprof JSON artifact.
        txt_path: Optional path to the rocprof text artifact.
        log: Optional logger for warnings on failure.
    """
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
    """Resolve the rocprof roofline subprocess timeout in seconds.

    Reads ``HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC`` and clamps it to a
    minimum of 60 seconds.

    Returns:
        The timeout in seconds (defaults to 1800).
    """
    try:
        return max(60, int(os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800")))
    except (TypeError, ValueError):
        return 1800


def _rocprof_profile_command(test_command: str) -> str:
    """Rewrite a test command to run in rocprof profiling mode.

    Swaps a ``--correctness`` flag for ``--profile`` only when the command
    targets a recognized unittest harness; otherwise returns it unchanged.

    Args:
        test_command: The original kernel test command.

    Returns:
        The (possibly rewritten) command string.
    """
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

    Args:
        kernel_id: The kernel id that was just integrated.
        session_dir: Session directory for state and artifacts.
        log: Logger for status/warning messages.

    Returns:
        A small status dict describing the rocprof outcome.
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
        "python3",
        str(tool),
        "--workdir",
        str(run_workdir),
        "--cmd",
        profiling_command,
        "--out-json",
        str(out_json),
        "--out-txt",
        str(out_txt),
        "--timeout-sec",
        str(timeout_sec),
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
        # Author-time breakdown capture: transcribe the external tool's sidecar
        # (reports/kernel_roofline.json) into the recorder right after it lands.
        _record_kernel_roofline_sidecar(session_dir)
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
    """Schedule a background rocprof roofline run after a kernel integrate.

    Honors ``HYPERLOOM_ROCPROF_ROOFLINE`` to disable profiling; otherwise
    records a ``scheduled`` status and launches the run as a tracked
    background task.

    Args:
        kernel_id: Identifier of the integrated kernel.
        session_dir: Session directory for status sidecars.
        log: Logger for status and error reporting.

    Returns:
        A status dict indicating whether the run was scheduled or skipped.
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

    _record_after_kernel_opt_rocprof_status(
        session_dir=session_dir,
        kernel_id=kernel_id,
        status="scheduled",
        reason="background_task",
        log=log,
    )
    task = asyncio.create_task(
        _run_after_kernel_opt_rocprof(
            kernel_id=kernel_id,
            session_dir=session_dir,
            log=log,
        )
    )
    _BACKGROUND_ROCPROF_TASKS.add(task)

    def _done(done_task: asyncio.Task[Any]) -> None:
        """Completion callback that drops the task and logs failures.

        Args:
            done_task: The finished background rocprof task.
        """
        _BACKGROUND_ROCPROF_TASKS.discard(done_task)
        try:
            done_task.result()
        except Exception as exc:  # noqa: BLE001 — best-effort background task
            log.warning("integrate: after_kernel_opt rocprof background failed: %s", exc)

    task.add_done_callback(_done)
    return {"status": "scheduled", "reason": "background_task"}


async def integrate_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Applies an optimized kernel artifact, re-runs the active Magpie baseline,
    and KEEPs only when measured E2E throughput clears the threshold (source +
    artifacts are backed up first so non-KEEP can restore without a rebuild).

    Required payload: ``base_tput``. Optional: patch_path, target_file,
    kernel_id, config_path, extra_server_args, keep_threshold_pct (1.0),
    budget_minutes (20). Returns ``{status, decision, base_tput, new_tput,
    gain_pct, kernel_id, patch_path, report_path, workspace}``.

    Args:
        payload: The integrate request payload (requires ``base_tput``).
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` with the KEEP/REVERT decision and re-baseline
        metrics (``status``, ``decision``, ``base_tput``, ``new_tput``,
        ``gain_pct``, ``kernel_id``, ``patch_path``, ``report_path``,
        ``workspace``).
    """
    from .action_executors.baseline import BaselineExecutor
    from .action_executors.benchmark_result import is_valid_measurement
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
        payload,
        session_dir=session_dir,
    )
    if missing_inputs is not None:
        return missing_inputs

    patch_path = payload.get("patch_path")
    kernel_id = payload.get("kernel_id")
    apply_result = _maybe_apply_kernel_patch(
        payload,
        session_dir=session_dir,
        kernel_id=kernel_id,
    )
    log.info("integrate_handler: apply_result=%s", apply_result)
    if apply_result.get("status") == "failed":
        # Apply crash: the patch was never measured. Stamp a fault error_class
        # (top-level, not just nested in apply_result) so SharedState routes
        # this through the fault retry budget instead of the REVERT quota.
        return {
            "status": "failed",
            "error_class": "apply_failed",
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
            "output_dir": str(workspace),
            "timeout_sec": int(payload.get("budget_minutes", 20)) * 60,
            "extra_server_args": extra_args,
            "extra_envs": dict(payload.get("extra_envs") or {}),
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
        """Restore the ``AITER_REBUILD`` env var to its prior value.

        No-op unless a forced rebuild was applied for this re-baseline;
        otherwise pops or restores the original value (GH #458).
        """
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
        # The re-baseline server crashed / timed out / produced no usable
        # measurement, so the patch was never fairly scored. Surface a fault
        # error_class at the top level — propagating the re-baseline's own
        # error_class when present (e.g. subprocess_timeout) and otherwise
        # defaulting to bench_exception — so this routes through the fault
        # retry budget rather than being discarded as a genuine REVERT.
        rebaseline_error_class = (
            str((bench_result or {}).get("error_class") or "").strip() if isinstance(bench_result, dict) else ""
        ) or "bench_exception"
        return {
            "status": "failed",
            "error_class": rebaseline_error_class,
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
    # base_tput > 0 already guaranteed by the early guard above.
    gain_pct = (new_tput - base_tput) / base_tput * 100.0
    stack_positive_keep = False
    stack_incremental_gain_pct: float | None = None
    try:
        from .shared_state import SharedState

        state = SharedState.load_or_init(session_dir)
        current_best = state.current_best or {}
        current_best_tput = float(current_best.get("tput") or 0.0)
        if current_best_tput > 0:
            stack_incremental_gain_pct = (new_tput - current_best_tput) / current_best_tput * 100.0
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
        "status": "ok",
        "decision": decision,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "base_tput": base_tput,
        "new_tput": new_tput,
        "gain_pct": gain_pct,
        "report_path": bench_result.get("report_path"),
        "workspace": bench_result.get("workspace"),
        "extra_server_args": extra_args,
        "apply_result": apply_result,
        "revert_result": revert_result,
        "rebuild_check": rebuild_check,
    }
    if stack_positive_keep and gain_pct <= keep_threshold_pct:
        result["decision_reason"] = "stack_positive_increment"
        result["stack_incremental_gain_pct"] = stack_incremental_gain_pct
        result["stack_incremental_keep_threshold_pct"] = STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
    if rocprof_after_info:
        result["rocprof_after_kernel_opt"] = rocprof_after_info
    return result


# Kernel-agent programmatic dispatch table (LLM-driven requests routed via ``Coordinator._handle_request``).
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "trace_analyze": trace_analyze_handler,
    "run_gemm_tuning": run_gemm_tuning_handler,
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


# ===========================================================================
# Kernel-decision write-owner functions (folded back from the former
# shared_state_kernel.py satellite; phase 6C). SharedState is a passive
# persisted record; the functions that *own kernel decisions* (recording
# kernel-opt / integrate / gemm-tuning outcomes, kernel-patch identity,
# pending-keep bookkeeping, hot-kernel reuse) belong to this kernel domain.
# They take ``state`` as their first argument and read/mutate it exactly as
# the original SharedState methods did; SharedState keeps thin forwarding
# shims so existing callers (``state.record_kernel_opt(...)`` etc.) and the
# ~54 tests that hit them are unchanged.
#
# The shared_state default constants are imported lazily inside the functions
# that need them: ``kernel_request_handlers`` has no module-level shared_state
# import (shared_state imports only stdlib), so a function-local import keeps
# the dependency one-way and cycle-free.
# ===========================================================================
def _format_last_kernel_opt(state) -> str:
    """Single-line repr of last kernel-opt outcome for prompt injection.

    Returns:
        str: A compact ``kernel_id=... decision=... speedup=...`` line
            (with optional per-kernel attempts/retired history), or
            ``"(none)"`` when no kernel_opt has run.
    """
    if not state.last_kernel_opt:
        return "(none)"
    ko = state.last_kernel_opt
    kid = str(ko.get("kernel_id") or "")
    attempts_entry = state.kernel_opt_attempts.get(kid) or {}
    history_tag = ""
    if attempts_entry:
        history_tag = (
            f" history=attempts={attempts_entry.get('attempts', 0)}"
            f"/partial={attempts_entry.get('partial_count', 0)}"
        )
        rej_reason = attempts_entry.get("rejected_reason")
        if rej_reason:
            history_tag += f"/retired={rej_reason}"
    return (
        f"kernel_id={kid or '?'} "
        f"decision={ko.get('decision','?')} "
        f"speedup={ko.get('micro_speedup','?')}"
        f"{history_tag}"
    )


def _resolve_kernel_patch_identity(
    state, payload: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    """Resolve a kernel patch's identity tuple from a result/intent payload.

    Pulls ``kernel_id`` / patch path / target file / extra server args
    from the envelope, back-filling the patch path from
    :attr:`last_kernel_opt` when the payload omits it but names a
    matching kernel. The legacy ``extra_sglang_args`` alias is resolved
    through the compat helper.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or LLM
            intent envelope (``None`` treated as empty).

    Returns:
        tuple[str, str, str, str]: ``(kernel_id, patch_path,
            target_file, extra_args)``; any unresolved component is an
            empty string.
    """
    payload = payload or {}
    kernel_id = str(payload.get("kernel_id") or "")
    patch_path = str(
        payload.get("patch_path")
        or payload.get("best_artifact_path")
        or ""
    )
    if (
        not patch_path
        and kernel_id
        and str((state.last_kernel_opt or {}).get("kernel_id") or "") == kernel_id
    ):
        patch_path = str(
            (state.last_kernel_opt or {}).get("best_artifact_path")
            or (state.last_kernel_opt or {}).get("patch_path")
            or ""
        )
    target_file = str(
        payload.get("target_file")
        or payload.get("source_file")
        or ""
    )
    # External envelope; route through compat helper so legacy ``extra_sglang_args`` still resolves.
    from ..compat.payload_aliases import read_extra_server_args
    extra_args = read_extra_server_args(payload).strip()
    return kernel_id, patch_path, target_file, extra_args


def kernel_patch_key(state, payload: dict[str, Any] | None) -> str:
    """Compute the dedup key for a kernel patch.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope.

    Returns:
        str: ``"<kernel_id>|<patch_path>|<extra_args>"``, or ``""`` when
            either ``kernel_id`` or ``patch_path`` cannot be resolved.
    """
    kernel_id, patch_path, _target_file, extra_args = (
        _resolve_kernel_patch_identity(state, payload)
    )
    if not kernel_id or not patch_path:
        return ""
    return "|".join([kernel_id, patch_path, extra_args])


def find_rejected_kernel_patch(
    state,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Look up a previously-rejected patch matching ``payload``.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope identifying the patch.

    Returns:
        dict[str, Any] | None: The matching rejected-patch entry, or
            ``None`` when the key is unresolvable or not on record.
    """
    key = kernel_patch_key(state, payload)
    if not key:
        return None
    for entry in state.rejected_kernel_patches:
        if isinstance(entry, dict) and entry.get("key") == key:
            return entry
    return None


def record_kernel_integrate_result(
    state,
    result: dict[str, Any],
    *,
    max_attempts: int = 3,
    keep_threshold_pct: float = 1.0,
    max_fault_attempts: int | None = None,
) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts.

    Appends the attempt to the per-key ``kernel_integrate_attempts``
    ledger. Two terminal paths are kept distinct:

    * **Gate verdict** — a genuine REVERT (gain below threshold / accuracy
      regression), or ``max_attempts`` non-fault attempts without a KEEP,
      moves the patch into ``rejected_kernel_patches`` and records its
      ``kernel_id`` in ``rejected_kernel_ids`` (terminal).
    * **Integration fault** — an environment/apply/bench crash (see
      :meth:`SharedState._is_integrate_fault`) that never fairly measured the
      patch. Faults do *not* consume the REVERT quota; they get an independent
      ``max_fault_attempts`` budget and are marked ``retryable`` so the
      pending-integrate driver re-enqueues them, only being rejected once
      that fault budget is exhausted.

    Args:
        result (dict[str, Any]): The integrate E2E result envelope.
        max_attempts (int): Max non-fault attempts before rejecting a
            non-KEEP patch (default 3).
        keep_threshold_pct (float): The gain threshold recorded on the
            rejection row for context (default 1.0).
        max_fault_attempts (int): Independent budget for total integration-
            fault attempts (initial + retries) before they are rejected as
            ``fault_attempts_exhausted`` (default 2 = one retry).

    Returns:
        dict[str, Any] | None: The updated attempts entry (carrying a
            ``rejected`` sub-dict when rejection fired, or
            ``retryable=True`` for an un-exhausted fault), or ``None`` when
            ``result`` is not a dict or its patch key is unresolvable.
    """
    from .shared_state import _MAX_INTEGRATE_FAULT_ATTEMPTS, _now_iso

    if max_fault_attempts is None:
        max_fault_attempts = _MAX_INTEGRATE_FAULT_ATTEMPTS

    if not isinstance(result, dict):
        return None
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = (
        _resolve_kernel_patch_identity(state, result)
    )
    is_fault = state._is_integrate_fault(result)
    entry = dict(state.kernel_integrate_attempts.get(key) or {})
    attempts = list(entry.get("attempts") or [])
    attempt = {
        "decision": result.get("decision"),
        "status": result.get("status"),
        "error_class": result.get("error_class"),
        "is_fault": is_fault,
        "new_tput": result.get("new_tput"),
        "gain_pct": result.get("gain_pct"),
        "workspace": result.get("workspace"),
        "report_path": result.get("report_path"),
        "ts": _now_iso(),
    }
    attempts.append(attempt)
    best_gain = max(
        (
            float(a.get("gain_pct"))
            for a in attempts
            if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
        ),
        default=0.0,
    )
    # Quota accounting: faults and gate verdicts draw from separate budgets.
    fault_count = sum(
        1 for a in attempts if isinstance(a, dict) and a.get("is_fault")
    )
    verdict_attempt_count = len(attempts) - fault_count
    entry.update({
        "key": key,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "fault_count": fault_count,
        "verdict_attempt_count": verdict_attempt_count,
        "best_gain_pct": best_gain,
        "last_decision": result.get("decision"),
        "last_status": result.get("status"),
        "last_error_class": result.get("error_class"),
        "last_was_fault": is_fault,
        "updated_at": _now_iso(),
    })
    # Clear any stale retryable flag; re-set below only for un-exhausted faults.
    entry.pop("retryable", None)
    state.kernel_integrate_attempts[key] = entry

    # kernel_journey stage 4 (end-to-end integrate outcome): additive,
    # idempotent per kernel_id, best-effort.
    try:
        from ..breakdown.recorder import instrument
        sdir = getattr(state, "_session_dir", None)
        if sdir and kernel_id:
            _dec = str(result.get("decision") or "").upper()
            instrument.record_kernel_e2e(
                sdir,
                kernel_id=kernel_id,
                integrated=(_dec == "KEEP"),
                e2e_gain_pct=result.get("gain_pct"),
                validated=True if _dec == "KEEP" else None,
                decision=_dec,
                patch_path=patch_path,
                target_file=target_file,
                extra_server_args=extra_args,
            )
    except Exception:  # noqa: BLE001
        pass

    if result.get("decision") == "KEEP":
        return entry

    # Integration fault: never measured fairly. Don't burn the REVERT quota
    # — retry on its own budget and let the pending-integrate driver pick it
    # back up, only rejecting once the fault budget is exhausted.
    if is_fault:
        if fault_count < max_fault_attempts:
            entry["retryable"] = True
            state.kernel_integrate_attempts[key] = entry
            return entry
        reason = f"fault_attempts_exhausted_{max_fault_attempts}"
    else:
        # Gate verdict path: a genuine REVERT, or too many non-fault attempts
        # without a KEEP. Faults never count toward this quota.
        should_reject = (
            result.get("decision") == "REVERT"
            or verdict_attempt_count >= max_attempts
        )
        if not should_reject:
            return entry
        reason = (
            "revert_decision"
            if result.get("decision") == "REVERT"
            else f"max_e2e_attempts_{max_attempts}_without_keep"
        )
    rejected = {
        "key": key,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempt_count": len(attempts),
        "fault_count": fault_count,
        "best_gain_pct": best_gain,
        "keep_threshold_pct": keep_threshold_pct,
        "last_decision": result.get("decision"),
        "last_error_class": result.get("error_class"),
        "reason": reason,
        "ts": _now_iso(),
    }
    state.rejected_kernel_patches = [
        r for r in state.rejected_kernel_patches
        if not (isinstance(r, dict) and r.get("key") == key)
    ]
    state.rejected_kernel_patches.append(rejected)
    if kernel_id and kernel_id not in state.rejected_kernel_ids:
        state.rejected_kernel_ids.append(kernel_id)
    entry["rejected"] = rejected
    state.kernel_integrate_attempts[key] = entry
    return entry


def record_kernel_opt(state, result: dict[str, Any]) -> None:
    """Capture kernel_optimization_handler result for the next Orch turn; empty kernel_id no-op, non-KEEP can't overwrite a pending KEEP, retires kernel_id (r24 guard) after >= max_partial PARTIALs (INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL).

    Args:
        result (dict[str, Any]): The kernel_optimization_handler result
            envelope; non-dicts and empty ``kernel_id`` are no-ops.
    """
    from .shared_state import (
        _DEFAULT_KERNEL_OPT_MAX_FAILURES,
        _DEFAULT_KERNEL_OPT_MAX_PARTIAL,
        _now_iso,
    )

    if not isinstance(result, dict):
        return
    # Author-time breakdown capture: record geak/oob invocations (incl.
    # backend + pre-dispatch failures) before the metadata-less early
    # return so no failed attempt becomes invisible in the geak/oob view.
    try:
        from ..breakdown.recorder import instrument
        sdir = getattr(state, "_session_dir", None)
        instrument.record_kernel_invocations(sdir, result)
        # kernel_journey stage 2 (dispatch) + stage 3 (backend attempts):
        # additive, never overlaps the legacy geak/oob view above.
        _kid = str(result.get("kernel_id") or "")
        if sdir and _kid:
            _attempts = result.get("attempts")
            _attempts = _attempts if isinstance(_attempts, list) else []
            _backends = []
            for _a in _attempts:
                if isinstance(_a, dict):
                    _b = str(_a.get("backend") or "").lower()
                    if _b and _b not in _backends:
                        _backends.append(_b)
            if not _backends:
                _sel = result.get("selected_backends") or result.get("backends")
                if isinstance(_sel, list):
                    _backends = [str(b).lower() for b in _sel if b]
            # A backend that failed before dispatching attempts (e.g. geak
            # rejecting an empty/non-reusable kernel shape) still counts as
            # dispatched: the backend was invoked. Mirror the failure-detect
            # used by record_kernel_backend_result so the synthetic FAILED
            # attempt and the dispatch flag stay consistent.
            _status = str(result.get("status") or "").lower()
            _err_class = str(result.get("error_class") or "")
            _decision = str(
                (result.get("proposal") or {}).get("decision") or "").upper()
            _failed_predispatch = (not _attempts) and (
                _status in {"failed", "error", "crashed", "timeout"}
                or (_decision == "REVERT" and bool(_err_class))
            )
            if _failed_predispatch and not _backends:
                # Never default an unattributable failure to GEAK — the backend
                # that ran is stamped on the result upstream when known; only a
                # genuine pre-dispatch gating failure (no backend launched)
                # falls through, and "unknown" reflects that honestly. #602
                _b = str(result.get("backend") or "").lower() or "unknown"
                _backends = [_b]
            _dispatched = bool(_attempts) or _failed_predispatch
            instrument.record_kernel_dispatch(
                sdir,
                kernel_id=_kid,
                dispatched=_dispatched,
                backends=_backends,
                skip_reason="" if _dispatched else str(
                    result.get("error_class") or result.get("status") or ""),
                orchestration_commit=str(getattr(state, "code_revision", "") or ""),
            )
            instrument.record_kernel_backend_result(sdir, result)
    except Exception:  # noqa: BLE001
        pass
    kernel_id = str(result.get("kernel_id") or "")
    if not kernel_id:
        # Metadata-less failure: preserve prior streaming-record KEEP.
        return

    verification = result.get("verification") or {}
    proposal = result.get("proposal") or {}
    decision = str(proposal.get("decision", "")).upper()
    micro_speedup = verification.get("micro_speedup", 0.0)
    try:
        micro_float = float(micro_speedup)
    except (TypeError, ValueError):
        micro_float = 0.0
    best_artifact_path = str(verification.get("best_artifact_path", "") or "")
    source_file = str(
        result.get("source_file")
        or (result.get("candidate") or {}).get("source_file")
        or ""
    )
    # Extract test_command from the first attempt that recorded one so
    # after_kernel_opt rocprof can reuse it without re-deriving from scratch.
    test_command = ""
    for _attempt in (result.get("attempts") or []):
        if isinstance(_attempt, dict):
            _tc = str((_attempt.get("backend_paths") or {}).get("test_command") or "").strip()
            if _tc:
                test_command = _tc
                break
    status = str(result.get("status") or "").lower()
    err_class = str(result.get("error_class") or "")
    # Pure infra failure = backend ladder with no verdict; kept distinct from REVERT/PARTIAL so retirement counters don't double-count.
    is_infra_failure = (
        decision == ""
        and (
            status in {"failed", "error", "timeout"}
            or err_class in {
                "subtask_exception",
                "handler_exception",
                "subprocess_timeout",
                "kernel_agent_root_missing",
                "missing_integration_inputs",
            }
        )
    )
    ts = _now_iso()

    entry = dict(state.kernel_opt_attempts.get(kernel_id) or {})
    history = list(entry.get("history") or [])
    history.append({
        "decision": decision, "micro": micro_float,
        "status": status, "ts": ts,
    })
    history = history[-10:]
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    # Per-source attempts so a Python wrapper and its device file don't share a retry quota.
    per_source = dict(entry.get("attempts_per_source") or {})
    src_key = source_file or ""
    per_source[src_key] = int(per_source.get(src_key, 0)) + 1
    entry["attempts_per_source"] = per_source
    if decision == "PARTIAL":
        entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
    elif decision == "KEEP":
        # Success resets streaks so a future regression isn't auto-retired on stale history.
        entry["partial_count"] = 0
        entry["failure_count"] = 0
    if is_infra_failure:
        entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
    entry["last_decision"] = decision
    entry["last_status"] = status
    entry["last_micro_speedup"] = micro_float
    entry["last_artifact_path"] = best_artifact_path
    entry["last_source_file"] = source_file
    entry["last_ts"] = ts
    entry["history"] = history
    if test_command:
        entry["test_command"] = test_command

    # last_kernel_opt overwrite policy: KEEP always wins; non-KEEP writes only when no pending KEEP to protect.
    prev = state.last_kernel_opt or {}
    prev_decision = str(prev.get("decision", "")).upper()
    prev_kid = str(prev.get("kernel_id", ""))
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    prev_pending = (
        prev_decision == "KEEP"
        and bool(prev_kid)
        and prev_kid not in (state.rejected_kernel_ids or [])
        and prev_kid not in integrated_ids
    )
    if decision == "KEEP" or not prev_pending:
        state.last_kernel_opt = {
            "kernel_id": kernel_id,
            "decision": decision,
            "reasons": proposal.get("reasons", []),
            "micro_speedup": micro_float,
            "compile_passed": verification.get("compile_passed"),
            "correctness_passed": verification.get("correctness_passed"),
            "best_artifact_path": best_artifact_path,
            "source_file": source_file,
            "ts": ts,
        }

    max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
    env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
    if env_v:
        try:
            max_partial = max(1, int(env_v))
        except (TypeError, ValueError):
            # Malformed env override → keep the default partial-attempt cap.
            pass

    # One backend ladder without a KEEP retires the kernel by default; raise threshold for flaky backends.
    max_failures = _DEFAULT_KERNEL_OPT_MAX_FAILURES
    env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
    if env_f:
        try:
            max_failures = max(1, int(env_f))
        except (TypeError, ValueError):
            # Malformed env override → keep the default failure cap.
            pass

    should_reject = (
        decision == "REVERT"
        or int(entry.get("partial_count", 0)) >= max_partial
        or int(entry.get("failure_count", 0)) >= max_failures
    )
    if should_reject:
        if kernel_id not in state.rejected_kernel_ids:
            state.rejected_kernel_ids.append(kernel_id)
        entry["rejected_reason"] = (
            "revert_decision"
            if decision == "REVERT"
            else (
                f"max_partial_attempts_{max_partial}_without_keep"
                if int(entry.get("partial_count", 0)) >= max_partial
                else f"max_failures_{max_failures}_without_keep"
            )
        )

    state.kernel_opt_attempts[kernel_id] = entry


def record_gemm_tuning(state, result: dict[str, Any]) -> None:
    """Capture the GEAK GEMM tuning result for sequencing and prompts.

    Snapshots the result into ``last_gemm_tuning`` and appends it to the
    capped ``gemm_tuning_attempts`` history. A non-dict result is
    normalized into a failure record.

    Args:
        result (dict[str, Any]): The GEMM tuning result envelope.
    """
    from .shared_state import _DEFAULT_ATTEMPTS_HISTORY, _now_iso

    if not isinstance(result, dict):
        result = {"status": "failed", "error": "non-dict gemm tuning result"}
    entry = dict(result)
    entry.setdefault("ts", _now_iso())
    state.last_gemm_tuning = entry
    attempts = list(state.gemm_tuning_attempts or [])
    attempts.append(entry)
    state.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]


def _kernel_ids_in_optimization_stack(state) -> set[str]:
    """kernel_ids already absorbed into optimization_stack as integrate entries.

    Returns:
        set[str]: The set of ``kernel_id`` values that appear on an
            ``integrate`` entry of :attr:`optimization_stack`.
    """
    return {
        str(e.get("kernel_id"))
        for e in (state.optimization_stack or [])
        if isinstance(e, dict)
        and e.get("action") == "integrate"
        and e.get("kernel_id")
    }


def _source_files_in_optimization_stack(state) -> set[str]:
    """source_file paths already touched by an integrate entry; enforces "same source_file, only strongest KEEP integrated" (apply_kernel_patch is a whole-file overwrite).

    Returns:
        set[str]: The set of ``target_file`` / ``source_file`` paths
            referenced by ``integrate`` entries of
            :attr:`optimization_stack`.
    """
    sources: set[str] = set()
    for e in (state.optimization_stack or []):
        if not isinstance(e, dict) or e.get("action") != "integrate":
            continue
        src = str(e.get("target_file") or e.get("source_file") or "")
        if src:
            sources.add(src)
    return sources


def _kernel_ids_with_integrate_attempts(state) -> set[str]:
    """kernel_ids that already received a *terminal* E2E integrate verdict.

    A kernel_id whose only integrate attempts are un-exhausted integration
    faults (``retryable``) is intentionally excluded so the pending-integrate
    driver re-enqueues it for a fault retry. A kernel_id is treated as
    attempted once *any* of its entries reached a non-retryable terminal
    state (KEEP / real REVERT / fault budget exhausted); a terminal entry on
    one patch key wins over a retryable entry on another.

    Returns:
        set[str]: The kernel_ids with at least one non-retryable terminal
            integrate entry.
    """
    terminal: set[str] = set()
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "").strip()
        if not kid:
            continue
        if entry.get("retryable") and not entry.get("rejected"):
            continue
        terminal.add(kid)
    return terminal


def integrate_attempt_count_for_kernel(state, kernel_id: str) -> int:
    """Total *recorded* integrate attempts for a kernel_id.

    Sums ``attempt_count`` across every ``kernel_integrate_attempts`` entry
    sharing this ``kernel_id`` (one kernel may produce more than one patch
    key). The count only advances inside
    :func:`record_kernel_integrate_result`, so it is a reliable in-flight
    signal for the KERNEL-phase auto-integrate driver: an unchanged count
    means a dispatched integrate has not yet been recorded (still in flight),
    an advanced count means it completed.

    Args:
        kernel_id (str): The kernel identifier to total attempts for.

    Returns:
        int: Recorded integrate attempts (0 when unknown/blank).
    """
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0
    total = 0
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("kernel_id") or "").strip() != kid:
            continue
        try:
            total += int(entry.get("attempt_count") or 0)
        except (TypeError, ValueError):
            continue
    return total


def _kernel_trace_impact_pct(state, kernel_id: str) -> float:
    """Return TraceLens gpu_pct for a kernel_id; unknown kernels sort last.

    Args:
        kernel_id (str): The kernel identifier to look up in the latest
            trace-analyze ``hot_kernels_top15``.

    Returns:
        float: The kernel's ``gpu_pct`` impact, or ``0.0`` when blank,
            unknown, or unparseable.
    """
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0.0
    trace = state.last_trace_analyze or {}
    for row in trace.get("hot_kernels_top15") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("kernel_id") or "").strip() != kid:
            continue
        try:
            return float(row.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def next_pending_keep_kernel_id(state) -> str:
    """Return next KEEP kernel_id awaiting integrate ("" if drained).

    Ordering favors trace impact (``gpu_pct``) over kernel micro speedup:
    E2E validation should test the highest-impact hot kernel first, not
    merely the patch with the largest isolated microbenchmark win.

    Returns:
        str: The highest-impact pending KEEP ``kernel_id``, or ``""``
            when the queue is drained.
    """
    pending = pending_keep_kernel_ids(state)
    return pending[0] if pending else ""


def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first.

    Kernels that already have an integrate attempt (including
    ``NEEDS_REVIEW``) are excluded so a noisy near-threshold result does not
    automatically rerun the same patch up to the historical max-attempt cap.
    Positive ``NEEDS_REVIEW`` rows are handled by stack validation instead.

    Returns:
        list[str]: Pending KEEP ``kernel_id`` values sorted impact-first
            (trace ``gpu_pct``, then micro speedup), one per source file.
    """
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    attempted_ids = _kernel_ids_with_integrate_attempts(state)
    rejected = set(state.rejected_kernel_ids or [])
    # Mirror next_pending_keep_kernel_id same-file guard: only strongest KEEP per source_file is queueable.
    claimed_sources: set[str] = set()
    ranked: list[tuple[float, float, str, str]] = []
    for kid, entry in (state.kernel_opt_attempts or {}).items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("last_decision", "")).upper() != "KEEP":
            continue
        if kid in integrated_ids or kid in attempted_ids or kid in rejected:
            continue
        src = str(entry.get("last_source_file") or "")
        if src and src in integrated_sources:
            continue
        try:
            micro = float(entry.get("last_micro_speedup") or 0.0)
        except (TypeError, ValueError):
            micro = 0.0
        ranked.append((_kernel_trace_impact_pct(state, kid), micro, kid, src))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    result: list[str] = []
    for _impact, _micro, kid, src in ranked:
        if src and src in claimed_sources:
            continue
        if src:
            claimed_sources.add(src)
        result.append(kid)
    return result


def has_keep_pending_integrate(state) -> bool:
    """Whether any KEEP kernel is still awaiting integrate.

    Returns:
        bool: ``True`` when :meth:`next_pending_keep_kernel_id` is
            non-empty.
    """
    return bool(next_pending_keep_kernel_id(state))


def kernel_opt_attempts_count(state) -> int:
    """Number of distinct kernels with recorded kernel_opt attempts.

    Returns:
        int: The size of the ``kernel_opt_attempts`` ledger.
    """
    return len(state.kernel_opt_attempts or {})


def untried_hot_reusable_kernels(
    state,
    *,
    min_gpu_pct: float | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by gpu_pct, one kernel_id per task_group.

    Args:
        min_gpu_pct (float | None): Minimum GPU-share threshold; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``.
        top_n (int | None): Cap on enforced kernels by gpu_pct; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_GATE_TOP_N``.

    Returns:
        list[str]: The untried hot-reusable ``kernel_id`` values (one per
            task_group), sorted strongest-first.
    """
    from .shared_state import (
        _DEFAULT_HOT_KERNEL_GATE_TOP_N,
        _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
    )

    info = state.last_trace_analyze or {}
    hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
    task_groups = info.get("task_groups") or []
    if not isinstance(hot, list):
        return []

    if min_gpu_pct is None:
        try:
            min_gpu_pct = float(os.environ.get(
                "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
            ))
        except (TypeError, ValueError):
            min_gpu_pct = _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
    if top_n is None:
        try:
            top_n = int(os.environ.get(
                "HYPERLOOM_KERNEL_OPT_GATE_TOP_N",
                _DEFAULT_HOT_KERNEL_GATE_TOP_N,
            ))
        except (TypeError, ValueError):
            top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
    top_n = max(1, int(top_n))

    kid_to_group: dict[str, list[str]] = {}
    for g in task_groups:
        if not isinstance(g, dict):
            continue
        members = [str(m) for m in (g.get("kernel_ids") or []) if m]
        for m in members:
            kid_to_group[m] = members

    integrated_ids = _kernel_ids_in_optimization_stack(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    rejected = set(state.rejected_kernel_ids or [])
    attempts = state.kernel_opt_attempts or {}

    # Sort by gpu_pct desc so dedup picks the strongest member of each task_group.
    rows: list[tuple[float, str, str, list[str]]] = []
    for k in hot:
        if not isinstance(k, dict):
            continue
        if k.get("reusable_native_kernel") is not True:
            continue
        try:
            gpu_pct = float(k.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            gpu_pct = 0.0
        if gpu_pct < min_gpu_pct:
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        src = str(k.get("source_file") or "")
        members = sorted(kid_to_group.get(kid, [kid]))
        rows.append((gpu_pct, kid, src, members))
    rows.sort(key=lambda x: x[0], reverse=True)

    ranked: list[tuple[float, str, str, list[str]]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for row in rows:
        group_key = tuple(row[3])
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        ranked.append(row)
    ranked = ranked[:top_n]

    untried: list[str] = []
    for _pct, kid, src, members in ranked:
        if members and all(m in rejected for m in members):
            continue
        if any(m in integrated_ids for m in members):
            continue
        if src and src in integrated_sources:
            continue
        if any(int((attempts.get(m) or {}).get("attempts", 0)) > 0
               for m in members):
            continue
        untried.append(kid)
    return untried


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
