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
import sys
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..trace.llm_trace import LLMCallRecord, append_llm_call
from ..trace.parse_usage import (
    parse_forge_steps,
    parse_forge_usage,
    parse_geak_usage,
    parse_oob_json_usage,
)

# tree-reform.MD P2.2: cohesive clusters extracted to sibling modules;
# re-exported here so the module namespace + monkeypatch surface is intact.
from ._kernel_decisions import (
    _HONEST_E2E_UMBRELLA_ENV as _HONEST_E2E_UMBRELLA_ENV,
    _TRUEY as _TRUEY,
    _FALSEY as _FALSEY,
    _honest_flag as _honest_flag,
    _format_last_kernel_opt as _format_last_kernel_opt,
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
)



log = logging.getLogger(__name__)
_BACKGROUND_ROCPROF_TASKS: set[asyncio.Task[Any]] = set()
STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT = 1.0





def _vram_guarded_server_args(extra_args: str) -> str:
    """Optionally cap ``--gpu-memory-utilization`` for the integrate re-baseline.

    v4-parity VRAM barrier: when ``HL_INTEGRATE_VRAM_GUARD`` (or the umbrella)
    is on and the caller has NOT already pinned ``--gpu-memory-utilization``,
    append a conservative cap (``HL_INTEGRATE_VRAM_UTIL_CAP``, default 0.90) so a
    re-baseline server launched on a node with less headroom than the original
    roofline cannot OOM. Deterministic (no live VRAM probe), additive, and a
    strict no-op when the flag is off or a util is already specified — so the
    returned string is byte-identical to the input on the legacy path.

    Args:
        extra_args: The resolved ``extra_server_args`` string for the server.

    Returns:
        str: ``extra_args`` unchanged, or with a util cap appended.
    """
    if not _honest_flag("HL_INTEGRATE_VRAM_GUARD"):
        return extra_args
    # ``--gpu-memory-utilization`` is a vLLM-only flag; sglang REJECTS it
    # ("unrecognized arguments") and the server exits immediately, failing the
    # integrate re-baseline for EVERY sglang workload. sglang caps memory via
    # ``--mem-fraction-static`` (set by its launcher), so it needs no guard here.
    # Apply the cap only for vLLM; sglang/unknown framework -> strict no-op.
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

    v4-parity import-grep: greps the re-baseline server log for evidence the
    patched module's basename was imported/loaded/compiled by the served
    process, so a measured E2E delta is attributed to code the workload really
    ran (not a phantom win on an un-imported patch). Returns a *tri-state*:

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
    # Basename present but not in an obvious import context — treat as unknown
    # rather than penalize (avoids false negatives on terse logs).
    return None


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
    from ..framework.paths import resolve_patch_target_roots

    roots = resolve_patch_target_roots()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    # Add FlyDSL kernel checkout roots for moe_flydsl_* candidates.
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
# Default ladder: forge first, then geak, then the OOB backends (claude,
# codex, cursor). Cursor is key-gated and dropped from the auto-derived ladder
# when CURSOR_API_KEY is unset (see _backend_order). An explicit
# KERNEL_OPT_BACKEND_ORDER / KERNEL_OPT_BACKENDS (or payload backend_order)
# still overrides this default as-is.
_DEFAULT_KERNEL_BACKEND_ORDER = ("forge", "geak", "claude", "codex", "cursor")
# Soft cap on concurrent kernel-backend coroutines (pin with KERNEL_OPT_MAX_PARALLEL).
_DEFAULT_KERNEL_BATCH_PARALLEL = 8
_DEFAULT_OOB_BUDGET_MINUTES = 60.0
# Minimum wall-clock a fallback backend needs to do anything useful (and still
# salvage partial artifacts). When less than this remains in the per-kernel
# ladder budget, the ladder stops instead of launching a backend it cannot
# finish. Mirrors the +180s wrapper grace.
_KERNEL_LADDER_MIN_BACKEND_SEC = 180
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 3 * 60 * 60


@functools.lru_cache(maxsize=1)
def _default_geak_budget_minutes() -> float:
    """Default per-GEAK-attempt budget tracking ``$GEAK_RUN_MODE`` (quick→70, full→180); mirrors kernel-agent tool defaults.

    Full mode gets 180 min (3 h) so preprocess (CK/.cu harness build) + the
    optimization round + the post-round FULL_BENCHMARK verification all fit;
    at 130 min the verification step was being starved and every attempt
    finalized as an unverified NEEDS_REVIEW. Matches the kernel-agent tool
    default (kernel_optimization.py).

    Returns:
        The default per-attempt GEAK budget in minutes.
    """
    raw = (os.environ.get("GEAK_RUN_MODE") or "").strip().lower()
    return 70.0 if raw == "quick" else 180.0


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
    ``src/hyperloom/inference_optimizer/tests/conftest.py`` autouse fixture handles
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
            "src/hyperloom/inference_optimizer/scripts/install.sh and source $KERNEL_AGENT_ENV "
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
    from hyperloom.inference_optimizer import paths

    return paths.tracelens_root()


def _tracelens_root_error(root: Path) -> str | None:
    """Validate that the resolved TraceLens root is a usable git checkout.

    A directory that exists but lacks ``.git`` (e.g. an installer's half-done
    clone) is NOT usable and must be reported so a non-default override fails
    fast and a default path is self-healed (#722/PR#789).

    Returns:
        str | None: A human-readable error when the checkout is missing or
            incomplete, or ``None`` when it is a usable git checkout.
    """
    if not root.is_dir():
        return (
            f"TraceLens root not found: {root}; run kernel-agent/scripts/install.sh "
            "or set TRACELENS_ROOT to an existing checkout"
        )
    if not (root / ".git").exists():
        return (
            f"TraceLens root incomplete (not a git checkout): {root}; "
            "run kernel-agent/scripts/install.sh or set TRACELENS_ROOT to a valid checkout"
        )
    return None


def _maybe_selfheal_tracelens_root(root: Path, *, log: Any = None) -> None:
    """Rebuild the pod-local TraceLens checkout if it vanished mid-run (#722).

    Only the installer-managed default path is healed; an explicit
    ``TRACELENS_ROOT`` override pointing elsewhere is operator-maintained and
    must fail fast when missing (mirrors kernel-agent/scripts/install.sh).
    Best-effort: any failure is swallowed so the caller's normal validation
    produces the user-facing error.

    NB: the default path is itself persisted as ``TRACELENS_ROOT`` in
    kernel-agent.env.sh, so "env is set" is NOT a reliable override signal —
    compare the resolved path against the installer default instead.
    """
    from hyperloom.inference_optimizer import paths

    default_root = paths.open_source_root() / "TraceLens"
    try:
        is_default = Path(root).resolve() == default_root.resolve()
    except OSError:
        is_default = False
    if not is_default:
        return  # explicit operator override at a non-default path: never auto-clone
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
    # Route through the per-framework env-name source of truth so atom reads ``EXTRA_ATOM_ARGS`` instead of dropping flags via a sglang/vllm default.
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
    from hyperloom.inference_optimizer.session_paths import patches_dir

    kid = str(kernel_id or payload.get("kernel_id") or "")
    backup_root = payload.get("backup_root") or (patches_dir(session_dir, kid or "anon") / "backup")
    tool = _load_apply_tool()
    # Snapshot mode (content-addressed deploy): when a snapshot dir of byte-exact
    # final files is present, the patch lands atomically across all its files.
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
    from ..state.shared_state import SharedState

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

    if kernel_id and str(last_kernel.get("kernel_id") or "") == kernel_id:
        # Snapshot deploy: prefer the original patch (manifest) + its byte-exact
        # snapshot dir so the WHOLE multi-file patch lands atomically.
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

    # Multi-KEEP queue fallback: ``last_kernel_opt`` holds only the strongest pending KEEP, so pull patch_path/source_file from the per-kernel ledger for other queued KEEPs.
    if kernel_id:
        attempt = (state.kernel_opt_attempts or {}).get(kernel_id) or {}
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

    def _run() -> tuple[int, str, str]:
        """Run the command synchronously in a worker thread.

        Copies the current environment, injects the Ray GCS address when running
        in multi-node mode, and prepends the venv ``bin`` directory to ``PATH``
        before invoking the command with output capture and the timeout.

        Launches the child in its own POSIX session and, on timeout, reaps the
        WHOLE process group so a hung grandchild (the claude CLI / oob / curl
        holding a stalled LLM streaming socket) dies with the wrapper instead of
        being orphaned — the Sandbox-hang RCA left such grandchildren running,
        keeping the pod alive and the GPU idle. Mirrors ``subprocess.run``'s
        contract: captures stdout/stderr and re-raises ``TimeoutExpired``.

        Returns:
            tuple[int, str, str]: ``(returncode, stdout, stderr)``.

        Raises:
            subprocess.TimeoutExpired: When the command exceeds ``timeout_sec``.
        """
        env = os.environ.copy()
        from ..actions.executors._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
            dynamo_ssh_env_from_state,
        )
        from ..actions.executors._subprocess_kill import run_with_session_kill

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
        # run_with_session_kill launches the child in its own POSIX session and
        # reaps the whole descendant tree on every exit path (incl. on timeout,
        # where it re-raises TimeoutExpired) — the same contract this wrapper
        # used to hand-roll with Popen + kill_my_spawned_server.
        cp = run_with_session_kill(cmd, env=env, timeout=timeout_sec, text=True)
        return cp.returncode, cp.stdout or "", cp.stderr or ""

    return await asyncio.to_thread(_run)


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
        # Hand forge the fp8 GEMM path the model actually runs instead of letting
        # it fall back to its internal blockscale default. Explicit per-token env
        # wins; otherwise derive from the checkpoint's static format (config.json
        # quantization_config). Unreadable config -> "auto" so forge sniffs the
        # kernel_signature_log (keeps the legacy no-config behaviour unchanged).
        if per_token_signal:
            quant_type = "per_token"
        else:
            model_path = str(
                payload.get("model_path") or getattr(state, "model_path", "") or ""
            ).strip()
            quant_type = _resolve_fp8_quant_type(model_path)
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


# Map the resolved (precision, quant_type) to the aiter untuned-GEMM CSV that
# the specialist phase already records under each worktree. fp8 "auto" resolves
# to blockscale to match forge's own _resolve_quant_type default.
_FORGE_UNTUNED_CSV_BY_QUANT: dict[str, str] = {
    "auto": "a8w8_blockscale_untuned_gemm.csv",
    "blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "per_token": "a8w8_untuned_gemm.csv",
    "per_tensor": "a8w8_untuned_gemm.csv",
    "bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "fp4": "a4w4_blockscale_untuned_gemm.csv",
    "mxfp4": "a4w4_blockscale_untuned_gemm.csv",
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
    cfg = Path(model_path) / "config.json"
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


def _resolve_fp8_quant_type(model_path: str) -> str:
    """Pick the fp8 dense tuner quant_type from the checkpoint's static format.

    forge accepts an explicit ``quant_type``; rather than letting it fall back to
    its internal blockscale default, hand it the path the model actually runs:

    - ``blockscale`` only when the checkpoint ships block-quantized weights
      (``config.json`` ``quantization_config.weight_block_size`` or a block-style
      ``quant_method``) -- the a8w8 blockscale GEMM path.
    - ``per_token`` for a plain fp16/bf16 checkpoint served under dynamic
      ``--quantization fp8`` (the a8w8 per-token path).
    - ``auto`` when ``config.json`` cannot be read, so forge sniffs the
      ``kernel_signature_log`` itself (preserves the legacy behaviour and keeps
      the no-readable-config case unchanged).
    """
    data = _read_model_config(model_path)
    if data is None:
        return "auto"
    # Check both the top-level config and a nested ``text_config`` (multimodal
    # checkpoints sometimes carry the quantization_config there), mirroring
    # ``_model_hidden_size``.
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg_dict in candidates:
        qc = cfg_dict.get("quantization_config")
        if isinstance(qc, dict):
            if qc.get("weight_block_size"):
                return "blockscale"
            method = str(qc.get("quant_method") or qc.get("fmt") or "").lower()
            if "block" in method:
                return "blockscale"
    return "per_token"


def _csv_matches_model(csv_path: Path, model_path: str) -> bool:
    """Return True when an untuned CSV plausibly belongs to ``model_path``.

    A real per-model dense untuned CSV always contains GEMMs whose ``K`` equals
    the model ``hidden_size`` (QKV / gate-up / o-proj are column-parallel, so
    their contraction dim stays ``hidden_size`` regardless of TP). When the
    model's ``hidden_size`` is known and absent from the CSV's ``K`` column, the
    CSV was recorded for a different model (e.g. AITER's checked-in DeepSeek
    default with ``K=7168``) and must be rejected so forge derives shapes from
    the model config instead of "tuning" mismatched shapes to a 1.0x no-op.

    Returns True when validation is not possible (``hidden_size`` unreadable or
    the CSV exposes no ``K`` column) to preserve existing behaviour and avoid
    false rejections.
    """
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


def _resolve_forge_untuned_csv(
    session_dir: Path, precision: str, quant_type: str, model_path: str = ""
) -> str:
    """Find an aiter untuned-GEMM CSV recorded by the specialist phase.

    Dense fp8/fp4 forge tuners (a8w8*/a4w4*) skip themselves unless real GEMM
    shapes are supplied. Hyperloom's specialist runs already write these shapes
    to ``runs/specialist/<hash>/worktree/aiter/configs/*_untuned_gemm.csv`` in
    forge's ``M,N,K`` format. This resolver picks the newest non-empty CSV
    matching the resolved quant type and wires it into the forge tuners.

    When ``model_path`` is given, candidate CSVs whose GEMM shapes do not match
    the model (the AITER repo default carries fixed DeepSeek shapes, ``K=7168``,
    regardless of the model under test) are rejected. Returning "" then lets
    forge derive correct per-model shapes from ``config.json`` rather than
    tuning mismatched shapes that always yield ``improved=0/0, 1.0x``.

    Returns the CSV path, or "" when none is available (bf16 derives shapes from
    config.json and needs no CSV).
    """
    precision = (precision or "").strip().lower()
    quant_type = (quant_type or "").strip().lower()

    fname = _FORGE_UNTUNED_CSV_BY_QUANT.get(quant_type)
    if fname is None:
        if precision == "fp8":
            fname = "a8w8_blockscale_untuned_gemm.csv"
        elif precision in ("fp4", "mxfp4"):
            fname = "a4w4_blockscale_untuned_gemm.csv"
        else:
            return ""

    from hyperloom.inference_optimizer.session_paths import runs_root

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

    A caller can hand us inline JSON content instead of a path. ``Path(...)``
    accepts any string, but ``is_file()`` raises ``OSError(ENAMETOOLONG)`` when
    a path component exceeds the filesystem limit, which previously crashed the
    forge dense tuner instantly. Treat that (and any OSError) as "not a file".
    """
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _normalize_tokens(value: Any) -> str:
    """Return a clean comma-separated token string for forge's ``--tokens``.

    forge parses ``--tokens`` as ``int(t) for t in value.split(",")``. A caller
    that passes a list (or its string form ``"[4, 8, 64]"``) makes forge choke
    with ``Invalid --tokens value: invalid literal for int(): '[64]'``. Accept
    lists and bracketed strings; emit a bare comma-separated list.
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

    The orchestration layer sometimes passes the GEMM shapes as inline JSON
    (a list/dict, or its string form) in ``shapes_json`` instead of a file
    path. forge treats ``shapes_json`` strictly as a path and crashes with
    ``OSError(ENAMETOOLONG)`` on the long pseudo-path. Normalize here:

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
        # A real, existing path wins.
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
            # Non-JSON string that is not an existing file: unusable.
            return ""

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        out = workspace / "forge_shapes.json"
        out.write_text(json.dumps(parsed), encoding="utf-8")
        return str(out)
    except (OSError, TypeError, ValueError):
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
    from ..state.shared_state import SharedState

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
    tokens = _normalize_tokens(payload.get("tokens"))
    # Default mp = all visible GPUs (server is stopped during tuning).
    from ..policy.gate import detect_gpu_count

    detected_gpus = detect_gpu_count() or tp
    mp = int(payload.get("mp") or os.environ.get("FORGE_GEMM_TUNE_MP") or detected_gpus)

    # Resolve server log for 1-stage ASM detection
    kernel_sig_log = str(payload.get("kernel_signature_log") or "").strip()
    if not kernel_sig_log:
        kernel_sig_log = _resolve_forge_server_log(state, session_dir)

    # Resolve TraceLens shapes if available. Normalize first: callers sometimes
    # pass inline JSON content instead of a path, which makes forge crash with
    # OSError(ENAMETOOLONG); materialize it to a real file (or drop it).
    shapes_json = _normalize_forge_shapes_json(payload.get("shapes_json"), workspace)
    if not shapes_json:
        shapes_json = _resolve_forge_shapes(state, session_dir)

    # Dense fp8/fp4 tuners need real GEMM shapes. When TraceLens exposes no
    # shapes JSON, fall back to the aiter untuned-GEMM CSVs the specialist phase
    # already recorded; without this the dense tuner skips itself.
    untuned_csv = str(payload.get("untuned_csv") or "").strip()
    if untuned_csv and not _path_is_existing_file(untuned_csv):
        # Guard against inline content / stale paths handed in as untuned_csv.
        untuned_csv = ""
    if not untuned_csv and not shapes_json:
        untuned_csv = _resolve_forge_untuned_csv(session_dir, precision, quant_type, model_path)

    timeout = _gemm_tuning_timeout_sec(payload)
    session_max_min = float(getattr(state, "max_minutes", 0) or 0)
    input_payload = {
        "model_path": model_path,
        "framework": framework,
        "precision": precision,
        "quant_type": quant_type,
        "gpu_type": gpu_type,
        "tp": tp,
        "conc": conc,
        "mp": mp,
        "output_dir": str(workspace),
        "timeout": timeout,
        # Global timeout ensures the whole session (all tuners combined) stays
        # within the budget. Forge skips lower-priority tuners if time runs out.
        "global_timeout": timeout,
        "skip_gpu_check": True,
        "tokens": tokens,
        "untuned_csv": untuned_csv,
        "shapes_json": shapes_json,
        "tunableop_input": str(payload.get("tunableop_input") or ""),
        "kernel_signature_log": kernel_sig_log,
        "tuner": str(payload.get("tuner") or ""),
        # Thorough mode: exhaustive search when session budget allows (>= 24h)
        # and enough GPUs are available (>= 4) to parallelize the sweep.
        "thorough": bool(session_max_min >= 1440 and mp >= 4),
    }
    input_json = workspace / "forge_gemm_tuning_input.json"
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        "python3",
        str(_kernel_agent_tool_path("forge_gemm_tuning.py")),
        "--input-json",
        str(input_json),
    ]

    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout)
        result = _parse_forge_gemm_sentinel(stdout)
        if result is None:
            result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess; shape a failed
        # result instead of letting TimeoutExpired stall the coordinator tick.
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout}s: {cmd_repr[:1500]}",
        }

    result.setdefault("backend", "forge")
    # Tag the tuning engine so the breakdown's ``collect_gemm_tuning`` attributes
    # this run to forge instead of falling back to its ``geak`` default (the
    # GEAK path sets ``engine`` too). Without this, forge runs recorded into
    # ``gemm_tuning_attempts`` are mis-counted as the GEAK source.
    result.setdefault("engine", "forge")
    result.setdefault("workspace", str(workspace))
    result.setdefault("precision", precision)
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)

    # Surface why forge skipped: forge records per-tuner skip reasons in its
    # on-disk result.json, but the stdout sentinel can omit them, leaving the
    # session state with an unexplained "skipped". Merge from disk and derive a
    # concise top-level skip_reason so state + breakdown show the cause.
    if not result.get("tuners_skipped"):
        disk_skipped = _read_forge_result_json(workspace).get("tuners_skipped")
        if disk_skipped:
            result["tuners_skipped"] = disk_skipped
    if not result.get("skip_reason"):
        reason = _derive_gemm_skip_reason(result.get("tuners_skipped"))
        if reason:
            result["skip_reason"] = reason

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
    from ..state.shared_state import SharedState

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

    _gemm_timeout = _gemm_tuning_timeout_sec(payload)
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=_gemm_timeout)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess; shape a failed
        # result instead of letting TimeoutExpired stall the coordinator tick.
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
        result = await _run_forge_gemm_tuning(payload, session_dir=session_dir)
    else:
        result = await _run_geak_gemm_tuning(payload, session_dir=session_dir)
    # Full-trace: record this run as a gemm_tuning audit row, backfilled into the
    # trace as a ``gemm_tuning:<engine>`` span so the deterministic tuner is
    # attributed to its own source. Best-effort; never breaks the run.
    _trace_gemm_tuning_run(result, session_dir=session_dir)
    return result


def _trace_gemm_tuning_run(result: Any, *, session_dir: Path) -> None:
    """Append one ``gemm_tuning.jsonl`` audit row for a GEMM-tuning run.

    Mirrors :func:`_trace_kernel_attempt_steps`: distils the run result into a
    compact source-attribution row (engine, decision, speedup, per-tuner
    summary) and appends it to ``reports/trace/gemm_tuning.jsonl``. The Langfuse
    emitter backfills it as a ``gemm_tuning:<engine>`` span under the
    ``gemm_tuning`` agent. Best-effort end to end: any read/serialise/write
    failure degrades to a debug log and is swallowed.

    Args:
        result: The GEMM-tuning handler result envelope.
        session_dir: Session directory the audit row is appended under.
    """
    if not isinstance(result, dict):
        return
    from datetime import datetime, timezone

    from hyperloom.inference_optimizer.session_paths import gemm_tuning_steps_path

    engine = str(result.get("engine") or result.get("backend") or "").strip().lower() or "unknown"
    tuners: list[dict[str, Any]] = []
    for t in result.get("tuners_run") or []:
        if not isinstance(t, dict):
            continue
        tuners.append(
            {
                "tuner": t.get("tuner") or t.get("name"),
                "best_micro_speedup": t.get("best_micro_speedup"),
                "kept": t.get("kept"),
            }
        )
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
        "error_class": result.get("error_class"),
    }
    row = {k: v for k, v in row.items() if v is not None}
    try:
        path = gemm_tuning_steps_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        log.debug("full-trace: gemm_tuning audit append failed", exc_info=True)


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
    # Resolve TraceLens root independently of inherited env (the coordinator may
    # not have sourced kernel-agent.env.sh). If the installer-managed checkout
    # vanished mid-run (#722), self-heal it before the fail-fast validation.
    tracelens_root = _resolve_tracelens_root()
    # Self-heal when the checkout is missing OR incomplete (exists but no .git,
    # e.g. a concurrent install's half-done clone). Gating only on is_dir()
    # would let an incomplete default checkout bypass self-heal (#722/PR#789).
    if not (tracelens_root / ".git").exists():
        _maybe_selfheal_tracelens_root(tracelens_root, log=log)
    tl_err = _tracelens_root_error(tracelens_root)
    if tl_err:
        return {"status": "failed", "error_class": "tracelens_root_missing", "error": tl_err}

    # Pass the session root so artefacts settle at ``<session_dir>/kernel-agent/runs/...`` (the suffix is hardcoded in the tool).
    workspace_path = payload.get("workspace_path") or str(session_dir)
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    # Backfill workload context from SharedState so the tool gets the right framework/platform/model/analysis_mode when Orchestration omits them.
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    framework = (payload.get("framework") or state.framework or "").strip()
    target_platform = (payload.get("target_platform") or state.gpu_type or "").strip()
    model_name = (payload.get("model_name") or state.model_name or state.model_path or "").strip()
    analysis_mode = (payload.get("analysis_mode") or "").strip()
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    # Scriptable image frameworks (xDiT diffusion) are server-less and have no
    # LLM decode steady-state window, so the TraceLens steady-state splitter
    # cannot produce chunks and hard-fails (trace_split_no_steady_state). Feed
    # the raw trace to TraceLens directly and drop the (LLM-only) --split-* hints.
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    scriptable = is_scriptable(framework)

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
        "--workspace-path",
        workspace_path,
        # Pass the resolved root explicitly so the tool never depends on the
        # subprocess inheriting TRACELENS_ROOT from the coordinator env.
        "--tracelens-root",
        str(tracelens_root),
    ]
    # Kernel-candidate POOL size (issue #667): only forward --top-k when the
    # request explicitly overrides it. Otherwise let tracelens_analysis.py apply
    # its own large-pool default (env: HYPERLOOM_KERNEL_CANDIDATES_TOP_K) so the
    # candidate-build cap and the dispatch-side budget (source-fn grouping + op
    # dedup + attempt cap) stay decoupled and share a single source of truth.
    if payload.get("top_k") is not None:
        cmd += ["--top-k", str(payload.get("top_k"))]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if framework:
        cmd += ["--framework", str(framework)]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if analysis_mode:
        cmd += ["--analysis-mode", str(analysis_mode)]

    if scriptable:
        # No steady-state window to extract; skip the splitter entirely.
        cmd += ["--skip-split"]
        # Forward the denoise-step count so the diffusion workload roofline can
        # emit per-denoise-step timings. Priority: payload override (steps in
        # the profiled window) > baseline workload metadata (full schedule).
        num_denoise = payload.get("num_denoise_steps") or workload.get("num_inference_steps")
        if num_denoise not in (None, ""):
            try:
                if int(num_denoise) > 0:
                    cmd += ["--num-denoise-steps", str(int(num_denoise))]
            except (TypeError, ValueError):
                pass
    else:
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
    # Forward TraceLens splitter steady-state mode (mixed/decode_only/prefilldecode) via payload or env.
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
    # Post-kernel-opt roofline writes a separate report so it never overwrites
    # the baseline kernel_roofline.json; the chart diffs the two snapshots.
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    if roofline_output_name:
        cmd += ["--roofline-output-name", roofline_output_name]
    # A stale ``--roofline-json`` payload key is silently ignored.
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess (an in-process
        # TraceLens SDK stream that stalled is now also bounded client-side);
        # shape a failed result instead of stalling the coordinator tick.
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

        # Record hot-kernel discovery provenance (best-effort): the discovery
        # run + its hot-kernel list + tool provenance, so the journey can thread
        # discovery -> dispatch -> backends -> e2e.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

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


def _exists_with_retry(
    path: str | Path,
    *,
    attempts: int = 5,
    delay_sec: float = 0.5,
) -> bool:
    """Check ``path`` existence, retrying briefly to absorb storage latency.

    On shared/network filesystems (e.g. wekafs) a file that was just written can
    take a moment to become visible to another client, so a single
    :meth:`Path.exists` immediately after the write may spuriously report
    missing. Retry a handful of times with a short pause before giving up.

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
            elif not _names_specific_kernel(single_payload):
                # Empty eligible queue and the request named no specific target
                # (e.g. the post-GEMM auto pass dispatches a batch with no id).
                # All candidates are already tried/rejected, below the size
                # cutoff, or not reusable. Finish cleanly instead of falling
                # into the single-kernel path, which would surface
                # "missing 'kernel_id'" and mis-report an empty work queue as a
                # GEAK failure. A "skipped" status (no error_class / REVERT
                # decision) is not counted as a failure by the breakdown
                # recorder.
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


def _raw_kernel_backend_order(payload: dict | None = None) -> list[str]:
    """Return the raw, lowercased kernel backend order from payload/env.

    This is the single source of truth for kernel-backend selection and is
    shared by both the per-kernel ladder (:func:`_backend_order`) and the
    phase-level PerfSkills check (:func:`perfskills_selected`).  Unknown tokens
    are kept here on purpose; callers filter to the set they understand.

    Precedence (highest to lowest): ``payload['backend_order']`` ->
    ``KERNEL_OPT_BACKEND_ORDER`` env -> ``KERNEL_OPT_BACKENDS`` env.  When none
    is set an empty list is returned so callers can apply their own default.

    Args:
        payload: Optional request payload that may carry ``backend_order``.

    Returns:
        list[str]: The ordered, lowercased backend tokens (may be empty).
    """
    raw = (
        (payload or {}).get("backend_order")
        or os.environ.get("KERNEL_OPT_BACKEND_ORDER")
        or os.environ.get("KERNEL_OPT_BACKENDS")
    )
    if not raw:
        return []
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def perfskills_selected(payload: dict | None = None) -> bool:
    """Whether ``perfskills`` is requested in the kernel backend order.

    ``perfskills`` is not a per-kernel backend: when it appears in the order it
    means "delegate the whole KERNEL_AGENT phase to the PerfSkills e2e optimizer".
    It therefore *owns* the phase whenever present (any other backends in the
    order are ignored for the kernel phase), so an order of just ``perfskills``
    runs only PerfSkills.

    Args:
        payload: Optional request payload that may carry ``backend_order``.

    Returns:
        bool: ``True`` when ``perfskills`` is in the resolved order.
    """
    return "perfskills" in _raw_kernel_backend_order(payload)


def _kernel_ladder_budget_sec(payload: dict) -> int:
    """Total wall-clock budget for one kernel's whole backend ladder.

    The ladder runs its backends sequentially (forge -> geak -> oob fallbacks),
    each as a subprocess with its own timeout. Without a shared ceiling a
    backend that hangs to its hard timeout followed by a fallback running its
    full budget could roughly double a kernel's wall clock and overshoot the
    KERNEL-phase budget cap (which is only re-checked between orchestration
    turns, never mid-``run_optimization``). This budget bounds the whole ladder
    so a fallback only runs within the time left and an exhausted budget exits
    the ladder cleanly.

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
    order = _raw_kernel_backend_order(payload)
    if order:
        explicit = True
    else:
        # Ignore legacy payload["backends"]; the default ladder (GEAK first) mirrors ``kernel_optimization.choose_backends`` so single/batch agree.
        order = list(_DEFAULT_KERNEL_BACKEND_ORDER)
        explicit = False
    # `forge` (Kernel-Forge autonomous-loop backend) is first in
    # _DEFAULT_KERNEL_BACKEND_ORDER; keep it in `allowed` so it survives the
    # filter for both the default and any explicit backend_order.  `perfskills`
    # is intentionally absent: it is a phase-level delegate (see
    # ``perfskills_selected``), not a per-kernel backend, so it is dropped here.
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
    from hyperloom.inference_optimizer.session_paths import kernel_agent_runs_dir

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
    is_retryable_infra = (
        last_decision == "" and last_status in {"failed", "error", "timeout"} and not rejected_reason
    )
    if failure_count < max_failures and is_retryable_infra:
        return max_failures
    # High-impact infra-retry (flag-gated, default off): a high-GPU%-share kernel
    # that keeps infra-failing ("didn't finish") gets extra attempts to COMPLETE
    # before retirement, mirroring the retirement-side cap in record_kernel_opt so
    # dispatch / record / kernel_work_pending agree. Off => unchanged.
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
    from ..state.shared_state import (
        _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
        resolve_kernel_opt_max_failures,
    )

    max_failures = resolve_kernel_opt_max_failures()
    # min_gpu_pct must mirror SharedState.untried_hot_reusable_kernels' 3.0 default so the two layers agree and tiny kernels don't eat ladder wall-clock.
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
            from ..state.shared_state import SharedState

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
        """A kernel_id is live (batch-eligible) iff NOT rejected, NOT in-flight, and < max_attempts recorded against the CURRENT source_file (per-source counting).

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
        attempt_cap = _kernel_dispatch_attempt_cap(entry, max_failures=max_failures)
        if current_source:
            per_source = entry.get("attempts_per_source")
            if isinstance(per_source, dict):
                src_attempts = int(per_source.get(current_source, 0))
                return src_attempts < attempt_cap
        if int(entry.get("attempts", 0)) >= attempt_cap:
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
    # Collect the eligible (live, reusable, source-resolved) ungrouped rows first;
    # the per-row min_gpu_pct gate / selection is applied below so the optional
    # op-fanout de-dup can sum sibling GPU% before gating.
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
        if not _is_live(kernel_id, str(item.get("source_file") or "")):
            skipped[kernel_id] = "not_live"
            continue
        try:
            row_pct = float(item.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            row_pct = 0.0
        legacy_eligible.append((kernel_id, item, row_pct))

    if _honest_flag("HL_KERNEL_OPFANOUT_DEDUP"):
        # Op-fanout de-dup (flag-gated): one logical kernel (same source_file)
        # often shows up as many op-instance rows, so each would dispatch
        # separately and its trace impact would be split across rows. Collapse
        # same-source rows into a single representative (the highest-GPU% row)
        # carrying the SUMMED GPU% of the fanned siblings, so the min_gpu_pct
        # gate and downstream impact ranking see the kernel's true aggregate
        # share. Off => the per-row loop below is byte-identical to legacy.
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
        return (0, 0, 0.0)
    proposal = result.get("proposal") or {}
    verification = result.get("verification") or {}
    keep = 1 if (result.get("status") == "ok" and proposal.get("decision") == "KEEP") else 0
    micro = float(verification.get("micro_speedup") or 0.0)
    # GEAK-only middle tier (flag-gated, default off): a correctness-verified, high-micro
    # NEEDS_REVIEW from the GEAK backend is promotable above a bare non-KEEP — GEAK emits
    # NEEDS_REVIEW (no auto-correctness-KEEP) so its real wins otherwise always lose to any
    # Claude/Codex/forge KEEP. Scoped to backend=="geak" so OOB backends are byte-identical;
    # off => verified_nr is always 0 => 3-tuple sorts exactly as the old 2-tuple.
    _promote = _honest_flag("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW")
    try:
        _thr = float(os.environ.get("HL_VERIFIED_MICRO_PROMOTE_THRESHOLD", "1.10") or 1.10)
    except ValueError:
        _thr = 1.10
    _backend = str(verification.get("best_backend") or result.get("backend") or "").lower()
    verified_nr = (
        1
        if (
            _promote
            and keep == 0
            and _backend == "geak"
            and verification.get("correctness_passed") is True
            and micro >= _thr
        )
        else 0
    )
    return (keep, verified_nr, micro)


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
                # Not enough of the per-kernel budget left to run another
                # backend usefully; stop instead of overshooting it.
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
    # cap. Each ladder call caps its backends to the time left.
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


def _resolve_local_model_path(model: str) -> str:
    """Map a model name/path to a local weights directory the server can load.

    A bare HF id like ``meta-llama-Llama-3.1-8B-Instruct`` passed straight to
    sglang/vllm triggers a HuggingFace fetch (401 for gated repos), so resolve it
    to a local directory first. Reuses the existing kernel-agent resolver
    (``tracelens_analysis._candidate_model_config_paths``, honouring
    ``$HYPERLOOM_MODELS_ROOT``, default ``/wekafs/models``): the directory holding
    the first existing ``config.json`` wins. Returns the input unchanged when it
    is already a directory or nothing resolves (caller/serve layer decides).

    Args:
        model: A model name or filesystem path.

    Returns:
        str: A local weights directory if one resolves, else ``model`` unchanged.
    """
    if not model:
        return model
    if Path(model).is_dir():
        return model
    try:
        tool = _kernel_agent_tool_path("tracelens_analysis.py")
        tools_dir = str(tool.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)  # tracelens_analysis imports sibling tools-dir modules
        import tracelens_analysis as _tla  # type: ignore[import-not-found]
        for cfg in _tla._candidate_model_config_paths(model):
            if Path(cfg).is_file():
                return str(Path(cfg).parent)
    except Exception as exc:  # noqa: BLE001  # resolution is best-effort; fall back to the raw value
        log.info("combined_e2e: model-path resolve failed (%s); using raw '%s'", exc, model)
    return model


def _collect_combined_e2e_pairs(results: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Collect (best_patch, target_source) pairs from a batch's per-kernel results.

    Uses each kernel's BEST microbench patch (``geak_per_task_best_patch``)
    regardless of the per-kernel KEEP/REVERT verdict -- the combined E2E applies
    ALL optimized patches together and lets the end-to-end A/B be the arbiter.
    Kernels with no usable patch (e.g. a kernel that produced none) are skipped.

    Args:
        results: The batch's per-kernel result dicts (``out["batch_results"]``).

    Returns:
        list[tuple[str, str]]: ``(patch_path, target_source_file)`` pairs, one
            per kernel that produced a best patch.
    """
    pairs: list[tuple[str, str]] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        kid = str(r.get("kernel_id") or "?")
        target = str(r.get("source_file") or "")
        best_patch = ""
        for att in r.get("attempts") or []:
            bp = att.get("backend_paths") if isinstance(att, dict) else None
            if isinstance(bp, dict) and bp.get("geak_per_task_best_patch"):
                best_patch = str(bp["geak_per_task_best_patch"])
                break
        if best_patch and target:
            pairs.append((best_patch, target))
            log.info("combined_e2e: kernel %s -> patch %s on %s", kid, best_patch, target)
        else:
            log.info("combined_e2e: kernel %s skipped (no best patch / target)", kid)
    return pairs


def _combined_e2e_serving_config(payload: dict) -> dict[str, Any]:
    """Resolve serving knobs (tp/isl/osl/conc/num_prompts/framework) for combined E2E.

    Precedence: explicit ``payload['serving_config']`` dict (what the driver
    passes, parsed from its --serving-config) -> the materialized workload
    metadata resolver -> apply_and_bench's own defaults (left unset here).
    Never hardcodes workload-specific values.

    Args:
        payload: The run_optimization request payload.

    Returns:
        dict[str, Any]: Keys among {tp, isl, osl, conc, num_prompts, framework}
            that were resolved; missing keys fall back to apply_and_bench defaults.
    """
    cfg: dict[str, Any] = {}
    explicit = payload.get("serving_config")
    if isinstance(explicit, dict):
        for k in ("tp", "isl", "osl", "conc", "num_prompts", "framework"):
            if explicit.get(k) is not None:
                cfg[k] = explicit[k]
    if "framework" not in cfg and payload.get("framework"):
        cfg["framework"] = str(payload["framework"]).strip().lower()
    # Fall back to materialized workload metadata for any unset numeric knob.
    config_path = str(payload.get("config_path") or "")
    if config_path:
        meta = _load_materialized_workload_metadata(config_path)
        wl = (meta.get("runtime_args") or {}).get("workload") if isinstance(meta, dict) else None
        if isinstance(wl, dict):
            for k in ("tp", "isl", "osl", "conc", "num_prompts"):
                if k not in cfg and wl.get(k) is not None:
                    cfg[k] = wl[k]
    return cfg


def _run_combined_e2e_sync(
    results: list[dict[str, Any]], payload: dict, session_dir: Path
) -> dict[str, Any] | None:
    """Apply ALL of a GEAK batch's best patches together and measure E2E (blocking).

    GEAK-only, opt-in. Returns None (skip, no-op) unless: ``payload['combined_e2e']``
    is truthy, the effective backend is geak, a servable ``model_path`` is present,
    and >=1 GPU is visible. Reuses the gate-less ``apply_and_bench`` primitive (the
    single shared apply->rebuild->serve->A/B->revert mechanism). Never raises.

    Args:
        results: The batch's per-kernel result dicts.
        payload: The run_optimization request payload.
        session_dir: Session directory (E2E artifacts written under it).

    Returns:
        dict | None: The apply_and_bench result dict, an ``{status:error}`` dict,
            or ``None`` when the step is skipped.
    """
    if not payload.get("combined_e2e"):
        return None
    order = _backend_order(payload)
    if "geak" not in [b.lower() for b in (order or [])]:
        log.info("combined_e2e: skipped (backend is not geak: %s)", order)
        return None
    model = _resolve_local_model_path(str(payload.get("model_path") or "").strip())
    if not model:
        log.info("combined_e2e: skipped (no model_path to serve)")
        return None
    n_gpus = _visible_gpu_count()
    if not n_gpus or n_gpus < 1:
        log.info("combined_e2e: skipped (no visible GPU)")
        return None
    pairs = _collect_combined_e2e_pairs(results)
    if not pairs:
        log.info("combined_e2e: skipped (no kernel produced a patch)")
        return None

    try:
        tool = _kernel_agent_tool_path("apply_and_bench.py")
    except RuntimeError as exc:
        return {"status": "error", "error": f"apply_and_bench tool not found: {exc}"}
    spec = importlib.util.spec_from_file_location("apply_and_bench", str(tool))
    if spec is None or spec.loader is None:
        return {"status": "error", "error": "could not load apply_and_bench module"}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001  # module import failure is non-fatal to the batch
        return {"status": "error", "error": f"apply_and_bench import failed: {exc}"}

    cfg = _combined_e2e_serving_config(payload)
    # Expose exactly the GPUs the serving TP needs. apply_and_bench sets
    # HIP/ROCR_VISIBLE_DEVICES=gpu, so a hardcoded "0" with tp>1 makes the server
    # request device ordinals that aren't visible -> "HIP error: invalid device
    # ordinal" and a baseline_failed A/B. Map the first `tp` visible GPUs (clamped
    # to n_gpus) so tp=8 serving gets "0,1,...,7".
    _tp = int(cfg.get("tp") or 1)
    _tp = max(1, min(_tp, int(n_gpus)))
    _gpu_ids = ",".join(str(i) for i in range(_tp))
    kwargs: dict[str, Any] = dict(
        pairs=pairs,
        backup_root=str(session_dir / "e2e_backups"),
        model=model,
        out_dir=str(session_dir / "combined_e2e"),
        reps=int(payload.get("e2e_reps", 5)),
        gpu=_gpu_ids,
        aiter_rebuild=True,
    )
    # Only forward knobs we actually resolved; apply_and_bench has its own defaults.
    if cfg.get("framework"):
        kwargs["backend"] = cfg["framework"]
    for k in ("tp", "isl", "osl", "conc", "num_prompts"):
        if cfg.get(k) is not None:
            kwargs[k] = int(cfg[k])
    kwargs["tp"] = _tp  # keep --tp consistent with the exposed GPU set
    log.info("combined_e2e: applying %d patch(es) -> E2E A/B (model=%s, cfg=%s)",
             len(pairs), model, cfg)
    try:
        return mod.apply_and_bench(**kwargs)
    except Exception as exc:  # noqa: BLE001  # serving/bench failure must not fail the opt result
        return {"status": "error", "error": f"apply_and_bench raised: {exc}"}


async def _maybe_run_combined_e2e(
    results: list[dict[str, Any]], payload: dict, session_dir: Path
) -> dict[str, Any] | None:
    """Async wrapper: run the blocking combined-E2E in an executor (off the loop).

    Args:
        results: The batch's per-kernel result dicts.
        payload: The run_optimization request payload.
        session_dir: Session directory for E2E artifacts.

    Returns:
        dict | None: As :func:`_run_combined_e2e_sync`; never raises.
    """
    try:
        # get_running_loop() is the correct call inside an async def (the deprecated
        # get_event_loop() raises "no current event loop" under Python 3.11 when no
        # loop is set on the thread). We are always inside the orchestrator's loop here.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_combined_e2e_sync, results, payload, session_dir)
    except Exception as exc:  # noqa: BLE001  # belt-and-suspenders: never fail the batch
        return {"status": "error", "error": f"combined_e2e wrapper failed: {exc}"}


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
    # Autonomous combined E2E (GEAK-only, opt-in): once all kernels are optimized,
    # apply ALL their best patches together and remeasure end-to-end throughput, with
    # no manual step. Skipped (returns None) unless explicitly requested + servable.
    combined = await _maybe_run_combined_e2e(results, payload, session_dir)
    if combined is not None:
        out["combined_e2e"] = combined
    return out


def _backends_cli_arg(value: Any) -> str:
    """Normalize a payload ``backends`` field into a bare ``--backends`` value.

    The orchestration payload may carry ``backends`` as a bare string
    (``"geak"``), a comma-joined string (``"geak,claude"``), or a JSON list
    (``["geak"]``) when an upstream request serializes it as an array. A list
    MUST be comma-joined into bare names, never ``str()``-ed into the repr of a
    list (``"['geak']"``) — the kernel-agent's ``parse_backends`` validator
    correctly rejects that opaque token and the dispatch fails with the
    self-contradictory "unsupported backend(s): ['geak']".

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
        # per-kernel budget so a fallback never overshoots it.
        timeout_sec = max(1, min(timeout_sec, int(timeout_override_sec)))

    from ..actions.executors._multi_node_env import is_multi_node

    if is_multi_node():
        from hyperloom.inference_optimizer.multi_node.cli import (
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
        # different optimizer (e.g. claude) actually ran.
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
        # when this run dispatched a single, unambiguous backend.
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
    from hyperloom.inference_optimizer.session_paths import forge_steps_path
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
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

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
        from ..state.shared_state import SharedState

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
        from ..loop.coordinator_helpers import format_exc_brief

        reason = format_exc_brief(exc)
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
    from ..actions.executors.baseline import BaselineExecutor
    from ..actions.executors.benchmark_result import is_valid_measurement
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare {kernel_id} payload isn't failed with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)

    base_tput = float(payload.get("base_tput", 0.0))
    if base_tput <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }

    # Route through the compat helper so a legacy ``extra_sglang_args`` envelope still resolves.
    from hyperloom.inference_optimizer.compat.payload_aliases import read_extra_server_args

    env_only_validation = (
        str(payload.get("source") or "").strip() in {"forge_gemm_tuning", "gemm_tuning"}
        and (bool(payload.get("extra_envs")) or bool(read_extra_server_args(payload).strip()))
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
    apply_result = _maybe_apply_kernel_patch(
        payload,
        session_dir=session_dir,
        kernel_id=kernel_id,
    )
    if apply_result.get("status") == "skipped" and env_only_validation:
        apply_result = {
            "status": "ok",
            "reason": "env_only_validation",
            "kernel_id": kernel_id,
        }
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
    extra_args = read_extra_server_args(payload).strip()
    # VRAM barrier (flag-gated, default off): cap re-baseline util so the
    # integrate server cannot OOM on a tighter node. No-op when off.
    extra_args = _vram_guarded_server_args(extra_args)

    # Wrap BaselineExecutor in a Task/RunnerContext; extra_server_args goes via task params (forward compat).
    from hyperloom.inference_optimizer.session_paths import runs_dir

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
    from ..actions.executors._multi_node_env import is_multi_node

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
    from ..gain_math import gain_pct_or_zero, incremental_gain_pct

    # base_tput > 0 already guaranteed by the early guard above.
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

    # import-grep source confirmation (flag-gated, default off). Advisory by
    # default: annotate whether the served process actually imported/compiled the
    # patched source. Only the *strict* sub-flag enforces it, and only on
    # positive non-import evidence (confirmed is False) — an "unknown" (None)
    # never penalizes a real win. Off => no annotation, decision unchanged.
    source_import_confirmed: bool | None = None
    source_not_imported_downgrade = False
    if _honest_flag("HL_CONFIRM_SOURCE_IMPORTED"):
        _src = str(payload.get("target_file") or payload.get("source_file") or "")
        source_import_confirmed = _confirm_source_imported(_src, bench_result.get("workspace"))
        if (
            decision == "KEEP"
            and source_import_confirmed is False
            and _honest_flag("HL_CONFIRM_SOURCE_IMPORTED_STRICT")
        ):
            decision = "NEEDS_REVIEW"
            source_not_imported_downgrade = True

    # Paired same-config A/B confirmation (Tier-2; opt-in via its OWN explicit
    # flag, deliberately NOT enabled by the HL_HONEST_E2E umbrella — it does a
    # second server launch and revert/re-apply, so it stays opt-in until a full
    # workload run validates it). The gain above is vs a STORED base_tput scalar
    # (possibly a different config/run). When a candidate clears KEEP against the
    # stored scalar, re-confirm it against a PAIRED pristine baseline measured
    # under the *same* config in this call: revert -> measure pristine ->
    # recompute gain. A confirmed KEEP is re-applied; a disconfirmed KEEP drops
    # to NEEDS_REVIEW (never a false KEEP). ANY failure restores the applied
    # state and falls back to the stored-scalar decision, so it never breaks a run.
    paired_ab: dict[str, Any] | None = None
    paired_pristine_revert: HandlerResult | None = None
    if (
        os.environ.get("HL_INTEGRATE_PAIRED_AB", "").strip().lower() in _TRUEY
        and decision == "KEEP"
        and apply_result.get("status") == "ok"
    ):
        paired_ab = {"status": "attempted"}
        try:
            paired_pristine_revert = _maybe_revert_kernel_patch(apply_result)
            paired_ws = runs_dir(session_dir, "integrate", f"{fake_task_id}-pairedbase")
            paired_ws.mkdir(parents=True, exist_ok=True)
            paired_task = Task(
                task_id=f"{fake_task_id}-pairedbase",
                kind="baseline",
                state="running",
                params={
                    "config_path": payload.get("config_path"),
                    "output_dir": str(paired_ws),
                    "timeout_sec": int(payload.get("budget_minutes", 20)) * 60,
                    "extra_server_args": extra_args,
                    "extra_envs": dict(payload.get("extra_envs") or {}),
                },
                idempotency_key=f"{fake_task_id}-pairedbase",
            )
            paired_bench = await BaselineExecutor(session_dir=session_dir)(RunnerContext(task=paired_task, lease=None))
            if is_valid_measurement(paired_bench):
                paired_base_tput = float(paired_bench.get("output_throughput") or 0.0)
                paired_gain = gain_pct_or_zero(new_tput, paired_base_tput)
                paired_ab.update(
                    {
                        "status": "ok",
                        "paired_base_tput": paired_base_tput,
                        "paired_gain_pct": paired_gain,
                        "stored_base_tput": base_tput,
                        "stored_gain_pct": gain_pct,
                    }
                )
                if paired_gain > keep_threshold_pct:
                    # Confirmed: re-apply so the KEEP lands the patch.
                    reapply = _maybe_apply_kernel_patch(payload, session_dir=session_dir, kernel_id=kernel_id)
                    if reapply.get("status") == "ok":
                        apply_result = reapply
                        paired_pristine_revert = None  # patch is back; normal revert logic applies
                        base_tput = paired_base_tput
                        gain_pct = paired_gain
                        paired_ab["confirmed"] = True
                    else:
                        paired_ab.update({"confirmed": False, "reapply_failed": True})
                        decision = "NEEDS_REVIEW"
                else:
                    # Disconfirmed: leave reverted, drop to NEEDS_REVIEW.
                    base_tput = paired_base_tput
                    gain_pct = paired_gain
                    decision = "NEEDS_REVIEW"
                    paired_ab["confirmed"] = False
            else:
                # Paired measurement failed: restore applied state, keep stored decision.
                reapply = _maybe_apply_kernel_patch(payload, session_dir=session_dir, kernel_id=kernel_id)
                if reapply.get("status") == "ok":
                    apply_result = reapply
                    paired_pristine_revert = None
                paired_ab["status"] = "measurement_failed"
        except Exception as exc:  # noqa: BLE001 — never break integrate on paired-AB
            log.exception("paired A/B confirmation failed; falling back to stored-scalar decision")
            try:
                reapply = _maybe_apply_kernel_patch(payload, session_dir=session_dir, kernel_id=kernel_id)
                if reapply.get("status") == "ok":
                    apply_result = reapply
                    paired_pristine_revert = None
            except Exception:  # noqa: BLE001
                pass
            paired_ab = {"status": "error", "error": repr(exc)}

    revert_result = (
        {"status": "skipped", "reason": "KEEP decision"}
        if decision == "KEEP"
        # If the paired pass already reverted (disconfirmed KEEP), reuse that
        # result instead of double-reverting an already-reverted manifest.
        else (
            paired_pristine_revert if paired_pristine_revert is not None else _maybe_revert_kernel_patch(apply_result)
        )
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
    if source_import_confirmed is not None:
        result["source_import_confirmed"] = source_import_confirmed
    if source_not_imported_downgrade:
        result["decision_reason"] = "source_not_confirmed_imported"
    if paired_ab is not None:
        result["paired_ab"] = paired_ab
        if paired_ab.get("confirmed") is False:
            result["decision_reason"] = "paired_ab_disconfirmed"
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




































__all__ = [
    "KERNEL_REQUEST_HANDLERS",
    "get_handler",
    "has_handler",
    "integrate_handler",
    "run_gemm_tuning_handler",
    "run_optimization_handler",
    "trace_analyze_handler",
]
