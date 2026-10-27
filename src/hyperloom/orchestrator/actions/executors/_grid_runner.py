# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helper for the ``explore`` executor's grid runs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, NamedTuple, NoReturn

import yaml

from hyperloom.common.env import env_flag, is_truthy
from hyperloom.common.perf_metric import constraint_fields
from hyperloom.common.env_safety import (
    BLOCKED_EXTERNAL_ENV_NAMES,
    is_python_package_root,
    redact_secret_values,
    scrub_benchmark_process_env,
    ungrounded_env_file_reason,
)

from ...phases import machine_state as _phase_state
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows, report_progress
from ..stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ._benchmark_runtime import apply_runtime_benchmark_overrides as apply_runtime_benchmark_overrides
from ._benchmark_interpreter import _resolve_probe_python as _resolve_probe_python
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_ERROR_CLASS,
    AGENTX_PREFLIGHT_RETURNCODE,
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
    DETOKENIZER_STALL_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundWatch,
    resolve_benchmark_timeouts,
    run_with_session_kill,
    server_log_death_excerpt,
    session_deadline_to_remaining_sec,
    sync_agentx_phase_wait,
)
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    round_succeeded,
    select_run_workspace,
    served_complete_protocol,
    snapshot_workspaces,
)
from ._gpu_metrics import write_gpu_metrics_from_report
from .benchmark_backend import build_benchmark_command
from .benchmark_config import apply_benchmark_round_seed, prepare_benchmark_runtime
from ._launch_evidence import RequestedLaunch, build_launch_evidence, persist_launch_evidence
from ...measurement.runtime_findings import persist_runtime_findings, scan_server_log
from ._spec_decode_pin import SPECULATIVE_DECODING_PINNED, refuse_unpinned_speculative_change
from hyperloom.common.agentx_workload import pin_mlperf_round_concurrency
from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.runtime_environment import apply_runtime_override
from ._recipe_script import launcher_overwritten_envs
from ._server_argv import seal_server_argv
from .magpie_agentx import LaunchChanges, ServerArgvInvalidError, materialize_variant
from ._workload_envs import BenchmarkTarget, LaunchLayer

# Re-exported from sibling modules to keep the module namespace intact.
from ._grid_base import (
    DEFAULT_KEEP_THRESHOLD_PCT as DEFAULT_KEEP_THRESHOLD_PCT,
    GridVariant as GridVariant,
    VariantResult as VariantResult,
)
from hyperloom.inference_optimizer.grid_server_args import (
    SGLANG_MOE_RUNNER_BACKEND_RE,
    moe_runner_requires_aiter,
    remove_moe_runner_backend_arg,
    server_args_env_name as server_args_env_name,
    merge_server_args as merge_server_args,
    compose_server_args as compose_server_args,
    dedupe_extra_server_args,
)
from ._grid_variant_filter import (
    resolve_skip_spec as resolve_skip_spec,
    _MN_PARAMS_PRIORITY as _MN_PARAMS_PRIORITY,
    _MN_BACKENDS_PRIORITY as _MN_BACKENDS_PRIORITY,
    reorder_grid_for_multi_node as reorder_grid_for_multi_node,
    apply_multi_node_invalid_variants as apply_multi_node_invalid_variants,
    apply_aiter_moe_pin_filter as apply_aiter_moe_pin_filter,
    apply_compatibility_filter as apply_compatibility_filter,
    apply_user_skip_list as apply_user_skip_list,
)


log = logging.getLogger(__name__)

#: Sub-directory of a variant's slot that its discarded warmup round runs in.
_WARMUP_ROUND_DIR = "warmup_round"
#: A variant whose warm-up setup or warm-up YAML could not be built.
_WARMUP_YAML_BUILD_ERROR = "warmup_yaml_build_error"


def _validate_magpie_python_override(value: str) -> str:
    """Validate caller-supplied Magpie interpreter overrides before argv[0]."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if any(ch in raw for ch in ("\x00", "\n", "\r")):
        raise ValueError("magpie_python contains control characters")
    resolved = Path(raw)
    if not resolved.is_absolute():
        found = shutil.which(raw)
        if not found:
            raise ValueError(f"magpie_python is not executable on PATH: {raw}")
        resolved = Path(found)
    if not resolved.is_file():
        raise ValueError(f"magpie_python is not a file: {raw}")
    if not resolved.name.lower().startswith("python"):
        raise ValueError(f"magpie_python must point to a Python interpreter, got: {raw}")
    return str(resolved)


class _Halt(Enum):
    """What one variant's end means for the rest of the grid: walk on, stop unless told otherwise, or stop."""

    CONTINUE = auto()
    ON_FAILURE = auto()
    STOP = auto()


class _VariantAborted(Exception):
    """A variant ended before its result was assembled, carrying what the grid records for it."""

    def __init__(
        self, result: VariantResult, *, halt: _Halt = _Halt.ON_FAILURE, abandoned: StoppedByTheRun | None = None
    ) -> None:
        super().__init__(result.error_class)
        self.result = result
        self.halt = halt
        self.abandoned = abandoned


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root."""
    from hyperloom.inference_optimizer.session.paths import session_dir as _sd

    return _sd()


class SessionDirField:
    """An executor's ``session_dir``, resolved on read rather than at construction.

    The executors here are module-level singletons built at import, which is
    before the CLI pins the session. Resolving in ``__init__`` freezes them on
    the workspace root that concurrent sessions share.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._slot = f"_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        explicit = getattr(obj, self._slot, None)
        return explicit if explicit is not None else _resolve_session_dir()

    def __set__(self, obj: Any, value: Path | str | None) -> None:
        setattr(obj, self._slot, Path(value) if value else None)


# SKIP_VARIANTS: comma/whitespace patterns matched (exact or fnmatch) against ``GridVariant.name``.


# Env-flag capability probe: a serving env flag can be defined in the build yet still crash the server at engine init
# because the code path it activates imports a module the build did not package.
_UNSET = object()

# Cached probe result keyed by framework.
_CAP_PROBE_CACHE: dict[str, str | None] = {}

# Subprocess probe: locate the installed vLLM package via ``find_spec`` without importing it, read the aiter
# shared-expert router source, extract the ``fused_moe.*`` modules it imports, and verify each resolves to a real file
# in THIS build.
_AITER_SHARED_EXPERT_PROBE_SCRIPT = (
    "import importlib.util as u, os, re, json\n"
    "def go():\n"
    "    try:\n"
    "        spec = u.find_spec('vllm')\n"
    "    except Exception:\n"
    "        return {'status': 'unknown'}\n"
    "    if not spec or not spec.origin:\n"
    "        return {'status': 'unknown'}\n"
    "    vdir = os.path.dirname(spec.origin)\n"
    "    router = os.path.join(vdir, 'model_executor', 'layers', 'fused_moe',\n"
    "                          'router', 'aiter_shared_routed_fused_moe_router.py')\n"
    "    if not os.path.exists(router):\n"
    "        return {'status': 'unknown'}\n"
    "    try:\n"
    "        text = open(router, encoding='utf-8').read()\n"
    "    except Exception:\n"
    "        return {'status': 'unknown'}\n"
    "    mods = re.findall(r'from\\s+(vllm\\.model_executor\\.layers\\.fused_moe\\.[A-Za-z0-9_.]+)\\s+import', text)\n"
    "    missing = []\n"
    "    for m in sorted(set(mods)):\n"
    "        rel = m[len('vllm.'):].replace('.', os.sep)\n"
    "        if not (os.path.exists(os.path.join(vdir, rel + '.py'))\n"
    "                or os.path.exists(os.path.join(vdir, rel, '__init__.py'))):\n"
    "            missing.append(m)\n"
    "    return {'status': 'unsupported', 'missing': missing} if missing else {'status': 'ok'}\n"
    "print(json.dumps(go()))\n"
)


def _probe_vllm_aiter_shared_expert_unsupported() -> str | None:
    """Return a drop reason if the installed vLLM build can't honour the aiter shared-expert fusion flag, else ``None``."""
    cached = _CAP_PROBE_CACHE.get("vllm", _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]
    try:
        proc = subprocess.run(
            [_resolve_probe_python(), "-c", _AITER_SHARED_EXPERT_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        payload = json.loads(lines[-1]) if lines else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    status = payload.get("status")
    if status == "ok":
        _CAP_PROBE_CACHE["vllm"] = None
        return None
    if status == "unsupported":
        missing = ", ".join(payload.get("missing") or []) or "(unknown module)"
        reason = (
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS enabled but the installed "
            "vLLM build's aiter shared-expert router imports missing module(s): "
            f"{missing}. Flag unusable on this build (server crashes at engine "
            "init); upgrade vLLM to a build that ships the module."
        )
        _CAP_PROBE_CACHE["vllm"] = reason
        return reason
    return None  # unknown => do not drop or cache


def unsupported_capability_reason(variant: "GridVariant") -> str | None:
    """Return why a variant cannot boot (a config file not on disk, an env flag the build lacks), else ``None``."""
    envs = {str(k): str(v) for k, v in (getattr(variant, "extra_envs", None) or {}).items()}
    if missing_file := ungrounded_env_file_reason(envs):
        return missing_file
    if (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower() != "vllm":
        return None
    val = envs.get("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS")
    if val is None or not is_truthy(val, default=True):
        return None
    return _probe_vllm_aiter_shared_expert_unsupported()


# Sanitization for LLM-supplied overrides (benchmark_script / result_dir): reject path separators / shell
# metacharacters, raising ``ValueError`` instead of running an unsafe subprocess.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")
_RESULT_DIR_FORBID_RE = re.compile(r"[\s\"'`$;&|<>(){}\[\]\\*?!]")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe Magpie benchmark script file name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SCRIPT_NAME_RE.match(text):
        raise ValueError(
            f"benchmark_script={text!r} rejected: must be a bare *.sh "
            "file name (no path separators, no shell metacharacters)"
        )
    return text


def sanitize_result_dir(value: Any) -> str | None:
    """Return ``value`` if it's a safe absolute (or workspace-relative) dir."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _RESULT_DIR_FORBID_RE.search(text):
        raise ValueError(
            f"result_dir={text!r} rejected: contains whitespace or shell "
            "metacharacters; pass an absolute or workspace-relative path"
        )
    return text


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    """Load ``benchmark_report.json`` from a benchmark workspace."""
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# How long to keep re-reading ``benchmark_report.json`` when the report does not yet parse into a valid measurement.
REPORT_SETTLE_SECONDS = 30.0
REPORT_SETTLE_POLL_SECONDS = 1.0


async def _settled_measurement(
    workspace: Path,
    *,
    subprocess_started_unix: float | None,
    config_path: Path | None = None,
    settle_seconds: float = REPORT_SETTLE_SECONDS,
    poll_seconds: float = REPORT_SETTLE_POLL_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read the benchmark report, re-reading briefly while it is still settling."""
    deadline = time.monotonic() + max(0.0, float(settle_seconds))
    attempts = 0
    while True:
        report = _parse_report(workspace)
        measurement = extract_benchmark_measurement(
            report,
            config_path=config_path,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        attempts += 1
        succeeded = round_succeeded(measurement)
        if succeeded or time.monotonic() >= deadline:
            if attempts > 1 and succeeded:
                log.info(
                    "grid_runner: benchmark_report became valid after %d read(s); "
                    "the report was still being written when the subprocess was reaped",
                    attempts,
                )
            # Harvest already tried this, but it runs before the report is guaranteed to exist: a report Magpie
            # finishes writing after the subprocess is reaped would otherwise leave the round with no GPU artifact at
            # all. Here the report is in hand, so write it from that rather than reading the file a second time.
            if report is not None:
                write_gpu_metrics_from_report(workspace, report, source="benchmark_report.json")
            return report, measurement
        await asyncio.sleep(max(0.01, float(poll_seconds)))


def _run_grid_warmup_enabled() -> bool:
    """Whether ``run_grid`` should discard a cold warmup round when possible."""
    return env_flag("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", default=not os.environ.get("PYTEST_CURRENT_TEST"))


def _prepend_magpie_pythonpath(magpie_dir: str, current_pythonpath: str) -> str:
    """Prepend Magpie's import root to PYTHONPATH, skipping package-root dirs."""
    if not magpie_dir or is_python_package_root(magpie_dir):
        return current_pythonpath
    return f"{magpie_dir}:{current_pythonpath}" if current_pythonpath else magpie_dir


def sync_benchmark_timeout(config_path: Path, timeout_sec: float) -> None:
    """Give Magpie and bypass the same cap as the enclosing benchmark process."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = cfg["benchmark"]
    bench["timeout_seconds"] = timeout_sec
    if bench.get("server_lifecycle"):
        bench["server_lifecycle"]["server_ready_timeout_s"] = timeout_sec
    envs = bench.setdefault("envs", {})
    envs["PYTHONUNBUFFERED"] = "1"
    sync_agentx_phase_wait(envs, timeout_sec)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _benchmark_server_log(config_path: Path, output_dir: Path) -> str | None:
    """Scriptable benchmarks have no server whose readiness can arm silence."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    framework = str(cfg["benchmark"].get("framework", "")).lower()
    return None if framework in {"custom", "xdit"} else str(output_dir / "server.log")


@dataclass(frozen=True)
class GridRecipe:
    """The base config a grid's variants are materialized from, the layer they launch on, and their target."""

    base_yaml_path: Path
    base: LaunchLayer = LaunchLayer()
    target: BenchmarkTarget = BenchmarkTarget()


@dataclass(frozen=True)
class GridLaunch:
    """How each round of a grid is launched: its benchmark interpreter and result dir, and the server it runs on."""

    magpie_python: str | None = None
    result_dir: str | None = None
    serving_lease: Any = None
    server_lifecycle: dict[str, Any] | None = None
    server_already_ready: bool = False
    ready_server_log: str | None = None
    warmup_before_measure: bool | None = None
    lifecycle_boot_only: bool = False
    # The grid's measured round is itself a caller's discarded warmup, so it must not send the prompts of the
    # measured round that follows it on the same server.
    warmup_round: bool = False


@dataclass(frozen=True)
class GridStop:
    """When a grid must stop: the session deadline and its owner, a variant's expected cost, and failure policy."""

    session_deadline_sec: float | None = None
    variant_expected_sec: float | None = None
    deadline_stop: StoppedByTheRun = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
    keep_going_on_failure: bool = True


@dataclass(frozen=True)
class MagpieRound:
    """One Magpie benchmark round: the interpreter, config and directories it runs with, and its stop gates."""

    magpie_python: str
    config_path: Path
    output_dir: Path
    timeout_sec: int
    cwd: str
    result_dir: str | None = None
    silence_timeout_sec: float | None = None
    server_already_ready: bool = False
    session_deadline_sec: float | None = None
    warmup: bool = False


def _run_magpie(
    bench_round: MagpieRound,
    *,
    unset_envs: list[str] | None = None,
    serving_lease: Any = None,
    on_output: Callable[[], None] | None = None,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr)."""
    config_path = bench_round.config_path
    output_dir = bench_round.output_dir
    sync_benchmark_timeout(config_path, bench_round.timeout_sec)
    server_log_path = _benchmark_server_log(config_path, output_dir)
    env = scrub_benchmark_process_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    from ._workload_envs import resolve_reference_launch

    _args, _envs, reference_controls = resolve_reference_launch()
    for name in [*(reference_controls.get("unset_envs") or []), *(unset_envs or [])]:
        if name.strip().upper() not in BLOCKED_EXTERNAL_ENV_NAMES:
            env.pop(name, None)
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_PATH") or ""
    if magpie_dir:
        env["PYTHONPATH"] = _prepend_magpie_pythonpath(magpie_dir, env.get("PYTHONPATH", ""))

    # Multi-node: tell Magpie to skip its local-server launch and point benchmark_serving at the head pod's ClusterIP.
    from ._multi_node_env import magpie_remote_env

    env.update(magpie_remote_env())

    preparation_failure = prepare_benchmark_runtime(config_path, env)
    if preparation_failure is not None:
        return preparation_failure.returncode, "", preparation_failure.error
    apply_benchmark_round_seed(config_path, env, warmup=bench_round.warmup)
    # RESULT_DIR default; leaks are picked up by the salvage path.
    env["RESULT_DIR"] = bench_round.result_dir or str(output_dir)
    # InferenceX ``run_lm_eval`` cleans ``$EVAL_RESULT_DIR`` after processing lm-eval output.
    env["EVAL_RESULT_DIR"] = str(Path(env["RESULT_DIR"]) / "eval_output")
    # Pin SERVER_LOG / GPU_METRICS_CSV per-task so logs land alongside ``benchmark_report.json``.
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

    # Ray-managed GPU execution (§12 T1): route the round through the serving lease's actor, which holds ``num_gpus``
    # across every round sharing this server.
    if serving_lease is not None:
        from ._ray_backend import SessionKillRound, strip_visible_devices_from_config

        ray_config_path = strip_visible_devices_from_config(config_path)
        cmd = build_benchmark_command(
            python_exe=bench_round.magpie_python,
            config_path=ray_config_path,
            output_dir=output_dir,
        )
        # ``on_output`` cannot follow the round here: the benchmark runs inside a Ray actor in another process
        # (potentially on another node) and only its final ``(rc, stdout, stderr)`` comes back, so there is nothing
        # local to call per line.
        return serving_lease.run_session_kill(
            SessionKillRound(
                cmd,
                env=env,
                cwd=bench_round.cwd,
                timeout=bench_round.timeout_sec,
                silence_timeout_sec=bench_round.silence_timeout_sec,
                server_log_path=server_log_path,
                server_already_ready=bench_round.server_already_ready,
                # Converted here, at the last moment before the process boundary: the actor cannot read this process's
                # monotonic clock.
                session_remaining_sec=session_deadline_to_remaining_sec(bench_round.session_deadline_sec),
            )
        )

    cmd = build_benchmark_command(
        python_exe=bench_round.magpie_python,
        config_path=config_path,
        output_dir=output_dir,
    )
    # The launch puts Magpie in its own POSIX session and tears down the whole
    # descendant tree on every exit path.
    proc = run_with_session_kill(
        cmd,
        env=env,
        cwd=bench_round.cwd,
        watch=RoundWatch(
            timeout=bench_round.timeout_sec,
            silence_timeout_sec=bench_round.silence_timeout_sec,
            server_log_path=server_log_path,
            server_already_ready=bench_round.server_already_ready,
            session_deadline_sec=bench_round.session_deadline_sec,
        ),
        on_output=on_output,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _num_gpus_for_config(config_path: Path) -> float:
    """Read the tensor-parallel size (``TP``) from a materialized benchmark YAML."""
    try:
        with Path(config_path).open(encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
        envs = (cfg.get("benchmark") or {}).get("envs") or {}
        return float(int(envs.get("TP", 1) or 1))
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):  # best-effort; default to 1 GPU
        return 1.0


def _mn_restart_env(
    reference_envs: dict[str, str],
    variant: Any,
    unset_envs: list[str],
) -> dict[str, str]:
    """Merge the reference server envs under one variant's own overrides."""
    merged = {k: v for k, v in reference_envs.items() if k not in unset_envs}
    merged.update({str(k): str(v) for k, v in (variant.extra_envs or {}).items()})
    return merged


def _resolve_mn_effective_server_args(
    cfg_path: Path,
    base_yaml_path: Path,
    variant: Any,
    *,
    base_extra_args: str,
    base_args_mode: str,
) -> str:
    """Resolve the multi-node server args for a variant restart; prefer the materialized variant YAML, falling back to a recompose from the base YAML."""
    try:
        with cfg_path.open(encoding="utf-8") as _f:
            _variant_cfg = yaml.safe_load(_f) or {}
        _variant_bench = _variant_cfg.get("benchmark") or {}
        _variant_envs = _variant_bench.get("envs") or {}
        _variant_framework_env = server_args_env_name(_variant_bench.get("framework"))
        return str(_variant_envs.get(_variant_framework_env) or "")
    except (OSError, ValueError, yaml.YAMLError, AttributeError):
        log.debug(
            "grid_runner: failed to read materialized variant args from %s",
            cfg_path,
            exc_info=True,
        )
        with base_yaml_path.open(encoding="utf-8") as _f:
            _base_cfg = yaml.safe_load(_f) or {}
        _base_bench = _base_cfg.get("benchmark") or {}
        _base_envs = _base_bench.get("envs") or {}
        _base_framework_env = server_args_env_name(_base_bench.get("framework"))
        _fallback_inherited_args = str(_base_envs.get(_base_framework_env) or "")
        return dedupe_extra_server_args(
            compose_server_args(
                inherited_args="" if str(base_args_mode).strip().lower() == "replace" else _fallback_inherited_args,
                base_extra_args=base_extra_args,
                variant_extra_args=variant.extra_server_args,
                remove_args=getattr(variant, "remove_args", []),
                args_mode=getattr(variant, "args_mode", "append"),
            )
        )


def _variant_progress_note(
    grid: list[GridVariant],
    results: list[VariantResult],
    idx: int,
) -> dict[str, Any]:
    """Build the progress note for the variant at ``idx`` from that variant's own row."""
    landed = results[idx] if idx < len(results) else None
    return {
        "unit": "variant",
        "label": grid[idx].name,
        "index": idx + 1,
        "total": len(grid),
        "status": getattr(landed, "status", None),
        "output_throughput": getattr(landed, "output_throughput", None),
    }


# The returncode side of :mod:`...stop_attribution`: the same two causes, keyed by the sentinel a subprocess comes
# back with.
_STOPPED_BY_THE_RUN: dict[int, StoppedByTheRun] = {
    SESSION_TIME_EXHAUSTED_RETURNCODE: STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
    ORCHESTRATOR_CANCELLED_RETURNCODE: STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS],
}


def stopped_by_the_run(returncode: int | None) -> StoppedByTheRun | None:
    """Return how to record a round the run itself stopped, if it did."""
    if returncode is None:
        return None
    return _STOPPED_BY_THE_RUN.get(int(returncode))


def session_grid_bounds(shared_state: Any) -> tuple[float | None, float | None]:
    """Resolve ``(session_deadline_sec, variant_expected_sec)`` for a :func:`run_grid` call."""
    if shared_state is None:
        return (None, None)
    deadline_fn = getattr(shared_state, "grid_session_deadline_sec", None)
    deadline = deadline_fn() if callable(deadline_fn) else None
    variant_sec = _phase_state.one_more_measurement_sec(shared_state)
    if variant_sec is None:
        variant_sec = _phase_state.measured_seconds(shared_state, "baseline_runtime_sec")
    return (deadline, variant_sec)


class _RoundProcess(NamedTuple):
    """What the benchmark subprocess of one grid round returned."""

    returncode: Any
    stdout: str
    stderr: str


class _RoundError(NamedTuple):
    """Why a grid round measured nothing, and under which class it is recorded."""

    summary: str
    error_class: str


def _invalid_measurement_error(
    proc: _RoundProcess,
    *,
    report: Any,
    death_excerpt: str,
    workspace: Any,
    slot: Any,
) -> _RoundError:
    if proc.returncode != 0:
        error = death_excerpt or redact_secret_values((proc.stderr or proc.stdout)[-2000:])
        if not error.strip():
            error = redact_secret_values(_on_disk_stderr_tail(workspace, slot))
        if not error.strip():
            error = redact_secret_values(_report_errors_summary(report))
        return _RoundError(error, "magpie_nonzero_invalid_measurement")
    if not report:
        return _RoundError(death_excerpt or "benchmark_report missing", "benchmark_report_missing")
    return _RoundError(
        death_excerpt or "benchmark_report missing valid throughput/completed requests",
        "benchmark_report_invalid_metric",
    )


# A preflight abort is an environment fault: the AgentX client is missing for the whole grid, so every
# variant after it is recorded as never started rather than dropped from the returned points.
_AGENTX_PREFLIGHT_ABANDONED = StoppedByTheRun(
    error_class=AGENTX_PREFLIGHT_ERROR_CLASS,
    interrupted="AgentX preflight aborted the grid while this round was running",
    never_started="AgentX preflight aborted the grid before this variant ran",
    ends_the_batch=True,
)


def _multi_node_warmup_rounds(variant_expected_sec: Any) -> int:
    """The extra warmup pass a multi-node variant costs, counted only where rounds are budgeted at all."""
    if variant_expected_sec is None:
        return 0
    from ._multi_node_env import is_multi_node, mn_bench_warmup_enabled

    return 1 if (is_multi_node() and mn_bench_warmup_enabled()) else 0


class _VariantSlot(NamedTuple):
    """One variant's place in the grid and the directory its rounds write to."""

    idx: int
    variant: GridVariant
    path: Path


class _VariantFailure(NamedTuple):
    """Why one variant ended, and what its ending means for the rest of the grid.

    ``error`` is the row's own message when it says more than the marker's ``recorded``
    summary; ``abandoned`` is why the variants a halted grid will not reach never started,
    and is set exactly when ``halt`` is ``STOP``.
    """

    error_class: str
    recorded: str
    error: str | None = None
    halt: _Halt = _Halt.ON_FAILURE
    abandoned: StoppedByTheRun | None = None


class _RoundAttempt(NamedTuple):
    """One variant's run of one round, and what it left behind.

    ``server_survivors`` are the servers the variant booted that outlived its teardown.
    """

    at: _VariantSlot
    returncode: int | None
    started_unix: float
    server_log: Path
    server_survivors: tuple[int, ...]


class _GridRun(NamedTuple):
    """One grid: its variants, the recipe they derive from, how each is launched, and when it must stop."""

    grid: list[GridVariant]
    output_root: Path
    recipe: GridRecipe
    launch: GridLaunch
    stop: GridStop
    magpie_python: str
    silence_timeout_sec: float
    benchmark_timeout_sec: float
    rounds_per_variant: int
    mn_warmup_rounds: int
    mn_reference_envs: dict[str, str]
    auto_warmup_requested: bool


class _VariantRounds(NamedTuple):
    """The config one variant's rounds share, and the server they were given."""

    cfg_path: Path
    lifecycle: dict[str, Any]
    auto_warmup: bool
    mn_effective_args: str


class _BenchmarkPass(NamedTuple):
    """One benchmark pass of one variant, and what it left behind.

    ``server_survivors`` are the servers the pass booted that outlived the teardown that
    followed it; a pass whose servers the next round still needs leaves it empty.
    """

    proc: _RoundProcess
    started_unix: float
    workspaces_before: Any
    server_survivors: tuple[int, ...] = ()


class _MeasuredVariant(NamedTuple):
    """One variant's measured round, the workspace and salvage it left on disk, and the warmup it discarded."""

    rounds: _VariantRounds
    round: _BenchmarkPass
    workspace: Path
    artifacts: list[Any]
    warmup_tput: float | None


class _VariantOutcome(NamedTuple):
    """A measured variant together with what its benchmark report turned out to say."""

    measured: _MeasuredVariant
    report: dict[str, Any] | None
    measurement: dict[str, Any]
    warnings: list[str]


def _variant_tag(run: _GridRun, at: _VariantSlot) -> str:
    return f"{at.idx + 1}/{len(run.grid)} name={at.variant.name}"


def _abort_variant(at: _VariantSlot, failure: _VariantFailure, **fields: Any) -> NoReturn:
    _write_variant_abort_marker(
        at.path,
        variant_name=at.variant.name,
        error_class=failure.error_class,
        error_summary=failure.recorded,
        extra_args=at.variant.extra_server_args,
    )
    raise _VariantAborted(
        VariantResult(
            name=at.variant.name,
            extra_server_args=at.variant.extra_server_args,
            extra_envs=dict(at.variant.extra_envs),
            status="failed",
            error=failure.recorded if failure.error is None else failure.error,
            error_class=failure.error_class,
            note=at.variant.note,
            **fields,
        ),
        halt=failure.halt,
        abandoned=failure.abandoned,
    )


def _abort_on_round_stop(
    run: _GridRun,
    stopped: StoppedByTheRun,
    attempt: _RoundAttempt,
    *,
    round_label: str,
) -> NoReturn:
    if attempt.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE:
        stopped = run.stop.deadline_stop
    at = attempt.at
    runtime_sec = round(max(0.0, time.time() - attempt.started_unix), 2)
    log.warning(
        "grid_runner: variant %s %s round reaped after %.1fs: %s; recorded as skipped, not failed",
        _variant_tag(run, at),
        round_label,
        runtime_sec,
        stopped.interrupted,
    )
    teardown = "tree reaped"
    if attempt.server_survivors:
        teardown += f", but server pid(s) {', '.join(map(str, attempt.server_survivors))} survived SIGKILL"
    _write_variant_abort_marker(
        at.path,
        variant_name=at.variant.name,
        error_class=stopped.error_class,
        error_summary=f"{stopped.interrupted}; {teardown}",
        extra_args=at.variant.extra_server_args,
    )
    ends_the_batch = stopped.ends_the_batch
    raise _VariantAborted(
        VariantResult(
            name=at.variant.name,
            extra_server_args=at.variant.extra_server_args,
            extra_envs=dict(at.variant.extra_envs),
            status="skipped",
            returncode=attempt.returncode,
            runtime_sec=runtime_sec,
            error=stopped.interrupted,
            error_class=stopped.error_class,
            server_log_path=_measurement_server_log_path(attempt.server_log, slot=at.path),
            note=at.variant.note,
            nonfatal_warnings=[f"server_survived_teardown:{pid}" for pid in attempt.server_survivors],
        ),
        halt=_Halt.STOP if ends_the_batch else _Halt.ON_FAILURE,
        abandoned=stopped if ends_the_batch else None,
    )


def _abandon_rest_of_grid(run: _GridRun, at: _VariantSlot) -> NoReturn:
    stop = run.stop.deadline_stop
    raise _VariantAborted(
        _not_run_skip_result(at.variant, stop),
        halt=_Halt.STOP,
        abandoned=stop,
    )


def _warn_budget_cannot_admit(run: _GridRun, idx: int, *, rounds_left: int, spent_on: str) -> bool:
    """Warn when the session budget can no longer admit ``rounds_left`` rounds, and say whether it can."""
    if run.stop.session_deadline_sec is None:
        return False
    remaining_sec = run.stop.session_deadline_sec - time.monotonic()
    required_sec = (
        float(run.stop.variant_expected_sec) * rounds_left if run.stop.variant_expected_sec is not None else None
    )
    if remaining_sec > 0 and (required_sec is None or remaining_sec >= required_sec):
        return False
    log.warning(
        "grid_runner: %.0fs left cannot admit this variant's %d remaining round(s) "
        "(expected_sec=%s, spent on: %s); skipping %d remaining variant(s)",
        max(0.0, remaining_sec),
        rounds_left,
        required_sec,
        spent_on,
        len(run.grid) - idx,
    )
    return True


async def _unit_started(grid: list[GridVariant], idx: int, label: str) -> None:
    """Report that a unit of variant ``idx`` is about to start."""
    await report_progress(
        unit="variant_step",
        label=f"{grid[idx].name}:{label}",
        index=idx + 1,
        total=len(grid),
        status="started",
    )


async def _reported_magpie(run: _GridRun, at: _VariantSlot, label: str, bench_round: MagpieRound) -> _RoundProcess:
    """Run one Magpie pass, announced on entry and kept alive by its output."""
    await _unit_started(run.grid, at.idx, label)
    async with heartbeat_while_output_flows(
        unit="variant_step",
        label=f"{at.variant.name}:{label}",
        index=at.idx + 1,
        total=len(run.grid),
    ) as activity:
        returncode, stdout, stderr = await asyncio.to_thread(
            _run_magpie,
            bench_round,
            unset_envs=at.variant.unset_envs,
            serving_lease=run.launch.serving_lease,
            on_output=activity.note,
        )
    return _RoundProcess(returncode, stdout, stderr)


async def _report_finished_variant(grid: list[GridVariant], results: list[VariantResult], idx: int) -> None:
    """Report the variant that just landed."""
    await report_progress(**_variant_progress_note(grid, results, idx))


def _sweep_stale_aiter_locks() -> None:
    try:
        from ._aiter_jit import sweep_stale_aiter_locks_if_dead

        sweep = sweep_stale_aiter_locks_if_dead()
        if sweep.get("deleted"):
            log.warning(
                "grid_runner: reaped %d orphaned aiter JIT lock(s) under %s before server launch (compiler_alive=%s)",
                sweep.get("deleted"),
                sweep.get("dir"),
                sweep.get("compiler_alive"),
            )
    except (OSError, RuntimeError) as exc:
        log.debug("grid_runner: aiter lock sweep swallowed: %r", exc)


def _multi_node_reference_envs() -> dict[str, str]:
    """The reference recipe's server envs, which ``restart_server_for_round`` never reads itself."""
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return {}
    from ._workload_envs import resolve_reference_launch

    _, reference_envs, _ = resolve_reference_launch()
    return reference_envs


def _refuse_unsupported_capability(run: _GridRun, at: _VariantSlot) -> None:
    reason = unsupported_capability_reason(at.variant)
    if not reason:
        return
    log.warning(
        "grid_runner: variant %s aborted: capability_unsupported: %s",
        _variant_tag(run, at),
        reason,
    )
    _abort_variant(at, _VariantFailure("capability_unsupported", reason))


async def _variant_yaml(run: _GridRun, at: _VariantSlot, *, into: Path, lifecycle: dict[str, Any] | None) -> Path:
    return await asyncio.to_thread(
        _build_variant_yaml,
        run.recipe,
        at.variant,
        output_subdir=into,
        server_lifecycle=lifecycle,
    )


async def _variant_yaml_or_abort(
    run: _GridRun,
    at: _VariantSlot,
    *,
    into: Path,
    lifecycle: dict[str, Any] | None,
    failure_class: str | None = None,
) -> Path:
    """Materialize this variant's Magpie YAML under ``into``, or end the variant on the build error."""
    try:
        return await _variant_yaml(run, at, into=into, lifecycle=lifecycle)
    except Exception as exc:  # re-raised as _VariantAborted by _abort_variant
        error_class = failure_class or getattr(exc, "error_class", "yaml_build_error")
        log.warning(
            "grid_runner: variant %s aborted: %s: %r",
            _variant_tag(run, at),
            error_class,
            exc,
            exc_info=True,
        )
        _abort_variant(at, _VariantFailure(error_class, repr(exc), f"{error_class}: {exc!r}"))


async def _prepare_variant_rounds(run: _GridRun, at: _VariantSlot) -> _VariantRounds:
    cfg_path = await _variant_yaml_or_abort(run, at, into=at.path, lifecycle=run.launch.server_lifecycle)
    mn_effective_args = _resolve_mn_effective_server_args(
        cfg_path,
        run.recipe.base_yaml_path,
        at.variant,
        base_extra_args=run.recipe.base.extra_server_args,
        base_args_mode=run.recipe.base.args_mode,
    )
    cold = _VariantRounds(cfg_path, {"eligible": False}, False, mn_effective_args)
    if not run.auto_warmup_requested:
        return cold
    try:
        from ._server_lifecycle import resolve_lifecycle_params

        lifecycle = resolve_lifecycle_params(cfg_path)
        if not lifecycle.get("eligible"):
            log.info(
                "grid_runner: warmup-before-measure not eligible (%s); running single measured round.",
                lifecycle.get("reason") or "unknown",
            )
            return cold._replace(lifecycle=lifecycle)
        warmed_cfg_path = await _variant_yaml(
            run,
            at,
            into=at.path,
            lifecycle={"cleanup": True, "pid_dir": str(at.path), "port": int(lifecycle.get("port") or 0)},
        )
    except Exception as exc:  # re-raised as _VariantAborted by _abort_variant
        log.warning(
            "grid_runner: variant %s aborted: warmup-before-measure eligibility/materialization failed: %r",
            _variant_tag(run, at),
            exc,
            exc_info=True,
        )
        _abort_variant(at, _VariantFailure(_WARMUP_YAML_BUILD_ERROR, repr(exc), f"{_WARMUP_YAML_BUILD_ERROR}: {exc!r}"))
    return _VariantRounds(warmed_cfg_path, lifecycle, True, mn_effective_args)


async def _run_warmup_round(run: _GridRun, at: _VariantSlot, rounds: _VariantRounds) -> float | None:
    """Run and discard one benchmark pass so the measured round starts on a warm server."""
    warmup_slot = at.path / _WARMUP_ROUND_DIR
    warmup_server_log = warmup_slot / "server.log"
    cfg_path = await _variant_yaml_or_abort(
        run,
        at,
        into=warmup_slot,
        lifecycle={"cleanup": False, "pid_dir": str(at.path), "port": int(rounds.lifecycle.get("port") or 0)},
        failure_class=_WARMUP_YAML_BUILD_ERROR,
    )
    workspaces_before = snapshot_workspaces(warmup_slot)
    started_unix = time.time()
    # Held in a local because the abort line below has to name the cap the round was actually granted: the
    # declared one is a hang backstop, and a round killed at the reserved cap logged as a two-hour timeout
    # reads as a variant that hangs rather than a budget that ran out.
    warmup_cap_sec = run.benchmark_timeout_sec
    try:
        proc = await _reported_magpie(
            run,
            at,
            "warmup",
            MagpieRound(
                magpie_python=run.magpie_python,
                config_path=cfg_path,
                output_dir=warmup_slot,
                timeout_sec=warmup_cap_sec,
                silence_timeout_sec=run.silence_timeout_sec,
                cwd=str(run.output_root),
                result_dir=run.launch.result_dir,
                session_deadline_sec=run.stop.session_deadline_sec,
                warmup=True,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        _teardown_variant_server(at.path, rounds.lifecycle)
        log.warning(
            "grid_runner: variant %s aborted: warmup timeout (timeout_sec=%d): %s",
            _variant_tag(run, at),
            warmup_cap_sec,
            exc,
        )
        _abort_variant(
            at,
            _VariantFailure("warmup_magpie_timeout", str(exc), f"warmup_timeout: {exc}"),
            server_log_path=_existing_log_path(warmup_server_log),
            runtime_sec=round(max(0.0, time.time() - started_unix), 2),
            nonfatal_warnings=["run_grid_warmup_round_failed"],
        )
    stopped = stopped_by_the_run(proc.returncode)
    if stopped is not None:
        survivors = _teardown_variant_server(at.path, rounds.lifecycle)
        attempt = _RoundAttempt(at, proc.returncode, started_unix, warmup_server_log, tuple(survivors))
        _abort_on_round_stop(run, stopped, attempt, round_label="warmup")
    pass_ = _BenchmarkPass(proc, started_unix, workspaces_before)
    return await _warmup_throughput(run, at, rounds, pass_)


async def _warmup_throughput(
    run: _GridRun,
    at: _VariantSlot,
    rounds: _VariantRounds,
    pass_: _BenchmarkPass,
) -> float | None:
    """The throughput the warmup pass measured, or end the variant when it measured none."""
    warmup_slot = at.path / _WARMUP_ROUND_DIR
    run_ws = select_run_workspace(warmup_slot, known_before=pass_.workspaces_before)
    workspace = run_ws if run_ws is not None else warmup_slot
    harvested = harvest_leaked_artifacts(workspace, subprocess_started_unix=pass_.started_unix)
    returncode = pass_.proc.returncode
    if run_ws is not None:
        _, measurement = await _settled_measurement(
            workspace,
            config_path=rounds.cfg_path,
            subprocess_started_unix=pass_.started_unix,
            settle_seconds=REPORT_SETTLE_SECONDS if returncode == 0 else 0.0,
        )
    else:
        measurement = extract_benchmark_measurement(
            None,
            config_path=rounds.cfg_path,
            workspace=workspace,
            subprocess_started_unix=pass_.started_unix,
        )
    if returncode == 0 and measurement.get("valid_measurement"):
        warmup_tput = measurement.get("output_throughput")
        log.info(
            "grid_runner: variant %s warmup tput=%.1f tok/s discarded; measuring hot round next",
            at.variant.name,
            warmup_tput or 0.0,
        )
        return warmup_tput
    _teardown_variant_server(at.path, rounds.lifecycle)
    warmup_server_log = warmup_slot / "server.log"
    report_path = workspace / "benchmark_report.json"
    warmup_error = (
        server_log_death_excerpt(str(warmup_server_log))
        or redact_secret_values((pass_.proc.stderr or pass_.proc.stdout)[-2000:])
        if returncode != 0
        else "warmup benchmark_report missing valid throughput/completed requests"
    )
    log.warning(
        "grid_runner: variant %s aborted: warmup_round_failed (rc=%s): %s",
        _variant_tag(run, at),
        returncode,
        warmup_error[:200],
    )
    _abort_variant(
        at,
        _VariantFailure("warmup_round_failed", warmup_error),
        workspace=str(workspace) if run_ws is not None else None,
        report_path=str(report_path) if report_path.exists() else None,
        raw_result_path=measurement.get("raw_result_path"),
        reported_success=measurement.get("reported_success"),
        returncode=returncode,
        server_log_path=_existing_log_path(warmup_server_log),
        runtime_sec=round(max(0.0, time.time() - pass_.started_unix), 2),
        nonfatal_warnings=[
            "run_grid_warmup_round_failed",
            *[f"harvested_leaked_artifact:{src}" for src, _ in harvested],
        ],
    )


def _log_variant_banner(run: _GridRun, at: _VariantSlot) -> None:
    from ._multi_node_env import log_mn_banner

    log_mn_banner("grid_runner", log, variant=f"{at.idx + 1}/{len(run.grid)}:{at.variant.name}")
    log.info(
        "grid_runner: variant %s args=%s",
        _variant_tag(run, at),
        at.variant.extra_server_args,
    )


async def _restart_multi_node_server(run: _GridRun, at: _VariantSlot, rounds: _VariantRounds) -> None:
    """Restart sglang/vllm with this variant's flags so each multi-node row runs against a fresh server."""
    from ._multi_node_server_lifecycle import RoundServer, ServerRestartFailed, restart_server_for_round

    unset_envs = [str(k) for k in getattr(at.variant, "unset_envs", []) or [] if str(k).strip()]
    try:
        # PD knobs auto-resolved from $PD_* env; PD config stays constant across variants within one run.
        await restart_server_for_round(
            RoundServer(
                rounds.mn_effective_args,
                model_path=run.recipe.target.model_path,
                ep=int(os.environ.get("EP") or 0) or None,
            ),
            # Reference recipe envs under this variant's own overrides (e.g. MORI_* MoE-dispatch tuning) so
            # server-side env knobs proposed by specialists take effect on the restarted sglang without dropping
            # the envs the baseline was measured with.
            extra_env=_mn_restart_env(run.mn_reference_envs, at.variant, unset_envs),
            unset_env=unset_envs,
        )
    except ServerRestartFailed as exc:
        log.warning(
            "grid_runner: variant %s aborted: mn_server_restart_failed: %s",
            _variant_tag(run, at),
            exc,
        )
        _abort_variant(
            at,
            _VariantFailure("mn_server_restart_failed", str(exc), f"mn_server_restart_failed: {exc}"),
        )


async def _run_multi_node_warmup(run: _GridRun, at: _VariantSlot, rounds: _VariantRounds) -> None:
    """Run and discard one client pass against the just-restarted, persistent remote server."""
    from ._multi_node_env import is_multi_node, mn_bench_warmup_enabled

    if not is_multi_node() or not mn_bench_warmup_enabled():
        return
    warm_slot = at.path / "mn_warmup"
    started_unix = time.time()
    proc = await _reported_magpie(
        run,
        at,
        "mn_warmup",
        MagpieRound(
            magpie_python=run.magpie_python,
            config_path=rounds.cfg_path,
            output_dir=warm_slot,
            timeout_sec=run.benchmark_timeout_sec,
            silence_timeout_sec=run.silence_timeout_sec,
            server_already_ready=True,
            cwd=str(run.output_root),
            result_dir=None,
            session_deadline_sec=run.stop.session_deadline_sec,
            warmup=True,
        ),
    )
    log.info(
        "grid_runner: MN warmup pass done (discarded) %s rc=%s",
        _variant_tag(run, at),
        proc.returncode,
    )
    stopped = stopped_by_the_run(proc.returncode)
    if stopped is not None:
        survivors = _teardown_variant_server(at.path, rounds.lifecycle if rounds.auto_warmup else None)
        attempt = _RoundAttempt(at, proc.returncode, started_unix, warm_slot / "server.log", tuple(survivors))
        _abort_on_round_stop(run, stopped, attempt, round_label="mn_warmup")


async def _run_measured_round(run: _GridRun, at: _VariantSlot, rounds: _VariantRounds) -> _BenchmarkPass:
    from ._multi_node_env import is_multi_node

    # Snapshot wall-clock before launch so the salvage path can mtime-gate leak destinations per-variant.
    workspaces_before = snapshot_workspaces(at.path)
    started_unix = time.time()
    measure_cap_sec = run.benchmark_timeout_sec
    try:
        proc = await _reported_magpie(
            run,
            at,
            "benchmark",
            MagpieRound(
                magpie_python=run.magpie_python,
                config_path=rounds.cfg_path,
                output_dir=at.path,
                timeout_sec=measure_cap_sec,
                cwd=str(run.output_root),
                result_dir=run.launch.result_dir,
                silence_timeout_sec=run.silence_timeout_sec,
                server_already_ready=(run.launch.server_already_ready or rounds.auto_warmup or is_multi_node()),
                session_deadline_sec=run.stop.session_deadline_sec,
                warmup=run.launch.warmup_round,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        # Harvest pre-timeout leaks.
        destination = select_run_workspace(at.path, known_before=workspaces_before) or at.path
        harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=started_unix)
        log.warning(
            "grid_runner: variant %s aborted: magpie timeout (timeout_sec=%d): %s",
            _variant_tag(run, at),
            measure_cap_sec,
            exc,
        )
        _abort_variant(
            at,
            _VariantFailure("magpie_timeout", str(exc), f"timeout: {exc}"),
            server_log_path=_existing_log_path(at.path / "server.log"),
            runtime_sec=round(max(0.0, time.time() - started_unix), 2),
            nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in harvested],
        )
    finally:
        round_survivors = _teardown_variant_server(at.path, rounds.lifecycle if rounds.auto_warmup else None)
    return _BenchmarkPass(proc, started_unix, workspaces_before, tuple(round_survivors))


def _abort_on_watchdog(run: _GridRun, at: _VariantSlot, round_: _BenchmarkPass) -> NoReturn:
    runtime_sec = round(max(0.0, time.time() - round_.started_unix), 2)
    destination = select_run_workspace(at.path, known_before=round_.workspaces_before) or at.path
    # server.log is itself one of the salvaged artifacts, so the cause of death only reaches the slot here.
    harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=round_.started_unix)
    server_log = at.path / "server.log"
    if round_.proc.returncode == SERVER_DEAD_RETURNCODE:
        cause = "server_init_dead (engine/worker bootstrap failed; parent hung)"
        failure = _VariantFailure(
            "server_init_dead",
            server_log_death_excerpt(str(server_log))
            or "server engine/worker init failed; parent process hung and was reaped by the liveness watchdog",
        )
    else:
        cause = "detokenizer_stall (server ready but log went silent)"
        failure = _VariantFailure(
            "detokenizer_stall",
            "server reported ready but emitted no log output (hung engine / detokenizer stall); "
            "reaped by the detokenizer-stall watchdog",
            "detokenizer_stall: server ready but log went silent",
        )
    log.warning(
        "grid_runner: variant %s aborted: %s after %.1fs",
        _variant_tag(run, at),
        cause,
        runtime_sec,
    )
    _abort_variant(
        at,
        failure,
        returncode=round_.proc.returncode,
        runtime_sec=runtime_sec,
        server_log_path=_existing_log_path(server_log),
        nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in harvested],
    )


def _refuse_reaped_round(run: _GridRun, at: _VariantSlot, round_: _BenchmarkPass) -> None:
    returncode = round_.proc.returncode
    if returncode == EVAL_PROBE_UNPATCHABLE_RETURNCODE:
        # Eval bounds could not be installed for a variant that runs eval, so nothing launched.
        log.error(
            "grid_runner: variant %s aborted: eval_probe_unpatchable: %s",
            _variant_tag(run, at),
            round_.proc.stderr,
        )
        _abort_variant(
            at,
            _VariantFailure("eval_probe_unpatchable", round_.proc.stderr),
            returncode=returncode,
            runtime_sec=round(max(0.0, time.time() - round_.started_unix), 2),
        )
    if returncode in (SERVER_DEAD_RETURNCODE, DETOKENIZER_STALL_RETURNCODE):
        _abort_on_watchdog(run, at, round_)
    stopped = stopped_by_the_run(returncode)
    if stopped is not None:
        attempt = _RoundAttempt(at, returncode, round_.started_unix, at.path / "server.log", round_.server_survivors)
        _abort_on_round_stop(run, stopped, attempt, round_label="measured")


def _harvest_round_workspace(run: _GridRun, at: _VariantSlot, round_: _BenchmarkPass) -> tuple[Path, list[Any]]:
    workspace = select_run_workspace(at.path, known_before=round_.workspaces_before)
    artifacts = harvest_leaked_artifacts(
        workspace if workspace is not None else at.path,
        subprocess_started_unix=round_.started_unix,
    )
    if artifacts:
        log.info(
            "_grid_runner: variant=%s harvested %d leaked artifact(s): %s",
            at.variant.name,
            len(artifacts),
            ", ".join(src.name for src, _ in artifacts),
        )
    if workspace is None:
        _abort_without_workspace(run, at, round_, artifacts)
    return workspace, artifacts


def _abort_without_workspace(
    run: _GridRun,
    at: _VariantSlot,
    round_: _BenchmarkPass,
    artifacts: list[Any],
) -> NoReturn:
    proc = round_.proc
    returncode = proc.returncode
    preflight_abort = returncode == AGENTX_PREFLIGHT_RETURNCODE
    server_log = at.path / "server.log"
    if preflight_abort:
        found = _RoundError(
            redact_secret_values((proc.stderr or "").strip()[-2000:]) or "AgentX preflight aborted the round",
            AGENTX_PREFLIGHT_ERROR_CLASS,
        )
    else:
        found = _RoundError(
            server_log_death_excerpt(str(server_log))
            or (
                redact_secret_values((proc.stderr or proc.stdout)[-2000:])
                if returncode != 0
                else "no benchmark_* workspace produced"
            ),
            "no_benchmark_workspace",
        )
    log.warning(
        "grid_runner: variant %s aborted: %s (rc=%s)",
        _variant_tag(run, at),
        found.error_class,
        returncode,
    )
    halt = _Halt.ON_FAILURE if returncode != 0 else _Halt.CONTINUE
    abandoned: StoppedByTheRun | None = None
    if preflight_abort:
        halt, abandoned = _Halt.STOP, _AGENTX_PREFLIGHT_ABANDONED
        # Environment, not variant.
        log.error(
            "grid_runner: variant %s aborted on the AgentX preflight; "
            "abandoning the remaining %d point(s) -- the client is missing for the "
            "whole grid, not for this variant: %s",
            _variant_tag(run, at),
            len(run.grid) - (at.idx + 1),
            found.summary,
        )
    _abort_variant(
        at,
        _VariantFailure(found.error_class, found.summary, halt=halt, abandoned=abandoned),
        returncode=returncode,
        server_log_path=_existing_log_path(server_log),
        nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in artifacts],
    )


def _round_warnings(measurement: dict[str, Any], measured: _MeasuredVariant) -> list[str]:
    warnings = list(measurement.pop("nonfatal_warnings", []) or [])
    warnings += [f"harvested_leaked_artifact:{src}" for src, _ in measured.artifacts]
    if measured.warmup_tput is not None:
        warnings.append("run_grid_warmup_discarded_first")
        warnings.append(f"warmup_round_tput:{float(measured.warmup_tput):.1f}")
    return warnings


def _abort_on_invalid_measurement(run: _GridRun, at: _VariantSlot, outcome: _VariantOutcome) -> NoReturn:
    returncode = outcome.measured.round.proc.returncode
    workspace = outcome.measured.workspace
    report_path = workspace / "benchmark_report.json"
    server_log = at.path / "server.log"
    found = _invalid_measurement_error(
        outcome.measured.round.proc,
        report=outcome.report,
        death_excerpt=server_log_death_excerpt(str(server_log)),
        workspace=workspace,
        slot=at.path,
    )
    log.warning(
        "grid_runner: variant %s aborted: %s (rc=%s): %s",
        _variant_tag(run, at),
        found.error_class,
        returncode,
        found.summary[:200],
    )
    _abort_variant(
        at,
        _VariantFailure(
            found.error_class,
            found.summary,
            halt=_Halt.ON_FAILURE if returncode != 0 else _Halt.CONTINUE,
        ),
        workspace=str(workspace),
        report_path=str(report_path) if report_path.exists() else None,
        raw_result_path=outcome.measurement.get("raw_result_path"),
        reported_success=outcome.measurement.get("reported_success"),
        returncode=returncode,
        nonfatal_warnings=outcome.warnings,
        server_log_path=_existing_log_path(server_log),
    )


async def _variant_row(run: _GridRun, at: _VariantSlot, measured: _MeasuredVariant) -> VariantResult:
    report, measurement = await _settled_measurement(
        measured.workspace,
        config_path=measured.rounds.cfg_path,
        subprocess_started_unix=measured.round.started_unix,
        settle_seconds=REPORT_SETTLE_SECONDS,
    )
    outcome = _VariantOutcome(measured, report, measurement, _round_warnings(measurement, measured))
    if round_succeeded(measurement):
        return _measured_row(at, outcome)
    if run.launch.lifecycle_boot_only and measured.round.proc.returncode == 0:
        return _booted_server_row(at, outcome)
    _abort_on_invalid_measurement(run, at, outcome)


def _booted_server_row(at: _VariantSlot, outcome: _VariantOutcome) -> VariantResult:
    workspace = outcome.measured.workspace
    report_path = workspace / "benchmark_report.json"
    return VariantResult(
        name=at.variant.name,
        extra_server_args=at.variant.extra_server_args,
        extra_envs=dict(at.variant.extra_envs),
        status="succeeded",
        workspace=str(workspace),
        report_path=str(report_path) if report_path.exists() else None,
        returncode=outcome.measured.round.proc.returncode,
        nonfatal_warnings=outcome.warnings,
        note="server_lifecycle_boot_only",
    )


def _kept_nonzero_exit(at: _VariantSlot, outcome: _VariantOutcome) -> str | None:
    """The exit code this measurement is kept despite, or None when the round exited clean."""
    returncode = outcome.measured.round.proc.returncode
    if returncode == 0:
        return None
    proc = outcome.measured.round.proc
    nonzero_error = redact_secret_values((proc.stderr or proc.stdout)[-2000:])
    measurement = outcome.measurement
    if not served_complete_protocol(measurement):
        log.warning(
            "grid_runner: variant %s aborted: magpie_nonzero_after_valid_measurement (rc=%d, served %s of %s requests)",
            at.variant.name,
            returncode,
            measurement.get("completed_requests"),
            measurement.get("requested_requests"),
        )
        workspace = outcome.measured.workspace
        report_path = workspace / "benchmark_report.json"
        _abort_variant(
            at,
            _VariantFailure("magpie_nonzero_after_valid_measurement", nonzero_error),
            workspace=str(workspace),
            report_path=str(report_path) if report_path.exists() else None,
            raw_result_path=measurement.get("raw_result_path"),
            reported_success=measurement.get("reported_success"),
            returncode=returncode,
            nonfatal_warnings=outcome.warnings,
            server_log_path=None,
        )
    # Every request the client recorded as requested was served, so the exit code came from
    # something the round had already finished with. The cause stays on the result: a reader
    # comparing this point to its neighbours is owed the reason it is not a clean zero.
    outcome.warnings.append(f"nonzero_rc_after_complete_protocol:{returncode}")
    log.warning(
        "grid_runner: variant %s exited %d after serving its whole protocol "
        "(%s of %s requests); keeping the measurement: %s",
        at.variant.name,
        returncode,
        measurement.get("completed_requests"),
        measurement.get("requested_requests"),
        nonzero_error,
    )
    return nonzero_error


def _measured_row(at: _VariantSlot, outcome: _VariantOutcome) -> VariantResult:
    nonzero_kept_error = _kept_nonzero_exit(at, outcome)
    measurement = outcome.measurement
    workspace = outcome.measured.workspace
    report_path = workspace / "benchmark_report.json"
    row = VariantResult(
        name=at.variant.name,
        extra_server_args=at.variant.extra_server_args,
        extra_envs=dict(at.variant.extra_envs),
        status="succeeded",
        output_throughput=measurement.get("output_throughput"),
        request_throughput=measurement.get("request_throughput"),
        total_token_throughput=measurement.get("total_token_throughput"),
        completed_requests=measurement.get("completed_requests"),
        duration_seconds=measurement.get("duration_seconds"),
        ttft_mean_ms=measurement.get("ttft_mean_ms"),
        **constraint_fields(measurement),
        tpot_mean_ms=measurement.get("tpot_mean_ms"),
        input_throughput=measurement.get("input_throughput"),
        tpot_p90_ms=measurement.get("tpot_p90_ms"),
        intvty_p90=measurement.get("e2e_norm_intvty_p90"),
        intvty_p50=measurement.get("e2e_norm_intvty_p50"),
        request_error_rate=measurement.get("request_error_rate"),
        workspace=str(workspace),
        report_path=str(report_path) if report_path.exists() else None,
        raw_result_path=measurement.get("raw_result_path"),
        reported_success=measurement.get("reported_success"),
        returncode=outcome.measured.round.proc.returncode,
        nonfatal_warnings=outcome.warnings,
        error=nonzero_kept_error,
        server_log_path=None,
        note=at.variant.note,
        runtime_sec=round(max(0.0, time.time() - outcome.measured.round.started_unix), 2),
    )
    log.info("grid_runner: variant %s tput=%.1f tok/s", at.variant.name, row.output_throughput or 0.0)
    return row


async def _measure_variant(run: _GridRun, at: _VariantSlot) -> VariantResult:
    """Measure one variant and return its row, or raise :class:`_VariantAborted` carrying why it has none."""
    _refuse_unsupported_capability(run, at)
    rounds = await _prepare_variant_rounds(run, at)
    warmup_tput = await _run_warmup_round(run, at, rounds) if rounds.auto_warmup else None
    _log_variant_banner(run, at)
    await _restart_multi_node_server(run, at, rounds)
    # The restart above is a launch of its own and is under no cap, so the budget admitted at the top of the
    # loop may no longer be there.
    spent_on = "this variant's server restart"
    if _warn_budget_cannot_admit(run, at.idx, rounds_left=1 + run.mn_warmup_rounds, spent_on=spent_on):
        if rounds.auto_warmup:
            _teardown_variant_server(at.path, rounds.lifecycle)
        _abandon_rest_of_grid(run, at)
    await _run_multi_node_warmup(run, at, rounds)
    round_ = await _run_measured_round(run, at, rounds)
    _refuse_reaped_round(run, at, round_)
    workspace, artifacts = _harvest_round_workspace(run, at, round_)
    return await _variant_row(run, at, _MeasuredVariant(rounds, round_, workspace, artifacts, warmup_tput))


async def run_grid(
    *,
    grid: list[GridVariant],
    output_root: Path,
    recipe: GridRecipe,
    launch: GridLaunch = GridLaunch(),
    stop: GridStop = GridStop(),
) -> list[VariantResult]:
    """Execute variants; ``stop.deadline_stop`` names the owner of the supplied deadline.

    ``launch.ready_server_log`` is the log of the caller's ready server, measured when no variant log exists.
    """
    silence_timeout_sec, benchmark_timeout_sec = resolve_benchmark_timeouts()
    magpie_python = launch.magpie_python
    if not magpie_python:
        # Backend-aware: bypass uses a plain python3, not Magpie's venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        magpie_python = resolve_benchmark_interpreter()
    else:
        magpie_python = _validate_magpie_python_override(magpie_python)
    warmup_before_measure = launch.warmup_before_measure
    if warmup_before_measure is None:
        warmup_before_measure = _run_grid_warmup_enabled()
    auto_warmup_requested = bool(warmup_before_measure and launch.server_lifecycle is None)
    # This function names the working directory, so it creates it: callers and the per-variant config writer both
    # happen to create it first today, and neither is a contract.
    output_root.mkdir(parents=True, exist_ok=True)
    _sweep_stale_aiter_locks()
    mn_warmup_rounds = _multi_node_warmup_rounds(stop.variant_expected_sec)
    run = _GridRun(
        grid=grid,
        output_root=output_root,
        recipe=recipe,
        launch=launch,
        stop=stop,
        magpie_python=magpie_python,
        silence_timeout_sec=silence_timeout_sec,
        benchmark_timeout_sec=benchmark_timeout_sec,
        rounds_per_variant=1 + (1 if auto_warmup_requested else 0) + mn_warmup_rounds,
        mn_warmup_rounds=mn_warmup_rounds,
        mn_reference_envs=_multi_node_reference_envs(),
        auto_warmup_requested=auto_warmup_requested,
    )
    results: list[VariantResult] = []
    for i, variant in enumerate(grid):
        at = _VariantSlot(i, variant, output_root / f"variant_{i:02d}_{_safe(variant.name)}")
        try:
            if _warn_budget_cannot_admit(
                run, i, rounds_left=run.rounds_per_variant, spent_on="the variants before this one"
            ):
                _abandon_rest_of_grid(run, at)
            # Reported on entry, not on completion: the finished-variant note only follows a row, so a first
            # variant that hangs — or a branch that raises before reaching it — would emit nothing at all, which
            # is exactly the silence the heartbeat exists to break.
            await _unit_started(grid, i, "variant")
            results.append(await _measure_variant(run, at))
        except _VariantAborted as abort:
            results.append(abort.result)
            await _report_finished_variant(grid, results, i)
            if abort.abandoned is not None:
                results.extend(_not_run_skip_result(rest, abort.abandoned) for rest in grid[i + 1 :])
            if abort.halt is _Halt.STOP or (abort.halt is _Halt.ON_FAILURE and not stop.keep_going_on_failure):
                break
        else:
            await _report_finished_variant(grid, results, i)
    await asyncio.to_thread(
        _attach_grid_launch_evidence,
        results,
        grid=grid,
        output_root=output_root,
        caller_reused_ready_server=launch.server_already_ready,
        ready_server_log=launch.ready_server_log,
    )
    return results


def _teardown_variant_server(slot: Path, lifecycle: dict[str, Any] | None) -> list[int]:
    """Stop every server this variant booted for its own rounds; return the pids still alive after SIGKILL.

    ``lifecycle`` names the persistent server the rounds shared, ``None`` when they shared none.
    """
    from ._server_lifecycle import stop_agentx_servers, teardown_lifecycle_server

    if lifecycle is not None:
        teardown_lifecycle_server(
            pid_dir=slot,
            framework=str(lifecycle.get("framework") or ""),
            port=int(lifecycle.get("port") or 0),
        )
    return stop_agentx_servers(slot)


def _not_run_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Build the ``skipped`` result for a variant the run never got to."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        error=stopped.never_started,
        error_class=stopped.error_class,
        note=variant.note,
    )


def _existing_log_path(path: Path) -> str | None:
    """Return ``path`` as a string when it exists, else ``None``."""
    return str(path) if path.exists() else None


def _measurement_server_log_path(
    server_log: Path,
    workspace: Path | None = None,
    *,
    slot: Path | None = None,
) -> str | None:
    """Return the server log attributable to this variant's measurement."""
    owning_slot = slot or server_log.parent
    direct = _existing_log_path(server_log)
    if not direct and workspace is not None:
        direct = _existing_log_path(workspace / "server.log")
    if direct:
        return direct
    warmup_dir = owning_slot / _WARMUP_ROUND_DIR
    try:
        candidates = [p for d in (owning_slot, warmup_dir) for p in d.glob("*/server.log") if p.is_file()]
    except OSError:
        candidates = []
    if not candidates:
        return None
    try:
        return str(max(candidates, key=lambda path: path.stat().st_mtime))
    except OSError:
        return None


def _attach_grid_launch_evidence(
    results: list[VariantResult],
    *,
    grid: list[GridVariant],
    output_root: Path,
    caller_reused_ready_server: bool,
    ready_server_log: str | None = None,
) -> None:
    """Persist declared and observed launch evidence for each grid result."""
    from ._workload_envs import VLLM_SOURCE_ROOT_ENVS

    for idx, result in enumerate(results):
        if idx >= len(grid):
            break
        if result.status == "skipped" or result.error_class in {  # skipped or refused: never launched a measurement
            "capability_unsupported",
            "yaml_build_error",
            _WARMUP_YAML_BUILD_ERROR,
            SPECULATIVE_DECODING_PINNED,
            ServerArgvInvalidError.error_class,
        }:
            continue
        slot = output_root / f"variant_{idx:02d}_{_safe(grid[idx].name)}"
        config_path = slot / "config.yaml"
        workspace = Path(result.workspace) if result.workspace else None
        primary_log = Path(result.server_log_path) if result.server_log_path else slot / "server.log"
        actual_log = _measurement_server_log_path(primary_log, workspace, slot=slot)
        if not actual_log and ready_server_log:
            actual_log = _existing_log_path(Path(ready_server_log))
        result.server_log_path = actual_log
        from hyperloom.inference_optimizer.agentx.report import agentx_measurement, native_launch_evidence

        if result.benchmark_backend == "magpie_agentx" and workspace is not None:
            measurement = agentx_measurement(_parse_report(workspace) or {})
            evidence = native_launch_evidence(measurement, config_path, actual_server_log=actual_log)
        else:
            evidence = build_launch_evidence(
                RequestedLaunch(config_path, os.environ.get("FRAMEWORK", "")),
                actual_server_log=actual_log,
                slot=slot,
                caller_reused_ready_server=caller_reused_ready_server,
            )
        result.launch_evidence = evidence
        result.launch_evidence_path = persist_launch_evidence(evidence, slot=slot)
        declared_env = evidence["requested_server_env"].keys() - set(VLLM_SOURCE_ROOT_ENVS)
        report = scan_server_log(actual_log, evidence["framework"], declared_env=declared_env)
        persist_runtime_findings(report, slot=slot)


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory."""
    try:
        slot.mkdir(parents=True, exist_ok=True)
        marker = {
            "variant": variant_name,
            "error_class": error_class,
            "error": (error_summary or "")[:2000],
            "extra_args": extra_args,
            "aborted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        (slot / "abort_reason.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "_grid_runner: failed to write abort_reason.json at %s: %s",
            slot,
            exc,
        )


def _report_errors_summary(report: dict[str, Any] | None, limit: int = 2000) -> str:
    """Join ``benchmark_report.json`` ``errors`` into a single diagnostic."""
    if not isinstance(report, dict):
        return ""
    errors = report.get("errors")
    if not isinstance(errors, list):
        return ""
    text = "; ".join(str(item).strip() for item in errors if str(item).strip())
    return text[-limit:] if text else ""


def _on_disk_stderr_tail(*dirs: Path, limit: int = 2000) -> str:
    """Return the tail of the first non-empty on-disk benchmark log."""
    for d in dirs:
        if not d:
            continue
        for name in ("benchmark_stderr.log", "benchmark_stdout.log"):
            try:
                p = d / name
                if not p.is_file():
                    continue
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return f"[{name}] {text[-limit:]}"
            except OSError:
                continue
    return ""


__all__ = [
    "GridVariant",
    "ORCHESTRATOR_CANCELLED_CLASS",
    "SESSION_TIME_EXHAUSTED_CLASS",
    "StoppedByTheRun",
    "VariantResult",
    "apply_multi_node_invalid_variants",
    "apply_aiter_moe_pin_filter",
    "apply_runtime_benchmark_overrides",
    "reorder_grid_for_multi_node",
    "merge_server_args",
    "run_grid",
    "sanitize_result_dir",
    "sanitize_script_name",
    "server_args_env_name",
    "session_grid_bounds",
    "stopped_by_the_run",
    # Re-exported from the sibling modules to keep the namespace intact.
    "_resolve_probe_python",
    "resolve_skip_spec",
    "_MN_PARAMS_PRIORITY",
    "_MN_BACKENDS_PRIORITY",
    "apply_compatibility_filter",
    "apply_user_skip_list",
]


def _apply_variant_env(envs, variant, base_extra_envs, base_unset_envs):
    """Apply accepted environment edits before the candidate's set/unset operations."""
    # Composed base-then-variant, so a variant unsetting a key the base sets
    # removes it: the last layer to name a key is the one that decides it.
    for k in to_str_list(base_unset_envs):
        if k.strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            continue
        envs.pop(k, None)
    for k, v in (base_extra_envs or {}).items():
        envs[str(k)] = str(v)
    base_envs = dict(envs)
    for k in getattr(variant, "unset_envs", []) or []:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(k).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("grid: refusing to unset pinned env %s for variant %s", k, variant.name)
            continue
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    return base_envs


def _runnable_moe_backend_args(combined: str, *, args_env: str, model_path: str | None, variant: GridVariant) -> str:
    """``combined`` with a MoE runner backend this checkpoint cannot run dropped."""
    if not combined or not SGLANG_MOE_RUNNER_BACKEND_RE.search(combined):
        return combined
    if args_env != "EXTRA_SGLANG_ARGS" or not moe_runner_requires_aiter(combined, model_path):
        return combined
    log.warning(
        "grid: dropping inherited --moe-runner-backend for variant %s: "
        "this checkpoint's MoE quant scheme is only implemented on the "
        "aiter runner and would crash on the first forward pass.",
        variant.name,
    )
    return remove_moe_runner_backend_arg(combined)


def _importable_overlay_dir(variant: GridVariant) -> str:
    """This variant's overlay directory, or ``""`` when it names none the server may import."""
    overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if not overlay:
        return ""
    contained = ":" not in overlay and ".." not in Path(overlay).parts
    if contained and not any(c in overlay for c in ("\n", "\r", "\x00")) and Path(overlay).is_dir():
        return overlay
    log.warning(
        "grid: dropping unsafe overlay_pythonpath %r (not a single existing directory / contains separator or traversal)",
        overlay,
    )
    return ""


def _pin_xdit_tool_path(envs: dict[str, Any], framework: Any) -> None:
    if str(framework or "").strip().lower() != "xdit":
        return
    parts = [p for p in str(envs.get("PATH", "") or "").split(":") if p]
    for essential in ("/opt/rocm/bin", "/venv/bin"):
        if essential not in parts:
            parts.insert(0, essential)
    envs["PATH"] = ":".join(parts)


def _build_variant_yaml(
    recipe: GridRecipe,
    variant: GridVariant,
    *,
    output_subdir: Path,
    server_lifecycle: dict[str, Any] | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk."""
    base = recipe.base
    target = recipe.target
    cfg = yaml.safe_load(recipe.base_yaml_path.read_text(encoding="utf-8"))
    bench = cfg.setdefault("benchmark", {})
    if is_magpie_agentx(bench):
        return materialize_variant(
            recipe.base_yaml_path,
            variant,
            output_subdir,
            model_path=target.model_path,
            base=LaunchChanges(
                base.extra_server_args,
                base.extra_envs or {},
                base.remove_args or [],
                base.unset_envs or [],
                base.args_mode == "replace",
            ),
        )
    replacing = str(base.args_mode).strip().lower() == "replace"
    envs = apply_runtime_benchmark_overrides(
        bench,
        model_path=target.model_path,
        gpu_type=target.gpu_type,
        benchmark_script=target.benchmark_script,
        conc=variant_conc(variant),
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    # A replacing base drops the inherited string wholesale, so its own removals have nothing left to strip.
    inherited, base_remove = ("", []) if replacing else (str(envs.get(extra_args_env, "")), base.remove_args)
    base_args = compose_server_args(
        inherited_args=inherited, base_extra_args=base.extra_server_args, remove_args=base_remove
    )
    combined = compose_server_args(
        inherited_args=inherited,
        base_extra_args=base.extra_server_args,
        variant_extra_args=variant.extra_server_args,
        remove_args=to_str_list(base_remove) + to_str_list(getattr(variant, "remove_args", [])),
        args_mode=getattr(variant, "args_mode", "append"),
    )
    combined = _runnable_moe_backend_args(
        combined, args_env=extra_args_env, model_path=target.model_path, variant=variant
    )
    if combined:
        envs[extra_args_env] = dedupe_extra_server_args(combined)
    elif extra_args_env in envs or variant.args_mode == "replace" or base.args_mode == "replace":
        envs[extra_args_env] = ""
    base_envs = _apply_variant_env(envs, variant, base.extra_envs, base.unset_envs)
    from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path

    pin_mlperf_round_concurrency(envs)
    # The launcher re-exports these unconditionally, so a value carried here is
    # one the run never used.
    for k in launcher_overwritten_envs(bench) & envs.keys():
        log.warning("grid: dropping %s for variant %s; the recipe overwrites it", k, variant.name)
        envs.pop(k, None)
    # The three AgentX bounds took this rung's CONC through ``variant_conc`` above, not through this merge: raising
    # the client's grace alone would make the round wait inside a cap that did not move with it.
    _overlay = _importable_overlay_dir(variant)
    if _overlay:
        prepend_launch_path(envs, "PYTHONPATH", _overlay)

    # Attempt runtime override: inject path_prefix/pythonpath_prefix/framework_bin etc. into benchmark.envs so the
    # server subprocess resolves the attempt runtime.
    _rt_override = getattr(variant, "runtime_override", None) or {}
    if _rt_override:
        apply_runtime_override(envs, _rt_override)
    refuse_unpinned_speculative_change(base_args, base_envs, combined, envs)

    _pin_xdit_tool_path(envs, bench.get("framework"))

    if server_lifecycle is not None:
        from ._server_lifecycle import inject_lifecycle

        inject_lifecycle(
            bench,
            cleanup=bool(server_lifecycle.get("cleanup", True)),
            pid_dir=server_lifecycle["pid_dir"],
            port=int(server_lifecycle["port"]),
        )

    # The final write to the argument env; nothing below may touch it.
    seal_server_argv(envs, bench.get("framework"))
    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def variant_conc(variant: Any) -> int | None:
    """The concurrency a variant will run at, or ``None`` when it names none."""
    raw = (getattr(variant, "extra_envs", None) or {}).get("CONC")
    try:
        conc = int(raw)
    except (TypeError, ValueError):
        return None
    return conc if conc > 0 else None
