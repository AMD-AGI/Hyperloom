# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark round records, stopped-round results and the round's config, env and log helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, NamedTuple

import yaml

from hyperloom.common.env_safety import scrub_benchmark_process_env
from ...stop_attribution import StoppedByTheRun
from .._grid_runner import stopped_by_the_run
from .._subprocess_kill import post_ready_runtime_sec

if TYPE_CHECKING:
    from ....loop.sub_agent_runner import RunnerContext

log = logging.getLogger(__name__)

_BENCHMARK_MODE_ATTR = "benchmark_mode"


# The round kept its warmup pass as the baseline because the budget could not pay for the measured pass after it.
MEASURE_ROUND_DROPPED_WARNING = "baseline_measure_round_dropped_low_budget"

# The cold-start guard's two round directories.
_WARMUP_ROUND_DIR = "warmup_round"
_MEASURE_ROUND_DIR = "measure_round"
_DOUBLE_RUN_ROUND_DIRS = (_WARMUP_ROUND_DIR, _MEASURE_ROUND_DIR)
#: The staged accuracy pass's slot, run on the server the measured pass leaves hot.
_ACCURACY_ROUND_DIR = "accuracy_round"


def _config_framework(config_path: Path | str) -> str:
    """Framework name from a materialized benchmark YAML (``""`` when unreadable)."""
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return ""
    return str((cfg.get("benchmark") or {}).get("framework") or "").strip().lower()


def _watchdog_server_log_path(output_dir: Path, framework: str) -> str | None:
    """``server.log`` path for the subprocess watchdogs, or None when server-less."""
    from hyperloom.inference_optimizer import framework_registry

    if framework_registry.is_scriptable(framework):
        return None
    return str(output_dir / "server.log")


def _round_post_ready_sec(
    server_log_path: str | None,
    *,
    started_unix: float,
    runtime_sec: float,
) -> float | None:
    """How much of a round's wall-clock was the benchmark rather than the boot."""
    if not server_log_path:
        return None
    return post_ready_runtime_sec(
        server_log_path,
        started_unix=started_unix,
        runtime_sec=runtime_sec,
    )


class _RoundEnd(NamedTuple):
    """How far a round the run stopped got: its returncode, the stop detail and its wall-clock."""

    returncode: int | None
    error: str
    runtime_sec: float


def _round_stop_result(
    stopped: StoppedByTheRun,
    end: _RoundEnd,
    *,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the failed result of a round the run itself stopped."""
    return {
        "status": "failed",
        "error_class": stopped.error_class,
        "returncode": end.returncode,
        "error": end.error,
        "subprocess_runtime_sec": round(end.runtime_sec, 2),
        "output_dir": str(output_dir),
        **capture_meta,
    }


def _stopped_round_result(
    returncode: int | None,
    *,
    round_label: str,
    runtime_sec: float,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the result for a round the run itself stopped, or ``None`` when it did not."""
    stopped = stopped_by_the_run(returncode)
    if stopped is None:
        return None
    log.warning(
        "baseline_executor: %s reaped after %.1fs: %s; error_class=%s.",
        round_label,
        runtime_sec,
        stopped.interrupted,
        stopped.error_class,
    )
    return _round_stop_result(
        stopped,
        _RoundEnd(returncode=returncode, error=stopped.interrupted, runtime_sec=runtime_sec),
        output_dir=output_dir,
        capture_meta=capture_meta,
    )


def _unlaunched_round_result(
    stopped: StoppedByTheRun,
    *,
    round_label: str,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the result for a round the run stopped before launching it."""
    log.warning(
        "baseline_executor: %s not launched: %s; error_class=%s.",
        round_label,
        stopped.never_started,
        stopped.error_class,
    )
    return _round_stop_result(
        stopped,
        _RoundEnd(returncode=None, error=stopped.never_started, runtime_sec=0.0),
        output_dir=output_dir,
        capture_meta=capture_meta,
    )


def _cold_anchor_from_warmup(
    warmup_result: dict[str, Any],
    *,
    dropped: dict[str, Any],
) -> dict[str, Any]:
    """Keep the warmup's cold figure as the anchor, marked as the cold one."""
    warnings = warmup_result.setdefault("nonfatal_warnings", [])
    if MEASURE_ROUND_DROPPED_WARNING not in warnings:
        warnings.append(MEASURE_ROUND_DROPPED_WARNING)
    warmup_result["measure_round_dropped"] = dropped
    return warmup_result


def _set_materialized_run_eval(config_path: Path, *, enabled: bool) -> None:
    """Set the effective eval mode after lifecycle eligibility is known."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark = cfg.setdefault("benchmark", {})
    benchmark.setdefault("envs", {})["RUN_EVAL"] = "true" if enabled else "false"
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_result_dir(output_dir: Path, override_result_dir: str | None) -> Path:
    """Resolve the benchmark ``$RESULT_DIR`` exactly as the subprocess sees it."""
    if not override_result_dir:
        return output_dir
    result_dir = Path(override_result_dir)
    if result_dir.is_absolute():
        return result_dir
    return (output_dir / result_dir).resolve()


def _benchmark_process_env(
    config_path: Path, output_dir: Path, result_dir: Path, inferencex_path: str
) -> dict[str, str]:
    """Build the client environment without overriding Magpie's native checkout selection."""
    from ..magpie_agentx import native_config

    env = scrub_benchmark_process_env(os.environ.copy())
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    if inferencex_path and not native_config(config_path):
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
    env["RESULT_DIR"] = str(result_dir)
    # InferenceX removes EVAL_RESULT_DIR after processing lm-eval output.
    env["EVAL_RESULT_DIR"] = str(result_dir / "eval_output")
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")
    return env


def _should_establish_quality_ref(task_kind: str | None, params: dict[str, Any] | None = None) -> bool:
    """Only a genuine ``baseline`` task may establish/overwrite the quality reference."""
    if str(task_kind or "") != "baseline":
        return False
    return not (params or {}).get("quality_ref_exempt")


def _requested_framework(params: dict[str, Any]) -> str | None:
    return str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None


def _is_double_run_accuracy_handoff(
    result: dict[str, Any],
    salvaged: dict[str, Any] | None,
) -> bool:
    """Whether accuracy came from the warmup round because that is the design."""
    out_dir = str((result or {}).get("output_dir") or "")
    if not out_dir or Path(out_dir).name != _MEASURE_ROUND_DIR:
        return False
    source = str((salvaged or {}).get("source_file") or "")
    return _WARMUP_ROUND_DIR in Path(source).parts


_LOG_NAMES = ("benchmark_stderr.log", "benchmark_stdout.log", "server.log")


def _iter_log_tails(root: Path, *, max_bytes: int, limit: int = 64) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, tail)`` for each known log under *root*, over at most *limit* files."""
    seen = 0
    try:
        for path in root.rglob("*.log"):
            if path.name not in _LOG_NAMES:
                continue
            seen += 1
            if seen > limit:
                break
            try:
                with path.open("rb") as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    handle.seek(max(0, size - max_bytes))
                    tail = handle.read().decode("utf-8", "replace")
            except OSError:
                continue
            yield path, tail
    except OSError:
        return


@dataclass(frozen=True)
class BenchmarkAttempt:
    """What every benchmark round of one attempt runs with, and the recorder its rounds report to."""

    timeout_sec: int
    override_result_dir: str | None
    resolved_model: str
    materialized_config_path: Path
    inferencex_path: str
    effective_extra_server_args: str
    params: dict[str, Any]
    ctx: RunnerContext
    run_eval_disabled: bool = False
    serving_lease: Any = None
    recorder: Any = None
    run_index: int = 0


class _BenchmarkRound(NamedTuple):
    config_path: Path
    materialized_config_path: Path
    output_dir: Path
    result_dir: Path
    ctx: RunnerContext
    ctx_extra: Mapping[str, Any]
    cmd: list[str]
    env: dict[str, Any]
    framework: str
    watchdog_server_log: str | None
    capture_meta: dict[str, Any]
    timeout_sec: int
    silence_timeout_sec: Any
    session_deadline_sec: Any
    serving_lease: Any
    server_already_ready: bool
    run_eval_disabled: bool


class _BenchProcess(NamedTuple):
    returncode: Any
    stdout: str
    stderr: str
    runtime_sec: float
    started_unix: float
    server_log_text: str = ""
    boot_observation_ref: Any = None
    workspace: Path | None = None


__all__ = ["BenchmarkAttempt"]
