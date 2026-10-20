# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workload sweep that relaunches the GEAK-optimized server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple

import yaml

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS, effective_mask_tokens, is_rocr_level
from hyperloom.orchestrator.kernel.geak_config import (
    _accepted_config_as_variant,
)
from hyperloom.orchestrator.actions.executors._gpu_pin import (
    coerce_tp,
    resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
)
from ._accuracy_gate import parse_eval_results
from ._gpu_power import (
    ServerLogPhaseDriver,
    build_gpu_power_recorder,
    read_measured_gpu_power,
    read_measured_gpu_power_by_gpu,
)
from ._subprocess_kill import _LogCursor
from ._launch_evidence import build_launch_evidence, persist_launch_evidence
from ._workload_envs import GEAK_METRIC_OUTPUT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GeakPointReport:
    """One replayed sweep point as its ``benchmark_report.json`` records it."""

    conc: int
    isl: int
    osl: int
    success: bool
    output_throughput_tok_s: float | None
    mean_ttft_ms: float | None
    mean_tpot_ms: float | None
    mean_e2el_ms: float | None
    error: str | None = None


def _write_benchmark_report(out_dir: Path, point: _GeakPointReport) -> None:
    """Write a session-breakdown-compatible ``benchmark_report.json``."""
    report = {
        "success": bool(point.success),
        "conc": point.conc,
        "isl": point.isl,
        "osl": point.osl,
        "output_throughput_tok_s": point.output_throughput_tok_s,
        "mean_ttft_ms": point.mean_ttft_ms,
        "mean_tpot_ms": point.mean_tpot_ms,
        "mean_e2el_ms": point.mean_e2el_ms,
        "source": "geak",
    }
    if point.error:
        report["error"] = point.error
    try:
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # Best-effort: a failed write must never break the sweep.
        log.warning("geak_sweep: could not write %s: %s", out_dir / "benchmark_report.json", exc)


def _parse_isl_osl(spec: str) -> tuple[int, int]:
    isl_s, _, osl_s = str(spec).partition(":")
    return int(isl_s or 1024), int(osl_s or 1024)


def _geak_replay_server_log(out_dir: Path) -> str | None:
    """Return the newest server log created inside this replay's output only."""
    try:
        existing = _replay_server_logs(out_dir)
        return max(existing, key=lambda path: Path(path).stat().st_mtime) if existing else None
    except OSError:
        # Replay-log discovery is best effort.
        logging.debug("Unable to enumerate GEAK replay server logs", exc_info=True)
        return None


def _replay_server_logs(out_dir: Path) -> list[str]:
    """Every server log this replay has written so far, oldest first: one per replica boot."""
    candidates = [out_dir / "server.log", *sorted(out_dir.glob("replica_*/attempt_*/server.log"))]
    return [str(path) for path in candidates if path.is_file()]


def _run_with_power_sampling(
    run: Callable[[], subprocess.CompletedProcess], out_dir: Path, env: Mapping[str, str], entry: dict[str, Any]
) -> subprocess.CompletedProcess:
    """Run one replay under the power recorder and record its measured power into ``entry``.

    Both power fields stay ``None`` when sampling is off, nothing was measured, or the replay raised.
    """
    entry.update(gpu_power_avg_w=None, gpu_power_by_gpu_w=None)
    proc = _run_sampled(run, out_dir, env)
    entry["gpu_power_avg_w"] = read_measured_gpu_power(out_dir)[1]
    entry["gpu_power_by_gpu_w"] = read_measured_gpu_power_by_gpu(out_dir)[1]
    return proc


def _run_sampled(
    run: Callable[[], subprocess.CompletedProcess], out_dir: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess:
    """Run one replay with the measured-phase power recorder a native round gets.

    The replay boots its own servers, so the watchdog's phase marks are not available; the driver reads the same
    ready / warmup / measured / eval markers from the logs the replay writes. Writes ``gpu_power.json`` into
    ``out_dir`` whenever sampling is on.
    """
    recorder = build_gpu_power_recorder(str(out_dir / "server.log"), dict(env))
    if recorder is None:
        return run()
    driver = ServerLogPhaseDriver(recorder, lambda: _replay_server_logs(out_dir), cursor=_LogCursor())
    driver.start()
    try:
        return run()
    finally:
        try:
            driver.stop()
        finally:
            recorder.close()


def _replay_recipe_benchmark(handoff: Mapping[str, Any], env_spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``benchmark`` block of the launch recipe this replay inherits, empty when it names none."""
    recipe_path = str(handoff.get("launch_recipe") or env_spec.get("base_launch_recipe") or "")
    if not recipe_path:
        return {}
    parsed = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("benchmark"), dict):
        raise ValueError("invalid_replay_recipe")
    return parsed["benchmark"]


def _replay_logical_masks(pin: Mapping[str, Any], gpu_ids: str) -> dict[str, str]:
    """The ROCr-level mask that makes the handoff's logical GPU ids mean what they meant at handoff time."""
    var = str(pin.get("var") or "")
    tokens = effective_mask_tokens(pin.get("value"))
    if not is_rocr_level(var) or not tokens:
        raise ValueError("missing_replay_logical_gpu_pin")
    if any(not token.isdigit() or int(token) >= len(tokens) for token in effective_mask_tokens(gpu_ids)):
        raise ValueError("invalid_replay_logical_gpu_ids")
    return {var: ",".join(tokens)}


def _replay_device_env(handoff: Mapping[str, Any], pin: Mapping[str, Any], tp: int) -> dict[str, str]:
    """The visible-device variables a replay runs under, in the GPU-id space the handoff recorded."""
    gpu_ids = ",".join(effective_mask_tokens(handoff.get("gpu_ids"))) or _resolve_handoff_gpu_ids(gpu_pin=pin, tp=tp)
    space = str(handoff.get("gpu_ids_space") or _resolve_handoff_gpu_ids_space(gpu_pin=pin))
    if space not in {"absolute", "logical"} or _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "none":
        raise ValueError("invalid_replay_gpu_ids_space")
    if len(effective_mask_tokens(gpu_ids)) < tp:
        raise ValueError("replay_gpu_count_below_tp")
    masks = _replay_logical_masks(pin, gpu_ids) if space == "logical" else {}
    return {
        "GPU": gpu_ids,
        **masks,
        "HIP_VISIBLE_DEVICES": gpu_ids,
        "CUDA_VISIBLE_DEVICES": gpu_ids,
    }


def _replay_serving_env(handoff: Mapping[str, Any], env_spec: Mapping[str, Any]) -> dict[str, str]:
    """Resolve serving identity from the frozen handoff or its materialized recipe."""
    benchmark = _replay_recipe_benchmark(handoff, env_spec)
    recipe_envs = benchmark.get("envs") or {}
    if not isinstance(recipe_envs, dict):
        raise ValueError("invalid_replay_recipe_envs")
    legacy_env = {} if handoff or benchmark else os.environ
    tp = coerce_tp(handoff.get("tp"), recipe_envs.get("TP"), legacy_env.get("TP"))
    pin = handoff.get("gpu_pin")
    if not isinstance(pin, dict):
        pin = resolve_gpu_pin(recipe_envs=recipe_envs, environ=legacy_env)
    return {
        "MODEL": str(handoff.get("model_path") or benchmark.get("model") or legacy_env.get("MODEL_PATH") or "").strip(),
        "BACKEND": str(
            handoff.get("framework") or benchmark.get("framework") or legacy_env.get("FRAMEWORK") or "sglang"
        ).strip(),
        "TP": str(tp),
        **_replay_device_env(handoff, pin, tp),
    }


def _replay_point_failure(
    proc: subprocess.CompletedProcess, tput: object, basis: str, requested_basis: str
) -> dict[str, str] | None:
    """The fields of a replayed point that does not count, or ``None`` when it measured the requested basis."""
    measured = proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0
    if not measured:
        return {"status": "failed", "error": (proc.stderr or "")[-500:] or "no throughput"}
    if basis != requested_basis:
        error = f"measured {basis}, not the requested {requested_basis}"
        return {"status": "failed", "error": error, "metric_basis": basis}
    return None


class _ReplayUnavailable(Exception):
    """The GEAK sweep cannot be replayed at all; carries the failure the sweep publishes."""

    def __init__(self, error_class: str, error: str) -> None:
        super().__init__(error)
        self.failure: dict[str, Any] = {"status": "failed", "error_class": error_class, "error": error}


class _ReplayScript(NamedTuple):
    """The script a sweep point replays, and whether it is GEAK's deployed final launcher."""

    path: Path
    is_final_launch: bool

    @property
    def mode(self) -> str:
        """The ``replay_mode`` the sweep publishes for every point it runs."""
        return "final_launch_script" if self.is_final_launch else "bench_e2e_fallback"


@dataclass(frozen=True)
class _ReplayPlan:
    """Everything a GEAK sweep replays with, resolved once before its first point runs."""

    script: _ReplayScript
    flags: str
    accepted_env: dict[str, str]
    env_str: str
    overlay: str
    serving_env: dict[str, str]
    bench_client: str
    e2e_metric: str
    requested_basis: str
    protocol_env: dict[str, str]


class _SweepGrid(NamedTuple):
    """The CONC x (ISL, OSL) grid to replay and how each of its points is run."""

    conc_values: list[int]
    isl_osl_configs: list[str]
    repeats: int
    timeout_sec: int


@dataclass(frozen=True)
class _SweepPoint:
    """One (conc, isl, osl) point of the grid, with the workspace it replays in."""

    conc: int
    isl: int
    osl: int
    variant_name: str
    out_dir: Path
    repeats: int
    timeout_sec: int


@dataclass(frozen=True)
class _PointSummary:
    """The medians GEAK's ``bench_summary.json`` records for one replayed point."""

    throughput: Any
    basis: str
    ttft_ms: Any
    tpot_ms: Any
    e2el_ms: Any

    @property
    def on_output_axis(self) -> bool:
        """A number was measured, and on GEAK's output axis rather than another one."""
        return isinstance(self.throughput, (int, float)) and self.basis == GEAK_METRIC_OUTPUT[1]

    def as_report(self, point: _SweepPoint, *, success: bool, error: str | None) -> _GeakPointReport:
        """The session-breakdown-compatible report row this point writes beside its artifacts."""
        return _GeakPointReport(
            conc=point.conc,
            isl=point.isl,
            osl=point.osl,
            success=success,
            output_throughput_tok_s=self.throughput if self.on_output_axis else None,
            mean_ttft_ms=_measured_number(self.ttft_ms),
            mean_tpot_ms=_measured_number(self.tpot_ms),
            mean_e2el_ms=_measured_number(self.e2el_ms),
            error=error,
        )


_UNMEASURED = _PointSummary(throughput=None, basis="", ttft_ms=None, tpot_ms=None, e2el_ms=None)


@dataclass(frozen=True)
class _PointMeasurement:
    """What one replay of a sweep point produced: its entry fields and the summary behind them."""

    fields: dict[str, Any]
    summary: _PointSummary
    succeeded: bool
    error: str | None


def _measured_number(value: object) -> float | int | None:
    """*value* when GEAK recorded a number for it, else ``None``."""
    return value if isinstance(value, (int, float)) else None


def _requires_deployment(result: Mapping[str, Any]) -> bool:
    """GEAK's accepted tuning writes files or invalidates caches, so its replay needs the deployed launcher."""
    tuning = result.get("tuning_skillset")
    return (
        isinstance(tuning, dict)
        and tuning.get("gate") == "accepted"
        and bool(tuning.get("live_tree_files") or tuning.get("cache_invalidation"))
    )


def _executable_final_launcher(result: Mapping[str, Any]) -> Path | None:
    """GEAK's final launch script when it exists and can be executed, else ``None``."""
    path = Path(str(result.get("final_launch_script") or "").strip())
    if not path.name or not path.is_file() or not os.access(path, os.X_OK):
        return None
    return path


def _bench_script_fallback(result: Mapping[str, Any]) -> _ReplayScript:
    """The bench script the sweep replays when GEAK deployed no executable launcher."""
    bench_script = result.get("bench_script") or result.get("geak_bench_script")
    fallback = Path(str(bench_script or ""))
    if not fallback.is_file():
        final_launch_script = str(result.get("final_launch_script") or "").strip()
        raise _ReplayUnavailable(
            "missing_bench_script",
            (
                f"GEAK final launch script is not executable ({final_launch_script}); "
                f"bench script not found: {bench_script}"
            ),
        )
    return _ReplayScript(fallback, False)


def _replay_protocol_env(result: Mapping[str, Any], flags: str, *, pin_num_prompts: bool) -> dict[str, str]:
    """Forward the validated measurement config and client trust onto every variant of the sweep.

    Without it ``bench_e2e.sh`` falls back to its own defaults and the sweep stops measuring the workload
    shape the KERNEL_AGENT phase accepted. A single-point validated replay also pins ``num_prompts``.
    """
    protocol = result.get("bench_protocol")
    if not isinstance(protocol, dict):
        regimes = result.get("validated_regimes") or []
        protocol = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
    forwarded = [
        ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        ("num_warmups", "NUM_WARMUPS"),
        ("seed", "SEED"),
    ]
    if pin_num_prompts:
        forwarded.append(("num_prompts", "NUM_PROMPTS"))
    env = {name: str(protocol[key]) for key, name in forwarded if protocol.get(key) is not None}
    # Mirror the server's --trust-remote-code onto the bench client so its tokenizer load doesn't raise.
    if "trust-remote-code" in flags or "trust_remote_code" in flags:
        for name in ("BENCH_TRUST_REMOTE_CODE", "HF_HUB_TRUST_REMOTE_CODE", "MAGPIE_TRUST_REMOTE_CODE"):
            env.setdefault(name, "1")
    return env


def _build_replay_plan(
    result: dict[str, Any],
    handoff: Mapping[str, Any],
    env_spec: Mapping[str, Any],
    metric_axis: tuple[str, str],
    *,
    pin_num_prompts: bool,
) -> _ReplayPlan:
    """Resolve the script, serving identity and benchmark configuration every replayed point shares.

    The order the checks run in is the order they claim a failure: a deployment GEAK cannot launch is
    reported ahead of a configuration the sweep cannot read, and that ahead of a missing bench script.
    """
    launcher = _executable_final_launcher(result)
    if launcher is None and _requires_deployment(result):
        raise _ReplayUnavailable(
            "missing_deployment_launcher",
            "GEAK accepted tuning requires file deployment, but its final launch script is unavailable.",
        )
    try:
        flags, accepted_env = _accepted_config_as_variant(result.get("accepted_config"))
    except ValueError as exc:
        raise _ReplayUnavailable("invalid_accepted_config", str(exc)) from exc
    script = _ReplayScript(launcher, True) if launcher is not None else _bench_script_fallback(result)
    try:
        serving_env = _replay_serving_env(handoff, handoff.get("baseline_env_spec") or env_spec)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise _ReplayUnavailable("invalid_replay_identity", str(exc)) from exc
    e2e_metric, requested_basis = metric_axis
    return _ReplayPlan(
        script=script,
        flags=flags,
        accepted_env=accepted_env,
        env_str=shlex.join(f"{name}={value}" for name, value in accepted_env.items()),
        overlay=result.get("final_overlay") or "",
        serving_env=serving_env,
        bench_client=str(result.get("bench_client") or "native").strip() or "native",
        e2e_metric=e2e_metric,
        requested_basis=requested_basis,
        protocol_env=_replay_protocol_env(result, flags, pin_num_prompts=pin_num_prompts),
    )


def _plan_sweep_points(grid: _SweepGrid, output_root: Path) -> Iterator[_SweepPoint]:
    """Lay out the points of the grid one at a time, named as the sweep collector's scanner expects.

    The name must stay ``variant_<idx>_conc<c>_isl<i>_osl<o>`` or the collector never discovers the point.
    Laid out lazily, so a shape spec that cannot be parsed stops the sweep where the grid reaches it rather
    than before the points ahead of it have run.
    """
    shapes = [(conc, spec) for conc in grid.conc_values for spec in grid.isl_osl_configs]
    for index, (conc, spec) in enumerate(shapes):
        isl, osl = _parse_isl_osl(spec)
        variant_name = f"variant_{index}_conc{conc}_isl{isl}_osl{osl}"
        yield _SweepPoint(
            conc=conc,
            isl=isl,
            osl=osl,
            variant_name=variant_name,
            out_dir=output_root / variant_name,
            repeats=grid.repeats,
            timeout_sec=grid.timeout_sec,
        )


def _sweep_point_env(plan: _ReplayPlan, point: _SweepPoint) -> dict[str, str]:
    """The benchmark environment one replayed point runs under."""
    env = build_benchmark_env(
        {
            "OUT_DIR": str(point.out_dir),
            **plan.serving_env,
            "ISL": str(point.isl),
            "OSL": str(point.osl),
            "CONC": str(point.conc),
            "REPEATS": str(point.repeats),
            "PROFILE": "0",
            "OVERLAY_PYTHONPATH": plan.overlay,
            "EXTRA_SERVER_ARGS": plan.flags,
            "EXTRA_ENV": plan.env_str,
            "BENCH_CLIENT": plan.bench_client,
            "E2E_METRIC": plan.e2e_metric,
        }
    )
    for name in VISIBLE_DEVICE_VARS:
        env.pop(name, None)
    env.update(plan.serving_env)
    # setdefault: forwarded config/trust apply unless already pinned.
    for name, value in plan.protocol_env.items():
        env.setdefault(name, value)
    if plan.script.is_final_launch:
        env["REPLICAS"] = str(point.repeats)
    return env


def _replay_command(plan: _ReplayPlan, point: _SweepPoint) -> list[str]:
    """The shell invocation that replays one point; a final launch script takes its output directory."""
    tail = [str(point.out_dir)] if plan.script.is_final_launch else []
    return ["bash", str(plan.script.path), *tail]


def _point_summary(out_dir: Path) -> _PointSummary:
    """Read what one replayed point measured from the summary GEAK wrote for it.

    ``throughput_tok_s_median`` is the metric-neutral median of whatever basis GEAK measured, and the only
    field populated on every axis: GEAK nulls the output-named alias off the output axis precisely so nobody
    reads another axis under an "output" name. On output the two are the same number, so synthetic sweeps stay
    byte-identical. The output-named field stays as the fallback for summaries written before the neutral one
    existed, and a summary that names no basis predates the field and was measured on output.
    """
    summary = read_json(out_dir / "bench_summary.json", default={}, require_dict=True)
    throughput = summary.get("throughput_tok_s_median")
    if throughput is None:
        throughput = summary.get("output_throughput_tok_s_median")
    return _PointSummary(
        throughput=throughput,
        basis=str(summary.get("metric_basis") or GEAK_METRIC_OUTPUT[1]),
        ttft_ms=summary.get("ttft_ms_median"),
        tpot_ms=summary.get("tpot_ms_median"),
        e2el_ms=summary.get("e2el_ms_median"),
    )


def _measured_point_fields(plan: _ReplayPlan, out_dir: Path, summary: _PointSummary) -> dict[str, Any]:
    """The entry fields a point that measured the requested basis publishes."""
    evaluation = parse_eval_results(out_dir, framework=plan.serving_env["BACKEND"])
    return {
        "status": "succeeded",
        "measured_value": summary.throughput,
        "metric_basis": summary.basis,
        # ``output_throughput`` is GRADED_OUTPUT, which the perf snapshot reads, so a number on
        # any other axis published under it would be recorded as the session's output throughput.
        **({"output_throughput": summary.throughput} if summary.on_output_axis else {}),
        "ttft_mean_ms": summary.ttft_ms,
        "tpot_mean_ms": summary.tpot_ms,
        "accuracy": evaluation.get("accuracy"),
        "accuracy_source": evaluation.get("source_file"),
    }


async def _measure_sweep_point(plan: _ReplayPlan, point: _SweepPoint, env: Mapping[str, str]) -> _PointMeasurement:
    """Replay one sweep point under the power recorder and read the outcome it published.

    Anything the point raises -- the replay, the summary it wrote, or the accuracy result beside it -- is
    recorded on that point and goes no further, so one unreadable artifact costs its own point and not the
    rest of the grid. Whatever was already read stays on the report the point writes.
    """
    cmd = _replay_command(plan, point)

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=point.timeout_sec,
            env=env,
            cwd=str(point.out_dir),
        )

    power: dict[str, Any] = {}
    summary = _UNMEASURED
    on_requested_basis = False
    try:
        proc = await asyncio.to_thread(_run_with_power_sampling, _run, point.out_dir, env, power)
        summary = _point_summary(point.out_dir)
        failure = _replay_point_failure(proc, summary.throughput, summary.basis, plan.requested_basis)
        if failure is not None:
            return _PointMeasurement({**power, **failure}, summary, False, failure["error"])
        on_requested_basis = True
        measured = _measured_point_fields(plan, point.out_dir, summary)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        return _PointMeasurement({**power, "status": "failed", "error": error}, summary, on_requested_basis, error)
    return _PointMeasurement({**power, **measured}, summary, True, None)


def _launch_evidence_fields(plan: _ReplayPlan, out_dir: Path) -> dict[str, Any]:
    """The launch evidence one replayed point publishes, persisted beside its artifacts."""
    actual_log = _geak_replay_server_log(out_dir)
    evidence = build_launch_evidence(
        config_path=plan.script.path,
        actual_server_log=actual_log,
        framework=plan.serving_env["BACKEND"],
        slot=out_dir,
        requested_server_args=plan.flags,
        requested_server_env=plan.accepted_env,
        model_path=plan.serving_env["MODEL"],
    )
    return {
        "server_log_path": actual_log or "",
        "launch_evidence": evidence,
        "launch_evidence_path": persist_launch_evidence(evidence, slot=out_dir),
    }


async def _replay_sweep_point(plan: _ReplayPlan, point: _SweepPoint) -> dict[str, Any]:
    """Replay one sweep point and return the entry the sweep collector reads for it."""
    point.out_dir.mkdir(parents=True, exist_ok=True)
    measurement = await _measure_sweep_point(plan, point, _sweep_point_env(plan, point))
    entry: dict[str, Any] = {
        "conc": point.conc,
        "isl": point.isl,
        "osl": point.osl,
        "variant_name": point.variant_name,
        "workspace": str(point.out_dir),
        "replay_script": str(plan.script.path),
        "replay_mode": plan.script.mode,
        **measurement.fields,
        **_launch_evidence_fields(plan, point.out_dir),
    }
    report = measurement.summary.as_report(point, success=measurement.succeeded, error=measurement.error)
    _write_benchmark_report(point.out_dir, report)
    return entry


def _sweep_outcome(entries: list[dict[str, Any]], plan: _ReplayPlan, output_root: Path) -> dict[str, Any]:
    """Rank the replayed points and publish the sweep's headline measurement.

    The replay runs one (conc, isl, osl) repeated, so the best succeeded point is the headline. Every succeeded
    point was measured on the requested basis, and every axis GEAK measures is higher-is-better.
    """
    succeeded = [e for e in entries if e["status"] == "succeeded"]
    measured = [e for e in succeeded if isinstance(e.get("measured_value"), (int, float))]
    return {
        "status": "succeeded" if succeeded else "failed",
        **({"error": str(entries[0].get("error") or "replay_failed")} if entries and not succeeded else {}),
        "grid_size": len(entries),
        "points": entries,
        "workspace": output_root.as_posix(),
        "source": "geak",
        "replay_mode": plan.script.mode,
        "replay_script": str(plan.script.path),
        "promotion_measurement": max(measured, key=lambda e: e["measured_value"], default={}),
    }


async def sweep_via_geak(
    *,
    result: dict[str, Any],
    conc_values: list[int],
    isl_osl_configs: list[str],
    output_root: Path,
    variant_timeout_sec: int,
    repeats: int = 3,
    pin_num_prompts: bool = False,
    handoff: Mapping[str, Any] | None = None,
    env_spec: Mapping[str, Any] | None = None,
    metric_axis: tuple[str, str] = GEAK_METRIC_OUTPUT,
) -> dict[str, Any]:
    """Run a CONC × (ISL, OSL) sweep on the GEAK-optimized server.

    ``metric_axis`` is the ``(E2E_METRIC, metric_basis)`` pair from ``geak_metric_axis``. Every point is measured
    on it, and a point whose summary records another basis fails: a GEAK that cannot measure the requested axis
    and falls back to another would otherwise have that number rank the points.
    """
    try:
        plan = _build_replay_plan(result, handoff or {}, env_spec or {}, metric_axis, pin_num_prompts=pin_num_prompts)
    except _ReplayUnavailable as exc:
        return exc.failure
    output_root.mkdir(parents=True, exist_ok=True)
    grid = _SweepGrid(conc_values, isl_osl_configs, repeats, variant_timeout_sec)
    entries = [await _replay_sweep_point(plan, point) for point in _plan_sweep_points(grid, output_root)]
    return _sweep_outcome(entries, plan, output_root)
